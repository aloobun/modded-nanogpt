import os
import sys
import wandb
import argparse
import uuid
import glob
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon Optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2, f"zeropower input must be a 2D matrix, but got shape {G.shape}"
    a, b, c = (3.4445, -4.7750,  2.0315)
    # Use bfloat16 for consistency with the amp autocast context
    X = G.to(torch.bfloat16)
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
    def step(self):
        for group in self.param_groups:
            lr, momentum, zeropower_backend = group['lr'], group['momentum'], zeropower_backends[group['backend']]
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    if (g := p.grad) is None: continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']: g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


# -----------------------------------------------------------------------------
# Llama

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5): super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(); self.dim = dim; self.max_position_embeddings = max_position_embeddings; self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len; t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq); emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False); self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached: self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

def rotate_half(x): x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]; return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # One bug in attention was shape mismatch when applying positional embeddings. The query/key tensors have shape [B, H, T, D], but the
    # initial rotary embedding tensor was missing the batch and head dimensions and had to explicitly unsqueeze the cos/sin tensors to add these missing dimensions.
    cos = cos[position_ids].unsqueeze(0).unsqueeze(1); sin = sin[position_ids].unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin); k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); self.n_head = config.n_head; self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd; self.head_dim = self.n_embd // self.n_head; self.n_rep = self.n_head // self.n_kv_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False); self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False); self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings=config.sequence_length)
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        B, T, C = x.size(); q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2); v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(v, seq_len=T); q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        k = repeat_kv(k, self.n_rep); v = repeat_kv(v, self.n_rep)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True); y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module): # Llama's SwiGLU MLP
    def __init__(self, config):
        super().__init__(); self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False); self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__(); self.input_layernorm = LlamaRMSNorm(config.n_embd); self.attn = CausalSelfAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd); self.mlp = MLP(config)
    def forward(self, x, position_ids):
        x = x + self.attn(self.input_layernorm(x), position_ids); x = x + self.mlp(self.post_attention_layernorm(x))
        return x

from dataclasses import dataclass
@dataclass
class GPTConfig:
    vocab_size: int = 128256; sequence_length: int = 1024; n_layer: int = 8; n_head: int = 8
    n_embd: int = 512; n_kv_head: int = 4; intermediate_size: int = 1408

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(config.vocab_size, config.n_embd), h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), norm=LlamaRMSNorm(config.n_embd)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False); self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx, targets=None, return_logits=True):
        B, T = idx.size(); position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device); x = self.transformer.wte(idx)
        for block in self.transformer.h: x = block(x, position_ids)
        x = self.transformer.norm(x)
        if targets is not None: logits = self.lm_head(x); loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: logits = self.lm_head(x[:, [-1], :]); loss = None
        if not return_logits: logits = None
        return logits, loss
    def get_num_params(self): n_params = sum(p.numel() for p in self.parameters()); n_params -= self.lm_head.weight.numel(); return n_params

# -----------------------------------------------------------------------------
# Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f: header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520: raise ValueError("Magic number mismatch")
    if header[1] != 2: raise ValueError("Unsupported version")
    return header[2]

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32); assert header[0] == 20240520 and header[1] == 2; ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.int32)
    assert len(tokens) == ntok; return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank; self.num_processes = num_processes; self.B = B; self.T = T
        self.files = sorted(glob.glob(filename_pattern)); assert len(self.files) > 0, f"No files found: {filename_pattern}"
        self.ntok_total = sum(_peek_data_shard(fname) for fname in self.files); self.reset()
    def reset(self): self.current_shard = 0; self.tokens = _load_data_shard(self.files[self.current_shard]); self.current_position = self.process_rank * self.B * self.T
    def advance(self): self.current_shard = (self.current_shard + 1) % len(self.files); self.tokens = _load_data_shard(self.files[self.current_shard]); self.current_position = self.process_rank * self.B * self.T
    def next_batch(self):
        B, T = self.B, self.T; buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T); y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens): self.advance()
        return x.cuda(), y.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Llama-style model training with W&B Sweep")
    parser.add_argument('--embed_learning_rate', type=float, default=3e-4)
    parser.add_argument('--muon_learning_rate', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--warmup_iters', type=int, default=30)
    parser.add_argument('--input_bin', type=str, default='./token1b/train.bin')
    parser.add_argument('--input_val_bin', type=str, default='./token1b/val.bin')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device_batch_size', type=int, default=8)
    parser.add_argument('--sequence_length', type=int, default=1024)
    parser.add_argument('--num_iterations', type=int, default=300)
    parser.add_argument('--warmdown_iters', type=int, default=1450)
    parser.add_argument('--val_loss_every', type=int, default=125)
    parser.add_argument('--val_tokens', type=int, default=10_000_000)
    parser.add_argument('--wandb_project', type=str, default='muon-llama-sweep')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    ddp_rank, ddp_local_rank, ddp_world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'; torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)

    if master_process:
        wandb.init(project=args.wandb_project, config=args)
    
    B, T = args.device_batch_size, args.sequence_length
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)
    tokens_per_val_step = B * T * ddp_world_size
    val_steps = args.val_tokens // tokens_per_val_step

    model_config = GPTConfig(sequence_length=args.sequence_length)
    model = GPT(model_config).cuda()
    if master_process: print(f"Model initialized: ~{model.get_num_params()/1e6:.2f}M parameters")
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    adam_params, muon_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad: continue
        if 'transformer.h' in name and p.ndim == 2: muon_params.append(p)
        else: adam_params.append(p)

    optimizer1 = torch.optim.AdamW(adam_params, lr=args.embed_learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
    optimizer2 = Muon(muon_params, lr=args.muon_learning_rate, momentum=args.muon_momentum, nesterov=True, rank=ddp_rank, world_size=ddp_world_size)
    optimizers = [optimizer1, optimizer2]

    def get_lr(it):
        if it < args.warmup_iters: return (it + 1) / args.warmup_iters
        if it > args.num_iterations - args.warmdown_iters: return (args.num_iterations - it) / args.warmdown_iters
        return 1.0
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    train_loader.reset()
    x, y = train_loader.next_batch()
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        if (last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)):
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    with ctx: _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss.item():.4f}')
                wandb.log({"val_loss": val_loss.item()}, step=step)

        if last_step: break

        model.train()
        total_loss = 0.0
        for i in range(train_accumulation_steps):
            with ctx: _, loss = model(x, y, return_logits=False); loss = loss / train_accumulation_steps; total_loss += loss.detach()
            x, y = train_loader.next_batch()
            if i < train_accumulation_steps - 1:
                with model.no_sync(): loss.backward()
            else: loss.backward()
        
        for opt in optimizers: opt.step()
        for sched in schedulers: sched.step()
        model.zero_grad(set_to_none=True)

        if master_process:
            current_lr_muon = schedulers[1].get_last_lr()[0]
            current_lr_adam = schedulers[0].get_last_lr()[0]
            print(f"step:{step+1}/{args.num_iterations} loss:{total_loss.item():.4f} lr:{current_lr_muon:.6f}")
            wandb.log({ "train_loss": total_loss.item(), "lr_muon": current_lr_muon, "lr_adam": current_lr_adam }, step=step+1)

    if master_process:
        print(f"Peak memory consumption: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")
        wandb.finish()

    dist.destroy_process_group()
