import os
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from transformers import logging as hf_logging

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.runtime as xr
import torch_xla.test.test_utils as test_utils
from torch_xla.utils.checkpoint import checkpoint
import torch_xla.amp

from torch_xla.amp.syncfree import AdamW
from transformers.optimization import Adafactor

import wandb

#hf_logging.set_verbosity_error()
xr.use_spmd()
try:
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ.pop('CLOUD_TPU_TASK_ID', None)
    
    args = " --xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_enable_latency_hiding_scheduler=true --xla_tpu_enable_flash_attention=false --xla_tpu_enable_async_collective_fusion=false --xla_tpu_overlap_compute_collective_tc=false --xla_tpu_use_enhanced_launch_barrier=false"
    
    os.environ['LIBTPU_INIT_ARGS'] = args
    os.environ['XLA_FLAGS'] = "--xla_cpu_enable_fast_math=true --xla_gpu_force_compilation_parallelism=4"
    os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "17179869184"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    os.environ["XLA_PYTHON_CLIENT_MEM_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
except Exception as e:
    print(f"Could not set environment variables: {e}")

MAX_INPUT = 1024
PRETRAIN_DATASET = "JeanKaddour/minipile"
TOKENIZER_PATH = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "Llama-3-350M-Custom-Pretrain"

FLAGS = {
    'MAX_INPUT': MAX_INPUT, 
    'LOGGING_STEPS': 50, 
    'NUM_EPOCHS': 1, 
    'PAUSE_STEPS': 20000,
    'BATCH_SIZE': 8,
    'GRAD_ACCUMULATION_STEP': 8,
    'VAL_STEPS': 500,
    'EVAL_STEPS': 250,
    'VAL_BATCH': 4,
    'MAX_GRAD_CLIP': 0.5,
    'LEARNING_RATE': 3e-4,
    'WARMUP_RATIO': 0.01,
    'OPTIMIZER': 'adamw', 
    'SCHEDULAR': 'cosine', 
    'WEIGHT_DECAY': 0.1,
    'BETA1': 0.9,
    'BETA2': 0.95,
    'EPS': 1e-8,
    'TRAIN_DATASET': PRETRAIN_DATASET, 
    'WANDB': True, 
    'PROJECT': 'Llama3-350M-RoPE-Exp',
    'ATTENTION_TYPE': 'qk_rope',
}

ROPE_CONFIGS = {
    "nope":      {"apply_rope_to_q": False, "apply_rope_to_k": False, "apply_rope_to_v": False, "apply_inverse_rope_to_o": False},
    "qk_rope":   {"apply_rope_to_q": True,  "apply_rope_to_k": True,  "apply_rope_to_v": False, "apply_inverse_rope_to_o": False},
    "q_rope":    {"apply_rope_to_q": True,  "apply_rope_to_k": False, "apply_rope_to_v": False, "apply_inverse_rope_to_o": False},
    "k_rope":    {"apply_rope_to_q": False, "apply_rope_to_k": True,  "apply_rope_to_v": False, "apply_inverse_rope_to_o": False},
    "v_rope":    {"apply_rope_to_q": False, "apply_rope_to_k": False, "apply_rope_to_v": True,  "apply_inverse_rope_to_o": False},
    "o_rope":    {"apply_rope_to_q": False, "apply_rope_to_k": False, "apply_rope_to_v": False, "apply_inverse_rope_to_o": True},
    "qkv_rope":  {"apply_rope_to_q": True,  "apply_rope_to_k": True,  "apply_rope_to_v": True,  "apply_inverse_rope_to_o": False},
    "vo_rope":   {"apply_rope_to_q": True,  "apply_rope_to_k": True,  "apply_rope_to_v": True,  "apply_inverse_rope_to_o": True},
    "qkvo_rope": {"apply_rope_to_q": True,  "apply_rope_to_k": True,  "apply_rope_to_v": True,  "apply_inverse_rope_to_o": True},
    "default":   {}
}


num_devices = xr.global_runtime_device_count()
if num_devices >= 8:
    mesh_shape = (8, 1)
elif num_devices >= 4:
    mesh_shape = (4, 1)
else:
    mesh_shape = (num_devices, 1)

mesh = xs.Mesh(np.array(range(min(num_devices, mesh_shape[0]))), mesh_shape, ('fsdp', 'mp'))
xs.set_global_mesh(mesh)
device = xm.xla_device()

model_config = LlamaConfig(
    vocab_size=128256, 
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4, 
    hidden_act="silu", 
    max_position_embeddings=MAX_INPUT,
    initializer_range=0.02, 
    rms_norm_eps=1e-5, 
    use_cache=False, 
    rope_theta=500000.0,
    tie_word_embeddings=False,
)

model = AutoModelForCausalLM.from_config(model_config)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if 'pad_token' not in tokenizer.special_tokens_map: 
    tokenizer.pad_token = tokenizer.eos_token


class MinDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, text_field="text", max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field] + self.eos_token
        
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors=None
        )
        
        input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def build_rope_matrix(dim, seq_len, device=None, theta=10000.0):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = pos * freqs
    return torch.cat((torch.cos(t), torch.sin(t)), dim=-1)

def apply_rope(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = rope[..., ::2], rope[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

## https://kexue.fm/archives/10862
# https://github.com/kyegomez/VO-ROPE

def apply_inverse_rope(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = rope[..., ::2], -rope[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

class RoPEAttn(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_theta=10000.0,
                 apply_rope_to_q=True, apply_rope_to_k=True, apply_rope_to_v=False,
                 apply_inverse_rope_to_o=False, dtype=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rope_theta = rope_theta
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        
        self.apply_q = apply_rope_to_q
        self.apply_k = apply_rope_to_k
        self.apply_v = apply_rope_to_v
        self.apply_o = apply_inverse_rope_to_o
        
        # Use float32 for better stability
        self.q_proj = nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False, dtype=torch.float32)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False, dtype=torch.float32)
        self.o_proj = nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        
    def _repeat_kv(self, x, n_rep):
        if n_rep == 1:
            return x
        B, H, S, D = x.shape
        return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
        
        if self.apply_q or self.apply_k or self.apply_v:
            rope = build_rope_matrix(
                self.head_dim, N, 
                device=hidden_states.device, 
                theta=self.rope_theta
            ).unsqueeze(0).unsqueeze(2)
            
            if self.apply_q:
                q = apply_rope(q, rope)
            if self.apply_k:
                k = apply_rope(k, rope)
            if self.apply_v:
                v = apply_rope(v, rope)
        
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, N, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, N, head_dim)
        
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_probs, v)
        
        # inverse RoPE if configured !?!? **
        if self.apply_o and (self.apply_q or self.apply_k or self.apply_v):
            out = apply_inverse_rope(out.transpose(1, 2), rope).transpose(1, 2)
        
        out = self.o_proj(out.transpose(1, 2).contiguous().view(B, N, self.dim))
        return out, None

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"trainable params: {trainable_params:,} || all params: {all_params:,}")

def partition_module(model, mesh):
    for name, param in model.named_parameters():
        if param.ndim <= 1:
            # don't shard 1D tensors
            continue
        
        if any(key in name for key in ['embed_tokens', 'lm_head', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            xs.mark_sharding(param, mesh, ('fsdp', None))

def evaluate_loss(outputs, labels, pad_id):
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        labels.view(-1), 
        ignore_index=pad_id
    )

def train_step(batch, model, pad_id):
    with torch_xla.amp.autocast(device=device):
        outputs = model(**batch)
        return evaluate_loss(outputs, batch["labels"], pad_id)

def evaluate_model(model, val_loader, pad_id):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Mark sharding for batch tensors
            for k in batch:
                if torch.is_tensor(batch[k]):
                    xs.mark_sharding(batch[k], mesh, ('fsdp', None))
            
            outputs = model(**batch)
            loss = evaluate_loss(outputs, batch["labels"], pad_id)
            total_loss += loss.item()
            total_steps += 1
    
    model.train()
    return total_loss / total_steps if total_steps > 0 else 0

# https://github.com/IsNoobgrammer/XLA-Trainer

def train(FLAGS):
    if FLAGS['WANDB'] and xm.is_master_ordinal(): 
        wandb.init(project=FLAGS['PROJECT'], config=FLAGS, name=f"run-{FLAGS['ATTENTION_TYPE']}")
    
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_iter = (FLAGS['NUM_EPOCHS'] * FLAGS['LEN_TRAIN_DATA'] // FLAGS['BATCH_SIZE']) // FLAGS['GRAD_ACCUMULATION_STEP']
    
    optimizer = AdamW(
        params, 
        lr=FLAGS['LEARNING_RATE'], 
        weight_decay=FLAGS['WEIGHT_DECAY'],
        betas=(FLAGS['BETA1'], FLAGS['BETA2']),
        eps=FLAGS['EPS']
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        int(num_iter * FLAGS['WARMUP_RATIO']), 
        num_iter
    )
    
    total_steps = 0
    
    for epoch in range(1, FLAGS['NUM_EPOCHS'] + 1):
        model.train()
        xm.master_print(f'Epoch {epoch} | Train begin @ {test_utils.now()}')
        
        for step, batch in enumerate(training_loader):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    xs.mark_sharding(batch[k], mesh, ('fsdp', None))
            
            loss = train_step(batch, model, tokenizer.pad_token_id) / FLAGS['GRAD_ACCUMULATION_STEP']
            
            loss.backward()
            
            if (step + 1) % FLAGS['GRAD_ACCUMULATION_STEP'] == 0:
                total_steps += 1
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS['MAX_GRAD_CLIP'])
                
                xm.reduce_gradients(optimizer)
                xm.optimizer_step(optimizer, pin_layout=True, barrier=True)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                if total_steps % FLAGS['LOGGING_STEPS'] == 0:
                    actual_loss = loss.item() * FLAGS['GRAD_ACCUMULATION_STEP']
                    xm.master_print(f"Step: {total_steps}/{num_iter} | Loss: {actual_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    
                    if FLAGS['WANDB'] and xm.is_master_ordinal():
                        wandb.log({
                            'train/loss': actual_loss,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'train_step': total_steps
                        })
                
                if total_steps % FLAGS['EVAL_STEPS'] == 0:
                    val_loss = evaluate_model(model, validation_loader, tokenizer.pad_token_id)
                    xm.master_print(f"Validation Loss at Step {total_steps}: {val_loss:.4f}")
                    
                    if FLAGS['WANDB'] and xm.is_master_ordinal():
                        wandb.log({
                            'val/loss': val_loss,
                            'train_step': total_steps
                        })
                
                if total_steps % 100 == 0:
                    xm.mark_step()
    
    if FLAGS['WANDB'] and xm.is_master_ordinal():
        wandb.finish()

if __name__ == '__main__':
    ATTENTION_TYPE = FLAGS['ATTENTION_TYPE'].lower()
    if ATTENTION_TYPE in ROPE_CONFIGS and ATTENTION_TYPE != 'default':
        xm.master_print(f"Applying custom RoPE config: {ROPE_CONFIGS[ATTENTION_TYPE]}")
        
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn = RoPEAttn(
                    dim=model.config.hidden_size,
                    num_heads=model.config.num_attention_heads,
                    num_kv_heads=model.config.num_key_value_heads,
                    rope_theta=model.config.rope_theta,
                    **ROPE_CONFIGS[ATTENTION_TYPE],
                    dtype=torch.float32  # Use float32 for stability
                )
    
    model._set_gradient_checkpointing(True, gradient_checkpointing_func=checkpoint)
    
    for layer in model.model.layers:
        xs.apply_backward_optimization_barrier(layer)
    
    print_trainable_parameters(model)
    model.to(device)
    partition_module(model, mesh)
    
    xm.master_print(f"Loading dataset: {PRETRAIN_DATASET}")
    train_data = MinDataset(
        load_dataset(PRETRAIN_DATASET, split="train[:80%]"),
        tokenizer, 
        max_length=MAX_INPUT
    )
    FLAGS['LEN_TRAIN_DATA'] = len(train_data)
    
    val_data = MinDataset(
        load_dataset(PRETRAIN_DATASET, split="train[1%:2%]"),
        tokenizer, 
        max_length=MAX_INPUT
    )
    
    training_loader = pl.MpDeviceLoader(
        torch.utils.data.DataLoader(
            train_data, 
            batch_size=FLAGS["BATCH_SIZE"], 
            drop_last=True, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        ), 
        device
    )
    
    validation_loader = pl.MpDeviceLoader(
        torch.utils.data.DataLoader(
            val_data, 
            batch_size=FLAGS["VAL_BATCH"], 
            drop_last=True, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        ), 
        device
    )
    
    if FLAGS['WANDB'] and xm.is_master_ordinal():
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
        except:
            print("WANDB key not found. Please login manually.")
            wandb.login()
    
    xm.master_print("starting training...")
    train(FLAGS)
    xm.master_print("finished.")
