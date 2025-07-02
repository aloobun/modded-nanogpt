### nano llama runs

1. Tokenized 1B token subset of the FineWeb dataset using Llama 3 tokenizer.

2. Llama like arch. `RMSNorm`, `SwiGLU`, `RoPE` & `Grouped Query Attention`.

3. Two optims: `AdamW` for sparse gradients (embeddings & norms) and `Muon` (transformer blocks) - hypothesis is `Muon's` rotational updates will lead to more stable training. So our layers contain large matrices of weights and the we need to update them so they get better at transforming input sequences. Now optimizers do this by "nudging" each weight individually based on its gradient. So the hypothesis is based on, `Muon` not "nudging" individual weights and instead, it calculates the best way to rotate the entire weight matrix. (orthogonal transformation preserves `L2` norm, which avoids exploding gradients/activations**).

4. ![Screenshot from 2025-07-02 16-14-45](https://github.com/user-attachments/assets/11e7ff00-86c2-4bb2-9de8-1c7ee29b0f0d)

5. sage-sweep-9
   ```
     adamw lr: 0.0005798834963148141
	muon lr: 0.0026466087427858727
	muon momentum: 0.95
	weight decay: 0.01242299362067212
   ```

6. These training runs logged here are small scale, designed primarily to validate the multi gpu (2x 3090) training setup and debug the custom arch. They are not intended to produce a fully trained model yet. Future work will involve longer training runs and a systematic exploration of different optimizers, schedulers, and hyperparameters.
