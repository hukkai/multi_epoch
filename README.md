# ortho_llm

Small local PyTorch/DDP research codebase for orthogonal-training ablations.

## Train

All training goes through the package entrypoint:

```bash
python -m ortho_llm.scripts.train --config configs/360m_2048l/dense_adamw_360m_2048l.yaml
```

Multi-process runs use the same entrypoint:

```bash
bash run.sh configs/360m_2048l/orth_muon_360m_2048l.yaml
```

The 360M 2x-Chinchilla configs are organized by context length under
`configs/`; see `RUN.md` for the runnable command list.

## Config Shape

Configs use nested sections only:

```yaml
model:
  parameterization: grouped_matrix
  enabled_roles: [attn.q, attn.k, attn.v, attn.o, mlp.gate, mlp.up, mlp.down]

optim:
  default_role_optimizer: orth_adam
  submat_dim: 32
  role_overrides:
    attn.k: adamw
    attn.v: muon
```

Grouped-matrix storage uses:

```text
MLP gate/up/down: (num_layers, mlp_ratio * hidden_size, hidden_size)
Q/O:              (num_layers, hidden_size, hidden_size)
K/V:              (num_layers, num_kv_heads * head_dim, hidden_size)
```

Orthogonal optimizers split each matrix into row blocks of shape
`(submat_dim, hidden_size)`.

Dense runs use:

```yaml
model:
  parameterization: dense
  enabled_roles: []
```

Flat configs and top-level training scripts are intentionally unsupported.
