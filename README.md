# ortho_llm

Small local PyTorch/DDP research codebase for orthogonal-training ablations.

## Train

All training goes through the package entrypoint:

```bash
python -m ortho_llm.scripts.train --config configs/smoke/mixed_roles.yaml
```

Multi-process runs use the same entrypoint:

```bash
bash run.sh configs/ablations/muon_orthogonal_all_roles.yaml
```

The research configs are organized by ablation question in
`configs/ablations/`; see `configs/README.md` for the recommended run order.

## Config Shape

Configs use nested sections only:

```yaml
model:
  parameterization: grouped_matrix
  enabled_roles: [attn.q, attn.k, attn.v, attn.o, mlp.gate, mlp.up, mlp.down]

optim:
  default_role_optimizer: so
  submat_dim: 20
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
