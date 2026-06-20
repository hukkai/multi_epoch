# Config Layout

This directory is organized by experiment intent rather than by historical model
size or optimizer names.

## Main Ablations

`configs/ablations/` contains directly runnable 22B-token runs. They share:

- data: `./data/C4-50B/`
- model: 24 layers, hidden size 1280, 16 heads, MLP ratio 3, tied embeddings
- training: 21,000 steps, global batch 512, sequence length 2048
- tokens: `21000 * 512 * 2048 = 22.0B`

Run one with:

```bash
python -m ortho_llm.scripts.train --config configs/ablations/so_all_roles.yaml
```

Recommended order:

1. `dense_adamw.yaml`
2. `grouped_adamw_all_roles.yaml`
3. `so_all_roles.yaml`
4. `muon_all_roles.yaml`
5. `muon_orthogonal_all_roles.yaml`

These answer the first question: whether gains come from dense AdamW, grouped
storage plus block initialization, SO/Stiefel updates, Muon-style update
orthogonalization, or Muon plus Stiefel constraints.

Then run role coverage:

- `so_attention_roles.yaml`
- `so_mlp_roles.yaml`
- `muon_orthogonal_attention_roles.yaml`
- `muon_orthogonal_mlp_roles.yaml`

These ask whether attention projections, MLP projections, or both are carrying
the effect.

Finally run constraint sensitivity for the strongest optimizer:

- `muon_orthogonal_fro_cap.yaml`
- `muon_orthogonal_spectral_cap.yaml`
- `muon_orthogonal_submat_40.yaml`
- `muon_orthogonal_submat_80.yaml`

The default Stiefel block size is 20 rows. The block-size variants test whether
the effect depends on very small row blocks. The norm-cap variants test whether
large projected Muon updates are destabilizing training.

## Smoke Configs

`configs/smoke/` stays tiny and is meant for local correctness checks, not
research conclusions.

## Sweeps

`configs/sweeps/` contains short exploratory matrices. They are intended to
generate commands with unique output directories:

```bash
python -m ortho_llm.scripts.sweep --config configs/sweeps/optimizer_by_role.yaml --dry-run
```

Use sweeps for pilots, then promote only the useful cells to full 22B-token
ablations.
