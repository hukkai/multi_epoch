# multi_epoch / ortho_llm

This repository is a compact PyTorch/DDP pretraining sandbox for testing **orthogonal training** ideas on a LLaMA-like causal language model. The current code is an exploratory prototype. The refactor goal is to preserve the current math while making the code easy to run as reproducible ablation studies.

## Current status

The repository currently has two training entrypoints:

- `train.py`: normal parameters are optimized by AdamW; when `orthogonal_type != none`, `chunk_weights` are excluded from AdamW and optimized by `SOOptimizer`.
- `train_muon.py`: normal parameters are optimized by AdamW; `chunk_weights` are optimized by `Muon` or `MuonOrthogonal` depending on `ortho_update`.

The current README is only a minimal command stub. Treat this document as the Codex-facing project spec during refactoring.


## Framework decision for this refactor

Continue this repository as the primary research codebase. Do **not** port the project wholesale to HuggingFace Trainer, Lightning/LitGPT, TorchTitan, Megatron-LM, or another large training framework during the ablation refactor.

Use the following design stance:

```text
custom research core + local minimal infrastructure
```

Keep these parts first-class and local to this repository:

- chunked orthogonal model construction;
- `chunk_weights` parameter ownership;
- SO / Muon / MuonOrthogonal optimizer logic;
- Stiefel, polar, Taylor, projection, and norm-cap logic;
- training-step order;
- geometry diagnostics and ablation logging.

No optional integrations are in scope for the ablation refactor. Codex should implement the needed infrastructure locally and keep the dependency surface small:

- config: local dataclasses + PyYAML + a small local dotlist override parser;
- logging: stdout + local `metrics.jsonl` / `manifest.json`;
- evaluation: local validation loss/perplexity only;
- sweeps: local config expansion and shell-command generation;
- orthogonality checks: local tensor metrics implemented in this repository.

Do not add OmegaConf, Hydra, W&B, TensorBoard, lm-evaluation-harness, GeoTorch, Geoopt, PyTorch orthogonal parametrization baselines, Lightning, HuggingFace Trainer, TorchTitan, Megatron, or similar framework/integration dependencies unless explicitly requested later.

## Core experiment idea

The baseline model is a small LLaMA-like decoder-only language model with:

- token embedding and tied/untied LM head,
- RoPE causal self-attention,
- RMSNorm,
- SwiGLU MLP,
- optional dropout in attention,
- causal language-modeling loss.

Orthogonal variants replace selected learned dense matrices with slices from a shared tensor:

```text
chunk_weights: (total_chunks, hidden_size, hidden_size)
```

The selected matrices are:

- attention: q, k, v, o projections;
- MLP: gate, up, down components derived from `mlp_ratio` chunks;
- all: attention + MLP.

The main switch is:

```text
orthogonal_type ∈ {none, atten, mlp, all}
```

Expected chunk count per model:

```text
atten: num_layers * 4
mlp:   num_layers * (3 * mlp_ratio)
all:   num_layers * (4 + 3 * mlp_ratio)
none:  no chunk_weights
```

`chunk_weights` are initialized by QR in the current prototype. During training, they may be constrained or regularized using Stiefel/polar-style updates.

## Important terminology

- `chunk_weights`: the square matrix bank used by chunked attention/MLP layers.
- `chunk_affine1`, `chunk_affine2`: learned affine scalars/vectors that scale effective chunk weights before reshaping into layer matrices.
- `num_submatrices`: splits each `(hidden_size, hidden_size)` matrix into row blocks of shape `(hidden_size / num_submatrices, hidden_size)` for Stiefel updates.
- `strict_stiefel`: exact polar projection at a configured cadence or final step.
- `norm_cap`: optional update/parameter cap, currently `none`, `fro`, or `spectral` in the Muon-orthogonal path.
- `orthogonal_type=none`: dense baseline model, no `chunk_weights`.
- `ortho_update=False` in Muon configs: plain Muon update for chunk weights, not the custom Stiefel update.

## Current repository map

```text
.
├── README.md                    # current minimal command stub
├── model.py                     # model config, dense LLaMA, chunked LLaMA variants
├── train.py                     # AdamW + SOOptimizer training path
├── train_muon.py                # AdamW + Muon/MuonOrthogonal training path
├── run.sh                       # selects train_muon.py when config filename contains "muon"
├── data_gen.sh                  # prepares 8 C4 shards
├── configs/
│   ├── 0.5B/                    # AdamW and SOOptimizer configs
│   └── muon/                    # Muon and MuonOrthogonal configs
├── data/
│   └── prepare_tokens.py        # HuggingFace dataset/tokenizer -> uint32 token shards
└── utils/
    ├── distributed.py           # torch.distributed init helpers
    ├── misc.py                  # seed, checkpoint, AverageMeter
    ├── optimizer.py             # AdamW param grouping helpers
    ├── scheduler.py             # cosine LR schedule
    ├── orthogonal.py            # SOOptimizer
    ├── muon.py                  # Muon optimizer
    ├── muon_orthogonal.py       # Muon + Stiefel-style update
    ├── ops.py                   # polar projection
    └── polar_taylor.py          # Stiefel projection/update approximations
```

## Data format

`data/prepare_tokens.py` creates one binary file per shard:

```text
data/C4-50B/tokens_0.bin
...
data/C4-50B/tokens_7.bin
```

Each file is a flat `uint32` array of token ids. The training loop reads contiguous segments of length `seq_length + 1`, then uses:

```text
input_ids = tokens[:-1]
labels    = tokens[1:]
```

Current behavior is sequential memmap reading. A refactor should preserve this path first, then add explicit epoch, validation, and deterministic shuffle options.

## Current run commands

Generate the default C4 token shards:

```bash
bash data_gen.sh
```

Run existing configs on 8 processes:

```bash
bash run.sh configs/0.5B/adamw_21k.yaml
bash run.sh configs/0.5B/ortho_21k.yaml
bash run.sh configs/muon/muon_21k_lr2.yaml
bash run.sh configs/muon/muon_orth_21k_spectral_cap.yaml
bash run.sh configs/muon/muon_orth_21k_fro_cap.yaml
```

`run.sh` currently uses `torchrun --nproc_per_node 8` and switches to `train_muon.py` when the config filename contains `muon`.

## Known limitations to fix before ablations

1. Training logic is duplicated between `train.py` and `train_muon.py`.
2. Config validation is duplicated and flat; ablation axes are encoded as separate YAML files instead of composable overrides.
3. There is no validation loop, held-out split, perplexity, or checkpoint resume path.
4. Logs are human-readable text only; ablations need JSONL/CSV plus a stable run manifest.
5. Orthogonality diagnostics are missing from logs.
6. Data iteration is sequential and only tracks an in-memory position; multi-epoch behavior and resume state should be explicit.
7. Checkpoints save model/optimizer partially and should include config, git commit, dataloader state, optimizer states, RNG states, and global tokens consumed.
8. Dense baseline, chunked-unconstrained baseline, SO-constrained, Muon, and MuonOrthogonal should be expressible by one unified trainer.

## Refactor principles

- Preserve current math first; move code before changing algorithms.
- Make every ablation axis a config field, not a new training script.
- Keep PyTorch/DDP simple; do not introduce HuggingFace Trainer, Lightning/LitGPT, TorchTitan, Megatron, or any optional integration dependency in this refactor.
- Add tiny CPU/GPU smoke tests before moving optimizer/model internals.
- Separate dense-vs-chunked model construction from optimizer choice.
- Treat `chunk_weights` as a named parameter group with explicit ownership by exactly one optimizer.
- Log both training metrics and geometry metrics.

## Recommended target package layout

```text
ortho_llm/
├── __init__.py
├── config.py                    # dataclasses, PyYAML load/save, local CLI overrides
├── distributed.py               # DDP environment/device helpers
├── data/
│   ├── memmap_dataset.py         # train/val shards, deterministic positions, resume state
│   └── prepare_tokens.py         # moved from data/prepare_tokens.py
├── modeling/
│   ├── llama.py                  # LlamaConfig, dense model
│   ├── chunked_layers.py         # ChunkedAttention, ChunkedMLP, chunk mapping
│   └── factory.py                # build_model(config)
├── optim/
│   ├── param_groups.py           # AdamW/chunk/scalar grouping
│   ├── scheduler.py
│   ├── so.py
│   ├── muon.py
│   ├── muon_orthogonal.py
│   ├── stiefel.py                # polar/taylor/projection utilities
│   └── factory.py                # build_optimizers(config, model)
├── train/
│   ├── trainer.py                # one train loop
│   ├── evaluator.py              # validation loss/perplexity
│   ├── checkpoint.py
│   ├── metrics.py                # loss, grad, LR, orthogonality, throughput
│   └── logging.py                # stdout + JSONL/CSV
└── scripts/
    ├── train.py                  # thin CLI wrapper
    ├── eval.py
    ├── sweep.py
    └── inspect_checkpoint.py
```

## Minimum ablation axes

Start with these axes; do not expand until the run pipeline is stable.

### Model/parameterization axes

```text
orthogonal_type: none | atten | mlp | all
chunk_affine: enabled | disabled
init: qr | gaussian_then_project | gaussian_no_project
```

### Optimizer/geometry axes

```text
chunk_optimizer: none | adamw | so | muon | muon_orthogonal
num_submatrices: 8 | 16 | 32 | 64
strict_stiefel_every: never | every_n_steps | final_only
norm_cap: none | fro | spectral
so_lr: 0.25 | 0.5 | 1.0 | 2.0
muon_lr: tuned grid around current 2e-3
muon_ns_steps: 3 | 5 | 7
```

### Training axes

```text
seed: at least 3 seeds for main claims
num_steps/token_budget: 11k and 21k existing budgets first
cosine_power: 1.0 | 2.0
weight_decay: current value plus one lower value
```

## Required metrics for ablation tables

Log these at `log_interval` and at validation intervals:

```text
step
tokens_consumed
train_loss
val_loss
val_ppl
learning_rate_main
learning_rate_chunk
grad_norm_total
grad_norm_chunk
update_norm_chunk
orth_error_fro_mean
orth_error_fro_max
orth_error_spectral_mean
singular_value_min_mean
singular_value_max_mean
tokens_per_second
wall_time_seconds
peak_cuda_memory_mb
nan_or_inf_flag
```

Orthogonality error for a block `W` should include:

```text
||W W^T - I||_F
||W W^T - I||_2
```

Use row-block shape `(hidden_size / num_submatrices, hidden_size)` for Stiefel-block metrics when applicable.

## Acceptance criteria for Codex changes

A refactor PR is acceptable only if:

1. `python -m pytest tests/` passes.
2. `python -m ortho_llm.scripts.train --config configs/smoke/tiny.yaml` runs at least 5 steps on CPU or one GPU.
3. Existing configs can be migrated without changing intended hyperparameters.
4. Dense baseline output shape and loss behavior match the old code on a fixed seed tiny batch within numerical tolerance.
5. For `orthogonal_type in {atten, mlp, all}`, `chunk_weights` count matches the formulas above.
6. Exactly one optimizer owns `chunk_weights` in each config.
7. JSONL logs and checkpoints contain the complete resolved config.
8. A resumed 10-step run matches an uninterrupted 10-step run up to expected nondeterminism.


## Dependency policy

Allowed by default:

- PyTorch;
- NumPy;
- PyYAML;
- tqdm;
- pytest;
- ruff.

Do not add optional integration dependencies in the first refactor. Specifically, do not add:

- OmegaConf or Hydra;
- W&B or TensorBoard;
- lm-evaluation-harness;
- GeoTorch or Geoopt;
- PyTorch `torch.nn.utils.parametrizations.orthogonal` as a baseline path;
- HuggingFace `Trainer`;
- Lightning/LitGPT;
- Megatron or TorchTitan;
- new distributed-training frameworks or experiment-management frameworks.

The code may use standard-library modules freely. Any new third-party dependency must be justified in the PR description and should be rejected unless it is essential for the local PyTorch training path.

## Related implementation references to compare against

External projects may be read as conceptual references only. Do not copy large code blocks, add package dependencies, or route training through their lifecycle.

- KellerJordan/Muon: useful as a conceptual reference for hidden 2D weight ownership and Newton-Schulz orthogonalized updates.
- KellerJordan/modded-nanogpt: useful as a conceptual reference for a lean, transparent language-model training loop.
- expRNN / Dynamic Trivializations literature: useful as background for orthogonal constraints and constrained/unconstrained parameter separation.

All comparison baselines in this repository should be implemented locally using the existing model and optimizer interfaces.

