# Refactor plan: ablation-ready orthogonal training

## Goal

Turn `multi_epoch` from an exploratory orthogonal-training prototype into a reproducible ablation codebase. Preserve the current algorithms first, then add missing local experiment infrastructure: unified config, unified trainer, validation, resume, structured logs, metrics, and sweep generation. Do not add optional integrations.

## Non-goals for the first refactor

- Do not rewrite the model architecture beyond moving code into modules.
- Do not change the math of `SOOptimizer`, `Muon`, `MuonOrthogonal`, polar projection, or Taylor Stiefel update until parity tests exist.
- Do not adopt HuggingFace Trainer, Lightning/LitGPT, TorchTitan, Megatron, or a large framework.
- Do not add optional integrations such as OmegaConf/Hydra, W&B/TensorBoard, lm-evaluation-harness, GeoTorch/Geoopt, or PyTorch orthogonal-parametrization baselines.
- Do not optimize for maximum throughput before correctness and ablation reproducibility are in place.


## Phase -1 — Framework boundary decision

Before touching model or optimizer code, record the architectural decision that this project remains a small custom research framework rather than migrating to a general LLM framework.

### Tasks

1. Add a short decision note, for example:

```text
docs/decisions/0001-keep-custom-research-core.md
```

2. State the selected approach:

```text
custom research core + local minimal infrastructure
```

3. Mark these as non-goals for the first ablation refactor:
   - no HuggingFace Trainer migration;
   - no Lightning/LitGPT conversion;
   - no TorchTitan/Megatron parallelism migration;
   - no framework-controlled optimizer lifecycle for `chunk_weights`.
4. Define the no-optional-integration boundary:
   - config composition must be implemented locally with dataclasses, PyYAML, and a small dotlist override parser;
   - logging must be local stdout plus `metrics.jsonl` / `manifest.json`;
   - evaluation must be local validation loss/perplexity only;
   - sweeps must be local YAML expansion and command generation;
   - orthogonal/manifold baselines must be implemented locally if needed;
   - do not add OmegaConf, Hydra, W&B, TensorBoard, lm-evaluation-harness, GeoTorch, Geoopt, or PyTorch orthogonal parametrization baseline paths.

### Acceptance

The repository still runs through a transparent `torchrun`/PyTorch training loop, and `chunk_weights` ownership remains explicit in local code. No core training path subclasses or depends on HuggingFace `Trainer`, Lightning `Trainer`, LitGPT recipes, TorchTitan, Megatron, OmegaConf/Hydra, W&B/TensorBoard, lm-evaluation-harness, GeoTorch, or Geoopt.


## Phase 0 — Safety harness before moving code

### Tasks

1. Add `requirements-dev.txt` with `pytest` and `ruff` only.
2. Add tiny fixtures for a small model:
   - `hidden_size=64`
   - `num_layers=2`
   - `num_heads=4`
   - `mlp_ratio=2`
   - `seq_length=32`
   - `vocab_size=128`
3. Add tests:
   - dense forward returns `(batch, seq, vocab)` logits and finite loss;
   - each `orthogonal_type` builds successfully;
   - `chunk_weights` count matches expected formulas;
   - `SOOptimizer.step()` preserves shape and clears grad;
   - `Muon.step()` and `MuonOrthogonal.step()` preserve shape;
   - `cosine_lr()` reproduces old scheduler values.
4. Add a tiny token file generator for tests:
   - writes `tokens_0.bin` as `np.uint32`;
   - no network or HuggingFace dependency.

### Acceptance

```bash
pytest tests/test_model_shapes.py tests/test_optimizers.py tests/test_scheduler.py
```

passes on CPU. GPU-specific tests should be skipped when CUDA is unavailable.

## Phase 1 — Local config system

### Target files

```text
ortho_llm/config.py                    # dataclasses + PyYAML + local dotlist overrides
configs/base/model_0p5b.yaml
configs/base/data_c4.yaml
configs/base/train_21k.yaml
configs/base/optim_adamw.yaml
configs/base/optim_so.yaml
configs/base/optim_muon.yaml
configs/base/optim_muon_orthogonal.yaml
configs/smoke/tiny.yaml
configs/ablations/*.yaml
```

### Tasks

1. Replace duplicated flat config parsing in `train.py` and `train_muon.py` with dataclasses:
   - `DataConfig`
   - `ModelConfig`
   - `TrainConfig`
   - `AdamWConfig`
   - `ChunkOptimizerConfig`
   - `LoggingConfig`
   - `CheckpointConfig`
   - `ExperimentConfig`
2. Support CLI overrides with a small local parser, not OmegaConf/Hydra:

```bash
python -m ortho_llm.scripts.train --config configs/ablations/main.yaml \
  train.num_steps=1000 model.orthogonal_type=mlp chunk_optimizer.kind=so
```

3. Emit `resolved_config.yaml` into every run directory.
4. Validate constraints:
   - `hidden_size % num_heads == 0`
   - `head_dim` even for RoPE
   - `hidden_size % num_submatrices == 0` when Stiefel blocks are used
   - `chunk_optimizer.kind == none` iff `orthogonal_type == none`, except for explicit dense baselines
   - exactly one optimizer owns `chunk_weights`

### Acceptance

All existing YAML files can be translated into the new schema and print the same key hyperparameters.

## Phase 2 — Model package split

### Target files

```text
ortho_llm/modeling/llama.py
ortho_llm/modeling/chunked_layers.py
ortho_llm/modeling/factory.py
```

### Tasks

1. Move dense LLaMA components into `llama.py`:
   - `LlamaConfig`
   - RoPE cache/application
   - RMSNorm usage
   - dense attention/MLP/block/model
2. Move chunked components into `chunked_layers.py`:
   - `ChunkedAttention`
   - `ChunkedMLP`
   - chunked block variants
   - helper that maps layer index -> chunk slice names
3. Keep `build_model(config)` in `factory.py`.
4. Add `model.named_chunk_parameters()` or a helper returning the canonical chunk parameter list.
5. Add optional `chunk_affine.enabled` config so ablations can disable affine scaling.

### Acceptance

Old and new model outputs match on a fixed seed for dense mode. Chunked modes match old shapes and parameter names, with documented expected renames.

## Phase 3 — Data pipeline

### Target files

```text
ortho_llm/data/memmap_dataset.py
ortho_llm/data/prepare_tokens.py
```

### Tasks

1. Move `data/prepare_tokens.py` into the package while keeping a wrapper script for backward compatibility.
2. Implement `MemmapTokenDataset` with:
   - shard discovery: `tokens_{rank}.bin` first, fallback `tokens_0.bin` for single-process smoke;
   - explicit `position` and `epoch` state;
   - deterministic sequential mode matching the old code;
   - optional shuffle of start offsets for future ablations;
   - validation split support by separate path or tail fraction.
3. Add state dict methods:

```python
state = dataset.state_dict()
dataset.load_state_dict(state)
```

4. Add guardrails:
   - fail clearly when token file is too short for `batch_size * (seq_length + 1)`;
   - log number of available tokens and expected tokens consumed.

### Acceptance

For the same token file, batch size, sequence length, and initial position, the new loader returns the same first N batches as old `load_micro_batch()`.

## Phase 4 — Optimizer factory

### Target files

```text
ortho_llm/optim/param_groups.py
ortho_llm/optim/factory.py
ortho_llm/optim/so.py
ortho_llm/optim/muon.py
ortho_llm/optim/muon_orthogonal.py
ortho_llm/optim/stiefel.py
```

### Tasks

1. Move old optimizer code without algorithmic changes.
2. Create a unified optimizer factory returning:

```python
OptimBundle(
    main_optimizer=...,        # usually AdamW
    chunk_optimizer=...,       # None, AdamW, SO, Muon, or MuonOrthogonal
    schedulers=...,
    owns_chunk_weights=True/False,
)
```

3. Add explicit chunk optimizer modes:
   - `none`: only valid when no `chunk_weights` exist;
   - `adamw`: chunked Euclidean control baseline;
   - `so`: current `SOOptimizer` path;
   - `muon`: current Muon path with `ortho_update=False`;
   - `muon_orthogonal`: current MuonOrthogonal path.
4. Add optimizer state save/load for custom optimizers.
5. Add runtime check that every trainable parameter belongs to exactly one optimizer group.

### Acceptance

For every current config class, the optimizer factory reproduces old ownership:

```text
AdamW dense:              all params in AdamW
AdamW + SO:               non-chunk params in AdamW, chunk_weights in SO
Muon no ortho_update:     non-chunk params in AdamW, chunk_weights in Muon
Muon ortho_update:        non-chunk params in AdamW, chunk_weights in MuonOrthogonal
```

## Phase 5 — Unified trainer

### Target files

```text
ortho_llm/train/trainer.py
ortho_llm/scripts/train.py
```

### Tasks

1. Merge `train.py` and `train_muon.py` into one trainer.
2. Keep these old behaviors initially:
   - DDP init from `RANK`, `WORLD_SIZE`, `LOCAL_RANK`;
   - bf16 autocast on CUDA;
   - gradient accumulation based on `global_batch_size / (batch_size * world_size)`;
   - cosine LR scheduling;
   - grad clipping;
   - strict Stiefel projection cadence equivalent to old `num_steps // 50` default.
3. Add clear train-step stages:

```text
load batch -> forward -> backward -> all-reduce/log loss -> clip -> chunk optimizer step -> main optimizer step -> zero grad -> log -> checkpoint/eval
```

4. Write `metrics.jsonl` and human-readable stdout.
5. Save checkpoints with:
   - model state;
   - optimizer states;
   - dataset state;
   - RNG states;
   - resolved config;
   - step and tokens consumed;
   - git commit if available.

### Acceptance

A 5-step smoke run works for:

```bash
python -m ortho_llm.scripts.train --config configs/smoke/dense_adamw.yaml
python -m ortho_llm.scripts.train --config configs/smoke/all_so.yaml
python -m ortho_llm.scripts.train --config configs/smoke/all_muon.yaml
python -m ortho_llm.scripts.train --config configs/smoke/all_muon_orthogonal.yaml
```

## Phase 6 — Local evaluation and geometry metrics

### Target files

```text
ortho_llm/train/evaluator.py
ortho_llm/train/metrics.py
```

### Tasks

1. Add local validation loss/perplexity over a fixed number of tokens or batches. Do not call lm-evaluation-harness.
2. Add orthogonality diagnostics for chunked modes:
   - row-block Frobenius error;
   - row-block spectral error;
   - singular value min/max/mean;
   - chunk update norm;
   - chunk grad norm.
3. Add throughput and memory metrics:
   - tokens/sec;
   - step time;
   - peak CUDA memory;
   - dataloader time vs compute time.
4. Add NaN/Inf detection and fail-fast option.

### Acceptance

Every ablation run produces a JSONL row with all required fields. Missing fields should be explicit `null`, not silently omitted.

## Phase 7 — Sweep generation

### Target files

```text
ortho_llm/scripts/sweep.py
configs/sweeps/pilot.yaml
configs/sweeps/main.yaml
```

### Tasks

1. Implement a local sweep generator that expands axes into concrete resolved configs. Do not use W&B Sweeps, Hydra multirun, or external experiment managers.
2. Generate stable run names:

```text
{date}_{model_size}_{orthogonal_type}_{chunk_optimizer}_nsub{num_submatrices}_cap{norm_cap}_seed{seed}
```

3. Support dry run:

```bash
python -m ortho_llm.scripts.sweep --config configs/sweeps/pilot.yaml --dry-run
```

4. Support emitting shell commands for SLURM or local `torchrun`.
5. Keep sweep metadata in `runs/<run_name>/manifest.json`.

### Acceptance

A pilot sweep can generate configs and commands without launching training, and every generated config passes validation.

## Recommended pilot ablation matrix

Use a small model and short token budget first. Only promote to 21k-step runs after the pilot produces clean logs and checkpoints.

### Pilot 1 — optimizer ownership and parameterization controls

Hold fixed: data, model size, global batch, token budget, seed.

```text
A0: orthogonal_type=none, chunk_optimizer=none        # dense AdamW baseline
A1: orthogonal_type=all,  chunk_optimizer=adamw       # chunked Euclidean control
A2: orthogonal_type=all,  chunk_optimizer=so
A3: orthogonal_type=all,  chunk_optimizer=muon
A4: orthogonal_type=all,  chunk_optimizer=muon_orthogonal
```

Purpose: separate the effect of chunked parameterization from the effect of orthogonal/Stiefel updates.

### Pilot 2 — target submodule scope

Hold fixed: best chunk optimizer from Pilot 1.

```text
orthogonal_type ∈ {atten, mlp, all}
```

Purpose: identify whether attention, MLP, or both drive any gain/loss.

### Pilot 3 — Stiefel block granularity

Only for `so` and `muon_orthogonal`.

```text
num_submatrices ∈ {8, 16, 32, 64}
```

Purpose: trade off constraint granularity, update cost, and stability.

### Pilot 4 — strict projection and norm cap

Only for `muon_orthogonal`.

```text
strict_stiefel_every ∈ {never, final_only, num_steps/50, 100}
norm_cap ∈ {none, fro, spectral}
```

Purpose: quantify whether exact projection/capping stabilizes training or merely adds overhead.

### Main run promotion rule

Promote a condition from pilot to 21k-step runs only if:

1. no NaNs/Inf;
2. validation loss is not clearly worse than dense AdamW after equal tokens;
3. orthogonality metrics behave as expected;
4. throughput overhead is acceptable;
5. at least two seeds agree on qualitative behavior.

## Suggested main ablation table columns

```text
run_name
seed
orthogonal_type
chunk_optimizer
num_submatrices
norm_cap
strict_stiefel_every
main_lr
chunk_lr
num_steps
tokens_consumed
final_train_loss
best_val_loss
final_val_loss
final_val_ppl
orth_error_fro_mean
orth_error_spectral_mean
tokens_per_second
peak_memory_mb
status
```

## Codex implementation order

Implement in this exact order:

1. Add tests around current code.
2. Add config dataclasses while old scripts still run.
3. Move model code and keep old `model.py` as compatibility import.
4. Move optimizer code and keep old `utils/*` imports as compatibility wrappers.
5. Add optimizer factory.
6. Add data loader class.
7. Add unified trainer.
8. Add eval and metrics.
9. Add checkpoint resume.
10. Add sweep generator.
11. Migrate old configs.
12. Remove or deprecate old duplicated entrypoints only after parity checks pass.

## Backward compatibility plan

Keep these wrappers for one transition period:

```text
model.py                    -> imports from ortho_llm.modeling
train.py                    -> calls ortho_llm.scripts.train with translated config
train_muon.py               -> calls ortho_llm.scripts.train with translated config
utils/*.py                  -> imports from ortho_llm.optim / train / distributed
run.sh                      -> still works for old config paths
```

Emit a deprecation warning, but do not break existing commands until new smoke tests and at least one real ablation run succeed.

## First PR checklist

A good first Codex PR should contain only:

- `tests/` smoke tests;
- `ortho_llm/config.py`;
- `configs/smoke/*.yaml`;
- minimal wrappers to load old YAML files;
- no algorithmic changes.

The first PR should not move optimizer math yet. The point is to create a safety net before refactoring high-risk tensor geometry code.

