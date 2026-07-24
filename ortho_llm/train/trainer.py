from __future__ import annotations

import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ortho_llm.config import ExperimentConfig, dump_config
from ortho_llm.data import MemmapTokenDataset
from ortho_llm.modeling import build_model
from ortho_llm.optim import OptimBundle, build_optimizers, cosine_lr
from ortho_llm.train.checkpoint import load_checkpoint, rank_checkpoint_filename, save_checkpoint
from ortho_llm.train.distributed import init_distributed, is_main_process
from ortho_llm.train.evaluator import evaluate
from ortho_llm.train.logging import JsonlLogger
from ortho_llm.train.misc import AverageMeter, load_rng_state_dict, rng_state_dict, set_seed


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _strict_stiefel_due(setting: int | str, step: int, num_steps: int) -> bool:
    completed_step = step + 1
    if isinstance(setting, int):
        return setting > 0 and completed_step % setting == 0
    normalized = str(setting).lower()
    if normalized in {"never", "none", "0"}:
        return False
    if normalized == "final_only":
        return completed_step == num_steps
    if normalized == "num_steps/50":
        interval = num_steps // 50 or 1
        return completed_step % interval == 0
    try:
        interval = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported strict_stiefel_every value {setting!r}") from exc
    return interval > 0 and completed_step % interval == 0


def _set_optimizer_lrs(bundle: OptimBundle, config: ExperimentConfig, step: int, warmup_steps: int) -> tuple[float, float | None]:
    train = config.train
    optim = config.optim
    main_lr = cosine_lr(step, train.num_steps, warmup_steps, train.lr, train.min_lr)
    muon_lr = None

    if bundle.main_optimizer is not None:
        for param_group in bundle.main_optimizer.param_groups:
            param_group["lr"] = main_lr * param_group.get("lr_multiplier", 1.0)

    for kind, optimizer in bundle.role_optimizers.items():
        if kind == "orth_adam":
            for param_group in optimizer.param_groups:
                param_group["lr"] = main_lr * optim.orth_adam_lr
        elif kind in {"muon", "orth_muon"}:
            muon_lr = cosine_lr(
                step,
                train.num_steps,
                warmup_steps,
                optim.muon_lr,
                optim.muon_min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = muon_lr
                if kind == "muon":
                    param_group["decay_lr"] = main_lr
    return main_lr, muon_lr


def _step_optimizers(bundle: OptimBundle, *, is_strict_step: bool) -> None:
    for kind, optimizer in bundle.role_optimizers.items():
        if kind == "orth_adam":
            optimizer.step(is_last=is_strict_step)
        elif kind == "orth_muon":
            optimizer.step(is_last=is_strict_step)
        else:
            optimizer.step()
    if bundle.main_optimizer is not None:
        bundle.main_optimizer.step()


def _write_manifest(output_dir: Path, config: ExperimentConfig, bundle: OptimBundle) -> None:
    manifest = {
        "config_path": config.config_path,
        "git_commit": _git_commit(),
        "role_to_optimizer": bundle.role_to_optimizer,
        "config": config.to_dict(),
    }
    path = output_dir / config.logging.manifest_filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _reject_existing_fresh_metrics(metrics_path: Path, resume: str | None) -> None:
    if not resume and metrics_path.exists():
        raise FileExistsError(
            f"Metrics file {metrics_path} already exists; choose a new train.output "
            "or set train.resume to continue the existing run"
        )


def _validate_resume_metrics(metrics_path: Path, start_step: int) -> None:
    if not metrics_path.exists():
        return
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return
    try:
        logged_steps = [int(json.loads(line)["step"]) for line in lines]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read logged steps from {metrics_path}") from exc
    newest_step = max(logged_steps)
    if newest_step > start_step:
        raise ValueError(
            f"Metrics file {metrics_path} already contains step {newest_step}, "
            f"which is newer than resume checkpoint step {start_step}"
        )


def _load_resume(
    config: ExperimentConfig,
    model: torch.nn.Module,
    bundle: OptimBundle,
    dataset: MemmapTokenDataset,
    *,
    rank: int,
    world_size: int,
) -> int:
    if not config.train.resume:
        return 0
    # RNG state tensors are CPU state even for CUDA training.  Loading the
    # checkpoint on CPU keeps them valid for torch.set_rng_state; model and
    # optimizer loaders move tensor state to their parameter devices.
    state = load_checkpoint(config.train.resume, map_location="cpu")
    rank_state_files = state.get("rank_state_files")
    if world_size > 1 and rank_state_files is None:
        raise ValueError(
            "Distributed resume requires per-rank checkpoint state; this checkpoint predates that format"
        )
    checkpoint_world_size = int(state.get("world_size", 1))
    if checkpoint_world_size != world_size:
        raise ValueError(
            f"Checkpoint world size {checkpoint_world_size} does not match current world size {world_size}"
        )

    if "optimizers" not in state:
        raise ValueError("Checkpoint is missing optimizer state required for resume")

    model.load_state_dict(state["model"])
    bundle.load_state_dict(
        state["optimizers"],
        load_role_optimizers=rank_state_files is None,
    )
    step = int(state.get("step", 0))

    if rank_state_files is not None:
        if not isinstance(rank_state_files, list) or len(rank_state_files) != world_size:
            raise ValueError("Checkpoint rank_state_files does not match its world size")
        rank_state_path = Path(config.train.resume).parent / rank_state_files[rank]
        rank_state = load_checkpoint(rank_state_path, map_location="cpu")
        if int(rank_state.get("rank", -1)) != rank:
            raise ValueError(f"Rank checkpoint {rank_state_path} belongs to a different rank")
        if int(rank_state.get("world_size", -1)) != world_size:
            raise ValueError(f"Rank checkpoint {rank_state_path} has a different world size")
        if int(rank_state.get("step", -1)) != step:
            raise ValueError(f"Rank checkpoint {rank_state_path} has a different training step")
        if "role_optimizers" not in rank_state:
            raise ValueError(f"Rank checkpoint {rank_state_path} is missing role optimizer state")
        bundle.load_role_optimizer_states(rank_state["role_optimizers"])
        dataset.load_state_dict(rank_state["dataset"])
        load_rng_state_dict(rank_state["rng"])
    else:
        if "dataset" not in state:
            raise ValueError("Checkpoint is missing dataset state required for resume")
        dataset.load_state_dict(state["dataset"])
        if "rng" in state:
            load_rng_state_dict(state["rng"])
    return step


def train(config: ExperimentConfig) -> None:
    distributed, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(config.train.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if config.train.global_batch_size % (config.train.batch_size * world_size) != 0:
        raise ValueError("global_batch_size must be divisible by batch_size * world_size")
    accum_steps = config.train.global_batch_size // (config.train.batch_size * world_size)
    if accum_steps <= 0:
        raise ValueError("Accumulation steps must be positive")

    output_dir = Path(config.train.output)
    metrics_path = output_dir / config.logging.metrics_filename
    _reject_existing_fresh_metrics(metrics_path, config.train.resume)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.device(device):
        raw_model = build_model(config.model)
    raw_model = raw_model.to(device)
    raw_model = torch.compile(raw_model)

    dataset = MemmapTokenDataset(
        config.data.data_dir,
        rank=rank,
        batch_size=config.train.batch_size,
        seq_length=config.train.seq_length,
        device=device,
    )
    val_dataset = None
    if config.data.val_data_dir:
        val_dataset = MemmapTokenDataset(
            config.data.val_data_dir,
            rank=rank,
            batch_size=config.train.batch_size,
            seq_length=config.train.seq_length,
            device=device,
        )

    bundle = build_optimizers(config, raw_model)
    start_step = _load_resume(
        config,
        raw_model,
        bundle,
        dataset,
        rank=rank,
        world_size=world_size,
    )
    if config.train.resume:
        _validate_resume_metrics(metrics_path, start_step)

    model: torch.nn.Module = raw_model
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None)

    if is_main_process():
        dump_config(config, output_dir / config.logging.resolved_config_filename)
        _write_manifest(output_dir, config, bundle)
    logger = JsonlLogger(metrics_path) if is_main_process() else None

    bundle.zero_grad(set_to_none=True)
    model.train()
    loss_meter = AverageMeter("loss")
    local_loss = torch.tensor(0.0, device=device)
    start_time = time.time()
    warmup_steps = int(config.train.num_steps * 0.01)
    start_micro_step = start_step * accum_steps
    total_micro_steps = config.train.num_steps * accum_steps

    for micro_step in range(start_micro_step, total_micro_steps):
        step = micro_step // accum_steps
        main_lr, muon_lr = _set_optimizer_lrs(bundle, config, step, warmup_steps)

        input_ids, labels = dataset.next_batch()
        should_sync = (micro_step + 1) % accum_steps == 0
        sync_context = (
            model.no_sync() if distributed and hasattr(model, "no_sync") and not should_sync else nullcontext()
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        with sync_context:
            with autocast_ctx:
                output = model(input_ids=input_ids, labels=labels)
                loss = output["loss"]
            if loss is None:
                raise RuntimeError("Training model did not return a loss")
            if config.train.fail_on_nan and not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.detach().item()}")
            (loss / accum_steps).backward()

        local_loss += loss.detach() / accum_steps
        if not should_sync:
            continue

        if distributed:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            local_loss /= world_size

        loss_meter.update(local_loss.item(), input_ids.size(0) * accum_steps)
        local_loss.zero_()

        if config.train.clip_grad and config.train.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)

        module = model.module if hasattr(model, "module") else model
        is_strict_step = _strict_stiefel_due(config.optim.strict_stiefel_every, step, config.train.num_steps)
        _step_optimizers(bundle, is_strict_step=is_strict_step)
        bundle.zero_grad(set_to_none=True)

        completed_step = step + 1
        val_metrics: dict[str, float | int | None] = {"val_loss": None, "val_ppl": None, "val_batches": 0}
        eval_configured = (
            val_dataset is not None
            and config.train.eval_interval > 0
            and config.train.eval_batches > 0
        )
        eval_due = eval_configured and (
            completed_step % config.train.eval_interval == 0
            or completed_step == config.train.num_steps
        )
        if eval_due:
            val_metrics = evaluate(model, val_dataset, num_batches=config.train.eval_batches, device=device)

        if is_main_process() and (
            completed_step % config.train.log_interval == 0
            or completed_step == 1
            or completed_step == config.train.num_steps
            or is_strict_step
            or eval_due
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_seen = completed_step * config.train.global_batch_size * config.train.seq_length
            session_tokens = (completed_step - start_step) * config.train.global_batch_size * config.train.seq_length
            tokens_per_second = session_tokens / elapsed
            row: dict[str, Any] = {
                "step": completed_step,
                "tokens_consumed": tokens_seen,
                "train_loss": loss_meter.avg,
                "val_loss": val_metrics["val_loss"],
                "val_ppl": val_metrics["val_ppl"],
                "val_batches": val_metrics["val_batches"],
                "learning_rate_main": main_lr,
                "learning_rate_chunk": muon_lr if muon_lr is not None else main_lr * config.optim.orth_adam_lr,
                "tokens_per_second": tokens_per_second,
                "wall_time_seconds": elapsed,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else None,
                "nan_or_inf_flag": False,
            }
            if logger is not None:
                logger.write(row)
            val_text = f" Val {val_metrics['val_loss']:.4f}" if val_metrics["val_loss"] is not None else ""
            chunk_lr_text = f" ChunkLR {row['learning_rate_chunk']:.6e}" if row["learning_rate_chunk"] is not None else ""
            print(
                f"Step {completed_step:06d}/{config.train.num_steps:06d} "
                f"LR {main_lr:.6e}{chunk_lr_text} Loss {loss_meter.avg:.4f}{val_text} "
                f"Tokens/s {tokens_per_second:.1f}"
            )
            loss_meter.reset()

        save_due = config.checkpoint.enabled and (
            completed_step % config.train.save_freq == 0 or completed_step == config.train.num_steps
        )
        if save_due:
            filename = config.checkpoint.filename_template.format(step=completed_step)
            rank_state_files = [rank_checkpoint_filename(filename, item) for item in range(world_size)]

            if distributed:
                role_optimizer_states = {
                    name: optimizer.state_dict()
                    for name, optimizer in bundle.role_optimizers.items()
                }
                save_checkpoint(
                    {
                        "rank": rank,
                        "world_size": world_size,
                        "step": completed_step,
                        "role_optimizers": role_optimizer_states,
                        "dataset": dataset.state_dict(),
                        "rng": rng_state_dict(),
                    },
                    output_dir,
                    rank_state_files[rank],
                )
                dist.barrier()

            if is_main_process():
                optimizer_state = bundle.state_dict()
                if distributed:
                    optimizer_state["role_optimizers"] = {}
                checkpoint_state: dict[str, Any] = {
                    "model": module.state_dict(),
                    "optimizers": optimizer_state,
                    "step": completed_step,
                    "world_size": world_size,
                    "tokens_consumed": completed_step * config.train.global_batch_size * config.train.seq_length,
                    "config": config.to_dict(),
                    "role_to_optimizer": bundle.role_to_optimizer,
                }
                if distributed:
                    checkpoint_state["rank_state_files"] = rank_state_files
                else:
                    checkpoint_state["dataset"] = dataset.state_dict()
                    checkpoint_state["rng"] = rng_state_dict()
                save_checkpoint(checkpoint_state, output_dir, filename)

            if distributed:
                dist.barrier()

    if distributed:
        dist.destroy_process_group()
