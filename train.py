from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from model import LlamaConfig, build_model
from utils import (
    AverageMeter,
    SOOptimizer,
    cosine_lr,
    get_param_groups,
    init_distributed,
    is_main_process,
    save_checkpoint,
    set_seed,
)


DEFAULT_CONFIG_PATH = "configs/0.5B/adamw_21k.yaml"

CONFIG_TYPES = {
    "data_dir": str,
    "output": str,
    "seed": int,
    "log_interval": int,
    "save_freq": int,
    "orthogonal_type": str,
    "hidden_size": int,
    "num_layers": int,
    "num_heads": int,
    "mlp_ratio": int,
    "max_position_embeddings": int,
    "vocab_size": int,
    "rope_theta": float,
    "rms_norm_eps": float,
    "attention_dropout": float,
    "tie_word_embeddings": bool,
    "batch_size": int,
    "global_batch_size": int,
    "seq_length": int,
    "num_steps": int,
    "lr": float,
    "min_lr": float,
    "weight_decay": float,
    "clip_grad": float,
    "so_lr": float,
    "num_submatrices": int,
    "orth_beta1": float,
    "orth_beta2": float,
    "orth_eps": float,
    "transpose_o": bool,
}

ORTHOGONAL_TYPE_CHOICES = {"none", "mlp", "atten", "all"}
ORTHOGONAL_CONFIG_KEYS = {
    "so_lr",
    "num_submatrices",
    "orth_beta1",
    "orth_beta2",
    "orth_eps",
    "transpose_o",
}


def _coerce_config_value(key: str, value: Any) -> Any:
    expected_type = CONFIG_TYPES[key]
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a YAML bool, got {type(value).__name__}")
        return value
    if expected_type is int:
        if isinstance(value, bool):
            raise ValueError(f"{key} must be an int, got bool")
        return int(value)
    if expected_type is float:
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a float, got bool")
        return float(value)
    if expected_type is str:
        return str(value)
    return value


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    config = {}
    for key, value in raw_config.items():
        normalized_key = str(key).replace("-", "_")
        if normalized_key in config:
            raise ValueError(f"Duplicate config key after normalization: {key}")
        config[normalized_key] = value

    expected_keys = set(CONFIG_TYPES)
    unknown_keys = sorted(set(config) - expected_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")

    required_keys = expected_keys
    if config.get("orthogonal_type") == "none":
        required_keys = expected_keys - ORTHOGONAL_CONFIG_KEYS

    missing_keys = sorted(required_keys - set(config))
    if missing_keys:
        raise ValueError(f"Missing config keys in {config_path}: {', '.join(missing_keys)}")

    coerced_config = {
        key: _coerce_config_value(key, config[key])
        for key in CONFIG_TYPES
        if key in config
    }

    if coerced_config["orthogonal_type"] not in ORTHOGONAL_TYPE_CHOICES:
        choices = ", ".join(sorted(ORTHOGONAL_TYPE_CHOICES))
        raise ValueError(f"orthogonal_type must be one of: {choices}")
    
    if "transpose_o" not in coerced_config and coerced_config["orthogonal_type"] != "none":
        coerced_config["transpose_o"] = False

    return argparse.Namespace(config=config_path, **coerced_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Orthogonal LLaMA-2-style pretraining")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH})",
    )
    cli_args = parser.parse_args()
    return load_config(cli_args.config)


def resolve_data_path(data_dir: str, rank: int) -> str:
    shard_path = os.path.join(data_dir, f"tokens_{rank}.bin")
    if os.path.exists(shard_path):
        return shard_path

    fallback_path = os.path.join(data_dir, "tokens_0.bin")
    if rank == 0 and os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(f"Could not find token shard for rank {rank} under {data_dir}")


def build_config(args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        attention_dropout=args.attention_dropout,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    exclude = ["chunk_weights"] if args.orthogonal_type != "none" else []
    param_groups = get_param_groups(model, args.weight_decay, exclude_names=exclude)
    return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)


def load_micro_batch(
    all_tokens: np.memmap,
    micro_step: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_length = seq_length + 1
    tokens_per_micro = batch_size * sample_length
    start = micro_step * tokens_per_micro
    end = (micro_step + 1) * tokens_per_micro

    token_slice = np.asarray(all_tokens[start:end], dtype=np.int64)
    token_batch = torch.from_numpy(token_slice.reshape(batch_size, sample_length))
    token_batch = token_batch.to(device, non_blocking=True)
    return token_batch[:, :-1], token_batch[:, 1:]


def main() -> None:
    args = parse_args()
    distributed, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.global_batch_size % (args.batch_size * world_size) != 0:
        raise ValueError("global_batch_size must be divisible by batch_size * world_size")
    accum_steps = args.global_batch_size // (args.batch_size * world_size)
    if accum_steps <= 0:
        raise ValueError("Accumulation steps must be positive")
    if args.seq_length > args.max_position_embeddings:
        raise ValueError("seq_length must be <= max_position_embeddings")

    data_path = resolve_data_path(args.data_dir, rank)
    all_tokens = np.memmap(data_path, dtype=np.uint32, mode="r")

    total_micro_steps = args.num_steps * accum_steps
    tokens_per_micro = args.batch_size * (args.seq_length + 1)
    required_tokens = total_micro_steps * tokens_per_micro
    if all_tokens.shape[0] < required_tokens:
        raise ValueError(
            f"Not enough tokens in {data_path}: need {required_tokens}, found {all_tokens.shape[0]}"
        )

    config = build_config(args)
    init_chunk_weights = not distributed or rank == 0
    with torch.device(device):
        model = build_model(
            config,
            orthogonal_type=args.orthogonal_type,
            init_chunk_weights=init_chunk_weights,
            transpose_o=args.transpose_o,
        )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = create_optimizer(args, model)
    module = model.module if hasattr(model, "module") else model
    orth_opt = None
    if args.orthogonal_type != "none":
        orth_opt = SOOptimizer(
            module.chunk_weights,
            lr=args.lr * args.so_lr,
            betas=(args.orth_beta1, args.orth_beta2),
            eps=args.orth_eps,
            num_submatrices=args.num_submatrices,
        )

    optimizer.zero_grad(set_to_none=True)
    loss_meter = AverageMeter("loss")
    start_time = time.time()
    start_micro_step = 0

    warmup_steps = int(args.num_steps * 0.01)
    local_loss = torch.tensor(0.0, device=device)

    for micro_step in range(start_micro_step, total_micro_steps):
        step = micro_step // accum_steps
        lr = cosine_lr(step, args.num_steps, warmup_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids, labels = load_micro_batch(all_tokens, micro_step, args.batch_size, args.seq_length, device)
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
            (loss / accum_steps).backward()

        local_loss += loss.detach() / accum_steps

        if not should_sync:
            continue

        if distributed:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            local_loss /= world_size

        loss_meter.update(local_loss.item(), input_ids.size(0) * accum_steps)
        local_loss.zero_()

        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # strict stiefel 50 times in the entire training
        strict_stiefel_steps = args.num_steps // 50 or 1
        is_last_step = (step + 1) % strict_stiefel_steps == 0

        if orth_opt is not None:
            orth_opt.step(lr=lr * args.so_lr, is_last=is_last_step)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step + 1
        if is_main_process() and (
            completed_step % args.log_interval == 0 or completed_step == 1 or is_last_step
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_seen = completed_step * args.global_batch_size * args.seq_length
            tokens_per_second = tokens_seen / elapsed
            print(
                f"Step {completed_step:06d}/{args.num_steps:06d} "
                f"LR {lr:.6e} Loss {loss_meter.avg:.4f} Tokens/s {tokens_per_second:.1f}"
            )
            loss_meter.reset()

        if is_main_process() and (
            completed_step % args.save_freq == 0 or completed_step == args.num_steps
        ):
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "step": completed_step,
                    "config": vars(config),
                    "args": vars(args),
                },
                args.output,
                filename=f"checkpoint_{completed_step:06d}.pth",
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
