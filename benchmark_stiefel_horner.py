from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable

import torch

from ortho_llm.optim.stiefel_update import (
    _COEFFS2,
    _COEFFS3,
    _COEFFS4,
    _apply_series,
    _apply_series2,
    _apply_series2_eager,
)


COEFFS_BY_ORDER = {
    2: _COEFFS2,
    3: _COEFFS3,
    4: _COEFFS4,
}

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


@torch.no_grad()
def old_apply_series(
    a: torch.Tensor,
    gram_error: torch.Tensor,
    coeffs: tuple[float, ...],
) -> torch.Tensor:
    term = gram_error @ a
    q = a + coeffs[0] * term
    for coeff in coeffs[1:]:
        term = gram_error @ term
        q = q + coeff * term
    return q


@torch.no_grad()
def new_apply_series(
    a: torch.Tensor,
    gram_error: torch.Tensor,
    coeffs: tuple[float, ...],
    *,
    compiled_order2: bool = False,
) -> torch.Tensor:
    if coeffs == _COEFFS2:
        fn = _apply_series2 if compiled_order2 else _apply_series2_eager
        return fn(a, gram_error)
    return _apply_series(a, gram_error, coeffs)


@torch.no_grad()
def gram_error(a: torch.Tensor) -> torch.Tensor:
    err = a @ a.transpose(-1, -2)
    err.diagonal(dim1=-2, dim2=-1).sub_(1)
    return err


def make_inputs(
    batch: int,
    n: int,
    m: int,
    dtype: torch.dtype,
    device: torch.device,
    noise: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = noise / math.sqrt(m)
    a = scale * torch.randn((batch, n, m), device=device, dtype=dtype, generator=generator)
    a[..., :n] += torch.eye(n, device=device, dtype=dtype)
    return a


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    sync(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / iters, out

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    sync(device)
    return (time.perf_counter() - t0) * 1000.0 / iters, out


def compare_outputs(old: torch.Tensor, new: torch.Tensor) -> dict[str, float]:
    compare_dtype = torch.float64 if old.dtype == torch.float64 else torch.float32
    old_cmp = old.to(compare_dtype)
    new_cmp = new.to(compare_dtype)
    diff = new_cmp - old_cmp
    denom = old_cmp.abs().max().clamp_min(torch.finfo(compare_dtype).tiny)

    old_gram = old_cmp @ old_cmp.transpose(-1, -2)
    new_gram = new_cmp @ new_cmp.transpose(-1, -2)
    gram_diff = new_gram - old_gram

    return {
        "max_abs": diff.abs().max().item(),
        "max_rel": (diff.abs().max() / denom).item(),
        "rms_abs": diff.square().mean().sqrt().item(),
        "gram_max_abs": gram_diff.abs().max().item(),
    }


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def set_tf32(mode: str) -> None:
    if not torch.cuda.is_available():
        return
    if mode == "default":
        return
    enabled = mode == "on"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark old sequential Stiefel Taylor apply vs new Horner apply."
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch dimension B.")
    parser.add_argument("--n", type=int, default=64, help="Row dimension n.")
    parser.add_argument("--m", type=int, default=4096, help="Column dimension m.")
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        choices=sorted(COEFFS_BY_ORDER),
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Approximate row-norm scale of dense noise added to an identity-padded Stiefel matrix.",
    )
    parser.add_argument(
        "--tf32",
        choices=("default", "on", "off"),
        default="default",
        help="Set CUDA matmul TF32 mode before benchmarking.",
    )
    parser.add_argument(
        "--include-gram",
        action="store_true",
        help="Also time gram_error(A) plus Taylor apply, matching more of fast_polar's Taylor path.",
    )
    parser.add_argument(
        "--skip-compiled-order2",
        action="store_true",
        help="Skip benchmarking the actual compiled order-2 helper used by fast_polar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n > args.m:
        raise ValueError(f"expected n <= m, got n={args.n}, m={args.m}")

    set_tf32(args.tf32)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    a = make_inputs(args.batch, args.n, args.m, dtype, device, args.noise, args.seed)
    err = gram_error(a)
    max_fro = torch.linalg.matrix_norm(err.float(), ord="fro", dim=(-2, -1)).max().item()

    print("Stiefel Taylor apply benchmark")
    print(f"shape=(B={args.batch}, n={args.n}, m={args.m}) dtype={args.dtype} device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"tf32_allow={torch.backends.cuda.matmul.allow_tf32}")
    print(
        f"warmup={args.warmup} iters={args.iters} "
        f"noise={args.noise} input_gram_fro_max={max_fro:.6f}"
    )
    print()

    for order in args.orders:
        coeffs = COEFFS_BY_ORDER[order]
        print(f"order={order}")

        old_once = old_apply_series(a, err, coeffs)
        new_once = new_apply_series(a, err, coeffs)
        sync(device)
        metrics = compare_outputs(old_once, new_once)
        print(
            "  diff(new_eager, old_seq): "
            + " ".join(f"{key}={format_float(value)}" for key, value in metrics.items())
        )

        old_ms, _ = benchmark(
            lambda: old_apply_series(a, err, coeffs),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        new_ms, _ = benchmark(
            lambda: new_apply_series(a, err, coeffs),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        print(
            f"  apply old_seq_ms={old_ms:.4f} "
            f"new_horner_eager_ms={new_ms:.4f} speedup={old_ms / new_ms:.3f}x"
        )

        if order == 2 and not args.skip_compiled_order2:
            compiled_ms, compiled_out = benchmark(
                lambda: new_apply_series(a, err, coeffs, compiled_order2=True),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            compiled_metrics = compare_outputs(old_once, compiled_out)
            print(
                f"  apply new_horner_compiled_ms={compiled_ms:.4f} "
                f"speedup_vs_old={old_ms / compiled_ms:.3f}x "
                f"compiled_max_abs={format_float(compiled_metrics['max_abs'])}"
            )

        if args.include_gram:
            old_e2e_ms, _ = benchmark(
                lambda: old_apply_series(a, gram_error(a), coeffs),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            new_e2e_ms, _ = benchmark(
                lambda: new_apply_series(a, gram_error(a), coeffs),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            print(
                f"  with_gram old_seq_ms={old_e2e_ms:.4f} "
                f"new_horner_eager_ms={new_e2e_ms:.4f} speedup={old_e2e_ms / new_e2e_ms:.3f}x"
            )
        print()


if __name__ == "__main__":
    main()
