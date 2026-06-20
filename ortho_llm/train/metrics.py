from __future__ import annotations

import math

import torch


@torch.no_grad()
def grad_norm(parameters) -> float:
    total = torch.tensor(0.0)
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().float().norm(2)
        total = total.to(value.device)
        total += value.pow(2)
        found = True
    return float(total.sqrt().item()) if found else 0.0


@torch.no_grad()
def orthogonality_metrics_for_blocks(blocks: torch.Tensor) -> dict[str, float]:
    if blocks.ndim != 3:
        raise ValueError(f"expected blocks with shape (n, rows, cols), got {tuple(blocks.shape)}")
    rows, cols = blocks.shape[-2:]
    if rows > cols:
        raise ValueError(f"orthogonality metrics expect rows <= cols, got {tuple(blocks.shape)}")
    work = blocks.detach().float()
    ident = torch.eye(rows, device=work.device, dtype=work.dtype)
    gram_error = work @ work.transpose(-1, -2) - ident
    fro = torch.linalg.matrix_norm(gram_error, ord="fro", dim=(-2, -1))
    spectral = torch.linalg.matrix_norm(gram_error, ord=2, dim=(-2, -1))
    singular_values = torch.linalg.svdvals(work)
    return {
        "orth_error_fro_mean": float(fro.mean().item()),
        "orth_error_fro_max": float(fro.max().item()),
        "orth_error_spectral_mean": float(spectral.mean().item()),
        "singular_value_min_mean": float(singular_values.min(dim=-1).values.mean().item()),
        "singular_value_max_mean": float(singular_values.max(dim=-1).values.mean().item()),
    }


@torch.no_grad()
def role_orthogonality_metrics(model, submat_dim: int) -> dict[str, float | None]:
    if not hasattr(model, "role_parameters"):
        return {
            "orth_error_fro_mean": None,
            "orth_error_fro_max": None,
            "orth_error_spectral_mean": None,
            "singular_value_min_mean": None,
            "singular_value_max_mean": None,
        }
    metrics = []
    for param in model.role_parameters().values():
        if param.ndim != 3:
            continue
        rows, cols = param.shape[-2:]
        if submat_dim > cols or rows % submat_dim != 0:
            continue
        blocks = param.detach().reshape(-1, submat_dim, cols)
        metrics.append(orthogonality_metrics_for_blocks(blocks))
    if not metrics:
        return {
            "orth_error_fro_mean": None,
            "orth_error_fro_max": None,
            "orth_error_spectral_mean": None,
            "singular_value_min_mean": None,
            "singular_value_max_mean": None,
        }
    keys = metrics[0].keys()
    return {
        key: float(sum(item[key] for item in metrics) / len(metrics))
        if key != "orth_error_fro_max"
        else float(max(item[key] for item in metrics))
        for key in keys
    }


def perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    if loss > 50:
        return float("inf")
    return math.exp(loss)
