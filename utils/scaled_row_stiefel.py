from __future__ import annotations

import torch

from .ops import polar


SCALED_ROW_STIEFEL_C_MIN = 0.5


def _screen_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def project_scaled_row_stiefel_update(
    x: torch.Tensor,
    update: torch.Tensor,
    c_min: float = SCALED_ROW_STIEFEL_C_MIN,
    eps: float = 1e-12,
) -> torch.Tensor:
    work_dtype = torch.promote_types(_screen_dtype(x.dtype), _screen_dtype(update.dtype))
    x_work = x.to(work_dtype)
    update_work = update.to(work_dtype)

    n = x.shape[-2]
    c = x_work.square().sum(dim=(-2, -1), keepdim=True).div(n).clamp_min(c_min)
    sym_update_x_t = _symmetrize(update_work @ x_work.transpose(-1, -2))
    mu = sym_update_x_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).div(n)
    eye = torch.eye(n, device=x.device, dtype=work_dtype)
    traceless = sym_update_x_t - mu.unsqueeze(-1) * eye

    projected = update_work - (traceless @ x_work).div(c.clamp_min(eps))
    return projected.to(update.dtype)


def retract_scaled_row_stiefel(
    w: torch.Tensor,
    c_min: float = SCALED_ROW_STIEFEL_C_MIN,
    eps: float = 1e-7,
) -> torch.Tensor:
    work_dtype = _screen_dtype(w.dtype)
    w_work = w.to(work_dtype)

    n = w.shape[-2]
    c_new = w_work.square().sum(dim=(-2, -1), keepdim=True).div(n).clamp_min(c_min)
    q = polar(w_work, tolerance=0.0, eps=eps)
    return (c_new.sqrt() * q).to(w.dtype)


def scaled_row_stiefel_update(
    x: torch.Tensor,
    update: torch.Tensor,
    c_min: float = SCALED_ROW_STIEFEL_C_MIN,
    eps: float = 1e-7,
) -> torch.Tensor:
    update = project_scaled_row_stiefel_update(x, update, c_min=c_min, eps=eps)
    return retract_scaled_row_stiefel(x + update, c_min=c_min, eps=eps)
