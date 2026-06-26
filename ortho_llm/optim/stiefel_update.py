from __future__ import annotations

import torch

from .ops import polar as exact_polar

TAYLOR2_MAX_ERR = 0.06
TAYLOR3_MAX_ERR = 0.14
TAYLOR4_MAX_ERR = 0.28

_COEFFS2 = (-0.5, 0.375)
_COEFFS3 = (-0.5, 0.375, -0.3125)
_COEFFS4 = (-0.5, 0.375, -0.3125, 0.2734375)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def stiefel_project(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return grad - _symmetrize(grad @ x.transpose(-1, -2)) @ x


def _screen_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _compile_stiefel_enabled() -> bool:
    return True


def _validate_shape(a: torch.Tensor) -> None:
    if a.ndim != 3:
        raise ValueError(f"expected a to have shape (b, n, m), got {tuple(a.shape)}")
    if a.shape[-2] > a.shape[-1]:
        raise ValueError(f"expected n <= m, got shape {tuple(a.shape)}")


def _gram_error(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gram_error = a @ a.transpose(-1, -2)
    gram_error.diagonal(dim1=-2, dim2=-1).sub_(1)
    err = torch.linalg.matrix_norm(gram_error, ord="fro", dim=(-2, -1))
    return gram_error, err


def _apply_series(
    a: torch.Tensor,
    gram_error: torch.Tensor,
    coeffs: tuple[float, ...],
) -> torch.Tensor:
    ident = torch.eye(gram_error.shape[-1], device=gram_error.device, dtype=gram_error.dtype)
    poly = coeffs[-1] * gram_error
    for coeff in reversed(coeffs[:-1]):
        poly = (poly + coeff * ident) @ gram_error
    return a + poly @ a


def _apply_series2_eager(a: torch.Tensor, gram_error: torch.Tensor) -> torch.Tensor:
    ident = torch.eye(gram_error.shape[-1], device=gram_error.device, dtype=gram_error.dtype)
    poly = (_COEFFS2[1] * gram_error + _COEFFS2[0] * ident) @ gram_error
    return a + poly @ a


if _compile_stiefel_enabled() and hasattr(torch, "compile"):
    _apply_series2 = torch.compile(_apply_series2_eager, fullgraph=True)
else:
    _apply_series2 = _apply_series2_eager


@torch.no_grad()
def fast_polar(
    a: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
    taylor2_max_err: float = TAYLOR2_MAX_ERR,
    taylor3_max_err: float = TAYLOR3_MAX_ERR,
    taylor4_max_err: float = TAYLOR4_MAX_ERR,
) -> torch.Tensor:
    _validate_shape(a)

    work = a.to(_screen_dtype(a.dtype))
    gram_error, err = _gram_error(work)
    max_err = err.max().item()

    if max_err <= tolerance:
        return a
    if max_err < taylor2_max_err:
        return _apply_series2(work, gram_error).to(a.dtype)
    if max_err < taylor3_max_err:
        return _apply_series(work, gram_error, _COEFFS3).to(a.dtype)
    if max_err < taylor4_max_err:
        return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)
    return exact_polar(a, tolerance=0.0, eps=eps)


def stiefel_update_taylor(
    x: torch.Tensor,
    update: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
    taylor2_max_err: float = TAYLOR2_MAX_ERR,
    taylor3_max_err: float = TAYLOR3_MAX_ERR,
    taylor4_max_err: float = TAYLOR4_MAX_ERR,
    projected: bool = False,
) -> torch.Tensor:
    if not projected:
        update = stiefel_project(x, update)
    return fast_polar(
        x + update,
        tolerance=tolerance,
        eps=eps,
        taylor2_max_err=taylor2_max_err,
        taylor3_max_err=taylor3_max_err,
        taylor4_max_err=taylor4_max_err,
    )
