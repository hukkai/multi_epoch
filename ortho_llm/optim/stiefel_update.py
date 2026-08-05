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


def _gram_error_matrix(a: torch.Tensor) -> torch.Tensor:
    gram_error = a @ a.transpose(-1, -2)
    gram_error.diagonal(dim1=-2, dim2=-1).sub_(1)
    return gram_error


def _gram_error(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gram_error = _gram_error_matrix(a)
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


def _stiefel_update_series4_eager(
    x: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    update = stiefel_project(x, update)
    a = x + update
    work = a.to(_screen_dtype(a.dtype))
    gram_error = _gram_error_matrix(work)
    return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)


def _stiefel_update_series4_unprojected_eager(
    x: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    a = x + update
    work = a.to(_screen_dtype(a.dtype))
    gram_error = _gram_error_matrix(work)
    return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)


def _polar_taylor4_eager(a: torch.Tensor) -> torch.Tensor:
    work = a.to(_screen_dtype(a.dtype))
    gram_error = _gram_error_matrix(work)
    return _apply_series(work, gram_error, _COEFFS4).to(a.dtype)


if _compile_stiefel_enabled() and hasattr(torch, "compile"):
    _stiefel_update_series4 = torch.compile(
        _stiefel_update_series4_eager,
        fullgraph=True,
    )
    _stiefel_update_series4_unprojected = torch.compile(
        _stiefel_update_series4_unprojected_eager,
        fullgraph=True,
    )
    _polar_taylor4 = torch.compile(_polar_taylor4_eager, fullgraph=True)
else:
    _stiefel_update_series4 = _stiefel_update_series4_eager
    _stiefel_update_series4_unprojected = _stiefel_update_series4_unprojected_eager
    _polar_taylor4 = _polar_taylor4_eager


@torch.no_grad()
def polar_taylor4(a: torch.Tensor) -> torch.Tensor:
    """Apply a fixed fourth-order inverse-square-root polar approximation."""
    _validate_shape(a)
    return _polar_taylor4(a)


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


@torch.no_grad()
def stiefel_update_taylor(
    x: torch.Tensor,
    update: torch.Tensor,
    do_projection: bool = True,
) -> torch.Tensor:
    _validate_shape(x)
    if x.shape != update.shape:
        raise ValueError(
            "expected x and update to have the same shape, "
            f"got {tuple(x.shape)} and {tuple(update.shape)}"
        )
    if do_projection:
        return _stiefel_update_series4(x, update)
    return _stiefel_update_series4_unprojected(x, update)
