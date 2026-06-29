from __future__ import annotations

import torch

from ..stiefel_update import fast_polar


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def flow_update(
    x: torch.Tensor,
    delta: torch.Tensor,
    eta: float,
    *,
    project: bool = True,
) -> torch.Tensor:
    """
    Second-order Taylor step for dX/dt = P_X(delta).

    Args:
        x: Current row-Stiefel blocks with shape (batch, n, m).
        delta: Raw ambient update direction, with the same shape as x.
        eta: Signed flow time / step size.
        project: Whether to re-project the Taylor step with fast polar.

    Returns:
        The second-order projected-flow Taylor approximation.
    """
    if x.ndim != 3:
        raise ValueError(f"expected x to have shape (batch, n, m), got {tuple(x.shape)}")
    if x.shape != delta.shape:
        raise ValueError(f"expected matching shapes, got x.shape={tuple(x.shape)}, delta.shape={tuple(delta.shape)}")
    if x.shape[-2] > x.shape[-1]:
        raise ValueError(f"expected row-Stiefel blocks with n <= m, got shape {tuple(x.shape)}")

    delta_x_t = delta @ x.transpose(-1, -2)
    s = _symmetrize(delta_x_t)
    u = delta - s @ x
    t = _symmetrize(delta @ u.transpose(-1, -2))
    v_correction = t @ x + s @ u
    updated = x + eta * u - 0.5 * eta * eta * v_correction
    if project:
        updated = fast_polar(updated)
    return updated
