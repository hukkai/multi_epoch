from __future__ import annotations

import torch


DEFAULT_TAYLOR_ORDER = 4


def asym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x - x.mT)


def build_transition(
    x: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the reduced transition matrix T and basis B such that

        [I, 0] exp(T) B ~= X exp(Omega)

    This is the general version and does not assume x @ x.T == I.
    """
    G = x @ x.mT
    M = delta @ x.mT
    N = x @ delta.mT
    H = delta @ delta.mT
    K = asym(M)

    top = torch.cat((-(G @ K + N), G), dim=-1)
    bottom = torch.cat((-(M @ K + H), M), dim=-1)

    transition = torch.cat((top, bottom), dim=-2)
    basis = torch.cat((x, delta), dim=-2)

    return transition, basis


def taylor_coeff_exp(T: torch.Tensor, order: int = DEFAULT_TAYLOR_ORDER) -> torch.Tensor:
    batch, dim, _ = T.shape
    eye = torch.eye(dim, device=T.device, dtype=T.dtype).expand(batch, dim, dim)

    coeff = eye.clone()
    term = eye

    for degree in range(1, order + 1):
        term = (term @ T) / float(degree)
        coeff = coeff + term

    return coeff


@torch.no_grad()
def skew_update_taylor(
    x: torch.Tensor,
    delta: torch.Tensor,
    order: int = DEFAULT_TAYLOR_ORDER,
    exp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Fixed-order version of the exp-based fused update.

    This removes the adaptive generator norm branch and therefore avoids
    per-step GPU -> CPU synchronization from .item().

    It keeps the general transition construction, so it does not assume
    x @ x.T == I exactly.
    """
    if x.ndim != 3 or delta.ndim != 3:
        raise ValueError(
            f"Expected rank-3 inputs, got x.ndim={x.ndim}, delta.ndim={delta.ndim}."
        )
    if x.shape != delta.shape:
        raise ValueError(
            f"Expected matching shapes, got x.shape={x.shape}, delta.shape={delta.shape}."
        )

    transition, basis = build_transition(x, delta)

    if exp_dtype is None:
        exp_dtype = torch.float32 if transition.dtype in (torch.float16, torch.bfloat16) else transition.dtype

    coeff = taylor_coeff_exp(transition.to(dtype=exp_dtype), order)

    n = x.size(-2)
    return coeff[:, :n, :].to(dtype=x.dtype) @ basis


@torch.no_grad()
def skew_update(
    x: torch.Tensor,
    delta: torch.Tensor,
    exp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    M = x.mT @ delta
    N = delta @ x.mT
    K = M - M.mT - x.mT @ asym(N) @ x

    if exp_dtype is None:
        exp_dtype = torch.float32 if K.dtype in (torch.float16, torch.bfloat16) else K.dtype
    
    return x @ torch.matrix_exp(K.to(dtype=exp_dtype)).to(dtype=x.dtype)