from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from .muon import orthogonalize_newton_schulz
from .ops import polar
from .stiefel_update import stiefel_project, stiefel_update_taylor

_SPECTRAL_NORM_FIRST_ITERS = 20
_SPECTRAL_NORM_WARM_ITERS = 3
_SPECTRAL_NORM_SAFETY_FACTOR = 1.01


@torch.no_grad()
def spectral_norm(
    x: torch.Tensor,
    cached_u: torch.Tensor | None = None,
    n_iter: int = 10,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate the largest singular value of each matrix in x.

    Args:
        x: Tensor of shape [B, m, n].
        cached_u: Optional left singular-vector estimates from a previous call.
        n_iter: Number of power iterations.
        eps: Numerical stability constant.

    Returns:
        Spectral norm estimates of shape [B] and updated left singular-vector
        estimates, both returned in float32.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, m, n], got {tuple(x.shape)}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    batch, m, n = x.shape

    # Use the orientation with fewer rows, reducing the size of u.
    work = x if m <= n else x.transpose(-2, -1)
    work = work.to(torch.bfloat16 if x.device.type == "cuda" else torch.float32)
    expected_u_shape = (batch, work.shape[-2], 1)

    if cached_u is not None:
        if tuple(cached_u.shape) != expected_u_shape:
            raise ValueError(
                f"cached_u must have shape {expected_u_shape}, got {tuple(cached_u.shape)}"
            )
        u = cached_u.to(device=x.device, dtype=torch.float32)
    else:
        u = torch.randn(
            batch,
            work.shape[-2],
            1,
            device=x.device,
            dtype=torch.float32,
        )

    u_norm = torch.linalg.vector_norm(u, dim=-2, keepdim=True)
    u = torch.where(
        u_norm > eps,
        u / u_norm,
        torch.full_like(u, u.shape[-2] ** -0.5),
    )

    for _ in range(n_iter):
        if x.device.type == "cuda":
            v = torch.bmm(
                work.transpose(-2, -1),
                u.to(torch.bfloat16),
                out_dtype=torch.float32,
            )
        else:
            v = torch.bmm(work.transpose(-2, -1), u)
        v = v / torch.linalg.vector_norm(v, dim=-2, keepdim=True).clamp_min(eps)

        if x.device.type == "cuda":
            z = torch.bmm(
                work,
                v.to(torch.bfloat16),
                out_dtype=torch.float32,
            )
        else:
            z = torch.bmm(work, v)
        sigma = torch.linalg.vector_norm(z, dim=-2, keepdim=True)
        next_u = z / sigma.clamp_min(eps)
        # Preserve a valid direction for zero matrices so a later non-zero
        # update can still use this vector as a warm start.
        u = torch.where(sigma > eps, next_u, u)

    return sigma.flatten(), u.float()


class OrthMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        submat_dim: int = 64,
        strict_stiefel: bool = True,
        async_gather: bool = True,
    ) -> None:
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if submat_dim <= 0:
            raise ValueError(f"Invalid submat_dim value: {submat_dim}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
            "submat_dim": submat_dim,
            "strict_stiefel": strict_stiefel,
            "async_gather": async_gather,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _distributed_context() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    @staticmethod
    def _validate_param(param: torch.Tensor, submat_dim: int) -> tuple[int, int]:
        if param.ndim != 3:
            raise ValueError(
                "OrthMuon expects parameters with shape (num_matrices, rows, cols), "
                f"got {tuple(param.shape)}"
            )
        rows, cols = param.shape[-2:]
        if submat_dim > cols:
            raise ValueError(f"submat_dim {submat_dim} must be <= matrix cols {cols}")
        if rows % submat_dim != 0:
            raise ValueError(f"Matrix rows {rows} must be divisible by submat_dim {submat_dim}")
        return rows, cols

    @staticmethod
    def _local_slice(param: torch.Tensor, world_size: int, rank: int) -> slice:
        if world_size == 1:
            return slice(None)
        if param.shape[0] % world_size != 0:
            raise ValueError(
                "OrthMuon parameter leading dimension "
                f"{param.shape[0]} must be divisible by world size {world_size}"
            )
        per_rank = param.shape[0] // world_size
        return slice(rank * per_rank, (rank + 1) * per_rank)

    @torch.no_grad()
    def step(self, closure: Any | None = None, is_last: bool = False) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size, rank = self._distributed_context()
        pending_gathers = []

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            submat_dim = group["submat_dim"]
            strict_stiefel = group["strict_stiefel"]
            async_gather = group.get("async_gather", True)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("OrthMuon does not support sparse gradients")

                rows, cols = self._validate_param(param, submat_dim)
                local_slice = self._local_slice(param, world_size, rank)
                param_slice = param.data[local_slice]
                grad = param.grad[local_slice].detach().float()

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param_slice, dtype=torch.float32)

                buffer = state["momentum_buffer"]
                if buffer.shape != grad.shape:
                    raise ValueError("OrthMuon momentum buffer shape does not match the local parameter shard")
                buffer.lerp_(grad, 1.0 - momentum)

                update = torch.lerp(grad, buffer, momentum) if nesterov else buffer
                update = orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)

                scale = 0.2 * math.sqrt(max(rows, cols))

                x = param_slice.float().reshape(-1, submat_dim, cols)

                update = stiefel_project(x, update.reshape_as(x))
                update = update.reshape_as(grad)
                cached_u = state.get("spectral_norm_u")
                n_iter = (
                    _SPECTRAL_NORM_FIRST_ITERS
                    if cached_u is None
                    else _SPECTRAL_NORM_WARM_ITERS
                )
                norm_estimate, cached_u = spectral_norm(
                    update,
                    cached_u=cached_u,
                    n_iter=n_iter,
                )
                state["spectral_norm_u"] = cached_u
                clip_scale = (_SPECTRAL_NORM_SAFETY_FACTOR * norm_estimate).clamp_min(1.0)
                update.div_(clip_scale.view(-1, 1, 1))
                update = update.reshape_as(x)

                update.mul_(-lr * scale)
                new_x = stiefel_update_taylor(x, update, do_projection=False)

                if is_last and strict_stiefel:
                    new_x = polar(new_x)

                new_param_slice = new_x.reshape_as(param_slice).to(dtype=param.dtype)
                if world_size > 1:
                    send_buffer = new_param_slice.contiguous()
                    if async_gather:
                        work = dist.all_gather_into_tensor(param.data, send_buffer, async_op=True)
                        pending_gathers.append((work, send_buffer))
                    else:
                        dist.all_gather_into_tensor(param.data, send_buffer)
                else:
                    param_slice.copy_(new_param_slice)

        for work, _send_buffer in pending_gathers:
            work.wait()

        return loss
