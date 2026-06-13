from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from .muon import orthogonalize_newton_schulz
from .ops import polar
from .polar_taylor import stiefel_update_taylor


def l2_normalize(x: torch.Tensor, 
                 num_steps: int = 10,
                 eps: float = 1e-12
                 ) -> torch.Tensor:
    for _ in range(num_steps):
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(eps)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x


class MuonOrthogonal(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        decay_lr: float | None = None,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        num_submatrices: int = 8,
        strict_stiefel: bool = True,
        simple_update: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if decay_lr is not None and decay_lr < 0.0:
            raise ValueError(f"Invalid decay learning rate: {decay_lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if num_submatrices <= 0:
            raise ValueError(f"Invalid num_submatrices value: {num_submatrices}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "decay_lr": decay_lr,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
            "num_submatrices": num_submatrices,
            "strict_stiefel": strict_stiefel,
            "simple_update": simple_update,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _distributed_context() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    @staticmethod
    def _validate_param(param: torch.Tensor, num_submatrices: int) -> tuple[int, int]:
        if param.ndim != 3 or param.shape[-2] != param.shape[-1]:
            raise ValueError(
                "MuonOrthogonal expects parameters with shape (num_matrices, dim, dim), "
                f"got {tuple(param.shape)}"
            )

        dim = param.shape[-1]
        if dim % num_submatrices != 0:
            raise ValueError(f"Matrix dim {dim} must be divisible by num_submatrices {num_submatrices}")
        return dim, dim // num_submatrices

    @staticmethod
    def _local_slice(param: torch.Tensor, world_size: int, rank: int) -> slice:
        if world_size == 1:
            return slice(None)
        if param.shape[0] % world_size != 0:
            raise ValueError(
                "MuonOrthogonal parameter leading dimension "
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

        for group in self.param_groups:
            lr = group["lr"]
            decay_lr = group["decay_lr"]
            if decay_lr is None:
                decay_lr = lr
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            num_submatrices = group["num_submatrices"]
            strict_stiefel = group["strict_stiefel"]
            simple_update = group["simple_update"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("MuonOrthogonal does not support sparse gradients")

                dim, orth_dim = self._validate_param(param, num_submatrices)
                local_slice = self._local_slice(param, world_size, rank)
                param_slice = param.data[local_slice]
                grad = param.grad[local_slice].detach().float()

                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param_slice, dtype=torch.float32)

                buffer = state["momentum_buffer"]
                if buffer.shape != grad.shape:
                    raise ValueError(
                        "MuonOrthogonal momentum buffer shape does not match the local parameter shard; "
                        "recreate the optimizer after changing distributed world size or parameter shape"
                    )
                buffer.lerp_(grad, 1.0 - momentum)

                update = torch.lerp(grad, buffer, momentum) if nesterov else buffer

                if not simple_update:
                    update = orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)
                else:
                    update = l2_normalize(update, num_steps=ns_steps * 2, eps=eps)

                scale = 0.2 * math.sqrt(dim)
                if weight_decay != 0.0:
                    param_slice.mul_(1.0 - decay_lr * weight_decay)

                x = param_slice.reshape(-1, orth_dim, dim)
                update = update.reshape_as(x).mul_(-lr * scale)
                new_x = stiefel_update_taylor(x, update)

                if is_last and strict_stiefel:
                    new_x = polar(new_x)

                new_param_slice = new_x.reshape_as(param_slice).to(dtype=param.dtype)
                if world_size > 1:
                    dist.all_gather_into_tensor(param.data, new_param_slice.contiguous())
                else:
                    param_slice.copy_(new_param_slice)

        return loss
