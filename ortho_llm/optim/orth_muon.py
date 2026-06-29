from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from .muon import orthogonalize_newton_schulz
from .ops import polar
from .stiefel_update import stiefel_update_taylor


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

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            submat_dim = group["submat_dim"]
            strict_stiefel = group["strict_stiefel"]

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
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param_slice, dtype=torch.float32)

                buffer = state["momentum_buffer"]
                if buffer.shape != grad.shape:
                    raise ValueError("OrthMuon momentum buffer shape does not match the local parameter shard")
                buffer.lerp_(grad, 1.0 - momentum)

                update = torch.lerp(grad, buffer, momentum) if nesterov else buffer
                update = orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)

                scale = 0.2 * math.sqrt(max(rows, cols))

                x = param_slice.float().reshape(-1, submat_dim, cols)
                update = update.reshape_as(x)

                eta = lr * scale
                update.mul_(-eta)
                new_x = stiefel_update_taylor(x, update)

                if is_last and strict_stiefel:
                    new_x = polar(new_x)

                new_param_slice = new_x.reshape_as(param_slice).to(dtype=param.dtype)
                if world_size > 1:
                    dist.all_gather_into_tensor(param.data, new_param_slice.contiguous())
                else:
                    param_slice.copy_(new_param_slice)

        return loss
