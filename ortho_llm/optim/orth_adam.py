from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from .ops import polar
from .stiefel import stiefel_update_taylor


class OrthAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        submat_dim: int = 64,
        strict_stiefel: bool = True,
    ) -> None:
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        defaults = {
            "lr": lr,
            "betas": betas,
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
    def _local_slice(param: torch.Tensor, world_size: int, rank: int) -> slice:
        if world_size == 1:
            return slice(None)
        if param.ndim != 3:
            raise ValueError(
                "distributed OrthAdam expects shape (num_matrices, rows, cols), "
                f"got {tuple(param.shape)}"
            )
        if param.shape[0] % world_size != 0:
            raise ValueError("OrthAdam parameter leading dimension must be divisible by world size")
        per_rank = param.shape[0] // world_size
        return slice(rank * per_rank, (rank + 1) * per_rank)

    @staticmethod
    def _reshape_stiefel(param_slice: torch.Tensor, submat_dim: int) -> torch.Tensor:
        if param_slice.ndim != 3:
            raise ValueError(f"OrthAdam expects shape (chunks, rows, cols), got {tuple(param_slice.shape)}")
        rows, cols = param_slice.shape[-2:]
        if submat_dim > cols:
            raise ValueError(f"submat_dim {submat_dim} must be <= matrix cols {cols}")
        if rows % submat_dim != 0:
            raise ValueError(f"Matrix rows {rows} must be divisible by submat_dim {submat_dim}")
        return param_slice.reshape(-1, submat_dim, cols)

    @torch.no_grad()
    def step(
        self,
        closure: Any | None = None,
        *,
        lr: float | None = None,
        is_last: bool = False,
    ) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size, rank = self._distributed_context()

        for group in self.param_groups:
            group_lr = group["lr"] if lr is None else lr
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            submat_dim = group["submat_dim"]
            strict_stiefel = group["strict_stiefel"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                local_slice = self._local_slice(param, world_size, rank)
                param_slice = param.data[local_slice]
                grad_slice = param.grad[local_slice].detach().float()
                x = self._reshape_stiefel(param_slice.float(), submat_dim)
                grad = self._reshape_stiefel(grad_slice, submat_dim)

                state = self.state[param]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    state["step"] = torch.tensor(0.0, device=grad.device)

                state["step"] += 1
                m = state["m"]
                v = state["v"]
                m += (grad - m) * (1.0 - beta1)
                v += (grad**2 - v) * (1.0 - beta2)

                m_hat = m / (1.0 - beta1 ** state["step"])
                v_hat = v / (1.0 - beta2 ** state["step"])
                update = -group_lr * m_hat / (v_hat.sqrt() + eps)

                new_x = stiefel_update_taylor(x, update)
                if is_last and strict_stiefel:
                    new_x = polar(new_x)

                new_slice = new_x.reshape_as(param_slice).to(dtype=param.dtype)
                if world_size > 1:
                    dist.all_gather_into_tensor(param.data, new_slice.contiguous())
                else:
                    param_slice.copy_(new_slice)
                param.grad = None

        return loss
