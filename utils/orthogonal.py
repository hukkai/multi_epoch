from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .ops import polar
from .polar_taylor import stiefel_update_taylor


class SOOptimizer:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        num_submatrices: int = 8,
        strict_stiefel: bool = True,
    ) -> None:
        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.strict_stiefel = strict_stiefel

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        total = param.shape[0]
        if total % self.world_size != 0:
            raise ValueError("chunk_weights must be divisible by world size")

        per_rank = total // self.world_size
        self.local_slice = slice(self.rank * per_rank, (self.rank + 1) * per_rank)

        self.m = torch.zeros_like(param.data[self.local_slice])
        self.v = torch.zeros_like(self.m)

        self.step_count = torch.tensor(0.0, device=self.m.device)

        self.dim = self.m.shape[1]
        if self.dim % num_submatrices != 0:
            raise ValueError(f"Matrix dim {self.dim} must be divisible by num_submatrices {num_submatrices}")

        self.orth_dim = self.dim // num_submatrices


    def step(self, lr: float | None = None, is_last: bool = False) -> None:
        if self.param.grad is None:
            return

        lr = lr if lr is not None else self.lr
        self.step_count += 1

        x = self.param.data[self.local_slice]
        grad = self.param.grad[self.local_slice]

        self.m += (grad - self.m) * (1.0 - self.beta1)
        self.v += (grad ** 2 - self.v) * (1.0 - self.beta2)

        m_hat = self.m / (1.0 - self.beta1**self.step_count)
        v_hat = self.v / (1.0 - self.beta2**self.step_count)


        x = x.reshape(-1, self.orth_dim, self.dim)
        update = -lr * m_hat / (v_hat.sqrt() + self.eps)
        update = update.reshape_as(x)

        new_x = stiefel_update_taylor(x, update)

        if is_last and self.strict_stiefel:
            new_x = polar(new_x)

        new_x = new_x.reshape_as(self.m)

        if dist.is_initialized():
            dist.all_gather_into_tensor(self.param.data, new_x.contiguous())
        else:
            self.param.data.copy_(new_x)
        self.param.grad = None