from __future__ import annotations

import math

import torch
import torch.distributed as dist


from .polar_taylor import orthogonal_rows_update_taylor, orthogonal_rows_exact


class SOOptimizer:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        sub_matrix: int = 8,
        strict_stiefel: bool = True,
        weight_decay: float = 0.1,
        min_norm: float = 0.5,
        max_norm: float | None = None,
    ) -> None:
        if min_norm < 0:
            raise ValueError(f"min_norm must be non-negative, got {min_norm}")
        if max_norm is not None and max_norm < min_norm:
            raise ValueError(f"max_norm must be >= min_norm, got {max_norm} < {min_norm}")

        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.strict_stiefel = strict_stiefel
        self.weight_decay = weight_decay
        self.min_norm = min_norm
        self.max_norm = max_norm

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
        self.buffer = torch.zeros_like(param.data)
        self.step_count = torch.tensor(0.0, device=self.m.device)

        self.dim = self.m.shape[1]
        if self.dim % sub_matrix != 0:
            raise ValueError(f"Matrix dim {self.dim} must be divisible by sub_matrix {sub_matrix}")

        self.orth_dim = self.dim // sub_matrix


    def state_dict(self) -> dict:
        return {
            "m": self.m,
            "v": self.v,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "strict_stiefel": self.strict_stiefel,
            "weight_decay": self.weight_decay,
            "min_norm": self.min_norm,
            "max_norm": self.max_norm,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.m = state.get("m", self.m).to(device=self.m.device, dtype=self.m.dtype)
        self.v = state.get("v", self.v).to(device=self.v.device, dtype=self.v.dtype)
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.strict_stiefel = state.get("strict_stiefel", self.strict_stiefel)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.min_norm = state.get("min_norm", self.min_norm)
        self.max_norm = state.get("max_norm", self.max_norm)
        self.step_count = state.get("step_count", self.step_count).to(
            device=self.step_count.device, dtype=self.step_count.dtype
        )

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

        delta = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * x
        x = x.reshape(-1, self.orth_dim, self.dim)
        update = (-lr * delta).reshape_as(x)

        new_x = orthogonal_rows_update_taylor(
            x,
            update,
            min_norm=self.min_norm,
            max_norm=self.max_norm,
        )

        if is_last and self.strict_stiefel:
            new_x = orthogonal_rows_exact(new_x, min_norm=self.min_norm, max_norm=self.max_norm)

        new_x = new_x.reshape_as(self.m)

        self.buffer.zero_()
        self.buffer[self.local_slice] = new_x
        if dist.is_initialized():
            dist.all_reduce(self.buffer)
        self.param.data.copy_(self.buffer)
        self.param.grad = None
