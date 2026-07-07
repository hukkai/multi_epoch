from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalize_newton_schulz(
    x: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"expected x to have at least 2 dimensions, got shape {tuple(x.shape)}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    original_shape = x.shape
    rows, cols = original_shape[-2:]
    work = x.reshape(-1, rows, cols).float()

    transposed = rows > cols
    if transposed:
        work = work.transpose(-1, -2)

    work = work / (work.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = _NS_COEFFS

    for _ in range(steps):
        xx_t = work @ work.transpose(-1, -2)
        work = a * work + b * (xx_t @ work) + c * (xx_t @ xx_t @ work)

    if transposed:
        work = work.transpose(-1, -2)

    return work.reshape(original_shape).to(dtype=x.dtype)


class Muon(torch.optim.Optimizer):
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
        async_gather: bool = True,
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

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "decay_lr": decay_lr,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
            "async_gather": async_gather,
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
        if param.ndim < 3:
            raise ValueError(
                "distributed Muon expects sharded parameters with shape (num_matrices, rows, cols), "
                f"got {tuple(param.shape)}"
            )
        if param.shape[0] % world_size != 0:
            raise ValueError(
                f"Muon parameter leading dimension {param.shape[0]} must be divisible by world size {world_size}"
            )
        per_rank = param.shape[0] // world_size
        return slice(rank * per_rank, (rank + 1) * per_rank)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size, rank = self._distributed_context()
        pending_gathers = []

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
            async_gather = group.get("async_gather", True)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if param.ndim < 2:
                    raise ValueError(
                        f"Muon expects matrix-like parameters with at least 2 dimensions, got {tuple(param.shape)}"
                    )

                local_slice = self._local_slice(param, world_size, rank)
                param_slice = param.data[local_slice]
                grad = param.grad[local_slice].detach().float()

                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param_slice, dtype=torch.float32)

                buffer = state["momentum_buffer"]
                if buffer.shape != grad.shape:
                    raise ValueError("Muon momentum buffer shape does not match the local parameter shard")
                buffer.lerp_(grad, 1.0 - momentum)

                update = torch.lerp(grad, buffer, momentum) if nesterov else buffer
                update = orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)

                rows, cols = param.shape[-2:]
                scale = 0.2 * math.sqrt(max(rows, cols))

                if weight_decay != 0.0:
                    param_slice.mul_(1.0 - decay_lr * weight_decay)
                param_slice.add_(update.to(dtype=param.dtype), alpha=-lr * scale)

                if world_size > 1:
                    send_buffer = param_slice.detach().clone().contiguous()
                    if async_gather:
                        work = dist.all_gather_into_tensor(param.data, send_buffer, async_op=True)
                        pending_gathers.append((work, send_buffer))
                    else:
                        dist.all_gather_into_tensor(param.data, send_buffer)

        for work, _send_buffer in pending_gathers:
            work.wait()

        return loss
