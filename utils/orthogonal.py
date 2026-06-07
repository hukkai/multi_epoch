from __future__ import annotations

import torch

from .orthogonal_base import SOOptimizer as SOOptimizer_base
from .orthogonal_scalar import SOOptimizer as SOOptimizer_scalar
from .orthogonal_geo import SOOptimizer as SOOptimizer_geo


def get_so_optimizer(
    param: torch.nn.Parameter,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    num_submatrices: int = 8,
    strict_stiefel: bool = True,
    method: str = "scalar",
) -> SOOptimizer_base:
    if method == "scalar":
        return SOOptimizer_scalar(param, lr, betas, eps, num_submatrices, strict_stiefel)
    elif method == "geo":
        return SOOptimizer_geo(param, lr, betas, eps, num_submatrices, strict_stiefel)
    elif method == "base":
        return SOOptimizer_base(param, lr, betas, eps, num_submatrices, strict_stiefel)
    else:
        raise ValueError(f"Unknown method {method}")