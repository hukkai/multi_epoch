from __future__ import annotations

from collections.abc import Iterable

import torch


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    *,
    exclude_param_ids: Iterable[int] | None = None,
    extra_decay_params: Iterable[torch.nn.Parameter] | None = None,
    extra_no_decay_params: Iterable[torch.nn.Parameter] | None = None,
) -> list[dict]:
    exclude_param_ids = set(exclude_param_ids or [])
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in exclude_param_ids:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    no_decay.extend(extra_no_decay_params or [])
    decay.extend(extra_decay_params or [])

    groups = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    return groups
