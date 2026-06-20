from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MatrixSpec:
    role: str
    layer_idx: int
    logical_shape: tuple[int, int]
    storage_name: str
    storage_slice: tuple[int | slice, ...]
    materialization: str
    optimizer_owner: str = "frozen"


class ParameterRegistry:
    def __init__(self, specs: Iterable[MatrixSpec], role_params: dict[str, nn.Parameter]) -> None:
        self.specs = list(specs)
        self._role_params = dict(role_params)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self._role_params)

    def role_parameters(self) -> dict[str, nn.Parameter]:
        return dict(self._role_params)

    def chunk_count(self) -> int:
        return sum(param.shape[0] for param in self._role_params.values())

    def specs_for_role(self, role: str) -> list[MatrixSpec]:
        return [spec for spec in self.specs if spec.role == role]

    def validate_optimizer_owners(self, role_to_owner: dict[str, str]) -> None:
        missing = sorted(set(self._role_params) - set(role_to_owner))
        extra = sorted(set(role_to_owner) - set(self._role_params))
        if missing:
            raise ValueError(f"Missing optimizer owners for roles: {', '.join(missing)}")
        if extra:
            raise ValueError(f"Optimizer owners provided for disabled roles: {', '.join(extra)}")

    def trainable_role_parameter_ids(self) -> set[int]:
        return {id(param) for param in self._role_params.values() if param.requires_grad}


def ensure_unique_parameter_ownership(groups: dict[str, list[torch.nn.Parameter]]) -> None:
    seen: dict[int, str] = {}
    for owner, params in groups.items():
        for param in params:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                raise ValueError(f"Parameter is owned by both {seen[param_id]} and {owner}")
            seen[param_id] = owner
