from __future__ import annotations

from dataclasses import dataclass

import torch

from ortho_llm.config import ALL_ROLES, ExperimentConfig
from ortho_llm.modeling.registry import ParameterRegistry, ensure_unique_parameter_ownership

from .muon import Muon
from .muon_orthogonal import MuonOrthogonal
from .param_groups import get_param_groups
from .so import SOOptimizer


@dataclass
class OptimBundle:
    main_optimizer: torch.optim.Optimizer | None
    role_optimizers: dict[str, torch.optim.Optimizer]
    role_to_optimizer: dict[str, str]

    def optimizers(self) -> list[torch.optim.Optimizer]:
        items: list[torch.optim.Optimizer] = []
        if self.main_optimizer is not None:
            items.append(self.main_optimizer)
        items.extend(self.role_optimizers.values())
        return items

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "main_optimizer": self.main_optimizer.state_dict() if self.main_optimizer is not None else None,
            "role_optimizers": {name: optimizer.state_dict() for name, optimizer in self.role_optimizers.items()},
            "role_to_optimizer": dict(self.role_to_optimizer),
        }

    def load_state_dict(self, state: dict) -> None:
        if self.main_optimizer is not None and state.get("main_optimizer") is not None:
            self.main_optimizer.load_state_dict(state["main_optimizer"])
        for name, optimizer_state in state.get("role_optimizers", {}).items():
            if name in self.role_optimizers:
                self.role_optimizers[name].load_state_dict(optimizer_state)


def resolve_role_owners(config: ExperimentConfig, registry: ParameterRegistry | None) -> dict[str, str]:
    if registry is None:
        return {}
    owners = {}
    for role in registry.roles:
        owner = config.optim.role_overrides.get(role, config.optim.default_role_optimizer)
        owners[role] = owner
    registry.validate_optimizer_owners(owners)
    return owners


def _build_role_optimizer(kind: str, params: list[torch.nn.Parameter], config: ExperimentConfig) -> torch.optim.Optimizer:
    optim = config.optim
    train = config.train
    if kind == "so":
        return SOOptimizer(
            params,
            lr=train.lr * optim.so_lr,
            betas=(optim.orth_beta1, optim.orth_beta2),
            eps=optim.orth_eps,
            submat_dim=optim.submat_dim,
        )
    if kind == "muon":
        return Muon(
            params,
            lr=optim.muon_lr,
            momentum=optim.muon_momentum,
            weight_decay=optim.muon_weight_decay,
            decay_lr=train.lr,
            nesterov=optim.muon_nesterov,
            ns_steps=optim.muon_ns_steps,
            eps=optim.muon_eps,
        )
    if kind == "muon_orthogonal":
        return MuonOrthogonal(
            params,
            lr=optim.muon_lr,
            momentum=optim.muon_momentum,
            weight_decay=optim.muon_weight_decay,
            decay_lr=train.lr,
            nesterov=optim.muon_nesterov,
            ns_steps=optim.muon_ns_steps,
            eps=optim.muon_eps,
            submat_dim=optim.submat_dim,
            norm_cap=optim.norm_cap,
        )
    raise ValueError(f"Unsupported role optimizer kind {kind!r}")


def build_optimizers(config: ExperimentConfig, model: torch.nn.Module) -> OptimBundle:
    registry = model.get_parameter_registry() if hasattr(model, "get_parameter_registry") else None
    role_params = registry.role_parameters() if registry is not None else {}
    role_to_owner = resolve_role_owners(config, registry)

    for role, owner in role_to_owner.items():
        if owner == "frozen":
            role_params[role].requires_grad_(False)

    excluded_role_param_ids = {id(param) for param in role_params.values()}
    adamw_role_params = [role_params[role] for role, owner in role_to_owner.items() if owner == "adamw"]
    main_groups = get_param_groups(
        model,
        config.train.weight_decay,
        exclude_param_ids=excluded_role_param_ids,
        extra_decay_params=adamw_role_params,
    )
    main_optimizer = None
    if main_groups:
        main_optimizer = torch.optim.AdamW(
            main_groups,
            lr=config.train.lr,
            betas=(config.optim.adamw_beta1, config.optim.adamw_beta2),
            eps=config.optim.adamw_eps,
        )

    params_by_kind: dict[str, list[torch.nn.Parameter]] = {}
    for role, owner in role_to_owner.items():
        if owner in {"adamw", "frozen"}:
            continue
        params_by_kind.setdefault(owner, []).append(role_params[role])

    role_optimizers = {
        kind: _build_role_optimizer(kind, params, config)
        for kind, params in params_by_kind.items()
    }

    ownership_groups = {"adamw": [] if main_optimizer is None else [p for group in main_groups for p in group["params"]]}
    for kind, optimizer in role_optimizers.items():
        ownership_groups[kind] = [p for group in optimizer.param_groups for p in group["params"]]
    ensure_unique_parameter_ownership(ownership_groups)

    enabled_unknown = sorted(set(role_to_owner) - set(ALL_ROLES))
    if enabled_unknown:
        raise ValueError(f"Unexpected role owners: {', '.join(enabled_unknown)}")

    return OptimBundle(
        main_optimizer=main_optimizer,
        role_optimizers=role_optimizers,
        role_to_optimizer=role_to_owner,
    )
