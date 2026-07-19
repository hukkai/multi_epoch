from __future__ import annotations

from dataclasses import dataclass

import torch

from ortho_llm.config import ALL_ROLES, ExperimentConfig
from ortho_llm.modeling.registry import ParameterRegistry, ensure_unique_parameter_ownership

from .muon import Muon
from .orth_muon import OrthMuon
from .param_groups import get_param_groups
from .orth_adam import OrthAdam


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

    def _validate_role_to_optimizer(self, state: dict) -> None:
        if "role_to_optimizer" not in state:
            raise ValueError("Optimizer state is missing role_to_optimizer")
        saved_mapping = state["role_to_optimizer"]
        if not isinstance(saved_mapping, dict):
            raise ValueError("Optimizer role_to_optimizer must be a mapping")
        if list(saved_mapping.items()) != list(self.role_to_optimizer.items()):
            raise ValueError(
                "Saved role_to_optimizer does not match the current ordered role ownership"
            )

    def _validate_main_optimizer(self, state: dict) -> None:
        if "main_optimizer" not in state:
            raise ValueError("Optimizer state is missing main_optimizer")
        saved_has_main = state["main_optimizer"] is not None
        current_has_main = self.main_optimizer is not None
        if saved_has_main != current_has_main:
            raise ValueError(
                "Saved main optimizer presence does not match the current optimizer bundle"
            )

    def _validate_role_optimizer_states(self, states: dict) -> None:
        if not isinstance(states, dict):
            raise ValueError("Role optimizer states must be a mapping")
        saved_kinds = set(states)
        current_kinds = set(self.role_optimizers)
        if saved_kinds != current_kinds:
            missing = sorted(current_kinds - saved_kinds)
            unexpected = sorted(saved_kinds - current_kinds)
            raise ValueError(
                "Saved role optimizer kinds do not match the current optimizer bundle "
                f"(missing={missing}, unexpected={unexpected})"
            )

    def load_role_optimizer_states(self, states: dict) -> None:
        """Load a complete set of rank-local role optimizer states."""
        self._validate_role_optimizer_states(states)
        for kind, optimizer in self.role_optimizers.items():
            optimizer.load_state_dict(states[kind])

    def load_state_dict(self, state: dict, *, load_role_optimizers: bool = True) -> None:
        """Load a full state, or only common state when role payloads are rank-local."""
        self._validate_role_to_optimizer(state)
        self._validate_main_optimizer(state)

        role_optimizer_states = None
        if load_role_optimizers:
            if "role_optimizers" not in state:
                raise ValueError("Optimizer state is missing role_optimizers")
            role_optimizer_states = state["role_optimizers"]
            self._validate_role_optimizer_states(role_optimizer_states)

        if self.main_optimizer is not None:
            self.main_optimizer.load_state_dict(state["main_optimizer"])
        if role_optimizer_states is not None:
            for kind, optimizer in self.role_optimizers.items():
                optimizer.load_state_dict(role_optimizer_states[kind])


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
    if kind == "orth_adam":
        return OrthAdam(
            params,
            lr=train.lr * optim.orth_adam_lr,
            betas=(optim.orth_adam_beta1, optim.orth_adam_beta2),
            eps=optim.orth_adam_eps,
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
    if kind == "orth_muon":
        return OrthMuon(
            params,
            lr=optim.muon_lr,
            momentum=optim.muon_momentum,
            nesterov=optim.muon_nesterov,
            ns_steps=optim.muon_ns_steps,
            eps=optim.muon_eps,
            submat_dim=optim.submat_dim,
        )
    raise ValueError(f"Unsupported role optimizer kind {kind!r}")


def build_optimizers(config: ExperimentConfig, model: torch.nn.Module) -> OptimBundle:
    registry = model.get_parameter_registry() if hasattr(model, "get_parameter_registry") else None
    role_params = registry.role_parameters() if registry is not None else {}
    role_affine_params = model.role_affine_parameters() if hasattr(model, "role_affine_parameters") else {}
    role_to_owner = resolve_role_owners(config, registry)

    for role, owner in role_to_owner.items():
        if owner == "frozen":
            role_params[role].requires_grad_(False)

    affine_param_roles: dict[int, tuple[torch.nn.Parameter, set[str]]] = {}
    for role, params in role_affine_params.items():
        for param in params:
            param_id = id(param)
            if param_id not in affine_param_roles:
                affine_param_roles[param_id] = (param, set())
            affine_param_roles[param_id][1].add(role)
    for param, roles in affine_param_roles.values():
        if roles and all(role_to_owner[role] == "frozen" for role in roles):
            param.requires_grad_(False)

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
