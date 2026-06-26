from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


ATTN_ROLES = ("attn.q", "attn.k", "attn.v", "attn.o")
MLP_ROLES = ("mlp.gate", "mlp.up", "mlp.down")
ALL_ROLES = ATTN_ROLES + MLP_ROLES
ROLE_TO_SAFE_NAME = {role: role.replace(".", "_") for role in ALL_ROLES}
SAFE_NAME_TO_ROLE = {value: key for key, value in ROLE_TO_SAFE_NAME.items()}

OPTIMIZER_CHOICES = {"adamw", "orth_adam", "muon", "orth_muon", "frozen"}
PARAMETERIZATION_CHOICES = {"dense", "grouped_matrix"}
INIT_CHOICES = {"qr", "gaussian_then_project", "gaussian_no_project"}
ORTH_MUON_UPDATE_METHODS = {"flow", "polar", "skew"}


@dataclass
class DataConfig:
    data_dir: str = "./data/C4-50B/"
    val_data_dir: str | None = None
    validation_split: float = 0.0
    shuffle: bool = False


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: int = 3
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    parameterization: str = "dense"
    enabled_roles: list[str] = field(default_factory=list)
    chunk_affine: bool = True
    init: str = "qr"
    num_kv_heads: int | None = None
    row_block_size: int | None = None


@dataclass
class TrainConfig:
    output: str = "./output"
    seed: int = 42
    log_interval: int = 10
    save_freq: int = 9999999999
    batch_size: int = 8
    global_batch_size: int = 512
    seq_length: int = 2048
    num_steps: int = 21000
    lr: float = 1.2e-3
    min_lr: float = 1.2e-4
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    cosine_power: float = 1.0
    eval_interval: int = 0
    eval_batches: int = 0
    fail_on_nan: bool = True
    resume: str | None = None


@dataclass
class OptimConfig:
    default_role_optimizer: str = "frozen"
    role_overrides: dict[str, str] = field(default_factory=dict)
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8
    orth_adam_lr: float = 1.0
    orth_adam_beta1: float = 0.9
    orth_adam_beta2: float = 0.95
    orth_adam_eps: float = 1e-8
    submat_dim: int = 64
    strict_stiefel_every: int | str = "num_steps/50"
    muon_lr: float = 2.0e-3
    muon_min_lr: float = 2.0e-4
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-7
    orth_muon_update_method: str = "polar"


@dataclass
class LoggingConfig:
    metrics_filename: str = "metrics.jsonl"
    manifest_filename: str = "manifest.json"
    resolved_config_filename: str = "resolved_config.yaml"


@dataclass
class CheckpointConfig:
    enabled: bool = True
    filename_template: str = "checkpoint_{step:06d}.pth"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


def apply_overrides(raw: dict[str, Any], overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    merged = copy.deepcopy(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        key, value = override.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Override has empty key: {override!r}")
        cursor = merged
        for part in parts[:-1]:
            next_cursor = cursor.setdefault(part, {})
            if not isinstance(next_cursor, dict):
                raise ValueError(f"Cannot set nested override through non-mapping key {part!r}")
            cursor = next_cursor
        cursor[parts[-1]] = _parse_override_value(value)
    return merged


def _coerce_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    kwargs = {item.name: data[item.name] for item in fields(cls) if item.name in data}
    return cls(**kwargs)


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    normalized = {str(key).replace("-", "_"): value for key, value in raw.items()}
    allowed = {"data", "model", "train", "optim", "logging", "checkpoint"}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(
            "Config must use the nested schema. Unknown top-level keys: "
            + ", ".join(unknown)
        )
    if not any(key in normalized for key in allowed):
        raise ValueError("Config must contain at least one nested section")
    return normalized


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    seen = set() if seen is None else seen
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved_path))
        raise ValueError(f"Config extends cycle detected: {chain}")
    seen.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    parent_refs = raw.pop("extends", None)
    if parent_refs is None:
        seen.remove(resolved_path)
        return raw
    if isinstance(parent_refs, str):
        parent_refs = [parent_refs]
    if not isinstance(parent_refs, list) or not all(isinstance(ref, str) for ref in parent_refs):
        raise ValueError("Config extends must be a string path or a list of string paths")

    merged: dict[str, Any] = {}
    for parent_ref in parent_refs:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = resolved_path.parent / parent_path
        merged = _deep_merge_config(merged, _load_raw_config(parent_path, seen))

    seen.remove(resolved_path)
    return _deep_merge_config(merged, raw)


def _validate_roles(config: ExperimentConfig) -> None:
    model = config.model
    unknown_roles = sorted(set(model.enabled_roles) - set(ALL_ROLES))
    if unknown_roles:
        raise ValueError(f"Unknown enabled roles: {', '.join(unknown_roles)}")
    if len(set(model.enabled_roles)) != len(model.enabled_roles):
        raise ValueError("enabled_roles contains duplicates")

    if model.parameterization not in PARAMETERIZATION_CHOICES:
        choices = ", ".join(sorted(PARAMETERIZATION_CHOICES))
        raise ValueError(f"parameterization must be one of: {choices}")
    if model.init not in INIT_CHOICES:
        choices = ", ".join(sorted(INIT_CHOICES))
        raise ValueError(f"init must be one of: {choices}")
    if model.parameterization == "dense" and model.enabled_roles:
        raise ValueError("dense parameterization cannot define enabled_roles")
    if model.parameterization != "dense" and not model.enabled_roles:
        raise ValueError("chunked parameterizations require enabled_roles")

    for key, value in config.optim.role_overrides.items():
        if key not in ALL_ROLES:
            raise ValueError(f"Unknown role override {key!r}")
        if value not in OPTIMIZER_CHOICES:
            choices = ", ".join(sorted(OPTIMIZER_CHOICES))
            raise ValueError(f"Optimizer for {key} must be one of: {choices}")
    if config.optim.default_role_optimizer not in OPTIMIZER_CHOICES:
        choices = ", ".join(sorted(OPTIMIZER_CHOICES))
        raise ValueError(f"default_role_optimizer must be one of: {choices}")


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    model = config.model
    train = config.train
    optim = config.optim

    if model.hidden_size % model.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = model.hidden_size // model.num_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    if model.mlp_ratio <= 0 or int(model.mlp_ratio) != model.mlp_ratio:
        raise ValueError("mlp_ratio must be a positive integer")
    model.mlp_ratio = int(model.mlp_ratio)
    if model.num_kv_heads is None:
        model.num_kv_heads = model.num_heads
    if model.num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if model.num_heads % model.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if train.global_batch_size % train.batch_size != 0:
        raise ValueError("global_batch_size must be divisible by batch_size before DDP world-size adjustment")
    if train.seq_length > model.max_position_embeddings:
        raise ValueError("seq_length must be <= max_position_embeddings")
    if optim.submat_dim <= 0:
        raise ValueError("submat_dim must be positive")
    if optim.orth_muon_update_method not in ORTH_MUON_UPDATE_METHODS:
        choices = ", ".join(sorted(ORTH_MUON_UPDATE_METHODS))
        raise ValueError(f"orth_muon_update_method must be one of: {choices}")
    kv_heads = model.num_kv_heads or model.num_heads
    kv_dim = kv_heads * head_dim
    intermediate_size = model.hidden_size * model.mlp_ratio
    if model.parameterization != "dense":
        for name, rows in (
            ("hidden_size", model.hidden_size),
            ("intermediate_size", intermediate_size),
            ("kv_dim", kv_dim),
        ):
            if rows % optim.submat_dim != 0:
                raise ValueError(f"{name} must be divisible by submat_dim")
        if optim.submat_dim > model.hidden_size:
            raise ValueError("submat_dim must be <= hidden_size")
        if model.row_block_size is not None and model.row_block_size != optim.submat_dim:
            raise ValueError("model.row_block_size is internal and must match optim.submat_dim")
        model.row_block_size = optim.submat_dim
    _validate_roles(config)
    return config


def config_from_dict(raw: dict[str, Any], *, config_path: str | None = None) -> ExperimentConfig:
    normalized = _normalize_raw_config(raw)
    config = ExperimentConfig(
        data=_coerce_dataclass(DataConfig, normalized.get("data", {})),
        model=_coerce_dataclass(ModelConfig, normalized.get("model", {})),
        train=_coerce_dataclass(TrainConfig, normalized.get("train", {})),
        optim=_coerce_dataclass(OptimConfig, normalized.get("optim", {})),
        logging=_coerce_dataclass(LoggingConfig, normalized.get("logging", {})),
        checkpoint=_coerce_dataclass(CheckpointConfig, normalized.get("checkpoint", {})),
        config_path=config_path,
    )
    return validate_config(config)


def load_config(config_path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> ExperimentConfig:
    path = Path(config_path)
    raw = _load_raw_config(path)
    raw = apply_overrides(raw, overrides)
    return config_from_dict(raw, config_path=str(path))


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    output = config.to_dict()
    Path(path).write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj
