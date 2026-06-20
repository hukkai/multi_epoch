from __future__ import annotations

import argparse
import itertools
import shlex
from pathlib import Path

import yaml

from ortho_llm.config import apply_overrides, load_config


def _format_override(key: str, value) -> str:
    return f"{key}={repr(value) if isinstance(value, (list, str)) else value}"


def _axis_label(value) -> str:
    if value == ["attn.q", "attn.k", "attn.v", "attn.o"]:
        return "atten"
    if value == ["mlp.gate", "mlp.up", "mlp.down"]:
        return "mlp"
    if value == ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]:
        return "all"
    if isinstance(value, list):
        return "-".join(str(item).replace(".", "-") for item in value)
    return str(value).replace(".", "-").replace("/", "-").replace(" ", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate local ablation commands")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_config = raw["base_config"]
    fixed_axes = raw.get("fixed_overrides", {})
    output_root = raw.get("output_root")
    axes = raw.get("axes", {})
    keys = list(axes)
    values = [axes[key] for key in keys]
    for combo in itertools.product(*values):
        axis_overrides = [_format_override(key, value) for key, value in zip(keys, combo)]
        fixed_overrides = [_format_override(key, value) for key, value in fixed_axes.items()]
        name = "_".join(_axis_label(value) for value in combo)
        overrides = [*fixed_overrides, *axis_overrides]
        if output_root and not any(item.startswith("train.output=") for item in overrides):
            overrides.append(_format_override("train.output", str(Path(output_root) / name)))
        config = load_config(base_config, overrides)
        if args.dry_run:
            print(f"# {name}")
            command = ["python", "-m", "ortho_llm.scripts.train", "--config", base_config, *overrides]
            print(" ".join(shlex.quote(part) for part in command))
        else:
            resolved = apply_overrides(yaml.safe_load(Path(base_config).read_text(encoding="utf-8")), overrides)
            print(yaml.safe_dump(config.to_dict() if config else resolved, sort_keys=False))


if __name__ == "__main__":
    main()
