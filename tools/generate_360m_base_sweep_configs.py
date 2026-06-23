from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "360m_4096l"
BASE_DIR = CONFIG_ROOT / "base"
SWEEP_DIR = CONFIG_ROOT / "sweeps"

ADAMW_LRS = (0.0006, 0.0012, 0.002)
MUON_LRS = (0.001, 0.002, 0.004)
WEIGHT_DECAYS = (0.03, 0.1, 0.3)
COSINE_POWERS = (1.0, 1.5, 2.0)
PILOT_ADAMW_CONFIGS = {
    (0.0012, 0.1, 1.0),
    (0.0012, 0.1, 2.0),
}
PILOT_MUON_CONFIGS = {
    (0.002, 0.1, 1.0),
    (0.002, 0.1, 2.0),
}

ENABLED_ROLES = [
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.o",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
]


def token(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def min_lr_for(lr: float) -> float:
    return round(lr * 0.1, 12)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def common_train(output: str) -> dict:
    return {
        "output": output,
        "log_interval": 10,
        "save_freq": 999999,
        "batch_size": 4,
        "global_batch_size": 256,
        "seq_length": 4096,
        "num_steps": 13732,
        "lr": 0.0012,
        "min_lr": 0.00012,
        "weight_decay": 0.1,
        "clip_grad": 1.0,
        "cosine_power": 1.0,
        "eval_interval": 274,
        "eval_batches": 999999,
    }


def write_base_configs() -> None:
    adamw = {
        "data": {
            "data_dir": "./data/C4-50B/",
            "val_data_dir": "./data/C4-val-1M/",
        },
        "model": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 16,
            "mlp_ratio": 3,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": True,
            "parameterization": "dense",
            "enabled_roles": [],
        },
        "train": common_train("./output/360m_4096l/dense_adamw"),
        "optim": {
            "default_role_optimizer": "frozen",
        },
        "checkpoint": {
            "enabled": True,
        },
    }
    muon = {
        "data": adamw["data"],
        "model": {
            **adamw["model"],
            "parameterization": "grouped_matrix",
            "enabled_roles": ENABLED_ROLES,
        },
        "train": common_train("./output/360m_4096l/muon"),
        "optim": {
            "default_role_optimizer": "muon",
            "submat_dim": 32,
            "muon_lr": 0.002,
            "muon_min_lr": 0.0002,
            "muon_weight_decay": 0.1,
        },
        "checkpoint": adamw["checkpoint"],
    }
    dump_yaml(BASE_DIR / "adamw_360m_4096l.yaml", adamw)
    dump_yaml(BASE_DIR / "muon_360m_4096l.yaml", muon)


def write_adamw_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "adamw_lr_wd_cosine_power"
    for lr in ADAMW_LRS:
        for weight_decay in WEIGHT_DECAYS:
            for cosine_power in COSINE_POWERS:
                if (lr, weight_decay, cosine_power) in PILOT_ADAMW_CONFIGS:
                    continue
                name = f"adamw_lr{token(lr)}_wd{token(weight_decay)}_cos{token(cosine_power)}"
                path = sweep_root / f"{name}.yaml"
                paths.append(path)
                dump_yaml(
                    path,
                    {
                        "extends": "../../base/adamw_360m_4096l.yaml",
                        "train": {
                            "output": f"./output/360m_4096l/sweeps/adamw_lr_wd_cosine_power/{name}",
                            "lr": lr,
                            "min_lr": min_lr_for(lr),
                            "weight_decay": weight_decay,
                            "cosine_power": cosine_power,
                        },
                    },
                )
    return paths


def write_muon_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "muon_lr_wd_cosine_power"
    for lr in MUON_LRS:
        for weight_decay in WEIGHT_DECAYS:
            for cosine_power in COSINE_POWERS:
                if (lr, weight_decay, cosine_power) in PILOT_MUON_CONFIGS:
                    continue
                name = f"muon_lr{token(lr)}_wd{token(weight_decay)}_cos{token(cosine_power)}"
                path = sweep_root / f"{name}.yaml"
                paths.append(path)
                dump_yaml(
                    path,
                    {
                        "extends": "../../base/muon_360m_4096l.yaml",
                        "train": {
                            "output": f"./output/360m_4096l/sweeps/muon_lr_wd_cosine_power/{name}",
                            "cosine_power": cosine_power,
                        },
                        "optim": {
                            "muon_lr": lr,
                            "muon_min_lr": min_lr_for(lr),
                            "muon_weight_decay": weight_decay,
                        },
                    },
                )
    return paths


def write_manifest(paths: list[Path]) -> None:
    manifest = SWEEP_DIR / "base_adamw_muon_lr_wd_cosine_power.txt"
    lines = [str(path.relative_to(ROOT)) for path in paths]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    readme = SWEEP_DIR / "README.md"
    readme.write_text(
        """# 360M Base Sweeps

This directory contains the first-stage base optimizer sweep to run now:

- AdamW: `train.lr x train.weight_decay x train.cosine_power`
- Muon: `optim.muon_lr x optim.muon_weight_decay x train.cosine_power`

The sweep ranges are:

- AdamW lr: `0.0006, 0.0012, 0.002`
- Muon lr: `0.001, 0.002, 0.004`
- weight decay: `0.03, 0.1, 0.3`
- cosine power: `1.0, 1.5, 2.0`

The full grid has 54 settings. Four settings are omitted because matching
pilot runs already exist:

- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=1.0`
- AdamW `lr=0.0012, weight_decay=0.1, cosine_power=2.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=1.0`
- Muon `muon_lr=0.002, muon_weight_decay=0.1, cosine_power=2.0`

Run every config listed in `base_adamw_muon_lr_wd_cosine_power.txt`; it contains
the remaining 50 settings.
After selecting the top 5 AdamW and top 5 Muon configs by validation loss,
create OrthAdam/OrthMuon configs by extending those selected configs. The
Orth follow-up should inherit the selected lr values and use
`train.cosine_power: 2.0` as the fixed Orth recipe.
""",
        encoding="utf-8",
    )


def main() -> None:
    write_base_configs()
    paths = write_adamw_sweep() + write_muon_sweep()
    write_manifest(paths)
    write_readme()
    print(f"Wrote {len(paths)} sweep configs under {SWEEP_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
