from __future__ import annotations

import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "1p3b_4096l"
BASE_DIR = CONFIG_ROOT / "base"
SWEEP_DIR = CONFIG_ROOT / "sweeps"

ADAMW_LRS = (0.0006, 0.0012, 0.002)
ADAMW_WEIGHT_DECAYS = (0.1, 0.3, 0.6)
MUON_LRS = (0.0005, 0.001, 0.002)
MUON_WEIGHT_DECAYS = (0.1, 0.3, 0.6)
ORTH_ADAM_LRS = ADAMW_LRS
ORTH_MUON_LRS = MUON_LRS

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


def reset_config_dirs() -> None:
    if SWEEP_DIR.exists():
        shutil.rmtree(SWEEP_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def common_train(output: str) -> dict:
    return {
        "output": output,
        "log_interval": 10,
        "save_freq": 999999,
        "batch_size": 2,
        "global_batch_size": 256,
        "seq_length": 4096,
        "num_steps": 48580,
        "lr": 0.0012,
        "min_lr": 0.00012,
        "weight_decay": 0.1,
        "clip_grad": 1.0,
        "eval_interval": 970,
        "eval_batches": 999999,
    }


def base_model() -> dict:
    return {
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "num_kv_heads": 16,
        "mlp_ratio": 3,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
        "parameterization": "dense",
        "enabled_roles": [],
    }


def write_base_configs() -> None:
    adamw = {
        "data": {
            "data_dir": "./data/C4-50B/",
            "val_data_dir": "./data/C4-val-1M/",
        },
        "model": base_model(),
        "train": common_train("./output/1p3b_4096l/dense_adamw"),
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
            **base_model(),
            "parameterization": "grouped_matrix",
            "enabled_roles": ENABLED_ROLES,
        },
        "train": common_train("./output/1p3b_4096l/muon"),
        "optim": {
            "default_role_optimizer": "muon",
            "submat_dim": 32,
            "muon_lr": 0.001,
            "muon_min_lr": 0.0001,
            "muon_weight_decay": 0.1,
        },
        "checkpoint": adamw["checkpoint"],
    }
    dump_yaml(BASE_DIR / "adamw_1p3b_4096l.yaml", adamw)
    dump_yaml(BASE_DIR / "muon_1p3b_4096l.yaml", muon)


def write_adamw_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "adamw_lr_wd"
    for lr in ADAMW_LRS:
        for weight_decay in ADAMW_WEIGHT_DECAYS:
            name = f"adamw_lr{token(lr)}_wd{token(weight_decay)}"
            path = sweep_root / f"{name}.yaml"
            paths.append(path)
            dump_yaml(
                path,
                {
                    "extends": "../../base/adamw_1p3b_4096l.yaml",
                    "train": {
                        "output": f"./output/1p3b_4096l/sweeps/adamw_lr_wd/{name}",
                        "lr": lr,
                        "min_lr": min_lr_for(lr),
                        "weight_decay": weight_decay,
                    },
                },
            )
    return paths


def write_muon_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "muon_lr_wd"
    for lr in MUON_LRS:
        for weight_decay in MUON_WEIGHT_DECAYS:
            name = f"muon_lr{token(lr)}_wd{token(weight_decay)}"
            path = sweep_root / f"{name}.yaml"
            paths.append(path)
            dump_yaml(
                path,
                {
                    "extends": "../../base/muon_1p3b_4096l.yaml",
                    "train": {
                        "output": f"./output/1p3b_4096l/sweeps/muon_lr_wd/{name}",
                    },
                    "optim": {
                        "muon_lr": lr,
                        "muon_min_lr": min_lr_for(lr),
                        "muon_weight_decay": weight_decay,
                    },
                },
            )
    return paths


def write_orth_adam_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "orth_adam_lr"
    for lr in ORTH_ADAM_LRS:
        name = f"orth_adam_lr{token(lr)}"
        path = sweep_root / f"{name}.yaml"
        paths.append(path)
        dump_yaml(
            path,
            {
                "extends": "../../base/adamw_1p3b_4096l.yaml",
                "model": {
                    "parameterization": "grouped_matrix",
                    "enabled_roles": ENABLED_ROLES,
                },
                "train": {
                    "output": f"./output/1p3b_4096l/sweeps/orth_adam_lr/{name}",
                    "lr": lr,
                    "min_lr": min_lr_for(lr),
                },
                "optim": {
                    "default_role_optimizer": "orth_adam",
                    "submat_dim": 32,
                    "orth_adam_lr": 1.0,
                },
            },
        )
    return paths


def write_orth_muon_sweep() -> list[Path]:
    paths: list[Path] = []
    sweep_root = SWEEP_DIR / "orth_muon_lr"
    for lr in ORTH_MUON_LRS:
        name = f"orth_muon_lr{token(lr)}"
        path = sweep_root / f"{name}.yaml"
        paths.append(path)
        dump_yaml(
            path,
            {
                "extends": "../../base/muon_1p3b_4096l.yaml",
                "train": {
                    "output": f"./output/1p3b_4096l/sweeps/orth_muon_lr/{name}",
                },
                "optim": {
                    "default_role_optimizer": "orth_muon",
                    "muon_lr": lr,
                    "muon_min_lr": min_lr_for(lr),
                },
            },
        )
    return paths


def write_manifest(name: str, paths: list[Path]) -> None:
    manifest = SWEEP_DIR / name
    lines = [str(path.relative_to(ROOT)) for path in paths]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    readme = SWEEP_DIR / "README.md"
    readme.write_text(
        """# 1.3B Base Sweeps

This directory contains the 1.3B optimizer sweeps for the 2048 x 24 GQA G=2
model at 4096 sequence length.

- AdamW: `train.lr x train.weight_decay`
- Muon: `optim.muon_lr x optim.muon_weight_decay`
- OrthAdam: `train.lr`
- OrthMuon: `optim.muon_lr`

The sweep ranges are:

- AdamW lr: `0.0006, 0.0012, 0.002`; weight decay: `0.1, 0.3, 0.6`
- Muon lr: `0.0005, 0.001, 0.002`; weight decay: `0.1, 0.3, 0.6`
- OrthAdam lr: `0.0006, 0.0012, 0.002`
- OrthMuon lr: `0.0005, 0.001, 0.002`

Run every config listed in `all_sweeps.txt` for the full 24-run sweep.
""",
        encoding="utf-8",
    )


def main() -> None:
    reset_config_dirs()
    write_base_configs()
    adamw_paths = write_adamw_sweep()
    muon_paths = write_muon_sweep()
    orth_adam_paths = write_orth_adam_sweep()
    orth_muon_paths = write_orth_muon_sweep()
    all_paths = adamw_paths + muon_paths + orth_adam_paths + orth_muon_paths
    write_manifest("base_adamw_muon_lr_wd.txt", adamw_paths + muon_paths)
    write_manifest("all_sweeps.txt", all_paths)
    write_readme()
    print(f"Wrote {len(all_paths)} sweep configs under {SWEEP_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
