from __future__ import annotations

from pathlib import Path

import pytest

from ortho_llm.config import ATTN_ROLES, MLP_ROLES, config_from_dict, load_config


CONFIG_PATHS = sorted(Path("configs").rglob("*.yaml"))


def test_role_constants_cover_attention_and_mlp() -> None:
    assert ATTN_ROLES == ("attn.q", "attn.k", "attn.v", "attn.o")
    assert MLP_ROLES == ("mlp.gate", "mlp.up", "mlp.down")


def test_nested_orth_adam_config_sets_role_policy() -> None:
    config = config_from_dict(
        {
            "model": {
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2,
                "max_position_embeddings": 32,
                "vocab_size": 128,
                "parameterization": "grouped_matrix",
                "enabled_roles": [
                    "attn.q",
                    "attn.k",
                    "attn.v",
                    "attn.o",
                    "mlp.gate",
                    "mlp.up",
                    "mlp.down",
                ],
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 5,
            },
            "optim": {
                "default_role_optimizer": "orth_adam",
                "orth_adam_lr": 1.0,
                "submat_dim": 4,
            },
        }
    )
    assert config.model.parameterization == "grouped_matrix"
    assert config.model.enabled_roles == [
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.o",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    ]
    assert config.optim.default_role_optimizer == "orth_adam"
    assert config.model.row_block_size == 4


def test_flat_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="nested schema"):
        config_from_dict({"data_dir": "./data", "enabled_roles": []})


def test_legacy_orth_muon_update_method_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="orth_muon_update_method"):
        config_from_dict(
            {
                "model": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "mlp_ratio": 2,
                    "max_position_embeddings": 32,
                    "vocab_size": 128,
                    "parameterization": "grouped_matrix",
                    "enabled_roles": ["attn.q"],
                },
                "train": {
                    "batch_size": 2,
                    "global_batch_size": 2,
                    "seq_length": 16,
                    "num_steps": 5,
                },
                "optim": {
                    "default_role_optimizer": "orth_muon",
                    "submat_dim": 4,
                    "orth_muon_update_method": "flow",
                },
            }
        )


def test_migrated_repo_config_loads() -> None:
    config = load_config("configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0012.yaml")
    assert config.model.enabled_roles == [
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.o",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    ]
    assert config.optim.default_role_optimizer == "orth_adam"


def test_config_extends_deep_merges_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        """
model:
  hidden_size: 32
  num_layers: 2
  num_heads: 4
  mlp_ratio: 2
  max_position_embeddings: 32
  vocab_size: 128
train:
  batch_size: 2
  global_batch_size: 2
  seq_length: 16
  num_steps: 5
  lr: 0.001
  weight_decay: 0.1
optim:
  adamw_beta2: 0.95
""",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        """
extends: parent.yaml
train:
  lr: 0.002
optim:
  adamw_beta2: 0.98
""",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config.train.lr == 0.002
    assert config.train.weight_decay == 0.1
    assert config.optim.adamw_beta2 == 0.98


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: str(path))
def test_repo_configs_load(config_path: Path) -> None:
    load_config(config_path)
