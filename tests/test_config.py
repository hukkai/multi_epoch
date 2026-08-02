from __future__ import annotations

from pathlib import Path

import pytest

from ortho_llm.config import ATTN_ROLES, MLP_ROLES, config_from_dict, dump_config, load_config


CONFIG_PATHS = sorted(Path("configs").rglob("*.yaml"))
TMP_ABLATION_CASES = [
    ("attn_head_interleaved_submat8.yaml", True, False, 8),
    ("attn_head_interleaved_submat32.yaml", True, False, 32),
    ("attn_o_input_submat8.yaml", False, True, 8),
    ("attn_o_input_submat32.yaml", False, True, 32),
]


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
    assert config.optim.affine_lr_multiplier == 1.0
    assert config.model.row_block_size == 4
    assert config.model.attention_affine is True
    assert config.model.mlp_affine is True
    assert config.model.attention_head_interleaved is False
    assert config.model.attention_o_input_submat is False


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


@pytest.mark.parametrize(
    ("filename", "head_interleaved", "o_input_submat", "submat_dim"),
    TMP_ABLATION_CASES,
)
def test_tmp_ablation_configs_set_requested_layout_and_optimizers(
    filename: str,
    head_interleaved: bool,
    o_input_submat: bool,
    submat_dim: int,
) -> None:
    path = Path("configs/360m_4096l/tmp_ablation") / filename
    config = load_config(path)

    assert config.model.attention_head_interleaved is head_interleaved
    assert config.model.attention_o_input_submat is o_input_submat
    assert config.model.attention_affine is True
    assert config.model.mlp_affine is False
    assert config.optim.default_role_optimizer == "orth_muon"
    assert config.optim.role_overrides == {
        "mlp.gate": "muon",
        "mlp.up": "muon",
        "mlp.down": "muon",
    }
    assert config.optim.submat_dim == submat_dim
    assert config.optim.muon_lr == pytest.approx(0.002)
    assert config.optim.muon_weight_decay == pytest.approx(0.3)
    assert config.train.output.endswith(f"/tmp_ablation/{path.stem}")


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


def test_resolved_config_round_trips_without_source_path(tmp_path: Path) -> None:
    config = load_config("configs/360m_4096l/sweeps/orth_adam_lr/orth_adam_lr0p0012.yaml")
    resolved_path = tmp_path / "resolved_config.yaml"

    dump_config(config, resolved_path)
    reloaded = load_config(resolved_path)

    assert "config_path:" not in resolved_path.read_text(encoding="utf-8")
    assert reloaded.data == config.data
    assert reloaded.model == config.model
    assert reloaded.train == config.train
    assert reloaded.optim == config.optim
    assert reloaded.logging == config.logging
    assert reloaded.checkpoint == config.checkpoint


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: str(path))
def test_repo_configs_load(config_path: Path) -> None:
    load_config(config_path)
