from __future__ import annotations

import pytest

from ortho_llm.config import ATTN_ROLES, MLP_ROLES, config_from_dict, load_config


def test_role_constants_cover_attention_and_mlp() -> None:
    assert ATTN_ROLES == ("attn.q", "attn.k", "attn.v", "attn.o")
    assert MLP_ROLES == ("mlp.gate", "mlp.up", "mlp.down")


def test_nested_so_config_sets_role_policy() -> None:
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
                "default_role_optimizer": "so",
                "so_lr": 1.0,
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
    assert config.optim.default_role_optimizer == "so"
    assert config.model.row_block_size == 4


def test_flat_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="nested schema"):
        config_from_dict({"data_dir": "./data", "enabled_roles": []})


def test_migrated_repo_config_loads() -> None:
    config = load_config("configs/ablations/so_all_roles.yaml")
    assert config.model.enabled_roles == [
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.o",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    ]
    assert config.optim.default_role_optimizer == "so"
