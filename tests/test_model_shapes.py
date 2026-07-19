from __future__ import annotations

import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model

def tiny_config(enabled_roles: list[str]):
    parameterization = "grouped_matrix" if enabled_roles else "dense"
    return config_from_dict(
        {
            "model": {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2,
                "max_position_embeddings": 32,
                "parameterization": parameterization,
                "enabled_roles": enabled_roles,
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"submat_dim": 4},
        }
    )


def test_dense_forward_returns_logits_and_loss() -> None:
    config = tiny_config([])
    model = build_model(config.model)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    labels = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert torch.isfinite(output["loss"])


def test_grouped_matrix_modes_build_and_store_expected_shapes() -> None:
    expected_roles = {
        ("attn.q", "attn.k", "attn.v", "attn.o"): {
            "attn.q": (2, 32, 32),
            "attn.k": (2, 32, 32),
            "attn.v": (2, 32, 32),
            "attn.o": (2, 32, 32),
        },
        ("mlp.gate", "mlp.up", "mlp.down"): {
            "mlp.gate": (2, 64, 32),
            "mlp.up": (2, 64, 32),
            "mlp.down": (2, 64, 32),
        },
        ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"): {
            "attn.q": (2, 32, 32),
            "attn.k": (2, 32, 32),
            "attn.v": (2, 32, 32),
            "attn.o": (2, 32, 32),
            "mlp.gate": (2, 64, 32),
            "mlp.up": (2, 64, 32),
            "mlp.down": (2, 64, 32),
        },
    }
    for roles, expected_shapes in expected_roles.items():
        config = tiny_config(list(roles))
        model = build_model(config.model)
        role_shapes = {role: tuple(param.shape) for role, param in model.role_parameters().items()}
        assert role_shapes == expected_shapes
        input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
        labels = torch.randint(0, config.model.vocab_size, (2, 16))
        output = model(input_ids=input_ids, labels=labels)
        assert output["logits"].shape == (2, 16, config.model.vocab_size)
        assert torch.isfinite(output["loss"])


def test_chunk_bank_returns_layer_sliced_weights() -> None:
    config = tiny_config(["attn.q", "mlp.up"])
    model = build_model(config.model)
    layer_views = model.chunk_bank.layer_views()

    attn_weight = model.chunk_bank.attention_weight("attn.q", layer_idx=1, layer_views=layer_views)
    mlp_weight = model.chunk_bank.mlp_weight("mlp.up", layer_idx=0, layer_views=layer_views)

    assert tuple(attn_weight.shape) == (32, 32)
    assert tuple(mlp_weight.shape) == (64, 32)
    assert attn_weight is layer_views.weights["attn.q"][1]
    assert mlp_weight is layer_views.weights["mlp.up"][0]


def test_mlp_affines_follow_enabled_roles() -> None:
    attention_config = tiny_config(["attn.q", "attn.k", "attn.v", "attn.o"])
    attention_model = build_model(attention_config.model)
    attention_affines = [
        name for name, _ in attention_model.named_parameters() if name.endswith(("_affine",))
    ]
    assert attention_affines == []

    mlp_config = tiny_config(["mlp.gate", "mlp.up", "mlp.down"])
    mlp_model = build_model(mlp_config.model)
    affine_shapes = {
        name: tuple(param.shape)
        for name, param in mlp_model.named_parameters()
        if name.endswith(("_affine",))
    }
    assert affine_shapes == {
        "layers.0.mlp.gate_affine": (64,),
        "layers.0.mlp.mid_affine": (64,),
        "layers.1.mlp.gate_affine": (64,),
        "layers.1.mlp.mid_affine": (64,),
    }
    role_affine_counts = {
        role: len(params) for role, params in mlp_model.role_affine_parameters().items()
    }
    assert role_affine_counts == {"mlp.gate": 2, "mlp.up": 2, "mlp.down": 2}


def test_mlp_affines_can_be_disabled() -> None:
    config = config_from_dict(
        {
            "model": {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2,
                "max_position_embeddings": 32,
                "parameterization": "grouped_matrix",
                "enabled_roles": ["mlp.gate", "mlp.up", "mlp.down"],
                "mlp_affine": False,
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"submat_dim": 4},
        }
    )
    model = build_model(config.model)
    affine_names = [
        name for name, _ in model.named_parameters() if name.endswith(("_affine",))
    ]
    assert affine_names == []
    assert model.role_affine_parameters() == {}

    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    labels = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert torch.isfinite(output["loss"])


def test_grouped_matrix_gqa_kv_storage_is_rectangular() -> None:
    config = config_from_dict(
        {
            "model": {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "num_kv_heads": 2,
                "mlp_ratio": 2,
                "max_position_embeddings": 32,
                "parameterization": "grouped_matrix",
                "enabled_roles": ["attn.k", "attn.v"],
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"submat_dim": 4},
        }
    )
    model = build_model(config.model)
    assert tuple(model.role_parameters()["attn.k"].shape) == (2, 16, 32)
    assert tuple(model.role_parameters()["attn.v"].shape) == (2, 16, 32)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    labels = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert torch.isfinite(output["loss"])
