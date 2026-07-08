from __future__ import annotations

import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model
from ortho_llm.modeling import chunked_layers


def tiny_config(enabled_roles: list[str], *, layer_weight_access: str = "index"):
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
                "layer_weight_access": layer_weight_access,
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


def test_chunk_affine_is_applied_after_layer_slice(monkeypatch) -> None:
    seen_shapes = []

    def spy_mul_add_broadcast(
        weight: torch.Tensor,
        affine1: torch.Tensor,
        affine2: torch.Tensor,
    ) -> torch.Tensor:
        seen_shapes.append((tuple(weight.shape), tuple(affine1.shape), tuple(affine2.shape)))
        return weight * (affine1 + affine2 + 1.0)

    monkeypatch.setattr(chunked_layers, "mul_add_broadcast", spy_mul_add_broadcast)
    config = tiny_config(["attn.q", "mlp.up"])
    model = build_model(config.model)

    attn_weight = model.chunk_bank.attention_weight("attn.q", layer_idx=1)
    mlp_weight = model.chunk_bank.mlp_weight("mlp.up", layer_idx=0)

    assert tuple(attn_weight.shape) == (32, 32)
    assert tuple(mlp_weight.shape) == (64, 32)
    assert seen_shapes == [
        ((32, 32), (32, 1), (1, 32)),
        ((64, 32), (64, 1), (1, 32)),
    ]


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


def test_grouped_matrix_unbind_weight_access_matches_index_and_backprops() -> None:
    roles = ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]
    index_config = tiny_config(roles, layer_weight_access="index")
    unbind_config = tiny_config(roles, layer_weight_access="unbind")
    torch.manual_seed(1234)
    index_model = build_model(index_config.model)
    unbind_model = build_model(unbind_config.model)
    unbind_model.load_state_dict(index_model.state_dict())

    input_ids = torch.randint(0, index_config.model.vocab_size, (2, 16))
    labels = torch.randint(0, index_config.model.vocab_size, (2, 16))
    index_output = index_model(input_ids=input_ids, labels=labels)
    unbind_output = unbind_model(input_ids=input_ids, labels=labels)

    assert torch.allclose(index_output["logits"], unbind_output["logits"])
    assert torch.allclose(index_output["loss"], unbind_output["loss"])
    unbind_output["loss"].backward()
    assert unbind_model.chunk_bank.weights["attn_q"].grad is not None
