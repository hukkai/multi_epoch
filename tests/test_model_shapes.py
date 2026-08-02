from __future__ import annotations

import pytest
import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model

def tiny_config(
    enabled_roles: list[str],
    *,
    hidden_size: int = 32,
    num_heads: int = 4,
    num_kv_heads: int | None = None,
    submat_dim: int = 4,
    attention_head_interleaved: bool = False,
    attention_o_input_submat: bool = False,
):
    parameterization = "grouped_matrix" if enabled_roles else "dense"
    model_config = {
        "vocab_size": 128,
        "hidden_size": hidden_size,
        "num_layers": 2,
        "num_heads": num_heads,
        "mlp_ratio": 2,
        "max_position_embeddings": 32,
        "parameterization": parameterization,
        "enabled_roles": enabled_roles,
        "attention_head_interleaved": attention_head_interleaved,
        "attention_o_input_submat": attention_o_input_submat,
    }
    if num_kv_heads is not None:
        model_config["num_kv_heads"] = num_kv_heads
    return config_from_dict(
        {
            "model": model_config,
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"submat_dim": submat_dim},
        }
    )


def storage_to_logical_rows(model, role: str) -> list[int]:
    storage = model.role_parameters()[role]
    rows, cols = storage.shape[-2:]
    with torch.no_grad():
        row_ids = torch.arange(rows, dtype=storage.dtype)[:, None].expand(rows, cols)
        storage[0].copy_(row_ids)
    layer_views = model.chunk_bank.layer_views()
    weight = model.chunk_bank.attention_weight(role, 0, layer_views)
    logical_to_storage = weight[:, 0].to(dtype=torch.long)
    return torch.argsort(logical_to_storage).tolist()


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
    config = tiny_config(["attn.q", "attn.o", "mlp.up"])
    model = build_model(config.model)
    layer_views = model.chunk_bank.layer_views()

    attn_weight = model.chunk_bank.attention_weight("attn.q", layer_idx=1, layer_views=layer_views)
    output_weight = model.chunk_bank.attention_weight("attn.o", layer_idx=1, layer_views=layer_views)
    mlp_weight = model.chunk_bank.mlp_weight("mlp.up", layer_idx=0, layer_views=layer_views)

    assert tuple(attn_weight.shape) == (32, 32)
    assert tuple(output_weight.shape) == (32, 32)
    assert tuple(mlp_weight.shape) == (64, 32)
    assert attn_weight is layer_views.weights["attn.q"][1]
    assert output_weight is layer_views.weights["attn.o"][1]
    assert mlp_weight is layer_views.weights["mlp.up"][0]
    assert {
        spec.role: spec.materialization
        for spec in model.get_parameter_registry().specs
        if spec.layer_idx == 0
    } == {
        "attn.q": "identity",
        "attn.o": "identity",
        "mlp.up": "identity",
    }


@pytest.mark.parametrize(
    ("submat_dim", "expected_first_blocks"),
    [
        (
            8,
            [
                [head * 4 for head in range(0, 16, 2)],
                [head * 4 for head in range(1, 16, 2)],
            ],
        ),
        (16, [[head * 4 for head in range(16)]]),
        (
            32,
            [[head * 4 + offset for offset in range(2) for head in range(16)]],
        ),
    ],
)
def test_attention_head_interleaving_spreads_storage_blocks_across_heads(
    submat_dim: int,
    expected_first_blocks: list[list[int]],
) -> None:
    config = tiny_config(
        ["attn.q"],
        hidden_size=64,
        num_heads=16,
        submat_dim=submat_dim,
        attention_head_interleaved=True,
    )
    model = build_model(config.model)
    storage_order = storage_to_logical_rows(model, "attn.q")

    for block_idx, expected in enumerate(expected_first_blocks):
        start = block_idx * submat_dim
        assert storage_order[start : start + submat_dim] == expected
    assert sorted(storage_order) == list(range(config.model.hidden_size))
    assert model.get_parameter_registry().specs[0].materialization == "head_interleaved"


@pytest.mark.parametrize("role", ["attn.k", "attn.v"])
def test_attention_head_interleaving_uses_kv_heads_for_gqa(role: str) -> None:
    config = tiny_config(
        [role],
        hidden_size=64,
        num_heads=16,
        num_kv_heads=8,
        submat_dim=4,
        attention_head_interleaved=True,
    )
    model = build_model(config.model)
    storage_order = storage_to_logical_rows(model, role)

    assert storage_order[:4] == [0, 8, 16, 24]
    assert storage_order[4:8] == [4, 12, 20, 28]
    assert sorted(storage_order) == list(range(32))


def test_attention_o_input_submat_transposes_before_output_affine() -> None:
    config = tiny_config(["attn.o"], attention_o_input_submat=True)
    model = build_model(config.model)
    attention = model.layers[0].self_attn
    storage = model.role_parameters()["attn.o"]
    raw = torch.arange(32 * 32, dtype=storage.dtype).reshape(32, 32)
    affine = torch.linspace(0.5, 1.5, 32)
    with torch.no_grad():
        storage[0].copy_(raw)
        attention.o_affine.copy_(affine)

    layer_views = model.chunk_bank.layer_views()
    weight = model.chunk_bank.attention_weight("attn.o", 0, layer_views)
    x = torch.randn(2, 3, 32)
    actual = attention._linear(x, "attn.o", "o_proj", model.chunk_bank, 0, layer_views)

    torch.testing.assert_close(weight, raw.T)
    torch.testing.assert_close(actual, torch.nn.functional.linear(x, raw.T * affine[:, None]))
    assert model.get_parameter_registry().specs[0].materialization == "transpose"


def test_attention_o_combines_input_submat_with_head_interleaving() -> None:
    config = tiny_config(
        ["attn.o"],
        attention_head_interleaved=True,
        attention_o_input_submat=True,
    )
    model = build_model(config.model)
    storage = model.role_parameters()["attn.o"]
    raw = torch.arange(32 * 32, dtype=storage.dtype).reshape(32, 32)
    with torch.no_grad():
        storage[0].copy_(raw)

    storage_order = [head * 8 + offset for offset in range(8) for head in range(4)]
    input_major = torch.empty_like(raw)
    input_major[storage_order] = raw
    expected = input_major.T
    actual = model.chunk_bank.attention_weight("attn.o", 0, model.chunk_bank.layer_views())

    torch.testing.assert_close(actual, expected)
    assert model.get_parameter_registry().specs[0].materialization == "head_interleaved_transpose"
    assert not any("permutation" in key for key in model.state_dict())


@pytest.mark.parametrize(
    ("attention_head_interleaved", "attention_o_input_submat"),
    [(True, False), (False, True), (True, True)],
)
def test_attention_layout_modes_support_full_forward_and_backward(
    attention_head_interleaved: bool,
    attention_o_input_submat: bool,
) -> None:
    config = tiny_config(
        ["attn.q", "attn.k", "attn.v", "attn.o"],
        attention_head_interleaved=attention_head_interleaved,
        attention_o_input_submat=attention_o_input_submat,
    )
    model = build_model(config.model)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    labels = torch.randint(0, config.model.vocab_size, (2, 16))

    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert output["loss"] is not None
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert all(param.grad is not None for param in model.role_parameters().values())


def test_affines_follow_enabled_roles() -> None:
    attention_config = tiny_config(["attn.q", "attn.k", "attn.v", "attn.o"])
    attention_model = build_model(attention_config.model)
    attention_affine_shapes = {
        name: tuple(param.shape)
        for name, param in attention_model.named_parameters()
        if name.endswith(("_affine",))
    }
    assert attention_affine_shapes == {
        "layers.0.self_attn.q_affine": (32,),
        "layers.0.self_attn.k_affine": (32,),
        "layers.0.self_attn.v_affine": (32,),
        "layers.0.self_attn.o_affine": (32,),
        "layers.1.self_attn.q_affine": (32,),
        "layers.1.self_attn.k_affine": (32,),
        "layers.1.self_attn.v_affine": (32,),
        "layers.1.self_attn.o_affine": (32,),
    }
    attention_role_affine_counts = {
        role: len(params) for role, params in attention_model.role_affine_parameters().items()
    }
    assert attention_role_affine_counts == {
        "attn.q": 2,
        "attn.k": 2,
        "attn.v": 2,
        "attn.o": 2,
    }

    mlp_config = tiny_config(["mlp.gate", "mlp.up", "mlp.down"])
    mlp_model = build_model(mlp_config.model)
    affine_shapes = {
        name: tuple(param.shape)
        for name, param in mlp_model.named_parameters()
        if name.endswith(("_affine",))
    }
    assert affine_shapes == {
        "layers.0.mlp.gate_affine": (64,),
        "layers.0.mlp.up_affine": (64,),
        "layers.0.mlp.down_affine": (32,),
        "layers.1.mlp.gate_affine": (64,),
        "layers.1.mlp.up_affine": (64,),
        "layers.1.mlp.down_affine": (32,),
    }
    role_affine_counts = {
        role: len(params) for role, params in mlp_model.role_affine_parameters().items()
    }
    assert role_affine_counts == {"mlp.gate": 2, "mlp.up": 2, "mlp.down": 2}


def test_attention_affine_scales_weight_rows() -> None:
    config = tiny_config(["attn.q"])
    model = build_model(config.model)
    attention = model.layers[0].self_attn
    layer_views = model.chunk_bank.layer_views()
    x = torch.randn(2, 3, config.model.hidden_size)
    affine = torch.linspace(0.5, 1.5, config.model.hidden_size)
    with torch.no_grad():
        attention.q_affine.copy_(affine)

    actual = attention._linear(x, "attn.q", "q_proj", model.chunk_bank, 0, layer_views)
    weight = model.chunk_bank.attention_weight("attn.q", 0, layer_views)
    expected = torch.nn.functional.linear(x, weight * affine[:, None])
    torch.testing.assert_close(actual, expected)


def test_mlp_affines_scale_weight_rows() -> None:
    config = tiny_config(["mlp.gate", "mlp.up", "mlp.down"])
    model = build_model(config.model)
    mlp = model.layers[0].mlp
    layer_views = model.chunk_bank.layer_views()

    for role, attr, input_size, output_size in (
        ("mlp.gate", "gate_proj", config.model.hidden_size, 64),
        ("mlp.up", "up_proj", config.model.hidden_size, 64),
        ("mlp.down", "down_proj", 64, config.model.hidden_size),
    ):
        x = torch.randn(2, 3, input_size)
        affine = torch.linspace(0.5, 1.5, output_size)
        with torch.no_grad():
            getattr(mlp, f"{role.removeprefix('mlp.')}_affine").copy_(affine)

        actual = mlp._linear(x, role, attr, model.chunk_bank, 0, layer_views)
        weight = model.chunk_bank.mlp_weight(role, 0, layer_views)
        if role == "mlp.down":
            weight = weight.T
        expected = torch.nn.functional.linear(x, weight * affine[:, None])
        torch.testing.assert_close(actual, expected)


def test_attention_affines_can_be_disabled() -> None:
    config = tiny_config(["attn.q", "attn.k", "attn.v", "attn.o"])
    config.model.attention_affine = False
    model = build_model(config.model)

    affine_names = [
        name for name, _ in model.named_parameters() if name.endswith(("_affine",))
    ]
    assert affine_names == []
    assert model.role_affine_parameters() == {}

    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(input_ids=input_ids)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)


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
    assert tuple(model.layers[0].self_attn.k_affine.shape) == (16,)
    assert tuple(model.layers[0].self_attn.v_affine.shape) == (16,)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    labels = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert torch.isfinite(output["loss"])
