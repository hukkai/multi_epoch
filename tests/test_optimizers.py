from __future__ import annotations

import pytest
import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model
from ortho_llm.optim import Muon, OptimBundle, OrthAdam, OrthMuon, build_optimizers
from ortho_llm.optim.stiefel_update import (
    _COEFFS2,
    _COEFFS3,
    _COEFFS4,
    _apply_series,
    _apply_series2_eager,
)
from ortho_llm.train.trainer import _set_optimizer_lrs


def _apply_series_reference(
    a: torch.Tensor,
    gram_error: torch.Tensor,
    coeffs: tuple[float, ...],
) -> torch.Tensor:
    term = gram_error @ a
    q = a + coeffs[0] * term
    for coeff in coeffs[1:]:
        term = gram_error @ term
        q = q + coeff * term
    return q


def _make_state_test_bundle(
    *,
    with_main: bool = True,
    role_order: tuple[str, ...] = ("attn.q", "attn.o"),
) -> OptimBundle:
    role_params = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in role_order]
    main_optimizer = None
    if with_main:
        main_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(2, 2))],
            lr=0.1,
            momentum=0.9,
        )
    return OptimBundle(
        main_optimizer=main_optimizer,
        role_optimizers={
            "muon": torch.optim.SGD(role_params, lr=0.1, momentum=0.9),
        },
        role_to_optimizer={role: "muon" for role in role_order},
    )


def _step_with_distinct_gradients(optimizer: torch.optim.Optimizer) -> None:
    for value, param in enumerate(optimizer.param_groups[0]["params"], start=1):
        param.grad = torch.full_like(param, float(value))
    optimizer.step()


def test_optimizer_restore_rejects_reordered_same_shaped_roles() -> None:
    saved_bundle = _make_state_test_bundle(role_order=("attn.q", "attn.o"))
    _step_with_distinct_gradients(saved_bundle.role_optimizers["muon"])
    current_bundle = _make_state_test_bundle(role_order=("attn.o", "attn.q"))

    with pytest.raises(ValueError, match="ordered role ownership"):
        current_bundle.load_state_dict(saved_bundle.state_dict())


def test_optimizer_restore_requires_role_mapping_and_main_presence() -> None:
    bundle = _make_state_test_bundle()
    missing_mapping = bundle.state_dict()
    missing_mapping.pop("role_to_optimizer")
    with pytest.raises(ValueError, match="missing role_to_optimizer"):
        bundle.load_state_dict(missing_mapping)

    missing_main = bundle.state_dict()
    missing_main.pop("main_optimizer")
    with pytest.raises(ValueError, match="missing main_optimizer"):
        bundle.load_state_dict(missing_main)

    without_main = _make_state_test_bundle(with_main=False)
    with pytest.raises(ValueError, match="main optimizer presence"):
        bundle.load_state_dict(without_main.state_dict())
    with pytest.raises(ValueError, match="main optimizer presence"):
        without_main.load_state_dict(bundle.state_dict())


def test_complete_optimizer_restore_requires_exact_role_optimizer_kinds() -> None:
    bundle = _make_state_test_bundle()
    valid_role_state = bundle.state_dict()["role_optimizers"]["muon"]

    for role_states in ({}, {"muon": valid_role_state, "extra": valid_role_state}):
        state = bundle.state_dict()
        state["role_optimizers"] = role_states
        with pytest.raises(ValueError, match="role optimizer kinds"):
            bundle.load_state_dict(state)


def test_rank_role_optimizer_restore_requires_exact_kinds() -> None:
    bundle = _make_state_test_bundle()
    valid_role_state = bundle.state_dict()["role_optimizers"]["muon"]

    for role_states in ({}, {"muon": valid_role_state, "extra": valid_role_state}):
        with pytest.raises(ValueError, match="role optimizer kinds"):
            bundle.load_role_optimizer_states(role_states)


def test_distributed_optimizer_restore_splits_common_and_rank_state() -> None:
    saved_bundle = _make_state_test_bundle()
    assert saved_bundle.main_optimizer is not None
    _step_with_distinct_gradients(saved_bundle.main_optimizer)
    _step_with_distinct_gradients(saved_bundle.role_optimizers["muon"])
    full_state = saved_bundle.state_dict()
    common_state = {**full_state, "role_optimizers": {}}

    current_bundle = _make_state_test_bundle()
    reordered_common_state = {
        **common_state,
        "role_to_optimizer": dict(reversed(common_state["role_to_optimizer"].items())),
    }
    with pytest.raises(ValueError, match="ordered role ownership"):
        current_bundle.load_state_dict(reordered_common_state, load_role_optimizers=False)

    current_bundle.load_state_dict(common_state, load_role_optimizers=False)
    assert current_bundle.main_optimizer is not None
    assert len(current_bundle.main_optimizer.state) == 1
    assert not current_bundle.role_optimizers["muon"].state

    current_bundle.load_role_optimizer_states(full_state["role_optimizers"])
    assert len(current_bundle.role_optimizers["muon"].state) == 2


def test_optimizer_factory_assigns_mixed_role_owners_once() -> None:
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
                "num_steps": 2,
            },
            "optim": {
                "default_role_optimizer": "orth_adam",
                "role_overrides": {
                    "attn.k": "adamw",
                    "attn.v": "muon",
                    "mlp.up": "orth_muon",
                    "mlp.down": "frozen",
                },
                "submat_dim": 4,
            },
        }
    )
    model = build_model(config.model)
    bundle = build_optimizers(config, model)
    assert bundle.role_to_optimizer["attn.k"] == "adamw"
    assert bundle.role_to_optimizer["attn.v"] == "muon"
    assert bundle.role_to_optimizer["mlp.up"] == "orth_muon"
    assert bundle.role_to_optimizer["mlp.down"] == "frozen"
    assert not model.role_parameters()["mlp.down"].requires_grad

    muon_group = bundle.role_optimizers["muon"].param_groups[0]
    orth_muon_group = bundle.role_optimizers["orth_muon"].param_groups[0]
    assert muon_group["weight_decay"] == config.optim.muon_weight_decay
    assert "decay_lr" in muon_group
    assert "weight_decay" not in orth_muon_group
    assert "decay_lr" not in orth_muon_group
    assert "update_method" not in orth_muon_group

    _set_optimizer_lrs(bundle, config, step=0, warmup_steps=1)
    assert "decay_lr" in muon_group
    assert "decay_lr" not in orth_muon_group


def test_optimizer_factory_freezes_unowned_mlp_affines() -> None:
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
                "enabled_roles": ["mlp.down"],
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"default_role_optimizer": "frozen", "submat_dim": 4},
        }
    )
    model = build_model(config.model)
    build_optimizers(config, model)
    assert not model.layers[0].mlp.down_affine.requires_grad
    assert not model.layers[1].mlp.down_affine.requires_grad


def test_optimizer_factory_tracks_mlp_affines_by_role() -> None:
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
                "enabled_roles": ["mlp.up", "mlp.down"],
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {
                "default_role_optimizer": "frozen",
                "role_overrides": {"mlp.up": "orth_muon"},
                "submat_dim": 4,
            },
        }
    )
    model = build_model(config.model)
    build_optimizers(config, model)
    assert model.layers[0].mlp.up_affine.requires_grad
    assert model.layers[1].mlp.up_affine.requires_grad
    assert not model.layers[0].mlp.down_affine.requires_grad
    assert not model.layers[1].mlp.down_affine.requires_grad


def test_affine_lr_multiplier_scales_all_affine_parameters() -> None:
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
                "num_steps": 2,
                "lr": 0.001,
                "min_lr": 0.0001,
            },
            "optim": {
                "default_role_optimizer": "orth_muon",
                "affine_lr_multiplier": 2.0,
                "submat_dim": 4,
            },
        }
    )
    model = build_model(config.model)
    bundle = build_optimizers(config, model)
    assert bundle.main_optimizer is not None

    affine_params = {
        param
        for params in model.role_affine_parameters().values()
        for param in params
    }
    affine_groups = [
        group
        for group in bundle.main_optimizer.param_groups
        if group.get("lr_multiplier") == 2.0
    ]
    assert len(affine_groups) == 1
    assert set(affine_groups[0]["params"]) == affine_params
    assert affine_groups[0]["weight_decay"] == 0.0
    assert affine_groups[0]["lr"] == pytest.approx(0.002)

    main_lr, _ = _set_optimizer_lrs(bundle, config, step=1, warmup_steps=1)
    assert affine_groups[0]["lr"] == pytest.approx(main_lr * 2.0)
    assert all(
        group["lr"] == pytest.approx(main_lr)
        for group in bundle.main_optimizer.param_groups
        if "lr_multiplier" not in group
    )


def test_optimizer_factory_freezes_unowned_attention_affines() -> None:
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
                "enabled_roles": ["attn.q"],
            },
            "train": {
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 16,
                "num_steps": 2,
            },
            "optim": {"default_role_optimizer": "frozen", "submat_dim": 4},
        }
    )
    model = build_model(config.model)
    build_optimizers(config, model)
    assert not model.layers[0].self_attn.q_affine.requires_grad
    assert not model.layers[1].self_attn.q_affine.requires_grad


def test_orth_adam_preserves_square_and_rectangular_shapes() -> None:
    for shape in ((4, 8, 8), (4, 4, 16)):
        param = torch.nn.Parameter(torch.randn(*shape))
        param.grad = torch.zeros_like(param)
        opt = OrthAdam(param, lr=0.01, submat_dim=4)
        opt.step(is_last=True)
        assert tuple(param.shape) == shape
        assert param.grad is None


def _assert_waits_after_launching_async_gathers(
    monkeypatch,
    optimizer_cls,
    module_path: str,
    *,
    optimizer_kwargs: dict | None = None,
    step_kwargs: dict | None = None,
) -> None:
    params = [torch.nn.Parameter(torch.randn(4, 4, 4)) for _ in range(2)]
    for param in params:
        param.grad = torch.zeros_like(param)

    events: list[tuple[str, int]] = []

    class FakeWork:
        def __init__(self, index: int) -> None:
            self.index = index

        def wait(self) -> None:
            events.append(("wait", self.index))

    def fake_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor, async_op: bool = False) -> FakeWork:
        assert async_op
        index = sum(event == "launch" for event, _ in events) + 1
        events.append(("launch", index))
        output[: input.shape[0]].copy_(input)
        return FakeWork(index)

    monkeypatch.setattr(optimizer_cls, "_distributed_context", staticmethod(lambda: (2, 0)))
    monkeypatch.setattr(f"{module_path}.dist.all_gather_into_tensor", fake_all_gather_into_tensor)

    opt = optimizer_cls(params, lr=0.01, **(optimizer_kwargs or {}))
    opt.step(**(step_kwargs or {}))

    assert events == [("launch", 1), ("launch", 2), ("wait", 1), ("wait", 2)]


def test_custom_optimizers_wait_after_launching_async_gathers(monkeypatch) -> None:
    cases = (
        (OrthAdam, "ortho_llm.optim.orth_adam", {"submat_dim": 4}, {}),
        (Muon, "ortho_llm.optim.muon", {}, {}),
        (OrthMuon, "ortho_llm.optim.orth_muon", {"submat_dim": 4}, {}),
    )
    for optimizer_cls, module_path, optimizer_kwargs, step_kwargs in cases:
        _assert_waits_after_launching_async_gathers(
            monkeypatch,
            optimizer_cls,
            module_path,
            optimizer_kwargs=optimizer_kwargs,
            step_kwargs=step_kwargs,
        )


def test_muon_optimizers_preserve_shape() -> None:
    for optimizer_cls in (Muon, OrthMuon):
        param = torch.nn.Parameter(torch.randn(4, 8, 8))
        param.grad = torch.randn_like(param)
        kwargs = {"submat_dim": 4} if optimizer_cls is OrthMuon else {}
        opt = optimizer_cls([param], lr=0.01, **kwargs)
        opt.step(is_last=True) if optimizer_cls is OrthMuon else opt.step()
        assert tuple(param.shape) == (4, 8, 8)


def test_orth_muon_applies_projected_update_without_spectral_norm_clipping(monkeypatch) -> None:
    applied_updates: list[torch.Tensor] = []

    def capture_stiefel_update(
        x: torch.Tensor,
        update: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs == {"do_projection": False}
        applied_updates.append(update.clone())
        return x

    monkeypatch.setattr(
        "ortho_llm.optim.orth_muon.orthogonalize_newton_schulz",
        lambda update, **_kwargs: torch.ones_like(update),
    )
    monkeypatch.setattr(
        "ortho_llm.optim.orth_muon.stiefel_project",
        lambda _x, update: update,
    )
    monkeypatch.setattr(
        "ortho_llm.optim.orth_muon.stiefel_update_taylor",
        capture_stiefel_update,
    )

    param = torch.nn.Parameter(torch.zeros(2, 4, 3))
    optimizer = OrthMuon(
        param,
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=0,
        submat_dim=2,
    )

    param.grad = torch.ones_like(param)
    optimizer.step()
    param.grad = torch.ones_like(param)
    optimizer.step()

    step_scale = 0.2 * 4**0.5
    expected_update = torch.full((4, 2, 3), -step_scale)
    torch.testing.assert_close(applied_updates[0], expected_update)
    torch.testing.assert_close(applied_updates[1], expected_update)

    restored_param = torch.nn.Parameter(torch.zeros_like(param))
    restored_optimizer = OrthMuon(
        restored_param,
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=0,
        submat_dim=2,
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_param.grad = torch.ones_like(restored_param)
    restored_optimizer.step()

    assert "spectral_norm_u" not in optimizer.state[param]
    assert "spectral_norm_u" not in restored_optimizer.state[restored_param]


def test_stiefel_horner_series_matches_sequential_reference() -> None:
    generator = torch.Generator().manual_seed(0)

    for dtype, rtol, atol in ((torch.float32, 2e-5, 2e-6), (torch.float64, 1e-12, 1e-12)):
        raw = torch.randn(4, 4096, 8, generator=generator, dtype=dtype)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        a = q.transpose(-1, -2).contiguous()
        a = a + 1e-3 * torch.randn(a.shape, generator=generator, dtype=dtype)

        gram_error = a @ a.transpose(-1, -2)
        gram_error.diagonal(dim1=-2, dim2=-1).sub_(1)

        for coeffs in (_COEFFS2, _COEFFS3, _COEFFS4):
            actual = (
                _apply_series2_eager(a, gram_error)
                if coeffs == _COEFFS2
                else _apply_series(a, gram_error, coeffs)
            )
            expected = _apply_series_reference(a, gram_error, coeffs)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(
                actual @ actual.transpose(-1, -2),
                expected @ expected.transpose(-1, -2),
                rtol=rtol,
                atol=atol,
            )
