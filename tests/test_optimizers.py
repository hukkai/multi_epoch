from __future__ import annotations

import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model
from ortho_llm.optim import Muon, OrthAdam, OrthMuon, build_optimizers
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


def test_orth_optimizer_param_groups_use_role_submat_overrides() -> None:
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
                "default_role_optimizer": "orth_muon",
                "submat_dim": 4,
                "submat_dim_overrides": {"attn": 8, "mlp": 16},
            },
        }
    )
    model = build_model(config.model)
    bundle = build_optimizers(config, model)

    submat_dims = [group["submat_dim"] for group in bundle.role_optimizers["orth_muon"].param_groups]
    assert submat_dims.count(8) == 4
    assert submat_dims.count(16) == 3


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
