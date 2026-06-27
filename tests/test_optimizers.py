from __future__ import annotations

import math

import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model
from ortho_llm.optim import Muon, OrthAdam, OrthMuon, build_optimizers
from ortho_llm.optim.flow_update import flow_update
from ortho_llm.optim.stiefel_update import (
    _COEFFS2,
    _COEFFS3,
    _COEFFS4,
    _apply_series,
    _apply_series2_eager,
    stiefel_project,
)
from ortho_llm.train.trainer import _set_optimizer_lrs


def _row_stiefel(batch: int, rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(batch, cols, rows, generator=generator)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.transpose(-1, -2).contiguous()


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
                "orth_muon_update_method": "flow",
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
    assert orth_muon_group["update_method"] == "flow"

    _set_optimizer_lrs(bundle, config, step=0, warmup_steps=1)
    assert "decay_lr" in muon_group
    assert "decay_lr" not in orth_muon_group


def test_optimizer_factory_accepts_raw_orth_muon_update() -> None:
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
                "enabled_roles": ["mlp.up"],
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
                "orth_muon_update_method": "raw",
            },
        }
    )
    model = build_model(config.model)
    bundle = build_optimizers(config, model)

    orth_muon_group = bundle.role_optimizers["orth_muon"].param_groups[0]
    assert orth_muon_group["update_method"] == "raw"


def test_orth_adam_preserves_square_and_rectangular_shapes() -> None:
    for shape in ((4, 8, 8), (4, 4, 16)):
        param = torch.nn.Parameter(torch.randn(*shape))
        param.grad = torch.zeros_like(param)
        opt = OrthAdam(param, lr=0.01, submat_dim=4)
        opt.step(is_last=True)
        assert tuple(param.shape) == shape
        assert param.grad is None


def test_muon_optimizers_preserve_shape() -> None:
    for optimizer_cls in (Muon, OrthMuon):
        param = torch.nn.Parameter(torch.randn(4, 8, 8))
        param.grad = torch.randn_like(param)
        kwargs = {"submat_dim": 4} if optimizer_cls is OrthMuon else {}
        opt = optimizer_cls([param], lr=0.01, **kwargs)
        opt.step(is_last=True) if optimizer_cls is OrthMuon else opt.step()
        assert tuple(param.shape) == (4, 8, 8)


def test_flow_update_matches_projected_flow_taylor_terms() -> None:
    generator = torch.Generator().manual_seed(1)
    x = _row_stiefel(batch=3, rows=4, cols=8, generator=generator)
    delta = torch.randn(x.shape, generator=generator)
    eta = 0.03

    u = stiefel_project(x, delta)
    s = 0.5 * (delta @ x.transpose(-1, -2) + x @ delta.transpose(-1, -2))
    t = 0.5 * (delta @ u.transpose(-1, -2) + u @ delta.transpose(-1, -2))
    expected = x + eta * u - 0.5 * eta * eta * (t @ x + s @ u)

    torch.testing.assert_close(flow_update(x, delta, eta=eta, project=False), expected)


def test_flow_update_fast_polar_projection_reduces_orthogonality_error() -> None:
    generator = torch.Generator().manual_seed(3)
    x = _row_stiefel(batch=3, rows=4, cols=8, generator=generator)
    delta = torch.randn(x.shape, generator=generator)
    eta = 0.03

    unprojected = flow_update(x, delta, eta=eta, project=False)
    projected = flow_update(x, delta, eta=eta)
    ident = torch.eye(x.shape[-2], dtype=x.dtype)

    unprojected_err = torch.linalg.matrix_norm(
        unprojected @ unprojected.transpose(-1, -2) - ident,
        ord="fro",
        dim=(-2, -1),
    )
    projected_err = torch.linalg.matrix_norm(
        projected @ projected.transpose(-1, -2) - ident,
        ord="fro",
        dim=(-2, -1),
    )

    assert projected_err.max() < unprojected_err.max()
    assert projected_err.max() < 1e-5


def test_orth_muon_flow_uses_descent_direction() -> None:
    generator = torch.Generator().manual_seed(2)
    param = torch.nn.Parameter(_row_stiefel(batch=1, rows=4, cols=8, generator=generator))
    grad = torch.randn(param.shape, generator=generator)
    param.grad = grad.clone()
    before = param.detach().float().reshape(-1, 4, 8).clone()

    lr = 1e-4
    eps = 1e-7
    opt = OrthMuon(
        [param],
        lr=lr,
        momentum=0.0,
        nesterov=False,
        ns_steps=0,
        eps=eps,
        submat_dim=4,
        strict_stiefel=False,
        update_method="flow",
    )

    raw_update = grad.float().reshape_as(before)
    raw_update = raw_update / (raw_update.norm(dim=(-2, -1), keepdim=True) + eps)
    projected_update = stiefel_project(before, raw_update)
    eta = lr * 0.2 * math.sqrt(max(param.shape[-2:]))

    opt.step(is_last=False)

    actual_step = param.detach().float().reshape_as(before) - before
    expected_first_order_step = -eta * projected_update
    torch.testing.assert_close(actual_step, expected_first_order_step, rtol=1e-4, atol=1e-6)


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
