from __future__ import annotations

import torch

from ortho_llm.config import config_from_dict
from ortho_llm.modeling import build_model
from ortho_llm.optim import Muon, MuonOrthogonal, SOOptimizer, build_optimizers


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
                "default_role_optimizer": "so",
                "role_overrides": {
                    "attn.k": "adamw",
                    "attn.v": "muon",
                    "mlp.up": "muon_orthogonal",
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
    assert bundle.role_to_optimizer["mlp.up"] == "muon_orthogonal"
    assert bundle.role_to_optimizer["mlp.down"] == "frozen"
    assert not model.role_parameters()["mlp.down"].requires_grad


def test_so_optimizer_preserves_square_and_rectangular_shapes() -> None:
    for shape in ((4, 8, 8), (4, 4, 16)):
        param = torch.nn.Parameter(torch.randn(*shape))
        param.grad = torch.zeros_like(param)
        opt = SOOptimizer(param, lr=0.01, submat_dim=4)
        opt.step(is_last=True)
        assert tuple(param.shape) == shape
        assert param.grad is None


def test_muon_optimizers_preserve_shape() -> None:
    for optimizer_cls in (Muon, MuonOrthogonal):
        param = torch.nn.Parameter(torch.randn(4, 8, 8))
        param.grad = torch.randn_like(param)
        kwargs = {"submat_dim": 4} if optimizer_cls is MuonOrthogonal else {}
        opt = optimizer_cls([param], lr=0.01, **kwargs)
        opt.step(is_last=True) if optimizer_cls is MuonOrthogonal else opt.step()
        assert tuple(param.shape) == (4, 8, 8)
