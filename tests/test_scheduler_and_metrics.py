from __future__ import annotations

import torch

from ortho_llm.optim.scheduler import cosine_lr as new_cosine_lr
from ortho_llm.train.metrics import orthogonality_metrics_for_blocks


def test_scheduler_warmup_decay_and_floor() -> None:
    assert new_cosine_lr(0, 10, 2, 1.0, 0.1) == 0.5
    assert new_cosine_lr(1, 10, 2, 1.0, 0.1) == 1.0
    assert new_cosine_lr(10, 10, 2, 1.0, 0.1) == 0.1


def test_rectangular_orthogonality_metrics() -> None:
    x = torch.randn(3, 4, 16)
    q = torch.linalg.qr(x.transpose(-1, -2)).Q.transpose(-1, -2)
    metrics = orthogonality_metrics_for_blocks(q)
    assert metrics["orth_error_fro_mean"] < 1.0e-5
    assert metrics["orth_error_spectral_mean"] < 1.0e-5
