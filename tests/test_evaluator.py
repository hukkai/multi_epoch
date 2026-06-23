from __future__ import annotations

import torch

from ortho_llm.train import evaluator


class FixedLossModel(torch.nn.Module):
    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self.losses = iter(losses)

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        del input_ids, labels
        return {"loss": torch.tensor(next(self.losses), dtype=torch.float32)}


class DummyDataset:
    def __init__(self, *, num_tokens: int = 4, tokens_per_batch: int = 2) -> None:
        self.position = 0
        self.num_tokens = num_tokens
        self.tokens_per_batch = tokens_per_batch

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.position += 1
        tokens = torch.zeros((1, 2), dtype=torch.long)
        return tokens, tokens


def test_evaluate_all_reduces_validation_loss(monkeypatch) -> None:
    model = FixedLossModel([2.0, 2.0])
    dataset = DummyDataset()

    monkeypatch.setattr(evaluator.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor: torch.Tensor, op) -> None:
        if op == evaluator.dist.ReduceOp.MIN:
            tensor.fill_(2)
        else:
            tensor.add_(4.0)

    monkeypatch.setattr(evaluator.dist, "all_reduce", fake_all_reduce)

    metrics = evaluator.evaluate(model, dataset, num_batches=2, device=torch.device("cpu"))

    assert metrics["val_loss"] == 3.0
    assert metrics["val_batches"] == 2
    assert dataset.position == 0
    assert model.training


def test_evaluate_caps_batches_to_shortest_rank(monkeypatch) -> None:
    model = FixedLossModel([1.0, 1.0])
    dataset = DummyDataset(num_tokens=12, tokens_per_batch=2)

    monkeypatch.setattr(evaluator.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor: torch.Tensor, op) -> None:
        if op == evaluator.dist.ReduceOp.MIN:
            tensor.fill_(2)
        else:
            tensor.mul_(2)

    monkeypatch.setattr(evaluator.dist, "all_reduce", fake_all_reduce)

    metrics = evaluator.evaluate(model, dataset, num_batches=5, device=torch.device("cpu"))

    assert metrics["val_loss"] == 1.0
    assert metrics["val_batches"] == 2
    assert dataset.position == 0
