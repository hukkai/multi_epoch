from __future__ import annotations

from pathlib import Path

import torch
import pytest

from ortho_llm.config import ExperimentConfig
from ortho_llm.train.checkpoint import load_checkpoint, rank_checkpoint_filename, save_checkpoint
from ortho_llm.train.misc import rng_state_dict
from ortho_llm.train.trainer import _load_resume, _validate_resume_metrics


def test_checkpoint_loads_rng_state_with_current_torch_defaults(tmp_path: Path) -> None:
    path = save_checkpoint({"step": 3, "rng": rng_state_dict()}, tmp_path, "checkpoint.pth")
    assert load_checkpoint(path)["step"] == 3


def test_rank_checkpoint_filename_preserves_parent_and_suffix() -> None:
    assert rank_checkpoint_filename("nested/checkpoint_000003.pth", 7) == (
        "nested/rank_states/checkpoint_000003.rank_00007.pth"
    )


def test_distributed_resume_loads_the_matching_rank_sidecar(tmp_path: Path, monkeypatch) -> None:
    common_path = tmp_path / "checkpoint.pth"
    rank_files = [rank_checkpoint_filename(common_path.name, rank) for rank in range(2)]
    save_checkpoint(
        {
            "model": {"value": torch.tensor(1.0)},
            "optimizers": {"main_optimizer": {"source": "common"}, "role_optimizers": {}},
            "step": 4,
            "world_size": 2,
            "rank_state_files": rank_files,
        },
        tmp_path,
        common_path.name,
    )
    for rank, filename in enumerate(rank_files):
        save_checkpoint(
            {
                "rank": rank,
                "world_size": 2,
                "step": 4,
                "role_optimizers": {"muon": {"source_rank": rank}},
                "dataset": {"position": 100 + rank},
                "rng": {"source_rank": rank},
            },
            tmp_path,
            filename,
        )

    class FakeModel:
        loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    class FakeBundle:
        role_optimizers = {"muon": object()}

        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state):
            self.loaded.append(state)

    class FakeDataset:
        loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    loaded_rng = []
    monkeypatch.setattr("ortho_llm.train.trainer.load_rng_state_dict", loaded_rng.append)
    config = ExperimentConfig()
    config.train.resume = str(common_path)
    model = FakeModel()
    bundle = FakeBundle()
    dataset = FakeDataset()

    step = _load_resume(
        config,
        model,
        bundle,
        dataset,
        torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert step == 4
    assert model.loaded["value"].item() == 1.0
    assert bundle.loaded[-1]["role_optimizers"]["muon"]["source_rank"] == 1
    assert dataset.loaded == {"position": 101}
    assert loaded_rng == [{"source_rank": 1}]


def test_resume_rejects_metrics_newer_than_checkpoint(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"step": 1}\n{"step": 5}\n{"step": 3}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="step 5"):
        _validate_resume_metrics(metrics_path, start_step=4)
