from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ortho_llm.data.memmap_dataset import MemmapTokenDataset, resolve_data_path


def _dataset(
    data_dir: str | Path,
    *,
    position: int = 0,
    epoch: int = 0,
) -> MemmapTokenDataset:
    return MemmapTokenDataset(
        str(data_dir),
        batch_size=2,
        seq_length=4,
        device=torch.device("cpu"),
        position=position,
        epoch=epoch,
    )


def test_missing_rank_shard_does_not_fall_back_to_rank_zero(tiny_tokens: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tokens_1.bin"):
        resolve_data_path(str(tiny_tokens), rank=1)


def test_state_dict_records_normalized_shard_identity(tiny_tokens: Path) -> None:
    dataset = _dataset(tiny_tokens, position=12, epoch=3)

    assert dataset.state_dict() == {
        "position": 12,
        "epoch": 3,
        "path": str((tiny_tokens / "tokens_0.bin").resolve()),
        "num_tokens": 4096,
    }


def test_load_state_dict_accepts_legacy_relative_path_without_length(
    tiny_tokens: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tiny_tokens.parent)
    dataset = _dataset(tiny_tokens.name, position=1, epoch=2)

    dataset.load_state_dict(
        {
            "position": 20,
            "epoch": 4,
            "path": f"{tiny_tokens.name}/tokens_0.bin",
        }
    )

    assert dataset.position == 20
    assert dataset.epoch == 4


def test_load_state_dict_rejects_different_shard_without_mutating_state(tiny_tokens: Path) -> None:
    dataset = _dataset(tiny_tokens, position=7, epoch=2)

    with pytest.raises(ValueError, match="shard path"):
        dataset.load_state_dict(
            {
                "position": 100,
                "epoch": 9,
                "path": str(tiny_tokens / "tokens_1.bin"),
                "num_tokens": dataset.num_tokens,
            }
        )

    assert dataset.position == 7
    assert dataset.epoch == 2


def test_load_state_dict_rejects_changed_shard_length_without_mutating_state(tiny_tokens: Path) -> None:
    dataset = _dataset(tiny_tokens, position=7, epoch=2)

    with pytest.raises(ValueError, match="shard length"):
        dataset.load_state_dict(
            {
                "position": 100,
                "epoch": 9,
                "path": str(tiny_tokens / "tokens_0.bin"),
                "num_tokens": dataset.num_tokens + 1,
            }
        )

    assert dataset.position == 7
    assert dataset.epoch == 2
