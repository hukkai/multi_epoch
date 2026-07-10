from __future__ import annotations

from pathlib import Path

import pytest

from ortho_llm.data.memmap_dataset import resolve_data_path


def test_missing_rank_shard_does_not_fall_back_to_rank_zero(tiny_tokens: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tokens_1.bin"):
        resolve_data_path(str(tiny_tokens), rank=1)
