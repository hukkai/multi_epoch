from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def rank_checkpoint_filename(filename: str, rank: int) -> str:
    path = Path(filename)
    rank_name = f"{path.stem}.rank_{rank:05d}{path.suffix}"
    return str(path.parent / "rank_states" / rank_name)


def save_checkpoint(state: dict[str, Any], output_dir: str | Path, filename: str) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Checkpoints are trusted artifacts produced by this training code and include
    # Python and NumPy RNG state, which is not supported by weights_only=True.
    return torch.load(path, map_location=map_location, weights_only=False)
