from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def resolve_data_path(data_dir: str, rank: int) -> Path:
    root = Path(data_dir)
    shard_path = root / f"tokens_{rank}.bin"
    if shard_path.exists():
        return shard_path
    fallback_path = root / "tokens_0.bin"
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Could not find tokens_{rank}.bin or tokens_0.bin under {data_dir}")


@dataclass
class MemmapState:
    position: int
    epoch: int


class MemmapTokenDataset:
    def __init__(
        self,
        data_dir: str,
        *,
        rank: int = 0,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        position: int = 0,
        epoch: int = 0,
    ) -> None:
        self.path = resolve_data_path(data_dir, rank)
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.sample_length = seq_length + 1
        self.tokens_per_batch = batch_size * self.sample_length
        self.device = device
        self.position = position
        self.epoch = epoch
        if self.tokens.shape[0] < self.tokens_per_batch:
            raise ValueError(
                f"Token file {self.path} is too short: need at least {self.tokens_per_batch}, "
                f"found {self.tokens.shape[0]}"
            )

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[0])

    def state_dict(self) -> dict:
        return {"position": self.position, "epoch": self.epoch, "path": str(self.path)}

    def load_state_dict(self, state: dict) -> None:
        self.position = int(state["position"])
        self.epoch = int(state.get("epoch", 0))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position + self.tokens_per_batch > self.num_tokens:
            self.position = 0
            self.epoch += 1
        start = self.position
        end = start + self.tokens_per_batch
        self.position = end
        token_slice = np.asarray(self.tokens[start:end], dtype=np.int64)
        token_batch = torch.from_numpy(token_slice.reshape(self.batch_size, self.sample_length))
        token_batch = token_batch.to(self.device, non_blocking=True)
        return token_batch[:, :-1], token_batch[:, 1:]
