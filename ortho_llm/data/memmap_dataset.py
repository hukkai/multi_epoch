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
    raise FileNotFoundError(f"Could not find required shard tokens_{rank}.bin under {data_dir}")


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
        return {
            "position": self.position,
            "epoch": self.epoch,
            "path": str(self.path.resolve()),
            "num_tokens": self.num_tokens,
        }

    def load_state_dict(self, state: dict) -> None:
        try:
            saved_path = Path(state["path"]).resolve()
        except KeyError as exc:
            raise ValueError("Dataset checkpoint is missing its shard path") from exc
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError("Dataset checkpoint has an invalid shard path") from exc

        current_path = self.path.resolve()
        if saved_path != current_path:
            raise ValueError(
                f"Dataset checkpoint shard path {saved_path} does not match current shard {current_path}"
            )

        if "num_tokens" in state:
            try:
                saved_num_tokens = int(state["num_tokens"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Dataset checkpoint has an invalid shard length") from exc
            if saved_num_tokens != self.num_tokens:
                raise ValueError(
                    f"Dataset checkpoint shard length {saved_num_tokens} does not match "
                    f"current shard length {self.num_tokens}"
                )

        position = int(state["position"])
        epoch = int(state.get("epoch", 0))
        self.position = position
        self.epoch = epoch

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
