from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ortho_llm.config import config_from_dict
from ortho_llm.train import train


def test_two_step_smoke_training_writes_artifacts(tiny_tokens, tmp_path) -> None:
    output_dir = tmp_path / "run"
    val_dir = tmp_path / "val_tokens"
    val_dir.mkdir()
    val_tokens = np.arange(4096, dtype=np.uint32) % 128
    val_tokens.tofile(val_dir / "tokens_0.bin")
    config = config_from_dict(
        {
            "data": {"data_dir": str(tiny_tokens), "val_data_dir": str(val_dir)},
            "model": {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 1,
                "max_position_embeddings": 16,
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
                "output": str(output_dir),
                "seed": 3,
                "log_interval": 1,
                "save_freq": 99,
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 8,
                "num_steps": 2,
                "lr": 1.0e-3,
                "min_lr": 1.0e-4,
                "weight_decay": 0.0,
                "clip_grad": 1.0,
                "eval_interval": 1,
                "eval_batches": 1,
            },
            "optim": {"default_role_optimizer": "adamw", "submat_dim": 4},
        }
    )
    train(config)
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "resolved_config.yaml").exists()
    assert list(Path(output_dir).glob("checkpoint_*.pth"))
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["val_loss"] is not None for row in rows)

    checkpoint = max(Path(output_dir).glob("checkpoint_*.pth"))
    config.train.num_steps = 3
    config.train.resume = str(checkpoint)
    train(config)
    resumed_rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in resumed_rows] == [1, 2, 3]

    config.train.resume = None
    with pytest.raises(FileExistsError, match="already exists"):
        train(config)


def test_eval_steps_are_logged_and_final_step_is_evaluated(tiny_tokens, tmp_path) -> None:
    output_dir = tmp_path / "eval_run"
    val_dir = tmp_path / "eval_tokens"
    val_dir.mkdir()
    (np.arange(4096, dtype=np.uint32) % 128).tofile(val_dir / "tokens_0.bin")
    config = config_from_dict(
        {
            "data": {"data_dir": str(tiny_tokens), "val_data_dir": str(val_dir)},
            "model": {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 1,
                "max_position_embeddings": 16,
            },
            "train": {
                "output": str(output_dir),
                "log_interval": 3,
                "save_freq": 99,
                "batch_size": 2,
                "global_batch_size": 2,
                "seq_length": 8,
                "num_steps": 3,
                "eval_interval": 2,
                "eval_batches": 1,
            },
            "optim": {"strict_stiefel_every": "never"},
        }
    )

    train(config)

    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[1]["val_loss"] is not None
    assert rows[2]["val_loss"] is not None
