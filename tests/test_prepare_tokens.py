from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "prepare_tokens.py"
SPEC = importlib.util.spec_from_file_location("prepare_tokens", MODULE_PATH)
prepare_tokens = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare_tokens
SPEC.loader.exec_module(prepare_tokens)


class FakeTokenizer:
    def __init__(self, vocab: dict[str, list[int]], eos_token_id: int | None = 99) -> None:
        self.vocab = vocab
        self.eos_token_id = eos_token_id

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.vocab[text])


def read_tokens(path: Path) -> list[int]:
    return np.fromfile(path, dtype=np.uint32).tolist()


def test_help_does_not_require_huggingface_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-dir" in result.stdout
    assert "--val-output-dir" not in result.stdout


def test_parse_args_keeps_train_only_cli_compatible(tmp_path: Path) -> None:
    args = prepare_tokens.parse_args(
        [
            "--tokenizer",
            "tok",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.split == "train"
    assert not hasattr(args, "val_split")
    assert not hasattr(args, "val_output_dir")


def test_validate_args_rejects_invalid_shard_rank(tmp_path: Path) -> None:
    args = prepare_tokens.parse_args(
        [
            "--tokenizer",
            "tok",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--shard-rank",
            "2",
            "--num-shards",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="--shard-rank"):
        prepare_tokens.validate_args(args)


def test_write_token_shard_streams_tokens_with_eos(tmp_path: Path) -> None:
    dataset = [{"text": "hello"}, {"text": "world"}]
    tokenizer = FakeTokenizer({"hello": [1, 2], "world": [3]})

    result = prepare_tokens.write_token_shard(
        dataset,
        tokenizer,
        output_dir=tmp_path,
        text_column="text",
        shard_rank=0,
        num_shards=1,
        split="train",
        show_progress=False,
    )

    assert result.documents == 2
    assert result.tokens == 5
    assert result.path == tmp_path / "tokens_0.bin"
    assert read_tokens(result.path) == [1, 2, 99, 3, 99]


def test_shards_use_rank_stride_mapping(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    tokenizer = FakeTokenizer({"a": [1], "b": [2], "c": [3], "d": [4]})

    prepare_tokens.write_token_shard(
        [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}],
        tokenizer,
        output_dir=train_dir,
        text_column="text",
        shard_rank=1,
        num_shards=2,
        split="train",
        show_progress=False,
    )
    assert read_tokens(train_dir / "tokens_1.bin") == [2, 99, 4, 99]


def test_write_token_shard_rejects_missing_text_column(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer({"hello": [1]})
    with pytest.raises(KeyError, match="Missing text column"):
        prepare_tokens.write_token_shard(
            [{"body": "hello"}],
            tokenizer,
            output_dir=tmp_path,
            text_column="text",
            shard_rank=0,
            num_shards=1,
            show_progress=False,
        )


def test_write_token_shard_rejects_missing_eos(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer({"hello": [1]}, eos_token_id=None)
    with pytest.raises(ValueError, match="EOS token"):
        prepare_tokens.write_token_shard(
            [{"text": "hello"}],
            tokenizer,
            output_dir=tmp_path,
            text_column="text",
            shard_rank=0,
            num_shards=1,
            show_progress=False,
        )


def test_write_token_shard_rejects_tokens_outside_uint32(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer({"hello": [prepare_tokens.UINT32_MAX + 1]})
    with pytest.raises(ValueError, match="outside uint32 range"):
        prepare_tokens.write_token_shard(
            [{"text": "hello"}],
            tokenizer,
            output_dir=tmp_path,
            text_column="text",
            shard_rank=0,
            num_shards=1,
            show_progress=False,
        )
