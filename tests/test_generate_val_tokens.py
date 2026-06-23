from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "generate_val_tokens_1m.py"
SPEC = importlib.util.spec_from_file_location("generate_val_tokens_1m", MODULE_PATH)
generate_val_tokens_1m = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_val_tokens_1m
SPEC.loader.exec_module(generate_val_tokens_1m)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = vocab

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.vocab[text])


def test_write_val_token_shard_keeps_document_boundaries(tmp_path: Path) -> None:
    dataset = [{"text": "a"}, {"text": "b"}]
    tokenizer = FakeTokenizer({"a": [1, 2], "b": [3, 4, 5]})

    result = generate_val_tokens_1m.write_val_token_shard(
        dataset,
        tokenizer,
        output_dir=tmp_path,
        rank=0,
        num_ranks=1,
        target_tokens=5,
    )

    assert result.tokens == 7
    assert result.documents == 2
    assert np.fromfile(tmp_path / "tokens_0.bin", dtype=np.uint32).tolist() == [1, 2, 99, 3, 4, 5, 99]
