from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tiny_tokens(tmp_path):
    data_dir = tmp_path / "tokens"
    data_dir.mkdir()
    tokens = np.arange(4096, dtype=np.uint32) % 128
    tokens.tofile(data_dir / "tokens_0.bin")
    return data_dir
