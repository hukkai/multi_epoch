from __future__ import annotations

import torch.nn as nn

from ortho_llm.config import ModelConfig
from ortho_llm.modeling.chunked_layers import RoleChunkedLlamaForCausalLM
from ortho_llm.modeling.llama import LlamaForCausalLM


def build_model(config: ModelConfig) -> nn.Module:
    if config.parameterization == "dense" or not config.enabled_roles:
        return LlamaForCausalLM(config)
    if config.parameterization == "grouped_matrix":
        return RoleChunkedLlamaForCausalLM(config)
    raise ValueError(f"Unsupported parameterization {config.parameterization!r}")
