from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ortho_llm.config import (
    ATTN_ROLES,
    MLP_ROLES,
    ROLE_TO_SAFE_NAME,
    ModelConfig,
)
from ortho_llm.modeling.llama import (
    CausalSelfAttention,
    MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from ortho_llm.modeling.registry import MatrixSpec, ParameterRegistry


@dataclass(frozen=True)
class RoleLayerViews:
    weights: dict[str, tuple[torch.Tensor, ...]]


def _stiefel_block_init_3d(param: nn.Parameter, submat_dim: int) -> None:
    with torch.no_grad():
        work = param.data.to(torch.float32)
        rows, cols = work.shape[-2:]
        if submat_dim > cols:
            raise ValueError(f"submat_dim {submat_dim} must be <= cols {cols}")
        if rows % submat_dim != 0:
            raise ValueError(f"rows {rows} must be divisible by submat_dim {submat_dim}")
        blocks = work.reshape(-1, submat_dim, cols)
        q = torch.linalg.qr(blocks.transpose(-1, -2)).Q.transpose(-1, -2)
        param.data.copy_(q.reshape_as(work).to(dtype=param.dtype))


class RoleChunkBank(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.mlp_ratio = config.mlp_ratio
        self.intermediate_size = config.hidden_size * config.mlp_ratio
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.kv_hidden_size = self.num_kv_heads * self.head_dim
        self.enabled_roles = tuple(config.enabled_roles)
        self.weights = nn.ParameterDict()

        for role in self.enabled_roles:
            safe_name = ROLE_TO_SAFE_NAME[role]
            rows = self._rows_for_role(role)
            weight = nn.Parameter(torch.empty(config.num_layers, rows, config.hidden_size))
            nn.init.normal_(weight, mean=0.0, std=config.hidden_size**-0.5)
            self.weights[safe_name] = weight

        self._init_role_weights()
        self.registry = self._build_registry()

    def _init_role_weights(self) -> None:
        if self.config.init == "gaussian_no_project":
            return
        for param in self.weights.values():
            if self.config.row_block_size is None:
                raise ValueError("row_block_size must be set before initializing chunk weights")
            _stiefel_block_init_3d(param, self.config.row_block_size)

    def _rows_for_role(self, role: str) -> int:
        if role in {"attn.q", "attn.o"}:
            return self.hidden_size
        if role in {"attn.k", "attn.v"}:
            return self.kv_hidden_size
        if role in MLP_ROLES:
            return self.intermediate_size
        raise ValueError(f"Unsupported role {role!r}")

    def layer_views(self) -> RoleLayerViews:
        weights: dict[str, tuple[torch.Tensor, ...]] = {}
        for role in self.enabled_roles:
            safe_name = ROLE_TO_SAFE_NAME[role]
            weights[role] = tuple(self.weights[safe_name].unbind(0))
        return RoleLayerViews(weights=weights)

    def _role_weight(
        self,
        role: str,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        return layer_views.weights[role][layer_idx]

    def attention_weight(
        self,
        role: str,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        return self._role_weight(role, layer_idx, layer_views)

    def mlp_weight(
        self,
        role: str,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        return self._role_weight(role, layer_idx, layer_views)

    def role_parameters(self) -> dict[str, nn.Parameter]:
        return {role: self.weights[ROLE_TO_SAFE_NAME[role]] for role in self.enabled_roles}

    def _build_registry(self) -> ParameterRegistry:
        specs: list[MatrixSpec] = []
        for role in self.enabled_roles:
            safe_name = f"chunk_bank.weights.{ROLE_TO_SAFE_NAME[role]}"
            if role in ATTN_ROLES:
                for layer_idx in range(self.config.num_layers):
                    logical_rows = self._rows_for_role(role)
                    specs.append(
                        MatrixSpec(
                            role=role,
                            layer_idx=layer_idx,
                            logical_shape=(logical_rows, self.hidden_size),
                            storage_name=safe_name,
                            storage_slice=(layer_idx, slice(None), slice(None)),
                            materialization="identity",
                        )
                    )
            else:
                for layer_idx in range(self.config.num_layers):
                    materialization = "transpose" if role == "mlp.down" else "identity"
                    specs.append(
                        MatrixSpec(
                            role=role,
                            layer_idx=layer_idx,
                            logical_shape=(self.intermediate_size, self.hidden_size)
                            if role != "mlp.down"
                            else (self.hidden_size, self.intermediate_size),
                            storage_name=safe_name,
                            storage_slice=(layer_idx, slice(None), slice(None)),
                            materialization=materialization,
                        )
                    )
        return ParameterRegistry(specs, self.role_parameters())


class HybridAttention(nn.Module):
    def __init__(self, config: ModelConfig, enabled_roles: set[str]) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.kv_hidden_size = self.num_kv_heads * self.head_dim
        self.attention_dropout = config.attention_dropout
        self.enabled_roles = enabled_roles

        self.dense = CausalSelfAttention(config)
        for role, attr in (
            ("attn.q", "q_proj"),
            ("attn.k", "k_proj"),
            ("attn.v", "v_proj"),
            ("attn.o", "o_proj"),
        ):
            if role in enabled_roles:
                delattr(self.dense, attr)

        for role, attr, size in (
            ("attn.q", "q_affine", self.hidden_size),
            ("attn.k", "k_affine", self.kv_hidden_size),
            ("attn.v", "v_affine", self.kv_hidden_size),
            ("attn.o", "o_affine", self.hidden_size),
        ):
            affine = (
                nn.Parameter(torch.ones(size))
                if config.attention_affine and role in enabled_roles
                else None
            )
            setattr(self, attr, affine)

    def role_affine_parameters(self) -> dict[str, tuple[nn.Parameter, ...]]:
        params: dict[str, tuple[nn.Parameter, ...]] = {}
        for role, attr in (
            ("attn.q", "q_affine"),
            ("attn.k", "k_affine"),
            ("attn.v", "v_affine"),
            ("attn.o", "o_affine"),
        ):
            affine = getattr(self, attr)
            if isinstance(affine, nn.Parameter):
                params[role] = (affine,)
        return params

    def _linear(
        self,
        x: torch.Tensor,
        role: str,
        attr: str,
        bank: RoleChunkBank,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        if role in self.enabled_roles:
            weight = bank.attention_weight(role, layer_idx, layer_views)
            affine = getattr(self, f"{role.removeprefix('attn.')}_affine")
            if isinstance(affine, nn.Parameter):
                weight = weight * affine[:, None]
            return F.linear(x, weight)
        return getattr(self.dense, attr)(x)

    def forward(
        self,
        x: torch.Tensor,
        bank: RoleChunkBank,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._linear(x, "attn.q", "q_proj", bank, layer_idx, layer_views)
        k = self._linear(x, "attn.k", "k_proj", bank, layer_idx, layer_views)
        v = self._linear(x, "attn.v", "v_proj", bank, layer_idx, layer_views)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self._linear(attn_output, "attn.o", "o_proj", bank, layer_idx, layer_views)


@torch.compile
def swiglu_with_affine(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_affine: torch.Tensor,
    mid_affine: torch.Tensor,
) -> torch.Tensor:
    gate_affine = gate_affine.to(dtype=gate.dtype)
    mid_affine = mid_affine.to(dtype=up.dtype)
    return F.silu(gate * gate_affine) * up * mid_affine


@torch.compile
def swiglu_with_gate_affine(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_affine: torch.Tensor,
) -> torch.Tensor:
    gate_affine = gate_affine.to(dtype=gate.dtype)
    return F.silu(gate * gate_affine) * up


@torch.compile
def swiglu_with_mid_affine(
    gate: torch.Tensor,
    up: torch.Tensor,
    mid_affine: torch.Tensor,
) -> torch.Tensor:
    mid_affine = mid_affine.to(dtype=up.dtype)
    return F.silu(gate) * up * mid_affine


class HybridMLP(nn.Module):
    def __init__(self, config: ModelConfig, enabled_roles: set[str]) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp_ratio = config.mlp_ratio
        self.enabled_roles = enabled_roles
        self.dense = MLP(config)
        for role, attr in (
            ("mlp.gate", "gate_proj"),
            ("mlp.up", "up_proj"),
            ("mlp.down", "down_proj"),
        ):
            if role in enabled_roles:
                delattr(self.dense, attr)

        intermediate_size = config.hidden_size * config.mlp_ratio
        self.gate_affine: nn.Parameter | None
        self.mid_affine: nn.Parameter | None
        if config.mlp_affine and "mlp.gate" in enabled_roles:
            self.gate_affine = nn.Parameter(torch.ones(intermediate_size))
        else:
            self.gate_affine = None
        if config.mlp_affine and {"mlp.up", "mlp.down"} & enabled_roles:
            self.mid_affine = nn.Parameter(torch.ones(intermediate_size))
        else:
            self.mid_affine = None

    def role_affine_parameters(self) -> dict[str, tuple[nn.Parameter, ...]]:
        params: dict[str, tuple[nn.Parameter, ...]] = {}
        if isinstance(self.gate_affine, nn.Parameter):
            params["mlp.gate"] = (self.gate_affine,)
        if isinstance(self.mid_affine, nn.Parameter):
            if "mlp.up" in self.enabled_roles:
                params["mlp.up"] = (self.mid_affine,)
            if "mlp.down" in self.enabled_roles:
                params["mlp.down"] = (self.mid_affine,)
        return params

    def _in_proj(
        self,
        x: torch.Tensor,
        role: str,
        attr: str,
        bank: RoleChunkBank,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        if role in self.enabled_roles:
            return F.linear(x, bank.mlp_weight(role, layer_idx, layer_views))
        return getattr(self.dense, attr)(x)

    def _down_proj(
        self,
        x: torch.Tensor,
        bank: RoleChunkBank,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        if "mlp.down" in self.enabled_roles:
            return F.linear(x, bank.mlp_weight("mlp.down", layer_idx, layer_views).T)
        return self.dense.down_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        bank: RoleChunkBank,
        layer_idx: int,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        gated = self._in_proj(x, "mlp.gate", "gate_proj", bank, layer_idx, layer_views)
        up = self._in_proj(x, "mlp.up", "up_proj", bank, layer_idx, layer_views)
        if isinstance(self.gate_affine, nn.Parameter) and isinstance(self.mid_affine, nn.Parameter):
            hidden = swiglu_with_affine(gated, up, self.gate_affine, self.mid_affine)
        elif isinstance(self.gate_affine, nn.Parameter):
            hidden = swiglu_with_gate_affine(gated, up, self.gate_affine)
        elif isinstance(self.mid_affine, nn.Parameter):
            hidden = swiglu_with_mid_affine(gated, up, self.mid_affine)
        else:
            hidden = F.silu(gated) * up
        return self._down_proj(hidden, bank, layer_idx, layer_views)


class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, enabled_roles: set[str], layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HybridAttention(config, enabled_roles)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HybridMLP(config, enabled_roles)

    def forward(
        self,
        x: torch.Tensor,
        bank: RoleChunkBank,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_views: RoleLayerViews,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), bank, self.layer_idx, cos, sin, layer_views)
        x = x + self.mlp(self.post_attention_layernorm(x), bank, self.layer_idx, layer_views)
        return x


class RoleChunkedLlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.enabled_roles = set(config.enabled_roles)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, base=config.rope_theta)
        self.chunk_bank = RoleChunkBank(config)
        self.layers = nn.ModuleList(
            [HybridBlock(config, self.enabled_roles, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_dense_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_dense_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_parameter_registry(self) -> ParameterRegistry:
        return self.chunk_bank.registry

    @property
    def chunk_count(self) -> int:
        return self.chunk_bank.registry.chunk_count()

    def role_parameters(self) -> dict[str, nn.Parameter]:
        return self.chunk_bank.role_parameters()

    def role_affine_parameters(self) -> dict[str, tuple[nn.Parameter, ...]]:
        params: dict[str, list[nn.Parameter]] = {}
        for layer in self.layers:
            for role, role_params in layer.self_attn.role_affine_parameters().items():
                params.setdefault(role, []).extend(role_params)
            for role, role_params in layer.mlp.role_affine_parameters().items():
                params.setdefault(role, []).extend(role_params)
        return {role: tuple(role_params) for role, role_params in params.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence length exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x.shape[1], device=x.device, dtype=x.dtype)
        layer_views = self.chunk_bank.layer_views()

        for layer in self.layers:
            x = layer(x, self.chunk_bank, cos, sin, layer_views)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"logits": logits, "loss": loss}
