import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import Qwen3Config
from .utils import apply_rope_with_offset
from ..distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, tp_group=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.tp_group = tp_group

        self.tp_size = 1
        if tp_group is not None and tp_group.size() > 1:
            self.tp_size = tp_group.size()

        self.num_heads_local = self.num_heads // self.tp_size
        self.num_key_value_heads_local = self.num_key_value_heads // self.tp_size

        use_bias = getattr(config, "attention_bias", False)
        if self.tp_size > 1:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=use_bias,
                gather_output=False,
                tp_group=tp_group,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=use_bias,
                gather_output=False,
                tp_group=tp_group,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=use_bias,
                gather_output=False,
                tp_group=tp_group,
            )
            self.o_proj = RowParallelLinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                tp_group=tp_group,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=use_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=use_bias,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=use_bias,
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=False
            )

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, q_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads_local, self.head_dim).transpose(
            1, 2
        )
        v = v.view(bsz, q_len, self.num_key_value_heads_local, self.head_dim).transpose(
            1, 2
        )

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        # position_ids needed or assume contiguous?
        # For simplicity assuming contiguous right now OR relying on passed in cos/sin being large enough and indexed if needed
        # The verify logic in utils handles offset.

        # Calculate offset from past_key_value
        kv_seq_len = 0
        if past_key_value is not None:
            kv_seq_len = past_key_value[0].shape[-2]

        q = apply_rope_with_offset(q, cos, sin, position_offset=kv_seq_len)
        k = apply_rope_with_offset(k, cos, sin, position_offset=kv_seq_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_kv = (k, v) if self.config.use_cache else None

        # GQA repeat
        num_groups = self.num_heads_local // self.num_key_value_heads_local
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim**0.5)

        # Causal mask: position i can only attend to positions 0..i
        if attention_mask is None:
            q_len, kv_len = q.size(2), k.size(2)
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        # With TP, each rank has num_heads_local heads; o_proj expects (batch, seq, num_heads*head_dim) split
        attn_output = attn_output.view(bsz, q_len, self.num_heads_local * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_kv


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, tp_group=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_size = 1
        if tp_group is not None and tp_group.size() > 1:
            self.tp_size = tp_group.size()

        if self.tp_size > 1:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                tp_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                tp_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size, self.hidden_size, bias=False, tp_group=tp_group
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int, tp_group=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, tp_group=tp_group)
        self.mlp = Qwen3MLP(config, tp_group=tp_group)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, present_kv = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv
