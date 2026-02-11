import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Dict, Tuple

from .config import Qwen3ModelConfig

def _get_tp_modules():
    for mod in (
        "yuntun.distributed.tensor_parallel",
        "distributed.tensor_parallel",
    ):
        try:
            tp = __import__(mod, fromlist=[
                "ColumnParallelLinear", "RowParallelLinear",
                "VocabParallelEmbedding", "ParallelLMHead", "shard_weight_along_dim",
            ])
            return (
                True,
                tp.ColumnParallelLinear,
                tp.RowParallelLinear,
                tp.VocabParallelEmbedding,
                tp.ParallelLMHead,
                tp.shard_weight_along_dim,
            )
        except ImportError:
            continue
    return False, None, None, None, None, None


_TP_AVAILABLE, ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, ParallelLMHead, shard_weight_along_dim = _get_tp_modules()

# ---- low-level utilities (torch-only, device-aware) ----
def compute_rope_params(
    head_dim: int,
    theta_base: float = 10000.0,
    context_length: int = 4096,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos / sin tables (float32 recommended) and return (cos, sin)
    shapes: (context_length, head_dim)
    Uses torch everywhere so buffers are device-aware.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    half = head_dim // 2
    # inv_freq: (half,)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
    positions = torch.arange(0, context_length, device=device, dtype=torch.float32)  # (context,)
    angles = torch.outer(positions, inv_freq)  # (context, half)
    angles = torch.cat([angles, angles], dim=-1)  # (context, head_dim)
    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def apply_rope_with_offset(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
    """
    Apply RoPE to x.
    x: (batch, heads, seq, head_dim)
    cos/sin: (context_len, head_dim) - expected float32 (or same dtype)
    returns same dtype as x
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    assert cos.shape[1] >= head_dim, "cos table head_dim mismatch"
    positions = torch.arange(position_offset, position_offset + seq_len, device=cos.device)
    cos_slice = cos[positions].view(1, 1, seq_len, head_dim)
    sin_slice = sin[positions].view(1, 1, seq_len, head_dim)

    x_half1 = x[..., : head_dim // 2]
    x_half2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x_half2, x_half1), dim=-1)
    out = x * cos_slice.to(x.dtype) + rotated * sin_slice.to(x.dtype)
    return out


def rmsnorm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm implemented in torch. x: (..., dim), scale: (dim,)
    Returns x normalized and scaled with original dtype restored.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    variance = torch.mean(x_f * x_f, dim=-1, keepdim=True)
    x_norm = x_f * torch.rsqrt(variance + eps) * scale.view((1,) * (x.dim() - 1) + (-1,))
    return x_norm.to(orig_dtype)


def apply_qk_rms(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, heads_or_groups, seq, head_dim)
    scale: (head_dim,)
    Applies RMSNorm per last dim keeping shape.
    """
    b, h, s, d = x.shape
    flat = x.reshape(b * h * s, d)
    normed = rmsnorm(flat, scale)
    return normed.reshape(b, h, s, d)


# ---- single transformer layer with grouped-kv attention (module-based implementation) ----
class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig, tp_group=None):
        super().__init__()
        emb = cfg.hidden_size
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv_groups = cfg.num_key_value_heads
        hidden = cfg.intermediate_size
        self._tp_group = tp_group
        use_tp = tp_group is not None and tp_group.size() > 1 and _TP_AVAILABLE

        if use_tp:
            self.q_proj = ColumnParallelLinear(emb, n_heads * hd, bias=False, gather_output=False, tp_group=tp_group)
            self.k_proj = ColumnParallelLinear(emb, n_kv_groups * hd, bias=False, gather_output=False, tp_group=tp_group)
            self.v_proj = ColumnParallelLinear(emb, n_kv_groups * hd, bias=False, gather_output=False, tp_group=tp_group)
            self.out_proj = RowParallelLinear(n_heads * hd, emb, bias=False, tp_group=tp_group)
            self.ff_gate = ColumnParallelLinear(emb, hidden, bias=False, gather_output=False, tp_group=tp_group)
            self.ff_up = ColumnParallelLinear(emb, hidden, bias=False, gather_output=False, tp_group=tp_group)
            self.ff_down = RowParallelLinear(hidden, emb, bias=False, tp_group=tp_group)
        else:
            self.q_proj = nn.Linear(emb, n_heads * hd, bias=False)
            self.k_proj = nn.Linear(emb, n_kv_groups * hd, bias=False)
            self.v_proj = nn.Linear(emb, n_kv_groups * hd, bias=False)
            self.out_proj = nn.Linear(n_heads * hd, emb, bias=False)
            self.ff_gate = nn.Linear(emb, hidden, bias=False)
            self.ff_up = nn.Linear(emb, hidden, bias=False)
            self.ff_down = nn.Linear(hidden, emb, bias=False)

        # optional q/k RMSNorm scales (per head-dim)
        if cfg.qk_norm:
            # Use float32 storage for stability (can be cast when used)
            self.q_norm_scale = nn.Parameter(torch.ones((hd,), dtype=torch.float32), requires_grad=False)
            self.k_norm_scale = nn.Parameter(torch.ones((hd,), dtype=torch.float32), requires_grad=False)
            self._has_qk_norm = True
        else:
            self.q_norm_scale = None
            self.k_norm_scale = None
            self._has_qk_norm = False

        # layer norm scales: keep float32 for numerical stability, convert when applying
        self.norm1_scale = nn.Parameter(torch.ones((emb,), dtype=torch.float32), requires_grad=False)
        self.norm2_scale = nn.Parameter(torch.ones((emb,), dtype=torch.float32), requires_grad=False)

        self._cfg = cfg
        self._use_tp = use_tp

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set weights for the layer. Accepts keys:
          - q_proj, k_proj, v_proj, out_proj
          - q_norm, k_norm
          - gate_proj, up_proj, down_proj
          - norm1, norm2
        Each tensor is expected in the same layout as PyTorch Linear.weight (out_features, in_features).
        If your weights are transposed (in_features, out_features), pass them transposed already or supply .T
        With TP, full weights are sharded automatically.
        """
        def assign_linear(linear: nn.Module, tensor: torch.Tensor, shard_dim: Optional[int] = None):
            w = tensor.to(dtype=linear.weight.dtype, device=linear.weight.device)
            if shard_dim is not None and _TP_AVAILABLE and self._tp_group and self._tp_group.size() > 1:
                rank, world_size = self._tp_group.rank(), self._tp_group.size()
                w = shard_weight_along_dim(w, dim=shard_dim, rank=rank, world_size=world_size)
            linear.weight.data.copy_(w)

        # Column parallel: shard along output (dim 0)
        if "q_proj" in weights:
            assign_linear(self.q_proj, weights["q_proj"], shard_dim=0 if self._use_tp else None)
        if "k_proj" in weights:
            assign_linear(self.k_proj, weights["k_proj"], shard_dim=0 if self._use_tp else None)
        if "v_proj" in weights:
            assign_linear(self.v_proj, weights["v_proj"], shard_dim=0 if self._use_tp else None)
        # Row parallel: shard along input (dim 1)
        if "out_proj" in weights:
            assign_linear(self.out_proj, weights["out_proj"], shard_dim=1 if self._use_tp else None)

        if self._has_qk_norm:
            if "q_norm" in weights:
                self.q_norm_scale.data.copy_(weights["q_norm"].to(self.q_norm_scale.dtype, device=self.q_norm_scale.device))
            if "k_norm" in weights:
                self.k_norm_scale.data.copy_(weights["k_norm"].to(self.k_norm_scale.dtype, device=self.k_norm_scale.device))

        if "gate_proj" in weights:
            assign_linear(self.ff_gate, weights["gate_proj"], shard_dim=0 if self._use_tp else None)
        if "up_proj" in weights:
            assign_linear(self.ff_up, weights["up_proj"], shard_dim=0 if self._use_tp else None)
        if "down_proj" in weights:
            assign_linear(self.ff_down, weights["down_proj"], shard_dim=1 if self._use_tp else None)

        if "norm1" in weights:
            self.norm1_scale.data.copy_(weights["norm1"].to(self.norm1_scale.dtype, device=self.norm1_scale.device))
        if "norm2" in weights:
            self.norm2_scale.data.copy_(weights["norm2"].to(self.norm2_scale.dtype, device=self.norm2_scale.device))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_layer: Optional[Dict[str, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (batch, seq, emb)
        cos/sin: (context_len, head_dim) float32 buffers
        kv_cache_layer: {"keys": (batch, n_kv_groups_local, seq_kv, head_dim),
                        "values": (batch, n_kv_groups_local, seq_kv, head_dim)}
        returns (out: batch, seq, emb), new_kv_cache_layer
        """
        b, seq, emb = x.shape
        cfg = self._cfg
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv_groups = cfg.num_key_value_heads
        tp_size = self._tp_group.size() if (self._tp_group and self._tp_group.size() > 1) else 1
        n_heads_local = n_heads // tp_size
        n_kv_groups_local = n_kv_groups // tp_size
        group_size = n_heads_local // n_kv_groups_local  # same as n_heads // n_kv_groups
        scale = hd ** 0.5

        # --- attention block ---
        x_norm = rmsnorm(x, self.norm1_scale)  # (b, seq, emb)

        # linear projections (output is local with TP)
        q = self.q_proj(x_norm).view(b, seq, n_heads_local, hd).permute(0, 2, 1, 3)  # (b, n_heads_local, seq, hd)
        k = self.k_proj(x_norm).view(b, seq, n_kv_groups_local, hd).permute(0, 2, 1, 3)  # (b, n_kv_groups_local, seq, hd)
        v = self.v_proj(x_norm).view(b, seq, n_kv_groups_local, hd).permute(0, 2, 1, 3)

        if self._has_qk_norm:
            q = apply_qk_rms(q, self.q_norm_scale)
            k = apply_qk_rms(k, self.k_norm_scale)

        # apply rope with offset (cos/sin may be float32)
        q = apply_rope_with_offset(q, cos, sin, position_offset)
        k = apply_rope_with_offset(k, cos, sin, position_offset)

        if kv_cache_layer is not None and kv_cache_layer["keys"].size(2) > 0:
            k = torch.cat([kv_cache_layer["keys"], k], dim=2)
            v = torch.cat([kv_cache_layer["values"], v], dim=2)

        new_cache = {"keys": k, "values": v}

        # expand kv to heads
        k_exp = k.repeat_interleave(group_size, dim=1)
        v_exp = v.repeat_interleave(group_size, dim=1)

        # attention scores (b, n_heads, q_len, k_len)
        attn_scores = torch.einsum("bnqh,bnkh->bnqk", q, k_exp) / scale

        # causal mask (only applied if no prior keys OR you want to apply as-is)
        if kv_cache_layer is None or kv_cache_layer["keys"].size(2) == 0:
            q_len, k_len = q.size(2), k_exp.size(2)
            causal_mask = torch.triu(torch.ones((q_len, k_len), dtype=torch.bool, device=attn_scores.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.einsum("bnqk,bnkh->bnqh", attn_weights, v_exp)
        context = context.permute(0, 2, 1, 3).contiguous().view(b, seq, n_heads_local * hd)
        attn_out = self.out_proj(context)  # (b, seq, emb) - all-reduced with TP

        x = x + attn_out  # residual

        # --- feedforward block ---
        x_norm2 = rmsnorm(x, self.norm2_scale)
        gate = F.silu(self.ff_gate(x_norm2))
        up = self.ff_up(x_norm2)
        ff_out = self.ff_down(gate * up)

        x = x + ff_out

        return x, new_cache


# ---- full model built from per-layer modules ----
class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig, tp_group=None):
        super().__init__()
        self.cfg = cfg
        self._tp_group = tp_group
        use_tp = tp_group is not None and tp_group.size() > 1 and _TP_AVAILABLE
        self._use_tp = use_tp

        if use_tp:
            cfg.validate_tensor_parallel(tp_size=tp_group.size())

        # Embedding / head
        if use_tp:
            self.tok_emb = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, tp_group=tp_group)
            self.out_head = ParallelLMHead(cfg.hidden_size, cfg.vocab_size, bias=False, tp_group=tp_group)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.out_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tok_emb.weight.requires_grad = False
        self.out_head.weight.requires_grad = False

        self.trf_blocks = nn.ModuleList([Qwen3Block(cfg, tp_group=tp_group) for _ in range(cfg.num_hidden_layers)])

        # final norm scale (RMSNorm style)
        self.final_norm_scale = nn.Parameter(torch.ones((cfg.hidden_size,), dtype=torch.float32), requires_grad=False)

        # precompute RoPE on device (float32 buffers)
        cos, sin = compute_rope_params(
            cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings,
            device=torch.device(self.cfg.device), dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # cast module weights to desired runtime dtype for inference
        self.to(torch.device(self.cfg.device))
        self._cast_runtime_dtype(self.cfg.dtype)

    def _cast_runtime_dtype(self, runtime_dtype: torch.dtype):
        """
        Cast parameters that are safe to cast to the runtime dtype (e.g., bfloat16),
        while keeping norm scales in float32 for stability.
        """
        for name, p in self.named_parameters():
            # keep norm scales (final_norm_scale, layer norm scales, q/k norm) in float32
            if "norm" in name:
                continue
            p.data = p.data.to(dtype=runtime_dtype)

        # Embedding and out_head weight types
        self.tok_emb.weight.data = self.tok_emb.weight.data.to(dtype=runtime_dtype)
        self.out_head.weight.data = self.out_head.weight.data.to(dtype=runtime_dtype)

    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Load global weights. Expected keys: 'tok_emb', 'out_head', 'final_norm'
        Each tensor should have PyTorch layout: tok_emb (vocab, hidden), out_head (vocab, hidden).
        With TP, full weights are sharded automatically.
        """
        def _copy_emb(w: torch.Tensor):
            t = w.to(dtype=self.tok_emb.weight.dtype, device=self.tok_emb.weight.device)
            if self._use_tp and _TP_AVAILABLE:
                rank, world_size = self._tp_group.rank(), self._tp_group.size()
                t = shard_weight_along_dim(t, dim=0, rank=rank, world_size=world_size)
            self.tok_emb.weight.data.copy_(t)

        def _copy_head(w: torch.Tensor):
            t = w.to(dtype=self.out_head.weight.dtype, device=self.out_head.weight.device)
            if self._use_tp and _TP_AVAILABLE:
                rank, world_size = self._tp_group.rank(), self._tp_group.size()
                t = shard_weight_along_dim(t, dim=0, rank=rank, world_size=world_size)
            self.out_head.weight.data.copy_(t)

        if "tok_emb" in weights:
            _copy_emb(weights["tok_emb"])
        if "out_head" in weights:
            _copy_head(weights["out_head"])
        if "final_norm" in weights:
            self.final_norm_scale.data.copy_(weights["final_norm"].to(dtype=self.final_norm_scale.dtype, device=self.final_norm_scale.device))

    def set_layer_weights(self, layer_idx: int, layer_weights: Dict[str, torch.Tensor]):
        self.trf_blocks[layer_idx].set_weights(layer_weights)

    def forward(self, input_ids: torch.LongTensor, kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        input_ids: (batch, seq)
        kv_cache: list of length n_layers or None; each element is a dict with 'keys' and 'values'
        returns logits (b, seq, vocab) and new_kv_cache list
        """
        b, seq = input_ids.shape
        x = self.tok_emb(input_ids)  # (b, seq, emb)
        new_kv_cache: List[Dict[str, torch.Tensor]] = []

        # run layers
        for i, block in enumerate(self.trf_blocks):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            pos_offset = layer_kv["keys"].size(2) if (layer_kv is not None and layer_kv["keys"].size(2) > 0) else 0
            x, updated_cache = block(x, self.cos, self.sin, kv_cache_layer=layer_kv, position_offset=pos_offset)
            new_kv_cache.append(updated_cache)

        # final norm + lm head
        x = rmsnorm(x, self.final_norm_scale)
        logits = self.out_head(x)  # (b, seq, vocab)
        return logits, new_kv_cache

    # convenience: single-layer set + bulk set
    def set_all_layer_weights(self, weights_list: List[Dict[str, torch.Tensor]]):
        for i, w in enumerate(weights_list):
            self.set_layer_weights(i, w)


# ---- quick smoke when run as script ----
if __name__ == "__main__":
    cfg = Qwen3ModelConfig()
    # triggers post_init to set device/dtype
    model = Qwen3Model(cfg)

    batch = 1
    prompt_len = 8
    dummy_ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), dtype=torch.long, device=cfg.device)

    logits, new_cache = model(dummy_ids, kv_cache=None)
    print("logits.shape:", logits.shape)

    last_id = dummy_ids[:, -1:].contiguous()
    logits2, new_cache2 = model(last_id, kv_cache=new_cache)
    print("logits2.shape:", logits2.shape)
