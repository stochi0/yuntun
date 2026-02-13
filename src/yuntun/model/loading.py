"""Load Qwen3 weights from Hugging Face into our model (with optional TP sharding)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .model import Qwen3ForCausalLM
from ..distributed.tensor_parallel import shard_weight_along_dim


def load_qwen3_from_hf(
    model: Qwen3ForCausalLM,
    model_name: str = "Qwen/Qwen3-0.6B",
    tp_group: Optional[dist.ProcessGroup] = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Load Qwen3 weights from Hugging Face into our model.

    Handles tensor parallelism: when tp_group has size > 1, weights are sharded
    appropriately for ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
    and ParallelLMHead.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict.
    """
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    hf_sd = hf_model.state_dict()
    del hf_model

    tp_size = tp_group.size() if tp_group else 1
    rank = tp_group.rank() if tp_group else 0

    our_sd = {}
    missing = []
    unexpected = []

    def shard(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if tp_size <= 1:
            return tensor
        return shard_weight_along_dim(tensor, dim, rank, tp_size)

    for hf_key, hf_val in hf_sd.items():
        val = hf_val.float().clone()

        # Embedding: vocab parallel (shard dim 0)
        if hf_key == "model.embed_tokens.weight":
            our_sd["model.embed_tokens.weight"] = shard(val, 0)

        # LM head: vocab parallel (shard dim 0)
        elif hf_key == "lm_head.weight":
            our_sd["lm_head.weight"] = shard(val, 0)

        # Column parallel (shard output dim = 0): q,k,v, gate, up
        elif "self_attn.q_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "self_attn.k_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "self_attn.v_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "self_attn.q_proj.bias" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "self_attn.k_proj.bias" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "self_attn.v_proj.bias" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "mlp.gate_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 0)
        elif "mlp.up_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 0)

        # Row parallel (shard input dim = 1): o_proj, down_proj
        elif "self_attn.o_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 1)
        elif "mlp.down_proj.weight" in hf_key:
            our_sd[hf_key] = shard(val, 1)

        # Unsharded: norms, q_norm, k_norm
        else:
            our_sd[hf_key] = val

    result = model.load_state_dict(our_sd, strict=strict)
    return result.missing_keys, result.unexpected_keys
