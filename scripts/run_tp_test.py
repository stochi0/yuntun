#!/usr/bin/env python3
"""
Test Megatron-style tensor parallelism with Qwen3.
Uses Gloo backend to simulate multi-GPU on CPU (works on Mac, no CUDA).
Loads actual Qwen3-0.6B weights from Hugging Face.

Run:
  TP=1: uv run python scripts/run_tp_test.py
  TP=2: uv run python -m torch.distributed.run --nproc_per_node=2 scripts/run_tp_test.py
"""

import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig

from yuntun.model.config import Qwen3Config
from yuntun.model.model import Qwen3ForCausalLM
from yuntun.model.loading import load_qwen3_from_hf
from yuntun.distributed.parallel import create_groups

# Gloo works on CPU - use it to simulate TP without multiple GPUs
BACKEND = "gloo"
HF_MODEL = "Qwen/Qwen3-0.6B"


def main():
    # Support both single-process (TP=1) and multi-process (TP>1)
    if "RANK" in __import__("os").environ:
        dist.init_process_group(backend=BACKEND)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        groups = create_groups(tp_size=world_size, pp_size=1, dp_size=1)
    else:
        rank = 0
        world_size = 1
        groups = type("Groups", (), {"tp": None})()

    # Load config from HF to match Qwen3-0.6B
    hf_config = AutoConfig.from_pretrained(HF_MODEL)
    cfg = Qwen3Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        attention_bias=getattr(hf_config, "attention_bias", False),
        qk_norm=getattr(hf_config, "qk_norm", True),
        head_dim=getattr(hf_config, "head_dim", None),
    )

    device = "cpu"
    model = Qwen3ForCausalLM(cfg, tp_group=groups.tp)
    load_qwen3_from_hf(model, model_name=HF_MODEL, tp_group=groups.tp)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    prompt = "Hello, how are you?"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    logits = model(input_ids=input_ids, return_dict=True)["logits"]

    if rank == 0:
        assert logits.shape == (1, input_ids.shape[1], cfg.vocab_size)
        decoded = tokenizer.decode(logits.argmax(dim=-1)[0], skip_special_tokens=True)
        print(f"{prompt!r} -> {decoded!r}")
        print("TP test passed")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
