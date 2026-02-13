#!/usr/bin/env python3
"""
Test Megatron-style tensor parallelism with Qwen3.
Uses Gloo backend to simulate multi-GPU on CPU (works on Mac, no CUDA).

Run: torchrun --nproc_per_node=2 scripts/run_tp_test.py
"""

import os
import sys
from pathlib import Path

# Ensure src is on path (package root)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.distributed as dist

from yuntun.model.config import Qwen3Config
from yuntun.model.model import Qwen3ForCausalLM
from yuntun.distributed.parallel import create_groups

# Gloo works on CPU - use it to simulate TP without multiple GPUs
BACKEND = "gloo"


def main():
    dist.init_process_group(backend=BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create TP group (tp_size=world_size, pp=1, dp=1)
    groups = create_groups(tp_size=world_size, pp_size=1, dp_size=1)

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )

    device = "cpu"
    model = Qwen3ForCausalLM(cfg, tp_group=groups.tp)
    model.to(device)

    if rank == 0:
        print(f"Running Qwen3 in TP (tp_size={world_size}) with Gloo backend on CPU")

    # Forward
    batch, seq = 2, 8
    input_ids = torch.randint(
        0, min(1000, cfg.vocab_size), (batch, seq), device=device
    )
    outputs = model(input_ids=input_ids, return_dict=True)
    logits = outputs["logits"]

    if rank == 0:
        print(
            f"logits.shape: {logits.shape} (expected [{batch}, {seq}, {cfg.vocab_size}])"
        )
        assert logits.shape == (batch, seq, cfg.vocab_size)
        print("TP test passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
