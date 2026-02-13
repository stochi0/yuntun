#!/usr/bin/env python3
"""
Test Megatron-style tensor parallelism with Qwen3.
Uses Gloo backend to simulate multi-GPU on CPU (works on Mac, no CUDA).

Run: torchrun --nproc_per_node=2 scripts/run_tp_test.py
"""

import os
import sys

# Ensure src is on path for distributed.tensor_parallel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.distributed as dist

from models.qwen3.model import Qwen3Model
from models.qwen3.config import Qwen3ModelConfig
from distributed.parallel import create_groups

# Gloo works on CPU - use it to simulate TP without multiple GPUs
BACKEND = "gloo"


def main():
    dist.init_process_group(backend=BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create TP group (tp_size=world_size, pp=1, dp=1)
    groups = create_groups(tp_size=world_size, pp_size=1, dp_size=1)

    cfg = Qwen3ModelConfig(tp_size=world_size)
    # Gloo simulates multi-GPU on CPU; use cpu for all ranks
    cfg.device = "cpu"
    cfg.dtype = torch.float32  # bf16 not supported on CPU
    model = Qwen3Model(cfg, tp_group=groups.tp)
    model.to(cfg.device)

    if rank == 0:
        print(f"Running Qwen3 in TP (tp_size={world_size}) with Gloo backend on CPU")

    # Forward
    batch, seq = 2, 8
    input_ids = torch.randint(
        0, min(1000, cfg.vocab_size), (batch, seq), device=cfg.device
    )
    logits, kv_cache = model(input_ids, kv_cache=None)

    if rank == 0:
        print(
            f"logits.shape: {logits.shape} (expected [{batch}, {seq}, {cfg.vocab_size}])"
        )
        assert logits.shape == (batch, seq, cfg.vocab_size)
        print("TP test passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
