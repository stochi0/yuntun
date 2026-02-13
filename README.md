# Yuntun

<div align="center">
  <img src="https://images.openai.com/static-rsc-3/mUmYOtSSgJMUZV_GHzMcb4_vBK_vxkF_ZrP-i8oj7xlY17QPzFY5y9P8EMMUuvP3wFgwnhYK_umVNPr4aaVwsuQIY5MjczAqE_kCzI04HeI?purpose=fullsize&v=1" width="280" alt="Yuntun mascot" />
</div>

A minimal, from-scratch implementation of the **Qwen3** causal language model architecture with pre-training support on the FineWeb dataset. Includes Megatron-style tensor parallelism for multi-GPU training.

## Features

- **Qwen3 architecture** — Decoder-only transformer with RoPE, GQA, RMSNorm, and QK-norm
- **FineWeb pre-training** — Streaming data loading from HuggingFace FineWeb (sample-10BT)
- **Tensor parallelism** — Megatron-style TP for embeddings, attention, MLP, and LM head
- **HuggingFace compatible** — Load pretrained Qwen3 weights and verify architectural parity
- **Config-driven training** — JSON config for model, data, and parallelism settings

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.10
- CUDA (optional, for GPU training)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Clone and install
git clone <repo-url>
cd yuntun
uv sync
```

## Quick Start

### Train on FineWeb (single GPU)

```bash
uv run python scripts/train.py --config configs/train.json
```

### Quick test (override max steps)

```bash
uv run python scripts/train.py --config configs/train.json --max-steps 100
```

### Tensor parallelism (multi-GPU)

```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/train.py --config configs/train.json
```

Set `parallelism.tp_size` in your config to match `--nproc_per_node`.

### Verify architecture parity with HuggingFace Qwen3

```bash
uv run python scripts/verify_architecture.py
```

Runs parameter equality, forward pass parity, random stress tests, and generation checks against `Qwen/Qwen3-0.6B`.

## Project Structure

```
yuntun/
├── configs/
│   └── train.json          # Training config (model, data, parallelism)
├── scripts/
│   ├── train.py            # Training entrypoint
│   └── verify_architecture.py  # HF parity verification
├── src/yuntun/
│   ├── model/              # Qwen3 model, layers, config
│   ├── data/               # FineWeb data module
│   ├── distributed/        # Tensor parallelism (TP)
│   └── trainer.py          # Training loop
└── tests/
    └── test_tp.py          # TP smoke test
```

## Configuration

Example `configs/train.json`:

```json
{
  "dataset": "HuggingFaceFW/fineweb",
  "dataset_config": "sample-10BT",
  "tokenizer": "Qwen/Qwen3-0.6B",
  "max_seq_length": 512,
  "batch_size": 8,
  "grad_accum_steps": 4,
  "lr": 0.0003,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "max_steps": 10000,
  "eval_steps": 500,
  "save_steps": 1000,
  "output_dir": "outputs/fineweb",
  "parallelism": {
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1
  },
  "model": {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 40960
  }
}
```

## API

```python
from yuntun import Qwen3Config, Qwen3Model, Qwen3ForCausalLM

config = Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    num_hidden_layers=28,
    # ...
)
model = Qwen3ForCausalLM(config)
```
