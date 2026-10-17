# Yuntun

A Qwen3-style causal language model implemented in PyTorch, with FineWeb
streaming and Megatron-style tensor parallelism.

## What is included

- Decoder-only transformer layers with RoPE, grouped-query attention, RMSNorm,
  and query/key normalization.
- FineWeb data loading and configuration-driven pretraining.
- Tensor-parallel embeddings, attention, MLPs, and language-model head.
- Hugging Face weight loading and architecture comparison scripts.

## Installation

Use Python 3.13 or later and [uv](https://docs.astral.sh/uv/).
The project requires PyTorch 2.10 or later; multi-GPU training requires CUDA.

```sh
git clone https://github.com/stochi0/yuntun.git
cd yuntun
uv sync --locked
```

## Build a model

```python
import torch
from yuntun import Qwen3Config, Qwen3ForCausalLM

config = Qwen3Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = Qwen3ForCausalLM(config)
```

This constructs a small random model without downloading pretrained weights.
See [the model implementation](src/yuntun/model/model.py) for its forward API.

## Training

The training entrypoint reads [configs/train.json](configs/train.json). It uses
FineWeb's `sample-10BT` configuration and a Qwen tokenizer from Hugging Face,
so a training run can download data and model assets.

```sh
uv run python scripts/train.py --config configs/train.json --max-steps 100
```

For two-way tensor parallelism, set `parallelism.tp_size` to `2` in the config
and launch two processes:

```sh
uv run python -m torch.distributed.run --nproc_per_node=2 \
  scripts/train.py --config configs/train.json
```

The process count must match the tensor-parallel configuration. The config also
contains model dimensions, sequence length, batch size, learning rate, and output
directory. Start with a model and batch size that fit the available device.

## Architecture checks

```sh
uv run python scripts/verify_architecture.py
```

This downloads `Qwen/Qwen3-0.6B` and compares parameters, forward outputs, and
generation behavior against the Hugging Face implementation. The checks are
architecture comparisons, not a claim about pretrained model quality or
training throughput.

The [tensor-parallel smoke check](tests/test_tp.py) covers the distributed path.

## Repository guide

| Location | Contents |
| --- | --- |
| [src/yuntun/model/](src/yuntun/model/) | Configuration, layers, model, and weight loading |
| [src/yuntun/data/](src/yuntun/data/) | FineWeb streaming |
| [src/yuntun/distributed/](src/yuntun/distributed/) | Parallel context and tensor-parallel operations |
| [src/yuntun/trainer.py](src/yuntun/trainer.py) | Training loop |
| [scripts/](scripts/) | Training and architecture-check entrypoints |

Reference implementation and pretrained weights:
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
