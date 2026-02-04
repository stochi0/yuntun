from functools import partial
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def make_transformer_policy(block_cls):

    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_cls},
    )


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    dp_group,
    block_cls,
    mixed_precision=False,
):

    auto_wrap_policy = make_transformer_policy(block_cls)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        process_group=dp_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
    )

    return fsdp_model
