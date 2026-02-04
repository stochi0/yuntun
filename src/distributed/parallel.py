from dataclasses import dataclass
import torch.distributed as dist

_GROUPS = None # avoid re-creating NCCL communicators

def rank_from_coords(dp, pp, tp, pp_size, tp_size):
    return dp * (pp_size * tp_size) + pp * tp_size + tp


def coords_from_rank(rank, pp_size, tp_size):
    tp = rank % tp_size
    pp = (rank // tp_size) % pp_size
    dp = rank // (pp_size * tp_size)
    return dp, pp, tp

@dataclass
class ParallelGroups:
    dp: dist.ProcessGroup
    pp: dist.ProcessGroup
    tp: dist.ProcessGroup

    dp_rank: int
    pp_rank: int
    tp_rank: int

def create_groups(tp_size, pp_size, dp_size):
    global _GROUPS
    if _GROUPS is not None:
        return _GROUPS

    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == tp_size * pp_size * dp_size

    dp_rank, pp_rank, tp_rank = coords_from_rank(
        rank, pp_size, tp_size
    )

    # Tensor Parallel
    tp_ranks = [
        rank_from_coords(dp_rank, pp_rank, t,
                         pp_size, tp_size)
        for t in range(tp_size)
    ]
    tp_group = dist.new_group(tp_ranks)

    # Pipeline Parallel
    pp_ranks = [
        rank_from_coords(dp_rank, p, tp_rank,
                         pp_size, tp_size)
        for p in range(pp_size)
    ]
    pp_group = dist.new_group(pp_ranks)

    # Data Parallel
    dp_ranks = [
        rank_from_coords(d, pp_rank, tp_rank,
                         pp_size, tp_size)
        for d in range(dp_size)
    ]
    dp_group = dist.new_group(dp_ranks)

    _GROUPS = ParallelGroups(
        dp=dp_group,
        pp=pp_group,
        tp=tp_group,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
    )

    return _GROUPS
