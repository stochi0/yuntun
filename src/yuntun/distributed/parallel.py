"""Process group creation for tensor, pipeline, and data parallelism."""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist


def create_groups(
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
) -> "ParallelGroups":
    """
    Create process groups for TP, PP, and DP.

    Args:
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        dp_size: Data parallel size.

    Returns:
        ParallelGroups with .tp, .pp, .dp attributes (process groups or None).
    """
    world_size = dist.get_world_size()

    if world_size != tp_size * pp_size * dp_size:
        raise ValueError(
            f"world_size={world_size} must equal tp_size*pp_size*dp_size={tp_size}*{pp_size}*{dp_size}"
        )

    tp_group: Optional[dist.ProcessGroup] = None
    pp_group: Optional[dist.ProcessGroup] = None
    dp_group: Optional[dist.ProcessGroup] = None

    # For TP-only (pp=1, dp=1): tp_group = all ranks
    if tp_size > 1:
        tp_group = dist.new_group(ranks=list(range(world_size)))

    if pp_size > 1:
        pp_group = dist.new_group(ranks=list(range(world_size)))

    if dp_size > 1:
        dp_group = dist.new_group(ranks=list(range(world_size)))

    return ParallelGroups(tp=tp_group, pp=pp_group, dp=dp_group)


class ParallelGroups:
    """Container for TP, PP, DP process groups."""

    tp: Optional[dist.ProcessGroup]
    pp: Optional[dist.ProcessGroup]
    dp: Optional[dist.ProcessGroup]

    def __init__(
        self,
        tp: Optional[dist.ProcessGroup] = None,
        pp: Optional[dist.ProcessGroup] = None,
        dp: Optional[dist.ProcessGroup] = None,
    ):
        self.tp = tp
        self.pp = pp
        self.dp = dp
