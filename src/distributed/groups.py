import torch.distributed as dist


def init_distributed(backend="nccl"):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def rank_from_coords(dp, pp, tp, pp_size, tp_size):
    return dp * (pp_size * tp_size) + pp * tp_size + tp


def create_groups(tp_size, pp_size, dp_size):

    if not dist.is_initialized():
        raise RuntimeError("Distributed training not initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == tp_size * pp_size * dp_size

    tp_rank = rank % tp_size
    pp_rank = (rank // tp_size) % pp_size
    dp_rank = rank // (tp_size * pp_size)

    # -------------------------
    # Tensor Parallel Groups
    # -------------------------

    tp_groups = []

    for d in range(dp_size):
        for p in range(pp_size):

            ranks = [
                rank_from_coords(d, p, t, pp_size, tp_size)
                for t in range(tp_size)
            ]

            tp_groups.append(dist.new_group(ranks))


    # -------------------------
    # Pipeline Parallel Groups
    # -------------------------

    pp_groups = []

    for d in range(dp_size):
        for t in range(tp_size):

            ranks = [
                rank_from_coords(d, p, t, pp_size, tp_size)
                for p in range(pp_size)
            ]

            pp_groups.append(dist.new_group(ranks))


    # -------------------------
    # Data Parallel Groups
    # -------------------------

    dp_groups = []

    for p in range(pp_size):
        for t in range(tp_size):

            ranks = [
                rank_from_coords(d, p, t, pp_size, tp_size)
                for d in range(dp_size)
            ]

            dp_groups.append(dist.new_group(ranks))

    return {
        "tp": tp_groups[dp_rank * pp_size + pp_rank],
        "pp": pp_groups[dp_rank * tp_size + tp_rank],
        "dp": dp_groups[pp_rank * tp_size + tp_rank],
    }
