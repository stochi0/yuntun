"""
Megatron-style tensor parallelism.

Column parallel: weight split along output dim. Y = XA^T, A = [A_1|A_2|...|A_p].
Row parallel: weight split along input dim. Y = XA^T, A = [A_1;A_2;...;A_p]^T.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

# ---------------------------------------------------------------------------
# Communication primitives
# ---------------------------------------------------------------------------


def tp_all_reduce(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-reduce across tensor parallel group (sums sharded tensors)."""
    if group is None or group.size() == 1:
        return tensor
    tensor = tensor.contiguous()
    work = dist.all_reduce(tensor, group=group, async_op=True)
    work.wait()
    return tensor


def tp_all_gather(
    tensor: torch.Tensor,
    dim: int = -1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather sharded tensors along dim into full tensor on each rank."""
    if group is None or group.size() == 1:
        return tensor
    world_size = group.size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    work = dist.all_gather(tensor_list, tensor.contiguous(), group=group, async_op=True)
    work.wait()
    return torch.cat(tensor_list, dim=dim)


# ---------------------------------------------------------------------------
# Vocab partitioning
# ---------------------------------------------------------------------------


def get_vocab_partition_range(
    vocab_size: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Return (start, end) indices for this rank's vocabulary shard."""
    per_rank = vocab_size // world_size
    return rank * per_rank, (rank + 1) * per_rank


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------


class _ColumnParallelLinearForward(torch.autograd.Function):
    """Forward: Y = X @ W^T + b. W split along output dim."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.group = group
        output = torch.nn.functional.linear(input_, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D902
        input_, weight = ctx.saved_tensors
        group = ctx.group

        grad_input = grad_weight = grad_bias = None
        handle = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
            if group is not None and group.size() > 1:
                handle = dist.all_reduce(grad_input, group=group, async_op=True)

        if ctx.needs_input_grad[1]:
            go = grad_output.reshape(-1, grad_output.shape[-1])
            inp = input_.reshape(-1, input_.shape[-1])
            grad_weight = torch.matmul(go.t(), inp)

        if ctx.use_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        if handle is not None:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None


class ColumnParallelLinear(nn.Module):
    """
    Linear with column parallelism. Y = XA^T + b.
    A parallelized along its second dimension: A = [A_1 | A_2 | ... | A_p].
    Each rank holds A_i of shape (output_per_partition, input_size).

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample (global).
        bias: If True, add bias.
        gather_output: If True, all-gather output so all ranks have full Y.
        tp_group: Tensor parallel process group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group else 1
        assert out_features % tp_size == 0
        self.output_size_per_partition = out_features // tp_size

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_features, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.output_size_per_partition, device=device))
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_group and self.tp_group.size() > 1:
            out = _ColumnParallelLinearForward.apply(
                x, self.weight, self.bias, self.tp_group
            )
        else:
            out = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output and self.tp_group and self.tp_group.size() > 1:
            out = tp_all_gather(out, dim=-1, group=self.tp_group)
        return out


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------


class _RowParallelLinearForward(torch.autograd.Function):
    """Forward: Y = X @ W^T + b, then all-reduce. W split along input dim."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = torch.nn.functional.linear(input_, weight, bias)
        if group is not None and group.size() > 1:
            work = dist.all_reduce(output, group=group, async_op=True)
            work.wait()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D902
        input_, weight = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            go = grad_output.reshape(-1, grad_output.shape[-1])
            inp = input_.reshape(-1, input_.shape[-1])
            grad_weight = torch.matmul(go.t(), inp)

        if ctx.use_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias, None


class RowParallelLinear(nn.Module):
    """
    Linear with row parallelism. Y = XA^T + b.
    A parallelized along its first dimension: A = [A_1; A_2; ...; A_p]^T.
    Each rank holds A_i of shape (output_size, input_per_partition).
    Forward: local GEMM then all-reduce on output.

    Args:
        in_features: Size of each input sample (global).
        out_features: Size of each output sample.
        bias: If True, add bias.
        input_is_parallel: If True, input is already split (default True).
        tp_group: Tensor parallel process group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group else 1
        assert in_features % tp_size == 0
        self.input_size_per_partition = in_features // tp_size

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_partition, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_group and self.tp_group.size() > 1:
            return _RowParallelLinearForward.apply(
                x, self.weight, self.bias, self.tp_group
            )
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# VocabParallelEmbedding
# ---------------------------------------------------------------------------


class VocabParallelEmbedding(nn.Module):
    """
    Embedding parallelized along the vocabulary dimension.
    Each rank holds vocab_size // tp_size rows.
    Forward: masked lookup, then all-reduce to sum partial embeddings.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group else 1
        rank = tp_group.rank() if tp_group else 0
        assert num_embeddings % tp_size == 0

        self.vocab_start_index, self.vocab_end_index = get_vocab_partition_range(
            num_embeddings, rank, tp_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, device=device)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_group and self.tp_group.size() > 1:
            input_mask = (input_ids < self.vocab_start_index) | (
                input_ids >= self.vocab_end_index
            )
            masked_input = input_ids.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids

        output_parallel = torch.nn.functional.embedding(masked_input, self.weight)

        if self.tp_group and self.tp_group.size() > 1:
            output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0.0)
            output = tp_all_reduce(output_parallel, self.tp_group)
        else:
            output = output_parallel

        return output


# ---------------------------------------------------------------------------
# ParallelLMHead
# ---------------------------------------------------------------------------


class ParallelLMHead(nn.Module):
    """
    LM head parallelized along vocabulary (output) dimension.
    Weight [vocab_per_partition, hidden]. Output gathered for full logits.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.tp_group = tp_group
        tp_size = tp_group.size() if tp_group else 1
        assert vocab_size % tp_size == 0
        self.vocab_size_per_partition = vocab_size // tp_size

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size_per_partition, hidden_size, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.vocab_size_per_partition, device=device))
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.tp_group and self.tp_group.size() > 1:
            logits = tp_all_gather(logits, dim=-1, group=self.tp_group)
        return logits


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def shard_weight_along_dim(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Return this rank's shard of the tensor along the given dimension."""
    dim_size = tensor.size(dim)
    assert dim_size % world_size == 0
    per_rank = dim_size // world_size
    start = rank * per_rank
    return tensor.narrow(dim, start, per_rank).clone()
