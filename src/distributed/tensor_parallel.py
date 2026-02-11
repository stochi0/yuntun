"""
Megatron-style tensor parallelism primitives.

Column Parallel: output dimension split across TP ranks.
  - Weight shape per rank: (output_size // tp_size, input_size)
  - Forward: local matmul, no communication
  - Backward: all-reduce on grad_input

Row Parallel: input dimension split across TP ranks.
  - Weight shape per rank: (output_size, input_size // tp_size)
  - Forward: local matmul then all-reduce on output
  - Backward: grad_output is replicated, grad_input stays split
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def reduce_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-reduce across TP group. Forward: reduce. Backward: copy."""
    if group is None or group.size() == 1:
        return input_
    dist.all_reduce(input_.contiguous(), group=group)
    return input_


def gather_from_tensor_model_parallel_region(
    input_: torch.Tensor,
    dim: int = -1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather along the given dimension."""
    if group is None or group.size() == 1:
        return input_
    world_size = group.size()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(tensor_list, input_.contiguous(), group=group)
    return torch.cat(tensor_list, dim=dim)


def copy_to_tensor_model_parallel_region(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Forward: copy. Backward: all-reduce on grad_output."""
    if group is None or group.size() == 1:
        return input_
    return _CopyToTensorParallelRegion.apply(input_, group)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return reduce_from_tensor_model_parallel_region(grad_output, ctx.group), None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        return reduce_from_tensor_model_parallel_region(input_, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _split_tensor_along_dim(tensor: torch.Tensor, world_size: int, rank: int, dim: int = -1) -> torch.Tensor:
    """Return this rank's shard along the given dimension."""
    dim_size = tensor.size(dim)
    assert dim_size % world_size == 0
    per_rank = dim_size // world_size
    start = rank * per_rank
    return tensor.narrow(dim, start, per_rank).contiguous()


# -----------------------------------------------------------------------------
# Async Column Parallel Linear
# -----------------------------------------------------------------------------

class _AsyncColumnParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = torch.matmul(input_, weight.t())
        if bias is not None:
            output = output + bias
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        group = ctx.group

        grad_input = None
        grad_weight = None
        grad_bias = None
        handle = None

        # Standard backward for Linear:
        # grad_input = grad_output @ weight
        # grad_weight = grad_output.t() @ input_
        # grad_bias = grad_output.sum(0)

        # Async All-Reduce of grad_input
        # We compute grad_input (partial) locally, then all-reduce it.
        # Overlap: Compute grad_weight while all-reducing grad_input.

        if ctx.needs_input_grad[0]:
            # Compute partial grad_input
            grad_input = torch.matmul(grad_output, weight)
            # Start all-reduce
            if group is not None and group.size() > 1:
                handle = dist.all_reduce(grad_input, group=group, async_op=True)
            else:
                handle = None

        if ctx.needs_input_grad[1]:
            # Compute grad_weight
            # grad_output: (batch*, out_features/tp)
            # input: (batch*, in_features)
            # weight: (out_features/tp, in_features)
            # grad_weight: (out_features/tp, in_features)
            # Reshape if necessary (handle multi-dim input)
            grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
            input_2d = input_.view(-1, input_.shape[-1])
            grad_weight = torch.matmul(grad_output_2d.t(), input_2d)

        if ctx.use_bias and ctx.needs_input_grad[2]:
            grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
            grad_bias = grad_output_2d.sum(0)

        if handle is not None:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None



# -----------------------------------------------------------------------------
# Column Parallel Linear
# -----------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """
    Linear with column parallelism (output dimension split).
    Y = X @ A^T, A split along columns: A = [A_1 | A_2 | ... | A_p]
    Each rank holds A_i of shape (output_per_rank, input_size).
    When tp_group is None or size 1, behaves as a normal nn.Linear.
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

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_features, device=_get_device())
        )
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, device=_get_device())) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calls the async autograd function
        # Note: Functional.linear handles reshaping, so we might need to handle it in _AsyncColumnParallelLinear if inputs are > 2D.
        # _AsyncColumnParallelLinear assumes x is (..., in_features).
        
        if self.tp_group and self.tp_group.size() > 1:
            # We use our custom autograd function for async backward overlap
            out = _AsyncColumnParallelLinear.apply(x, self.weight, self.bias, self.tp_group)
        else:
            # Fallback to standard linear
            out = torch.nn.functional.linear(x, self.weight, self.bias)
            
        if self.gather_output and self.tp_group and self.tp_group.size() > 1:
            out = gather_from_tensor_model_parallel_region(out, dim=-1, group=self.tp_group)
        return out


# -----------------------------------------------------------------------------
# Async Row Parallel Linear
# -----------------------------------------------------------------------------

class _AsyncRowParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = torch.matmul(input_, weight.t())
        if group is not None and group.size() > 1:
            dist.all_reduce(output, group=group)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        # No async comms needed here for RowParallel backward (grad_output is already replicated).
        # We just compute gradients locally.
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        
        if ctx.needs_input_grad[1]:
            grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
            input_2d = input_.view(-1, input_.shape[-1])
            grad_weight = torch.matmul(grad_output_2d.t(), input_2d)
        
        if ctx.use_bias and ctx.needs_input_grad[2]:
            grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias, None


# -----------------------------------------------------------------------------
# Row Parallel Linear
# -----------------------------------------------------------------------------


class RowParallelLinear(nn.Module):
    """
    Linear with row parallelism (input dimension split).
    Y = X @ A^T, A split along rows: A = [A_1; A_2; ...; A_p]^T
    Each rank holds A_i of shape (output_size, input_per_rank).
    Input X is expected to be already split (from column parallel output).
    Output is all-reduced.
    When tp_group is None or size 1, behaves as a normal nn.Linear.
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

        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_partition, device=_get_device())
        )
        self.bias = nn.Parameter(torch.empty(out_features, device=_get_device())) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_group and self.tp_group.size() > 1:
            out = _AsyncRowParallelLinear.apply(x, self.weight, self.bias, self.tp_group)
        else:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out


# -----------------------------------------------------------------------------
# Vocab Parallel Embedding (split along vocab dimension)
# -----------------------------------------------------------------------------


class VocabParallelEmbedding(nn.Module):
    """
    Embedding parallelized along the vocabulary dimension.
    Each rank holds vocab_size // tp_size rows.
    Forward: mask tokens not in range, lookup, all-reduce to sum partial embeddings.
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
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global_vocab_size(
            num_embeddings, rank, tp_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, device=_get_device())
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_group and self.tp_group.size() > 1:
            input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
            masked_input = input_ids.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids

        output_parallel = torch.nn.functional.embedding(masked_input, self.weight)

        if self.tp_group and self.tp_group.size() > 1:
            output_parallel[input_mask, :] = 0.0
            output = reduce_from_tensor_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel

        return output


# -----------------------------------------------------------------------------
# Vocab Parallel LM Head (ColumnParallelLinear with gather_output=True)
# -----------------------------------------------------------------------------


class ParallelLMHead(nn.Module):
    """
    LM head parallelized along vocabulary (output) dimension.
    Weight [vocab_per_rank, hidden]. Output gathered for full logits.
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

        self.weight = nn.Parameter(
            torch.empty(self.vocab_size_per_partition, hidden_size, device=_get_device())
        )
        self.bias = nn.Parameter(
            torch.empty(self.vocab_size_per_partition, device=_get_device())
        ) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_size)
        # weight: (vocab_per_rank, hidden_size) -> logits: (..., vocab_per_rank)
        logits = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.tp_group and self.tp_group.size() > 1:
            logits = gather_from_tensor_model_parallel_region(logits, dim=-1, group=self.tp_group)
        return logits


# -----------------------------------------------------------------------------
# Vocab utilities for embedding / lm_head
# -----------------------------------------------------------------------------


def vocab_range_from_global_vocab_size(
    vocab_size: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Return (start, end) index for this rank's vocab partition."""
    per_rank = vocab_size // world_size
    start = rank * per_rank
    end = start + per_rank
    return start, end


# -----------------------------------------------------------------------------
# Weight loading helpers: split full weights for TP
# -----------------------------------------------------------------------------


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
