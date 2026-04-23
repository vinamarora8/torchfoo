__all__ = [
    "all_reduce_",
    "all_reduce_sum_",
    "all_equal",
    "all_concat_jagged",
    "all_concat",
]

import torch
from torch import Tensor
import torch.distributed as dist

from .distributed import get_world_size, get_rank


def all_reduce_(x: Tensor) -> Tensor:
    r"""Perform an in-place differentiable "all reduce" operation

    Args:
        x: a contiguous Tensor on this device

    Returns:
        The input tensor, modified in-place with the all-reduced mean.
    """
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")

    return _AllReduce.apply(x)


def all_reduce_sum_(x: Tensor) -> Tensor:
    r"""Perform an in-place differentiable "all reduce sum" operation

    Args:
        x: a contiguous Tensor on this device

    Returns:
        The input tensor, modified in-place with the all-reduced sum.
    """
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")

    return _AllReduceSum.apply(x)


def all_equal(x: Tensor) -> bool:
    r"""Check if x is the same on all GPUs

    Args:
        x: a contiguous Tensor on this device

    Returns:
        True if x is equal across all processes.
    """
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")

    nworld = get_world_size()
    if nworld == 1:
        return True

    tensor_list = [torch.zeros_like(x) for _ in range(nworld)]
    dist.all_gather(tensor_list, x)

    return all((val == x).all() for val in tensor_list)


def all_concat_jagged(x: Tensor) -> Tensor:
    r"""Concatenate a Tensor with varying dim 0 from multiple GPUs

    Not differentiable.

    Args:
        x: a Tensor on this device

    Returns:
        Concatenated tensor from all processes along dim 0.
    """
    nworld = get_world_size()
    if nworld == 1:
        return x

    dev = x.device

    # 1. gather sizes
    jagged_size = torch.tensor(x.size(0), device=dev)
    jagged_sizes = [torch.tensor(0, device=dev) for _ in range(nworld)]
    dist.all_gather(jagged_sizes, jagged_size)
    jagged_sizes: list[int] = [
        x.item() for x in jagged_sizes
    ]  # ty:ignore[invalid-assignment]

    # 2. Create empty tensor list
    tensor_list = [
        torch.empty((jagged_sizes[i], *x.shape[1:]), device=dev, dtype=x.dtype)
        for i in range(nworld)
    ]

    # 3. Gather the tensors
    dist.all_gather(tensor_list, x)
    return torch.concat(tensor_list)


def all_concat(x: Tensor) -> Tensor:
    r"""Concatenate a Tensor distributed on multiple GPUs. Backprop-able.

    Args:
        x: a Tensor on this device

    Returns:
        Concatenated tensor from all processes along dim 0.
    """
    return _AllConcat.apply(x)


# Inspired by:
# https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/distributed.py
class _AllConcat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if get_world_size() > 1:
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(_ctx, grads):
        nworld = get_world_size()
        if nworld > 1:
            rank = get_rank()
            s = (grads.shape[0] // nworld) * rank
            e = (grads.shape[0] // nworld) * (rank + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class _AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor):
        n = get_world_size()
        ctx.n = n
        if n > 1:
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        # gradient will be averaged over the world in loss.backward()
        # so, we scale it up here to counter that
        if ctx.n > 1:
            return grads * ctx.n
        return grads


class _AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(_ctx, x: Tensor):
        nworld = get_world_size()
        if nworld > 1:
            x.div_(nworld)
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(_ctx, grads):
        return grads
