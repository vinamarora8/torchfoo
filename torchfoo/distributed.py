__all__ = [
    "is_distributed",
    "get_world_size",
    "get_rank",
    "all_reduce_",
    "all_reduce_sum_",
    "all_equal",
    "all_concat_jagged",
    "all_concat",
    "rank_zero_only",
]

import torch
from torch import Tensor
import torch.distributed as dist


def is_distributed() -> bool:
    r"""Check if distributed mode is initialized

    Returns:
        True if distributed mode is initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    r"""Get number of distributed processes

    Returns:
        Number of distributed processes, or 1 if not in distributed mode.
    """
    if not is_distributed():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    r"""Get rank of the current distributed process

    Returns:
        Rank of the current process, or 0 if not in distributed mode.
    """
    if not is_distributed():
        return 0

    return dist.get_rank()


def all_reduce_(x: Tensor) -> Tensor:
    r"""Perform an in-place backprop-able "all reduce" operation

    Args:
        x: a contiguous Tensor on this device

    Returns:
        The input tensor, modified in-place with the all-reduced mean.
    """
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")

    return _AllReduce.apply(x)


def all_reduce_sum_(x: Tensor) -> Tensor:
    r"""Perform an in-place backprop-able "all reduce sum" operation

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

    output = [None for _ in range(nworld)]
    dist.all_gather_object(output, x)
    return torch.concat(output)  # type: ignore


def all_concat(x: Tensor) -> Tensor:
    r"""Concatenate a Tensor distributed on multiple GPUs. Backprop-able.

    Args:
        x: a Tensor on this device

    Returns:
        Concatenated tensor from all processes along dim 0.
    """
    return _AllConcat.apply(x)


def rank_zero_only(func):
    r"""Decorator that ensures a function only executes on rank 0.

    On all other ranks, the function returns None.

    Args:
        func: a function to wrap

    Returns:
        Wrapped function that is a no-op on non-zero ranks.
    """
    rank = get_rank()

    def wrapper(*args, **kwargs):
        if rank != 0:
            return
        return func(*args, **kwargs)

    return wrapper


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
    def forward(_ctx, x: Tensor):
        if get_world_size() > 1:
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(_ctx, grads):
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
