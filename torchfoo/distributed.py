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
    "setup_distributed",
    "cleanup_distributed",
    "distributed",
]

import logging

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


def setup_distributed(
    rank: int,
    world_size: int,
    master_addr: str | None = None,
    master_port: str | int | None = None,
    backend: str = "nccl",
    force: bool = False,
):
    r"""Initialize a distributed process group.

    If world_size is 1 and force is False, setup is skipped.

    Args:
        rank: rank of the current process
        world_size: total number of processes
        master_addr: address of the master node. Falls back to MASTER_ADDR env var, then "localhost".
        master_port: port of the master node. Falls back to MASTER_PORT env var, then an open port.
        backend: distributed backend to use (default: "nccl")
        force: if True, initialize even when world_size is 1
    """
    import os

    if (world_size > 1) or force:
        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", None)
        if master_addr is None:
            master_addr = "localhost"

        if master_port is None:
            master_port = os.environ.get("MASTER_PORT", None)
        if master_port is None:
            master_port = _get_open_port()

        logging.info(f"DDP MASTER_ADDR {master_addr}, MASTER_PORT {master_port}")

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=torch.device(rank),
        )
    else:
        logging.info("Skipping distributed setup")


def cleanup_distributed():
    r"""Destroy the distributed process group if one is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _get_open_port() -> int:
    r"""Find an available port on the system.

    Returns:
        An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def distributed(num_gpus: int | None = None):
    r"""Decorator that launches a function across multiple GPUs.

    Rank 0 runs in the main process; ranks 1..N-1 are spawned.
    The decorated function must accept ``rank`` and ``world_size`` as its
    first two arguments.

    Args:
        num_gpus: number of GPUs to use. Defaults to ``torch.cuda.device_count()``.

    Examples::

        @distributed()
        def train(rank, world_size, cfg):
            setup_distributed(rank, world_size)
            ...

        train(cfg)

    If you use Hydra, then ``@distributed`` must be the inner decorator::

        @hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
        @distributed()
        def train(rank, world_size, cfg: DictConfig):
            setup_distributed(rank, world_size)
            ...

        train()  # hydra passes cfg -> distributed prepends rank, world_size
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import torch.multiprocessing as mp

            ngpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
            ngpus = max(ngpus, 1)

            if ngpus > 1:
                mp.set_start_method("spawn", force=True)
                for rank in range(1, ngpus):
                    mp.Process(
                        target=func, args=(rank, ngpus, *args), kwargs=kwargs
                    ).start()

            func(0, ngpus, *args, **kwargs)

        return wrapper

    return decorator


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
