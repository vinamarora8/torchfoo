__all__ = [
    "is_distributed",
    "get_world_size",
    "get_rank",
    "rank_zero_only",
    "setup_distributed",
    "cleanup_distributed",
    "parallelize",
]

import logging

import torch
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


class _DistributedState:
    """torchfoo's internal distributed setup state"""

    initialized: bool = False
    """True iff setup_distributed() ran and did not skip creating process group"""
    use_cuda: bool = False
    """True iff cuda is available, and has enough GPUs given the world_size"""


_distributed_state = _DistributedState()


def setup_distributed(
    rank: int,
    world_size: int,
    master_addr: str | None = None,
    master_port: str | int | None = None,
    backend: str = "auto",
    force: bool = False,
) -> bool:
    r"""Initialize a distributed process group.

    If world_size is 1 and force is False, setup is skipped.

    Args:
        rank: rank of the current process
        world_size: total number of processes
        master_addr: address of the master node. Falls back to MASTER_ADDR env var, then "localhost".
        master_port: port of the master node. Falls back to MASTER_PORT env var.
        backend: distributed backend (default: "auto"). "auto" selects "nccl" if
            CUDA is available and there are enough GPUs for the world_size,
            "gloo" otherwise.
        force: if True, initialize even when world_size is 1

    Returns:
        False if setup was skipped, True otherwise
    """
    import os

    if _distributed_state.initialized:
        raise RuntimeError("A distributed process group has already been initialized.")

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size

    if (world_size > 1) or force:
        if backend == "auto":
            backend = "nccl" if use_cuda else "gloo"

        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", None)
        if master_addr is None:
            master_addr = "localhost"

        if master_port is None:
            master_port = os.environ.get("MASTER_PORT", None)

        if master_port is None:
            raise ValueError(
                "Either master_port must be provided "
                "or MASTER_PORT must be set in env"
            )

        logging.info(f"DDP MASTER_ADDR {master_addr}, MASTER_PORT {master_port}")

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        init_kwargs = dict(backend=backend, rank=rank, world_size=world_size)
        if use_cuda:
            init_kwargs["device_id"] = rank
        dist.init_process_group(**init_kwargs)  # ty:ignore[invalid-argument-type]

        _distributed_state.initialized = True
    else:
        logging.info("Skipping distributed setup")

    _distributed_state.use_cuda = use_cuda

    return _distributed_state.initialized


def cleanup_distributed():
    r"""Destroy the distributed process group if one is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()
    _distributed_state.initialized = False
    _distributed_state.use_cuda = False


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


def parallelize(
    world_size: int | None = None,
    master_addr: str | None = None,
    master_port: str | int | None = None,
    backend: str = "auto",
    force: bool = False,
):
    r"""Decorator that parallelizes a function across multiple devices with distributed setup.

    Handles process spawning, ``setup_distributed``, and ``cleanup_distributed``
    automatically.

    Use ``get_rank()`` and ``get_world_size()`` inside the decorated function
    to query distributed state.


    Args:
        world_size: number of devices to use. Defaults to ``max(torch.cuda.device_count(), 1)``.
            If greater than the number of CUDA GPUs available, CPUs are used instead.
        master_addr: address of the master node. Falls back to MASTER_ADDR env var, then "localhost".
        master_port: port of the master node. Falls back to MASTER_PORT env var, then an open port.
        backend: distributed backend (default: "auto"). "auto" selects "nccl" if
            enough CUDA GPUs are available for ``world_size``, "gloo" otherwise.
        force: Force distributed even if detected world_size is 1 (default ``False``)

    Examples::

        import torchfoo as tfo

        @tfo.dist.parallelize()
        def train(cfg):
            rank = tfo.dist.get_rank()
            world_size = tfo.dist.get_world_size()
            ...

        if __name__ == "__main__":
            train(cfg)
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import torch.multiprocessing as mp

            if world_size is not None:
                wsize = world_size
                assert wsize > 0, "world_size must be >= 1"
            elif torch.cuda.is_available():
                wsize = torch.cuda.device_count()
            else:
                wsize = 1

            port = master_port if master_port is not None else _get_open_port()

            spawn_args = (
                wrapper,
                wsize,
                master_addr,
                port,
                backend,
                force,
                args,
                kwargs,
            )

            if wsize > 1 or force:
                mp.spawn(_parallelize_worker, args=spawn_args, nprocs=wsize, join=True)
            else:
                _parallelize_worker(0, *spawn_args)

        return wrapper

    return decorator


def _parallelize_worker(
    rank,
    wrapper,
    world_size,
    master_addr,
    master_port,
    backend,
    force,
    args,
    kwargs,
):
    # wrapper is the @parallelize wrapper; __wrapped__ is the original function
    # This is needed to get around the problem with pickle not being able to work with
    # local functions
    func = wrapper.__wrapped__
    setup_distributed(
        rank,
        world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        force=force,
    )
    try:
        if _distributed_state.use_cuda:
            torch.cuda.set_device(rank)
        func(*args, **kwargs)
    finally:
        cleanup_distributed()
