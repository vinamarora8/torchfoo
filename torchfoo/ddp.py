__all__ = [
    "setup_ddp",
    "cleanup_ddp",
    "make_module_ddp",
]

import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_ddp(
    rank: int,
    world_size: int,
    master_addr: str | None = None,
    master_port: str | int | None = None,
    force: bool = False,
):
    import os

    if (world_size > 1) or force:
        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", None)
        if master_addr is None:
            master_addr = "localhost"

        if master_port is None:
            master_port = os.environ.get("MASTER_PORT", None)
        if master_port is None:
            master_port = get_open_port()

        logging.info(f"DDP MASTER_ADDR {master_addr}, MASTER_PORT {master_port}")

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(rank),
        )
    else:
        logging.info(f"Skipping DDP setup")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_open_port() -> int:
    """Find an available port on the system.

    This function creates a temporary socket, binds it to port 0 to let the OS assign
    an available port, and returns that port number. The socket is automatically closed
    after getting the port.

    Returns:
        int: An available port number that can be used for network communication.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def make_module_ddp(m: torch.nn.Module):
    from . import distributed as tfoodist

    if tfoodist.is_distributed():
        ret = DistributedDataParallel(m, device_ids=[tfoodist.get_rank()])
        ret = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ret)
        return ret
    else:
        return m
