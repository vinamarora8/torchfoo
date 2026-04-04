__all__ = [
    "make_module_ddp",
]

import torch
from torch.nn.parallel import DistributedDataParallel

from . import distributed as tfoodist


def make_module_ddp(m: torch.nn.Module):
    r"""Wrap a module with DDP and convert BatchNorm to SyncBatchNorm.

    Returns the module unchanged if not in distributed mode.

    Args:
        m: the module to wrap

    Returns:
        The DDP-wrapped module, or the original module if not distributed.
    """
    if tfoodist.is_distributed():
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return DistributedDataParallel(m, device_ids=[tfoodist.get_rank()])
    else:
        return m
