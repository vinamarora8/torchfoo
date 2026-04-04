__all__ = [
    "make_module_ddp",
]

import warnings

import torch
from torch.nn.parallel import DistributedDataParallel

from . import distributed as tfoodist


def make_module_ddp(m: torch.nn.Module):
    r"""Wrap a module with DDP and convert BatchNorm to SyncBatchNorm.

    Returns the module unchanged if not in distributed mode.

    .. warning::
        SyncBatchNorm only works with GPU modules. On CPU, BatchNorm layers
        will not be converted and a warning is emitted.

    Args:
        m: the module to wrap

    Returns:
        The DDP-wrapped module, or the original module if not distributed.
    """
    if tfoodist.is_distributed():
        use_cuda = next(m.parameters()).is_cuda
        if use_cuda:
            m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        else:
            has_batchnorm = any(
                isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                for mod in m.modules()
            )
            if has_batchnorm:
                warnings.warn(
                    "SyncBatchNorm only works with GPU modules. "
                    "BatchNorm layers will not be converted.",
                    stacklevel=2,
                )
        device_ids = [tfoodist.get_rank()] if use_cuda else None
        return DistributedDataParallel(m, device_ids=device_ids)
    else:
        return m
