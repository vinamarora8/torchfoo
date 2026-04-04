__all__ = [
    "count_parameters",
    "make_ddp",
]

import warnings

import torch
from torch.nn.parallel import DistributedDataParallel


def count_parameters(m: torch.nn.Module, only: str = "all") -> int:
    r"""Count parameters in a module.

    Each unique parameter is counted once even if it appears in multiple
    places (e.g. weight-tied models).

    Args:
        m: the module to inspect
        only: which parameters to count —
            ``"all"`` (default), ``"trainable"``, or ``"frozen"``

    Returns:
        The number of parameters matching *only*.
    """
    if only not in ("all", "trainable", "frozen"):
        raise ValueError(f"only must be 'all', 'trainable', or 'frozen', got '{only}'")
    seen = set()
    total = 0
    for p in m.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if only == "all" or (only == "trainable") == p.requires_grad:
            total += p.numel()
    return total


def make_ddp(m: torch.nn.Module):
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
    from . import distributed as tfoodist

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
