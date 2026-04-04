__all__ = [
    "seed_everything",
]

import random

import torch


def seed_everything(seed: int, deterministic: bool = False):
    r"""Seed all random number generators for reproducibility.

    Seeds ``random``, ``torch``, and ``torch.cuda`` (all devices). Optionally
    seeds ``numpy`` if it is installed.

    Args:
        seed: the seed value to use
        deterministic: if True, sets ``torch.use_deterministic_algorithms(True)``
            and ``torch.backends.cudnn.benchmark = False``. This will ensure
            reproducible results at the cost of reduced performance.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
