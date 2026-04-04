__all__ = [
    "distributed",
    "dist",
    "module",
    "seed_everything",
    "current_device",
]

from . import distributed
from . import distributed as dist
from . import module
from .utils import seed_everything, current_device

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchfoo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover
