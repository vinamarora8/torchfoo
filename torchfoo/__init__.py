__all__ = [
    "distributed",
    "dist",
    "ddp",
    "seed_everything",
]

from . import distributed
from . import distributed as dist
from . import ddp
from .utils import seed_everything

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchfoo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover
