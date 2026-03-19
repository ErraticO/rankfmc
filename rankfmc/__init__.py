from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rankfmc")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .rankfm import RankFM
from . import evaluation

__all__ = [
    "__version__",
    "RankFM",
    "evaluation",
]
