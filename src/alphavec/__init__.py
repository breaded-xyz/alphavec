"""
Public API for alphavec.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .sim import MarketData, SimConfig, SimulationResult, simulate
from .search import (
    GridSearchBest,
    GridSearchResults,
    Metrics,
    grid_search,
)
from .metrics import TEARSHEET_NOTES
from .tearsheet import tearsheet

try:
    __version__ = _pkg_version("alphavec")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "simulate",
    "tearsheet",
    "SimulationResult",
    "MarketData",
    "SimConfig",
    "TEARSHEET_NOTES",
    "grid_search",
    "GridSearchBest",
    "GridSearchResults",
    "Metrics",
    "__version__",
]
