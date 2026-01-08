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
from .metrics import MetricKey, MetricsAccessor, MetricsArtifacts, TEARSHEET_NOTES, metrics_artifacts
from .tearsheet import tearsheet
from .walk_forward import (
    FoldConfig,
    FoldResult,
    FoldAggregation,
    WalkForwardResult,
    walk_forward,
    DEFAULT_AGGREGATE_METRICS,
)

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
    "MetricKey",
    "MetricsAccessor",
    "TEARSHEET_NOTES",
    "MetricsArtifacts",
    "metrics_artifacts",
    "grid_search",
    "GridSearchBest",
    "GridSearchResults",
    "Metrics",
    "walk_forward",
    "FoldConfig",
    "FoldResult",
    "FoldAggregation",
    "WalkForwardResult",
    "DEFAULT_AGGREGATE_METRICS",
    "__version__",
]
