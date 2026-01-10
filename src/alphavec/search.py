"""
Parameter search utilities for alphavec simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
from collections.abc import Collection, Mapping, Sequence
import concurrent.futures as cf
import gc
import warnings

import numpy as np
import pandas as pd

from .sim import MarketData, SimConfig, SimulationResult, simulate


class Metrics(str, Enum):
    """Standard metric names for grid search optimization."""
    SHARPE = "Annualized Sharpe"
    RETURN = "Annualized return %"
    SORTINO = "Annualized Sortino"
    CALMAR = "Calmar ratio"
    INFO_RATIO = "Information ratio"
    MAX_DRAWDOWN = "Max drawdown (equity) %"
    VOLATILITY = "Annualized volatility %"
    ALPHA = "Annualized alpha %"
    BETA = "Beta"
    WEIGHT_IC_MEAN = "Weight IC mean (next)"
    WEIGHT_IC_SHARPE = "Weight IC Sharpe (next)"


@dataclass(frozen=True)
class _Grid:
    """Internal grid representation. Users pass dicts like {"param": [values]} instead."""
    params: tuple[str, ...]  # Parameter name(s)
    values: tuple[Sequence[Any], ...]  # Value sequences for each param
    name: str

    def is_1d(self) -> bool:
        return len(self.params) == 1

    def is_2d(self) -> bool:
        return len(self.params) == 2

    def label(self) -> str:
        if self.is_1d():
            return self.name or self.params[0]
        else:
            return self.name or f"{self.params[0]} x {self.params[1]}"

    @property
    def param_name(self) -> str:
        """For 1D grids only."""
        assert self.is_1d(), "param_name only valid for 1D grids"
        return self.params[0]

    @property
    def param_values(self) -> Sequence[Any]:
        """For 1D grids only."""
        assert self.is_1d(), "param_values only valid for 1D grids"
        return self.values[0]

    @property
    def param1_name(self) -> str:
        """For 2D grids only."""
        assert self.is_2d(), "param1_name only valid for 2D grids"
        return self.params[0]

    @property
    def param1_values(self) -> Sequence[Any]:
        """For 2D grids only."""
        assert self.is_2d(), "param1_values only valid for 2D grids"
        return self.values[0]

    @property
    def param2_name(self) -> str:
        """For 2D grids only."""
        assert self.is_2d(), "param2_name only valid for 2D grids"
        return self.params[1]

    @property
    def param2_values(self) -> Sequence[Any]:
        """For 2D grids only."""
        assert self.is_2d(), "param2_values only valid for 2D grids"
        return self.values[1]


@dataclass(frozen=True)
class GridSearchBest:
    """
    Best scoring simulation from `grid_search`.
    """

    grid_index: int
    grid_name: str
    params: dict[str, Any]
    objective_metric: str | Metrics
    objective_value: float
    result: SimulationResult


@dataclass(frozen=True)
class GridSearchResults:
    """
    Consolidated results from `grid_search`.

    Implements the `MetricsAccessor` protocol for unified metric extraction.
    Metric methods delegate to the best result's SimulationResult.

    Attributes:
        table: DataFrame with results for every parameter combination tested.
        param_grids: Internal grid representations.
        objective_metric: The metric being optimized.
        best: The best-scoring result with full SimulationResult, or None if
            all runs produced NaN/inf objective values.

    Metric Access (delegates to best.result):
        >>> results.metric_value(MetricKey.ANNUALIZED_SHARPE)
        >>> results.available_metrics("Performance")
        >>> results.metrics_dict("Risk")

        Raises ValueError if best is None (no valid results).

    Analysis Methods:
        >>> results.table          # Full results DataFrame
        >>> results.best.params    # Best parameters
        >>> results.top(5)         # Top 5 parameter combinations
        >>> results.summary()      # Summary statistics by grid
        >>> results.plot()         # Line for 1D, heatmap for 2D
    """

    table: pd.DataFrame
    param_grids: tuple[_Grid, ...]
    objective_metric: str | Metrics
    best: GridSearchBest | None

    def pivot(self, *, grid_index: int = 0) -> pd.DataFrame:
        s = self.param_grids[grid_index]
        if s.is_1d():
            raise ValueError(
                f"pivot() only works for 2D grids. Grid {grid_index} is 1D. "
                f"For 1D results, use .for_grid({grid_index}) or .plot({grid_index})."
            )
        df = self.table[self.table["grid_index"] == grid_index]
        pivot = df.pivot(index="param1_value", columns="param2_value", values="objective_value")
        pivot = pivot.reindex(index=list(s.param1_values), columns=list(s.param2_values))
        pivot.index.name = s.param1_name
        pivot.columns.name = s.param2_name
        return pivot

    def heatmap_figure(
        self,
        *,
        grid_index: int = 0,
        title: str | None = None,
        cmap: str = "RdBu_r",
        center: float | None = 0.0,
        annot: bool = False,
        fmt: str = ".3g",
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        s = self.param_grids[grid_index]
        if s.is_1d():
            raise ValueError(
                f"heatmap_figure() only works for 2D grids. Grid {grid_index} is 1D. "
                f"Use .plot({grid_index}) for 1D visualization."
            )
        z = self.pivot(grid_index=grid_index)

        z_num = z.apply(pd.to_numeric, errors="coerce")
        nrows, ncols = z_num.shape
        fig_w = max(6.0, 0.6 * ncols + 2.0)
        fig_h = max(4.0, 0.55 * nrows + 2.0)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            z_num,
            ax=ax,
            cmap=cmap,
            center=center,
            annot=annot,
            fmt=fmt,
            cbar_kws={"label": self.objective_metric},
        )
        ax.set_title(title or f"{self.objective_metric} heatmap ({s.label()})")
        ax.set_xlabel(s.param2_name)
        ax.set_ylabel(s.param1_name)
        fig.tight_layout()
        return fig

    def heatmap_figures(
        self,
        *,
        cmap: str = "RdBu_r",
        center: float | None = 0.0,
        annot: bool = False,
        fmt: str = ".3g",
    ) -> list:
        return [
            self.heatmap_figure(grid_index=i, cmap=cmap, center=center, annot=annot, fmt=fmt)
            for i, grid in enumerate(self.param_grids)
            if grid.is_2d()
        ]

    def top(self, n: int = 10) -> pd.DataFrame:
        """Return top N parameter combinations by objective value."""
        return self.table.nlargest(n, "objective_value")

    def for_grid(self, index: int = 0) -> pd.DataFrame:
        """Filter results for specific grid."""
        return self.table[self.table["grid_index"] == index]

    def summary(self) -> pd.DataFrame:
        """Summary statistics by grid."""
        return (
            self.table.groupby(["grid_index", "grid_name"])["objective_value"]
            .agg(["count", "mean", "std", "min", "max"])
        )

    def plot(
        self,
        *,
        grid_index: int = 0,
        title: str | None = None,
        **kwargs
    ):
        """
        Smart plotting - line plot for 1D grids, heatmap for 2D grids.

        For 2D grids, passes kwargs to heatmap_figure().
        For 1D grids, creates a line plot.
        """
        import matplotlib.pyplot as plt

        grid = self.param_grids[grid_index]

        if grid.is_1d():
            # Line plot for 1D
            df = self.for_grid(grid_index).sort_values("param_value")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["param_value"], df["objective_value"], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(grid.param_name, fontsize=12)
            ax.set_ylabel(self.objective_metric, fontsize=12)
            ax.set_title(title or f"{self.objective_metric} vs {grid.param_name}", fontsize=14)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            return fig
        else:  # 2D grid
            return self.heatmap_figure(grid_index=grid_index, title=title, **kwargs)

    def metric_value(self, metric: str, *, default: object = None) -> object:
        """
        Get metric value from the best result.

        Delegates to `self.best.result.metric_value()`.

        Args:
            metric: The metric key (use MetricKey constants).
            default: Value to return if metric not found or best is None.

        Returns:
            The metric value from the best result, or default if unavailable.

        Raises:
            ValueError: If no valid results exist (best is None) and no default provided.
        """
        if self.best is None:
            if default is not None:
                return default
            raise ValueError(
                "Cannot access metrics: no valid results exist (best is None). "
                "This occurs when all grid search runs produced NaN/inf objective values."
            )
        return self.best.result.metric_value(metric, default=default)

    def available_metrics(self, category: str | None = None) -> list[str]:
        """
        Return available metric keys from the best result.

        Delegates to `self.best.result.available_metrics()`.

        Args:
            category: Optional category filter.

        Returns:
            List of metric keys available in the best result.

        Raises:
            ValueError: If no valid results exist (best is None).
        """
        if self.best is None:
            raise ValueError(
                "Cannot access metrics: no valid results exist (best is None). "
                "This occurs when all grid search runs produced NaN/inf objective values."
            )
        return self.best.result.available_metrics(category)

    def metrics_dict(self, category: str | None = None) -> dict[str, object]:
        """
        Return metrics dictionary from the best result.

        Delegates to `self.best.result.metrics_dict()`.

        Args:
            category: Optional category filter.

        Returns:
            Dictionary of metrics from the best result.

        Raises:
            ValueError: If no valid results exist (best is None).
        """
        if self.best is None:
            raise ValueError(
                "Cannot access metrics: no valid results exist (best is None). "
                "This occurs when all grid search runs produced NaN/inf objective values."
            )
        return self.best.result.metrics_dict(category)


def _objective_value(metrics: pd.DataFrame, objective_metric: str | Metrics) -> float:
    if objective_metric not in metrics.index:
        available = [str(x) for x in metrics.index]
        preview = available[:25]
        suffix = (
            "" if len(available) <= len(preview) else f" … (+{len(available) - len(preview)} more)"
        )
        raise KeyError(
            f"objective_metric {objective_metric!r} not found in metrics.index. "
            f"Available metrics: {preview}{suffix}"
        )
    return float(metrics.loc[objective_metric, "Value"])


def _to_grid(spec: Mapping[str, Sequence[Any]], index: int) -> _Grid:
    """Convert dict to internal _Grid representation."""
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"Grid must be a dict like {{'param': [values]}}. Got {type(spec).__name__}. "
            f"The Grid1D and Grid2D classes have been removed - use dicts instead."
        )

    items = list(spec.items())
    if len(items) == 0:
        raise ValueError("Grid dict cannot be empty. Provide at least one parameter.")
    elif len(items) == 1:
        k, v = items[0]
        return _Grid(params=(k,), values=(v,), name=f"grid_{index}")
    elif len(items) == 2:
        (k1, v1), (k2, v2) = items
        return _Grid(params=(k1, k2), values=(v1, v2), name=f"grid_{index}")
    else:
        raise ValueError(
            f"Grid dict must have 1 or 2 parameters, got {len(items)}. "
            f"For more parameters, use multiple grids."
        )


def grid_search(
    *,
    generate_weights: Callable[[Mapping[str, Any]], pd.DataFrame | pd.Series],
    objective_metric: str | Metrics = Metrics.SHARPE,
    maximize: bool = True,
    base_params: Mapping[str, Any] | None = None,
    param_grids: Collection[Mapping[str, Sequence[Any]]],
    progress: bool | str = False,
    max_workers: int | None = None,
    market: MarketData,
    config: SimConfig | None = None,
    executor: cf.Executor | None = None,
) -> GridSearchResults:
    """
    Run one or more parameter grid searches where the objective is a simulation metric.

    For each parameter combination in each grid:

    - Merge `base_params` with the grid parameters to form a single parameter mapping.
    - Call `generate_weights(merged_params)` to produce the target weights for the run.
    - Run `simulate()` with the generated weights and the provided market data.
    - Read `objective_metric` from the returned metrics table
      (`metrics.loc[objective_metric, "Value"]`).
    - Record the objective value into an output table suitable for pivoting/heatmaps.

    Runs are executed concurrently via an executor (a `ThreadPoolExecutor` is created by default).
    After all runs finish, the function selects the best result by optimizing (maximizing or
    minimizing based on the `maximize` parameter) the recorded objective value (ignoring NaN/inf),
    then re-runs `simulate()` for that best parameter pair to populate `GridSearchResults.best`
    with the best `SimulationResult`.

    Args:
        generate_weights: Callable that takes a parameter dictionary as input and generates a weights DataFrame/Series
            compatible with `simulate()`.
        objective_metric: Name of the metric (row label) to optimize from the `simulate()` metrics table.
            Can be a Metrics enum value or a custom string.
        maximize: If True, maximize the objective metric; if False, minimize it. Defaults to True.
        base_params: Base parameter mapping passed to `generate_weights` for every run. Optional, defaults to empty dict.
        param_grids: One or more parameter grids as dicts.
            Each grid is a dict mapping parameter names to value sequences.
            1D grids: {"param": [value1, value2, ...]}
            2D grids: {"param1": [values], "param2": [values]}
        progress: If True, show default progress bar; if string, use as custom description; if False, no progress bar (requires `tqdm`).
        max_workers: Maximum worker threads used when this function creates its own executor.
        market: Market data passed through to `simulate()`.
        config: Simulation config passed through to `simulate()`.
        executor: Optional `concurrent.futures.Executor`. If provided, it is used and not shut down
            by this function. If omitted, a `ThreadPoolExecutor` is created and cleaned up.

    Returns:
        A `GridSearchResults` with:

        - `table`: One row per parameter combination, with columns varying by grid dimensionality.
          For 2D grids: `grid_index`, `grid_name`, `param1_name`, `param1_value`, `param2_name`, `param2_value`,
          `objective_metric`, and `objective_value`.
          For 1D grids: `grid_index`, `grid_name`, `param_name`, `param_value`,
          `objective_metric`, and `objective_value`.
        - `param_grids`: The internal grid representations (as a tuple) in the same order.
        - `objective_metric`: The resolved metric key used (stringified).
        - `best`: The best-scoring run (max objective), including its `returns` and `metrics`, or
          None if no finite objective values were produced.
    """

    # Convert Metrics enum to string if needed
    if isinstance(objective_metric, Metrics):
        objective_metric = objective_metric.value
    else:
        objective_metric = str(objective_metric)

    # Convert dict grids to internal representation
    grids: tuple[_Grid, ...] = tuple(_to_grid(g, i) for i, g in enumerate(param_grids))
    if len(grids) == 0:
        raise ValueError("param_grids must be non-empty.")

    def _validate_set(s: _Grid) -> None:
        for param_name, param_values in zip(s.params, s.values):
            if not isinstance(param_name, str) or not param_name:
                raise ValueError(f"Parameter name must be a non-empty string, got {param_name!r}.")
            if len(param_values) == 0:
                raise ValueError(
                    f"Parameter '{param_name}' values must be non-empty. "
                    f"Provide at least one value to search."
                )

        # Check for duplicate param names in same grid
        if len(s.params) != len(set(s.params)):
            raise ValueError(
                f"Grid param names must be distinct, got {s.params}. "
                f"Use different parameter names."
            )

    for s in grids:
        _validate_set(s)

    # Warn about parameter conflicts with base_params
    if base_params:
        for s in grids:
            for param_name in s.params:
                if param_name in base_params:
                    warnings.warn(
                        f"Grid parameter '{param_name}' overrides base_params. "
                        f"Consider removing from base_params to avoid confusion.",
                        UserWarning,
                        stacklevel=2,
                    )

    tasks: list[tuple[int, int, int, Any, Any, dict[str, Any]]] = []
    for set_index, s in enumerate(grids):
        if s.is_1d():
            for i1, v1 in enumerate(s.param_values):
                merged: dict[str, Any] = dict(base_params) if base_params else {}
                merged[s.param_name] = v1
                tasks.append((set_index, i1, -1, v1, None, merged))
        else:  # 2D grid
            for i1, v1 in enumerate(s.param1_values):
                for i2, v2 in enumerate(s.param2_values):
                    merged: dict[str, Any] = dict(base_params) if base_params else {}
                    merged[s.param1_name] = v1
                    merged[s.param2_name] = v2
                    tasks.append((set_index, i1, i2, v1, v2, merged))

    def _run_one(
        task: tuple[int, int, int, Any, Any, dict[str, Any]],
    ) -> tuple[int, int, int, Any, Any, float, str]:
        grid_index, i1, i2, v1, v2, params = task
        weights = generate_weights(params)
        metrics = simulate(weights=weights, market=market, config=config).metrics
        val = _objective_value(metrics, objective_metric)
        return grid_index, i1, i2, v1, v2, val, str(objective_metric)

    owns_executor = executor is None
    if executor is None:
        executor = cf.ThreadPoolExecutor(max_workers=max_workers)

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception as e:
            raise ImportError(
                "tqdm is required for progress tracking; install it with `pip install tqdm`."
            ) from e

        if isinstance(progress, str):
            desc = progress
        else:
            desc = f"grid_search ({objective_metric})"

        pbar = tqdm(total=len(tasks), desc=desc, unit="run")

    # Periodic garbage collection for large grids to manage memory
    gc_interval = 50  # Collect every N completions
    n_tasks = len(tasks)

    try:
        futures = [executor.submit(_run_one, task) for task in tasks]
        rows: list[tuple[int, int, int, Any, Any, float, str]] = []
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rows.append(fut.result())
            if pbar is not None:
                pbar.update(1)
            # Periodic GC for large searches to prevent memory buildup
            if n_tasks >= gc_interval and i % gc_interval == 0:
                gc.collect()
    finally:
        if pbar is not None:
            pbar.close()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=False)
        # Final cleanup after all simulations complete
        if n_tasks >= gc_interval:
            gc.collect()

    keys = {key for *_rest, key in rows}
    if len(keys) != 1:
        raise RuntimeError(
            "Objective metric resolution was inconsistent across runs: "
            f"{sorted(keys)!r}. This usually means metrics tables differed."
        )
    resolved_key = next(iter(keys))

    out_rows: list[dict[str, Any]] = []
    for grid_index, i1, i2, v1, v2, val, _key in rows:
        s = grids[grid_index]
        if s.is_1d():
            out_rows.append(
                {
                    "grid_index": grid_index,
                    "grid_name": s.label(),
                    "_param1_pos": i1,
                    "param_name": s.param_name,
                    "param_value": v1,
                    "objective_metric": resolved_key,
                    "objective_value": val,
                }
            )
        else:  # 2D grid
            out_rows.append(
                {
                    "grid_index": grid_index,
                    "grid_name": s.label(),
                    "_param1_pos": i1,
                    "_param2_pos": i2,
                    "param1_name": s.param1_name,
                    "param1_value": v1,
                    "param2_name": s.param2_name,
                    "param2_value": v2,
                    "objective_metric": resolved_key,
                    "objective_value": val,
                }
            )

    df = pd.DataFrame(out_rows)
    # Sort by grid_index and param positions
    sort_cols = ["grid_index", "_param1_pos"]
    if "_param2_pos" in df.columns:
        sort_cols.append("_param2_pos")
    df = df.sort_values(sort_cols, kind="mergesort")
    # Drop position columns
    drop_cols = ["_param1_pos"]
    if "_param2_pos" in df.columns:
        drop_cols.append("_param2_pos")
    df = df.drop(columns=drop_cols)

    best: GridSearchBest | None = None
    if len(df.index) > 0:
        obj = pd.to_numeric(df["objective_value"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(obj)
        if bool(finite_mask.any()):
            if maximize:
                best_pos = int(np.argmax(np.where(finite_mask, obj, -np.inf)))
            else:
                best_pos = int(np.argmin(np.where(finite_mask, obj, np.inf)))
            best_row = df.iloc[best_pos]
            best_grid_index = int(best_row["grid_index"])
            best_grid = grids[best_grid_index]

            best_params: dict[str, Any] = dict(base_params) if base_params else {}
            if best_grid.is_1d():
                best_params[best_grid.param_name] = best_row["param_value"]
            else:  # 2D grid
                best_params[best_grid.param1_name] = best_row["param1_value"]
                best_params[best_grid.param2_name] = best_row["param2_value"]

            best_weights = generate_weights(best_params)
            best_result = simulate(weights=best_weights, market=market, config=config)

            best = GridSearchBest(
                grid_index=best_grid_index,
                grid_name=str(best_row["grid_name"]),
                params=best_params,
                objective_metric=resolved_key,
                objective_value=float(best_row["objective_value"]),
                result=best_result,
            )

    return GridSearchResults(
        table=df,
        param_grids=grids,
        objective_metric=resolved_key,
        best=best,
    )
