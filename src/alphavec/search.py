"""
Parameter search utilities for alphavec simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Collection, Mapping, Sequence
import concurrent.futures as cf

import numpy as np
import pandas as pd

from .sim import MarketData, SimConfig, SimulationResult, simulate


@dataclass(frozen=True)
class Grid2D:
    """
    A 2D parameter grid definition over two parameter axes.
    """

    param1_name: str
    param1_values: Sequence[Any]
    param2_name: str
    param2_values: Sequence[Any]
    name: str | None = None

    def label(self) -> str:
        return self.name or f"{self.param1_name} x {self.param2_name}"


@dataclass(frozen=True)
class GridSearchBest:
    """
    Best scoring simulation from `grid_search`.
    """

    grid_index: int
    grid_name: str
    params: dict[str, Any]
    objective_metric: str
    objective_value: float
    result: SimulationResult


@dataclass(frozen=True)
class GridSearchResults:
    """
    Consolidated results from `grid_search`.
    """

    table: pd.DataFrame
    param_grids: tuple[Grid2D, ...]
    objective_metric: str
    best: GridSearchBest | None

    def pivot(self, *, grid_index: int = 0) -> pd.DataFrame:
        s = self.param_grids[grid_index]
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
            for i in range(len(self.param_grids))
        ]


def _objective_value(metrics: pd.DataFrame, objective_metric: str) -> float:
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


def grid_search(
    *,
    generate_weights: Callable[[Mapping[str, Any]], pd.DataFrame | pd.Series],
    objective_metric: str = "Annualized Sharpe",
    base_params: Mapping[str, Any],
    param_grids: Collection[Grid2D],
    progress: bool = False,
    progress_desc: str | None = None,
    max_workers: int | None = None,
    market: MarketData,
    config: SimConfig | None = None,
    executor: cf.Executor | None = None,
) -> GridSearchResults:
    """
    Run one or more 2D parameter grid searches where the objective is a simulation metric.

    For each `(param1_value, param2_value)` combination in each `Grid2D`:

    - Merge `base_params` with the two overrides to form a single parameter mapping.
    - Call `generate_weights(merged_params)` to produce the target weights for the run.
    - Run `simulate()` with the generated weights and the provided market data.
    - Read `objective_metric` from the returned metrics table
      (`metrics.loc[objective_metric, "Value"]`).
    - Record the objective value into an output table suitable for pivoting/heatmaps.

    Runs are executed concurrently via an executor (a `ThreadPoolExecutor` is created by default).
    After all runs finish, the function selects the best result by **maximizing** the recorded
    objective value (ignoring NaN/inf), then re-runs `simulate()` for that best parameter pair to
    populate `GridSearchResults.best` with the best `SimulationResult`.

    Args:
        generate_weights: Callable that takes a parameter dictionary as input and generates a weights DataFrame/Series
            compatible with `simulate()`.
        objective_metric: Name of the metric (row label) to maximize from the `simulate()` metrics table.
        base_params: Base parameter mapping passed to `generate_weights` for every run.
        param_grids: One or more 2D grids. Each grid searches two named parameters over their value sequences.
        progress: If True, show a progress bar (requires `tqdm`).
        progress_desc: Optional custom progress bar description.
        max_workers: Maximum worker threads used when this function creates its own executor.
        market: Market data passed through to `simulate()`.
        config: Simulation config passed through to `simulate()`.
        executor: Optional `concurrent.futures.Executor`. If provided, it is used and not shut down
            by this function. If omitted, a `ThreadPoolExecutor` is created and cleaned up.

    Returns:
        A `GridSearchResults` with:

        - `table`: One row per parameter combination, with columns:
          `grid_index`, `grid_name`, `param1_name`, `param1_value`, `param2_name`, `param2_value`,
          `objective_metric`, and `objective_value`.
        - `param_grids`: The input grids (as a tuple) in the same order.
        - `objective_metric`: The resolved metric key used (stringified).
        - `best`: The best-scoring run (max objective), including its `returns` and `metrics`, or
          None if no finite objective values were produced.
    """

    grids = tuple(param_grids)
    if len(grids) == 0:
        raise ValueError("param_grids must be non-empty.")

    def _validate_set(s: Grid2D) -> None:
        if not isinstance(s.param1_name, str) or not s.param1_name:
            raise ValueError("Grid2D.param1_name must be a non-empty string.")
        if not isinstance(s.param2_name, str) or not s.param2_name:
            raise ValueError("Grid2D.param2_name must be a non-empty string.")
        if s.param1_name == s.param2_name:
            raise ValueError("Grid2D param names must be distinct.")
        if len(s.param1_values) == 0 or len(s.param2_values) == 0:
            raise ValueError("Grid2D values must be non-empty.")

    for s in grids:
        _validate_set(s)

    tasks: list[tuple[int, int, int, Any, Any, dict[str, Any]]] = []
    for set_index, s in enumerate(grids):
        for i1, v1 in enumerate(s.param1_values):
            for i2, v2 in enumerate(s.param2_values):
                merged = dict(base_params)
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
        pbar = tqdm(
            total=len(tasks),
            desc=progress_desc or f"grid_search ({objective_metric})",
            unit="run",
        )

    try:
        futures = [executor.submit(_run_one, task) for task in tasks]
        rows: list[tuple[int, int, int, Any, Any, float, str]] = []
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=False)

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
    df = df.sort_values(["grid_index", "_param1_pos", "_param2_pos"], kind="mergesort")
    df = df.drop(columns=["_param1_pos", "_param2_pos"])

    best: GridSearchBest | None = None
    if len(df.index) > 0:
        obj = pd.to_numeric(df["objective_value"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(obj)
        if bool(finite_mask.any()):
            best_pos = int(np.argmax(np.where(finite_mask, obj, -np.inf)))
            best_row = df.iloc[best_pos]
            best_grid_index = int(best_row["grid_index"])
            best_grid = grids[best_grid_index]

            best_params = dict(base_params)
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
