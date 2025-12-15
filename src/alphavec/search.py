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

from .sim import simulate


@dataclass(frozen=True)
class ParamGrid2D:
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
class ParamGridBest:
    """
    Best scoring simulation from `grid_search_and_simulate`.
    """

    grid_index: int
    grid_name: str
    params: dict[str, Any]
    objective_metric: str
    objective_value: float
    returns: pd.Series
    metrics: pd.DataFrame


@dataclass(frozen=True)
class ParamGridResults:
    """
    Consolidated results from `grid_search_and_simulate`.
    """

    table: pd.DataFrame
    param_grids: tuple[ParamGrid2D, ...]
    objective_metric: str
    best: ParamGridBest | None

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
        colorscale: str = "RdBu",
        zmid: float | None = 0.0,
    ):
        import plotly.graph_objects as go

        s = self.param_grids[grid_index]
        z = self.pivot(grid_index=grid_index)
        fig = go.Figure(
            data=go.Heatmap(
                z=z.to_numpy(),
                x=[str(x) for x in z.columns],
                y=[str(y) for y in z.index],
                colorscale=colorscale,
                zmid=zmid,
                colorbar=dict(title=self.objective_metric),
            )
        )
        fig.update_layout(
            title=title or f"{self.objective_metric} heatmap ({s.label()})",
            xaxis_title=s.param2_name,
            yaxis_title=s.param1_name,
            template="plotly_white",
        )
        return fig

    def heatmap_figures(
        self,
        *,
        colorscale: str = "RdBu",
        zmid: float | None = 0.0,
    ) -> list:
        return [
            self.heatmap_figure(grid_index=i, colorscale=colorscale, zmid=zmid)
            for i in range(len(self.param_grids))
        ]


def _objective_value(metrics: pd.DataFrame, objective_metric: str) -> float:
    if objective_metric not in metrics.index:
        available = [str(x) for x in metrics.index]
        preview = available[:25]
        suffix = "" if len(available) <= len(preview) else f" … (+{len(available) - len(preview)} more)"
        raise KeyError(
            f"objective_metric {objective_metric!r} not found in metrics.index. "
            f"Available metrics: {preview}{suffix}"
        )
    return float(metrics.loc[objective_metric, "Value"])


def grid_search_and_simulate(
    *,
    generate_weights: Callable[[Mapping[str, Any]], pd.DataFrame | pd.Series],
    objective_metric: str = "Annualized Sharpe",
    base_params: Mapping[str, Any],
    param_grids: Collection[ParamGrid2D],
    progress: bool = False,
    progress_desc: str | None = None,
    max_workers: int | None = None,
    close_prices: pd.DataFrame | pd.Series,
    order_prices: pd.DataFrame | pd.Series,
    funding_rates: pd.DataFrame | pd.Series | None = None,
    benchmark_asset: str | None = None,
    order_notional_min: float = 0.0,
    fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
    init_cash: float = 1000.0,
    freq_rule: str = "1D",
    trading_days_year: int = 365,
    risk_free_rate: float = 0.0,
    executor: cf.Executor | None = None,
) -> ParamGridResults:
    """
    Run a 2D parameter grid search where the objective is a simulation metric.

    For each combination in each `ParamGrid2D`:
      - Merge `base_params` with the (param1, param2) override.
      - Generate weights via `generate_weights(merged_params)`.
      - Run `simulate()` with the generated weights and the provided market data.
      - Extract `objective_metric` from the returned metrics table.
      - Store the objective value for visualization (e.g. heatmaps).
    """

    grids = tuple(param_grids)
    if len(grids) == 0:
        raise ValueError("param_grids must be non-empty.")

    def _validate_set(s: ParamGrid2D) -> None:
        if not isinstance(s.param1_name, str) or not s.param1_name:
            raise ValueError("ParamGrid2D.param1_name must be a non-empty string.")
        if not isinstance(s.param2_name, str) or not s.param2_name:
            raise ValueError("ParamGrid2D.param2_name must be a non-empty string.")
        if s.param1_name == s.param2_name:
            raise ValueError("ParamGrid2D param names must be distinct.")
        if len(s.param1_values) == 0 or len(s.param2_values) == 0:
            raise ValueError("ParamGrid2D values must be non-empty.")

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
        task: tuple[int, int, int, Any, Any, dict[str, Any]]
    ) -> tuple[int, int, int, Any, Any, float, str]:
        grid_index, i1, i2, v1, v2, params = task
        weights = generate_weights(params)
        _, metrics = simulate(
            weights=weights,
            close_prices=close_prices,
            order_prices=order_prices,
            funding_rates=funding_rates,
            benchmark_asset=benchmark_asset,
            order_notional_min=order_notional_min,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            init_cash=init_cash,
            freq_rule=freq_rule,
            trading_days_year=trading_days_year,
            risk_free_rate=risk_free_rate,
        )
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
            desc=progress_desc or f"grid_search_and_simulate ({objective_metric})",
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

    best: ParamGridBest | None = None
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
            best_returns, best_metrics = simulate(
                weights=best_weights,
                close_prices=close_prices,
                order_prices=order_prices,
                funding_rates=funding_rates,
                benchmark_asset=benchmark_asset,
                order_notional_min=order_notional_min,
                fee_pct=fee_pct,
                slippage_pct=slippage_pct,
                init_cash=init_cash,
                freq_rule=freq_rule,
                trading_days_year=trading_days_year,
                risk_free_rate=risk_free_rate,
            )

            best = ParamGridBest(
                grid_index=best_grid_index,
                grid_name=str(best_row["grid_name"]),
                params=best_params,
                objective_metric=resolved_key,
                objective_value=float(best_row["objective_value"]),
                returns=best_returns,
                metrics=best_metrics,
            )

    return ParamGridResults(
        table=df,
        param_grids=grids,
        objective_metric=resolved_key,
        best=best,
    )
