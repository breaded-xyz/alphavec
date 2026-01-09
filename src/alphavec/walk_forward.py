"""
Walk-forward (fold-based) analysis for alphavec simulations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import MetricKey
from .sim import MarketData, SimConfig, SimulationResult, simulate, _to_frame


@dataclass(frozen=True)
class FoldConfig:
    """
    Configuration for walk-forward fold-based simulation.

    Attributes:
        fold_period: Pandas frequency string defining fold length (e.g., "3ME", "6ME", "1YE").
            Use month-end ("ME") or year-end ("YE") for calendar alignment.
        min_periods: Minimum number of periods required per fold. Folds with fewer
            periods are discarded. Defaults to 3.
        align_start: If True, align fold boundaries to calendar periods (e.g., month-start
            for "1ME"). If False, use raw time slicing from simulation start. Defaults to True.
    """

    fold_period: str
    min_periods: int = 3
    align_start: bool = True


@dataclass(frozen=True)
class FoldResult:
    """
    Result for a single fold in walk-forward analysis.

    Attributes:
        fold_index: Zero-based index of this fold.
        start_period: Start timestamp of the fold (inclusive).
        end_period: End timestamp of the fold (inclusive).
        n_periods: Number of periods in this fold.
        result: The SimulationResult for this fold.
    """

    fold_index: int
    start_period: pd.Timestamp
    end_period: pd.Timestamp
    n_periods: int
    result: SimulationResult


@dataclass(frozen=True)
class FoldAggregation:
    """
    Aggregated statistics for a single metric across folds.

    Attributes:
        metric_key: The metric being aggregated (e.g., MetricKey.ANNUALIZED_SHARPE).
        values: Per-fold metric values as a numpy array.
        median: Median value across folds.
        mean: Mean value across folds.
        std: Standard deviation across folds.
        min: Minimum value across folds.
        max: Maximum value across folds.
        count: Number of folds with valid (non-NaN) values.
    """

    metric_key: str
    values: np.ndarray
    median: float
    mean: float
    std: float
    min: float
    max: float
    count: int


@dataclass(frozen=True)
class WalkForwardResult:
    """
    Result object returned by `walk_forward()`.

    Implements the `MetricsAccessor` protocol for unified metric extraction.
    Metric methods delegate to `full_result` (the combined simulation).

    Attributes:
        folds: Tuple of FoldResult objects, one per fold.
        aggregations: Dictionary mapping metric keys to FoldAggregation objects.
        fold_config: The FoldConfig used for this analysis.
        sim_config: The SimConfig used for simulations.
        summary: DataFrame with per-fold metrics (rows=folds, columns=metrics).
        full_result: Optional combined SimulationResult spanning all folds.
            Set `include_full_result=True` in walk_forward() to enable.

    Full-Period Metrics (delegates to full_result):
        >>> result.metric_value(MetricKey.ANNUALIZED_SHARPE)
        >>> result.available_metrics("Performance")
        >>> result.metrics_dict("Risk")

        Raises ValueError if full_result is None (include_full_result=False).

    Fold-Level Metrics:
        >>> result.metric_aggregation(MetricKey.ANNUALIZED_SHARPE).median
        >>> result.fold_metric_values(MetricKey.ANNUALIZED_SHARPE)
        >>> result.summary_stats([MetricKey.ANNUALIZED_SHARPE])

    Individual Fold Access:
        >>> for fold in result.folds:
        ...     sharpe = fold.result.metric_value(MetricKey.ANNUALIZED_SHARPE)

    Generating Tearsheets:
        >>> from alphavec import tearsheet
        >>> tearsheet(sim_result=result.full_result, output_path="full.html")
        >>> tearsheet(sim_result=result.folds[0].result, output_path="fold_0.html")
    """

    folds: tuple[FoldResult, ...]
    aggregations: dict[str, FoldAggregation]
    fold_config: FoldConfig
    sim_config: SimConfig
    summary: pd.DataFrame
    full_result: SimulationResult | None = None

    def metric_aggregation(self, metric_key: str) -> FoldAggregation | None:
        """Get aggregation for a specific metric."""
        return self.aggregations.get(metric_key)

    def fold_metric_values(self, metric_key: str) -> pd.Series:
        """Get per-fold values for a metric as a Series indexed by fold_index."""
        agg = self.aggregations.get(metric_key)
        if agg is None:
            return pd.Series(dtype=float)
        return pd.Series(agg.values, name=metric_key)

    def summary_stats(
        self,
        metric_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get summary statistics DataFrame for selected metrics.

        If metric_keys is None, includes all aggregated metrics.
        """
        keys = metric_keys or list(self.aggregations.keys())
        rows = []
        for key in keys:
            agg = self.aggregations.get(key)
            if agg is not None:
                rows.append(
                    {
                        "Metric": key,
                        "Median": agg.median,
                        "Mean": agg.mean,
                        "Std": agg.std,
                        "Min": agg.min,
                        "Max": agg.max,
                        "Count": agg.count,
                    }
                )
        return pd.DataFrame(rows)

    def metric_value(self, metric: str, *, default: object = None) -> object:
        """
        Get metric value from the full-period result.

        Delegates to `self.full_result.metric_value()`.

        Args:
            metric: The metric key (use MetricKey constants).
            default: Value to return if metric not found or full_result is None.

        Returns:
            The metric value from the full-period result, or default if unavailable.

        Raises:
            ValueError: If full_result is None (include_full_result=False) and no default.

        Note:
            For fold-level metrics, use `metric_aggregation()` or `fold_metric_values()`.
        """
        if self.full_result is None:
            if default is not None:
                return default
            raise ValueError(
                "Cannot access full-period metrics: full_result is None. "
                "Set include_full_result=True in walk_forward() to enable this. "
                "For fold-level metrics, use metric_aggregation() or fold_metric_values()."
            )
        return self.full_result.metric_value(metric, default=default)

    def available_metrics(self, category: str | None = None) -> list[str]:
        """
        Return available metric keys from the full-period result.

        Delegates to `self.full_result.available_metrics()`.

        Args:
            category: Optional category filter.

        Returns:
            List of metric keys available in the full-period result.

        Raises:
            ValueError: If full_result is None (include_full_result=False).
        """
        if self.full_result is None:
            raise ValueError(
                "Cannot access full-period metrics: full_result is None. "
                "Set include_full_result=True in walk_forward() to enable this."
            )
        return self.full_result.available_metrics(category)

    def metrics_dict(self, category: str | None = None) -> dict[str, object]:
        """
        Return metrics dictionary from the full-period result.

        Delegates to `self.full_result.metrics_dict()`.

        Args:
            category: Optional category filter.

        Returns:
            Dictionary of metrics from the full-period result.

        Raises:
            ValueError: If full_result is None (include_full_result=False).
        """
        if self.full_result is None:
            raise ValueError(
                "Cannot access full-period metrics: full_result is None. "
                "Set include_full_result=True in walk_forward() to enable this."
            )
        return self.full_result.metrics_dict(category)


DEFAULT_AGGREGATE_METRICS: list[str] = [
    # Performance
    MetricKey.ANNUALIZED_RETURN_PCT,
    MetricKey.ANNUALIZED_VOLATILITY,
    MetricKey.ANNUALIZED_SHARPE,
    MetricKey.MAX_DRAWDOWN_EQUITY_PCT,
    MetricKey.TOTAL_RETURN_PCT,
    # Risk
    MetricKey.ANNUALIZED_SORTINO,
    MetricKey.DOWNSIDE_DEVIATION,
    MetricKey.VAR_95_PCT,
    MetricKey.CVAR_95_PCT,
    MetricKey.CALMAR_RATIO,
    MetricKey.OMEGA_RATIO,
    MetricKey.ULCER_INDEX,
    # Distribution
    MetricKey.SKEWNESS,
    MetricKey.KURTOSIS,
    MetricKey.HIT_RATE,
    MetricKey.PROFIT_FACTOR,
    # Costs
    MetricKey.ANNUAL_TURNOVER_1WAY,
    MetricKey.FEES,
    # Exposure
    MetricKey.GROSS_EXPOSURE_MEAN_PCT,
    MetricKey.NET_EXPOSURE_MEAN_PCT,
    # Signal
    MetricKey.WEIGHT_IC_MEAN_NEXT,
]


def _freq_to_period_freq(fold_period: str) -> str:
    """
    Convert date_range frequency to Period-compatible frequency.

    E.g., "ME" -> "M", "YE" -> "Y", "QE" -> "Q"
    """
    # Handle month-end, year-end, quarter-end variations
    mappings = {
        "ME": "M",
        "YE": "Y",
        "QE": "Q",
        "1ME": "M",
        "1YE": "Y",
        "1QE": "Q",
        "3ME": "3M",
        "6ME": "6M",
        "2QE": "2Q",
    }
    return mappings.get(fold_period, fold_period)


def _generate_fold_boundaries(
    *,
    index: pd.DatetimeIndex,
    fold_period: str,
    min_periods: int,
    align_start: bool,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate fold boundaries from a DatetimeIndex.

    Returns list of (start, end) timestamp tuples for each valid fold.
    """
    if len(index) < min_periods:
        return []

    start = index.min()
    end = index.max()
    tz = index.tz

    if align_start:
        # Align to calendar boundaries using period arithmetic
        # Convert fold_period to Period-compatible frequency
        period_freq = _freq_to_period_freq(fold_period)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converting to Period representation")
            period_start = pd.Timestamp(start).to_period(period_freq).start_time
        if tz is not None:
            period_start = period_start.tz_localize(tz)
        boundaries = pd.date_range(
            start=period_start,
            end=end + pd.Timedelta(days=1),
            freq=fold_period,
        )
    else:
        # Raw slicing from data start
        boundaries = pd.date_range(
            start=start,
            end=end + pd.Timedelta(days=1),
            freq=fold_period,
        )

    # Handle timezone
    if tz is not None and boundaries.tz is None:
        boundaries = boundaries.tz_localize(tz)
    elif tz is None and boundaries.tz is not None:
        boundaries = boundaries.tz_localize(None)

    # Convert to fold ranges
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(boundaries) - 1):
        fold_start = boundaries[i]
        fold_end = boundaries[i + 1] - pd.Timedelta(nanoseconds=1)

        # Find actual data range within this fold
        mask = (index >= fold_start) & (index <= fold_end)
        if mask.sum() >= min_periods:
            actual_start = index[mask].min()
            actual_end = index[mask].max()
            folds.append((actual_start, actual_end))

    # Handle last partial fold if it has enough data
    if len(boundaries) > 0:
        last_boundary = boundaries[-1]
        mask = index >= last_boundary
        if mask.sum() >= min_periods:
            folds.append((index[mask].min(), index[mask].max()))

    return folds


def _aggregate_fold_metrics(
    *,
    fold_results: list[FoldResult],
    metrics_to_aggregate: list[str] | None,
) -> dict[str, FoldAggregation]:
    """
    Aggregate metrics across folds.
    """
    if metrics_to_aggregate is None:
        metrics_to_aggregate = DEFAULT_AGGREGATE_METRICS

    aggregations: dict[str, FoldAggregation] = {}

    for metric_key in metrics_to_aggregate:
        values = []
        for fold in fold_results:
            val = fold.result.metric_value(metric_key)
            if isinstance(val, (int, float)) and np.isfinite(val):
                values.append(float(val))
            else:
                values.append(np.nan)

        arr = np.array(values, dtype=float)
        valid = arr[np.isfinite(arr)]

        if len(valid) > 0:
            aggregations[metric_key] = FoldAggregation(
                metric_key=metric_key,
                values=arr,
                median=float(np.median(valid)),
                mean=float(np.mean(valid)),
                std=float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                min=float(np.min(valid)),
                max=float(np.max(valid)),
                count=len(valid),
            )

    return aggregations


def _build_fold_summary(
    fold_results: list[FoldResult],
    metric_keys: list[str],
) -> pd.DataFrame:
    """
    Build a summary DataFrame with rows=folds, columns=metrics.
    """
    rows = []
    for fold in fold_results:
        row: dict[str, object] = {
            "fold_index": fold.fold_index,
            "start_period": fold.start_period,
            "end_period": fold.end_period,
            "n_periods": fold.n_periods,
        }
        for key in metric_keys:
            row[key] = fold.result.metric_value(key)
        rows.append(row)

    return pd.DataFrame(rows)


def walk_forward(
    *,
    weights: pd.DataFrame | pd.Series,
    market: MarketData,
    config: SimConfig | None = None,
    fold_config: FoldConfig,
    metrics_to_aggregate: list[str] | None = None,
    include_full_result: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward (fold-based) simulation analysis.

    Splits the time range into consecutive folds based on `fold_config.fold_period`,
    runs independent simulations on each fold, and aggregates metrics across folds.

    Args:
        weights: Target portfolio weights (same format as `simulate()`).
        market: Market data (same format as `simulate()`).
        config: Simulation configuration applied to each fold. Period slicing
            parameters (start_period, end_period) in config are applied BEFORE
            fold splitting.
        fold_config: Configuration for fold generation.
        metrics_to_aggregate: List of metric keys to aggregate. If None, aggregates
            all numeric metrics from DEFAULT_AGGREGATE_METRICS.
        include_full_result: If True, also run simulation on the full period and
            include in result.full_result.

    Returns:
        WalkForwardResult with per-fold results and aggregated statistics.

    Raises:
        ValueError: If no valid folds can be generated (e.g., data too short).

    Example:
        >>> from alphavec import walk_forward, FoldConfig, MetricKey, tearsheet
        >>> result = walk_forward(
        ...     weights=weights,
        ...     market=market,
        ...     fold_config=FoldConfig(fold_period="3ME"),
        ... )
        >>> # Per-fold Sharpe values
        >>> result.fold_metric_values(MetricKey.ANNUALIZED_SHARPE)
        >>> # Aggregated statistics
        >>> result.metric_aggregation(MetricKey.ANNUALIZED_SHARPE).median
        >>> # Summary table
        >>> result.summary_stats([MetricKey.ANNUALIZED_SHARPE, MetricKey.MAX_DRAWDOWN_EQUITY_PCT])
        >>> # Generate tearsheet for full period
        >>> tearsheet(sim_result=result.full_result, output_path="full_tearsheet.html")
        >>> # Generate tearsheet for a specific fold
        >>> tearsheet(sim_result=result.folds[0].result, output_path="fold_0_tearsheet.html")
    """
    cfg = config or SimConfig()

    # Get reference index from close_prices
    close_df = _to_frame(market.close_prices, "close")
    ref_index = close_df.index

    # Apply initial period slicing if specified
    if cfg.start_period is not None or cfg.end_period is not None:
        if isinstance(cfg.start_period, str) or isinstance(cfg.end_period, str):
            # Use loc slicing for string periods
            start_ts = pd.Timestamp(cfg.start_period) if cfg.start_period is not None else ref_index.min()
            end_ts = pd.Timestamp(cfg.end_period) if cfg.end_period is not None else ref_index.max()
            # Handle timezone-aware indexes
            tz = ref_index.tz
            if tz is not None:
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize(tz)
                if end_ts.tz is None:
                    end_ts = end_ts.tz_localize(tz)
            ref_index = ref_index[(ref_index >= start_ts) & (ref_index <= end_ts)]
        elif isinstance(cfg.start_period, int) or isinstance(cfg.end_period, int):
            # Use iloc slicing for int periods
            start_idx = cfg.start_period if cfg.start_period is not None else 0
            end_idx = cfg.end_period if cfg.end_period is not None else len(ref_index)
            ref_index = ref_index[start_idx:end_idx]

    # If trim_warmup is enabled, filter ref_index to start from first valid weights
    if cfg.trim_warmup:
        weights_df = _to_frame(weights, "weights")
        has_finite = weights_df.notna().any(axis=1)
        if has_finite.any():
            first_valid_idx = has_finite.idxmax()
            ref_index = ref_index[ref_index >= first_valid_idx]

    # Generate fold boundaries
    fold_boundaries = _generate_fold_boundaries(
        index=ref_index,
        fold_period=fold_config.fold_period,
        min_periods=fold_config.min_periods,
        align_start=fold_config.align_start,
    )

    if len(fold_boundaries) == 0:
        raise ValueError(
            f"No valid folds generated. Data has {len(ref_index)} periods, "
            f"fold_period='{fold_config.fold_period}', min_periods={fold_config.min_periods}."
        )

    # Run simulation for each fold
    fold_results: list[FoldResult] = []
    for i, (fold_start, fold_end) in enumerate(fold_boundaries):
        # Create fold-specific config with period slicing
        # Don't propagate trim_warmup since we've already handled it at the walk_forward level
        start_str = fold_start.strftime("%Y-%m-%d %H:%M:%S") if fold_start.hour != 0 or fold_start.minute != 0 or fold_start.second != 0 else fold_start.strftime("%Y-%m-%d")
        end_str = fold_end.strftime("%Y-%m-%d %H:%M:%S") if fold_end.hour != 0 or fold_end.minute != 0 or fold_end.second != 0 else fold_end.strftime("%Y-%m-%d")

        fold_sim_config = SimConfig(
            init_cash=cfg.init_cash,
            benchmark_asset=cfg.benchmark_asset,
            order_notional_min=cfg.order_notional_min,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            start_period=start_str,
            end_period=end_str,
            trim_warmup=False,  # Already handled at walk_forward level
            freq_rule=cfg.freq_rule,
            trading_days_year=cfg.trading_days_year,
            risk_free_rate=cfg.risk_free_rate,
        )

        sim_result = simulate(
            weights=weights,
            market=market,
            config=fold_sim_config,
        )

        fold_results.append(
            FoldResult(
                fold_index=i,
                start_period=fold_start,
                end_period=fold_end,
                n_periods=len(sim_result.returns),
                result=sim_result,
            )
        )

    # Aggregate metrics
    agg_keys = metrics_to_aggregate if metrics_to_aggregate is not None else DEFAULT_AGGREGATE_METRICS
    aggregations = _aggregate_fold_metrics(
        fold_results=fold_results,
        metrics_to_aggregate=agg_keys,
    )

    # Build summary DataFrame
    summary = _build_fold_summary(fold_results, list(aggregations.keys()))

    # Optionally run full simulation
    full_result = None
    if include_full_result:
        full_result = simulate(
            weights=weights,
            market=market,
            config=cfg,
        )

    return WalkForwardResult(
        folds=tuple(fold_results),
        aggregations=aggregations,
        fold_config=fold_config,
        sim_config=cfg,
        summary=summary,
        full_result=full_result,
    )
