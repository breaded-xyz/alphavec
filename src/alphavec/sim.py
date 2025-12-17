"""
Vector-based backtest simulation for perpetual futures strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import _metrics


@dataclass(frozen=True)
class _Inputs:
    weights: pd.DataFrame
    close_prices: pd.DataFrame
    order_prices: pd.DataFrame
    funding_rates: pd.DataFrame


@dataclass(frozen=True)
class _RunOutputs:
    equity: np.ndarray
    fees_paid: np.ndarray
    funding_earned: np.ndarray
    turnover_ratio: np.ndarray
    gross_exposure_ratio: np.ndarray
    net_exposure_ratio: np.ndarray
    order_count_period: np.ndarray
    order_notional_sum_period: np.ndarray
    slippage_paid: np.ndarray
    positions_hist: np.ndarray
    first_order_date: pd.Timestamp | None


@dataclass(frozen=True)
class SimulationResult:
    """
    Result object returned by `simulate()`.
    """

    returns: pd.Series
    metrics: pd.DataFrame

    @property
    def equity(self) -> pd.Series | None:
        equity = self.metrics.attrs.get("equity")
        return equity if isinstance(equity, pd.Series) else None

    @property
    def benchmark_equity(self) -> pd.Series | None:
        bench = self.metrics.attrs.get("benchmark_equity")
        return bench if isinstance(bench, pd.Series) else None

    def metric_value(self, metric: str, *, default: object = None) -> object:
        try:
            v = self.metrics.loc[metric, "Value"]
        except Exception:
            return default
        return default if pd.isna(v) else v

    def tearsheet(
        self,
        *,
        grid_results=None,
        output_path=None,
        signal_smooth_window: int = 30,
        rolling_sharpe_window: int = 30,
    ) -> str:
        from .tearsheet import tearsheet

        return tearsheet(
            sim_result=self,
            grid_results=grid_results,
            output_path=output_path,
            signal_smooth_window=signal_smooth_window,
            rolling_sharpe_window=rolling_sharpe_window,
        )


@dataclass(frozen=True)
class MarketData:
    close_prices: pd.DataFrame | pd.Series
    order_prices: pd.DataFrame | pd.Series | None = None
    funding_rates: pd.DataFrame | pd.Series | None = None


@dataclass(frozen=True)
class SimConfig:
    init_cash: float = 1000.0
    benchmark_asset: str | None = None
    order_notional_min: float = 0.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    freq_rule: str = "1D"
    trading_days_year: int = 365
    risk_free_rate: float = 0.0


def _to_frame(x: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name or name)
    return x


def _normalize_inputs(
    *,
    weights: pd.DataFrame | pd.Series,
    close_prices: pd.DataFrame | pd.Series,
    order_prices: pd.DataFrame | pd.Series | None,
    funding_rates: pd.DataFrame | pd.Series | None,
) -> _Inputs:
    """
    Validate and normalize inputs using close_prices as the reference.

    All inputs must be perfectly aligned - no automatic alignment is performed.
    """
    # Convert Series to DataFrame
    cp = _to_frame(close_prices, "close_prices")
    w = _to_frame(weights, "weights")
    op = _to_frame(order_prices, "order_prices") if order_prices is not None else None
    fr = _to_frame(funding_rates, "funding_rates") if funding_rates is not None else None

    # Validate reference (close_prices)
    if not pd.api.types.is_float_dtype(cp.values.dtype):
        raise ValueError("close_prices must be of type float")

    if cp.shape[1] < 1:
        raise ValueError("close_prices must have at least 1 column")

    if len(cp.index) < 3:
        raise ValueError("close_prices must have at least 3 rows")

    if not isinstance(cp.index, pd.DatetimeIndex):
        raise ValueError("close_prices must have a DatetimeIndex")

    if not cp.index.is_monotonic_increasing:
        raise ValueError("close_prices index must be monotonically increasing")

    # Get reference index and columns
    ref_index = cp.index
    ref_columns = cp.columns

    # Validate weights match reference exactly
    if not w.index.equals(ref_index):
        raise ValueError("weights index must exactly match close_prices index")
    if not w.columns.equals(ref_columns):
        raise ValueError("weights columns must exactly match close_prices columns")

    # Validate order_prices if provided
    if op is not None:
        if not op.index.equals(ref_index):
            raise ValueError("order_prices index must exactly match close_prices index")
        if not op.columns.equals(ref_columns):
            raise ValueError("order_prices columns must exactly match close_prices columns")
    else:
        op = cp.copy()

    # Validate funding_rates if provided
    if fr is not None:
        if not fr.index.equals(ref_index):
            raise ValueError("funding_rates index must exactly match close_prices index")
        if not fr.columns.equals(ref_columns):
            raise ValueError("funding_rates columns must exactly match close_prices columns")
    else:
        fr = pd.DataFrame(0.0, index=ref_index, columns=ref_columns)

    # Convert to float
    w = w.astype(float)
    cp = cp.astype(float)
    op = op.astype(float)
    fr = fr.astype(float)

    return _Inputs(weights=w, close_prices=cp, order_prices=op, funding_rates=fr)


def _run_simulation(
    *,
    inputs: _Inputs,
    init_cash: float,
    fee_rate: float,
    slippage_rate: float,
    order_notional_min: float,
) -> _RunOutputs:
    """
    Run the core simulation loop and return raw arrays for downstream metrics.
    """

    w = inputs.weights.to_numpy(dtype=float)
    cp = inputs.close_prices.to_numpy(dtype=float)
    op = inputs.order_prices.to_numpy(dtype=float)
    fr = inputs.funding_rates.to_numpy(dtype=float)

    n_periods, n_assets = w.shape
    positions = np.zeros(n_assets, dtype=float)
    cash: float = float(init_cash)

    equity = np.empty(n_periods, dtype=float)
    fees_paid = np.empty(n_periods, dtype=float)
    funding_earned = np.empty(n_periods, dtype=float)
    turnover_ratio = np.empty(n_periods, dtype=float)
    gross_exposure_ratio = np.empty(n_periods, dtype=float)
    net_exposure_ratio = np.empty(n_periods, dtype=float)
    order_count_period = np.empty(n_periods, dtype=int)
    order_notional_sum_period = np.empty(n_periods, dtype=float)
    slippage_paid = np.empty(n_periods, dtype=float)

    first_order_date: pd.Timestamp | None = None

    slip = float(slippage_rate)
    fee_rate = float(fee_rate)
    min_notional = float(order_notional_min)

    last_op = np.full(n_assets, np.nan, dtype=float)
    last_cp = np.full(n_assets, np.nan, dtype=float)
    positions_hist = np.empty((n_periods, n_assets), dtype=float)

    for i in range(n_periods):
        weights_raw = w[i]
        weights_i = np.nan_to_num(weights_raw, nan=0.0)
        op_raw = op[i]
        cp_raw = cp[i]
        fr_raw = fr[i]

        # Carry forward last seen prices so we can (a) value positions and (b) close positions even
        # when an instrument is temporarily not tradable (price=NaN).
        op_eff = np.where(np.isnan(op_raw), last_op, op_raw)
        cp_eff = np.where(np.isnan(cp_raw), last_cp, cp_raw)
        last_op = np.where(np.isnan(op_raw), last_op, op_raw)
        last_cp = np.where(np.isnan(cp_raw), last_cp, cp_raw)

        op_eff_safe = np.nan_to_num(op_eff, nan=0.0)
        cp_eff_safe = np.nan_to_num(cp_eff, nan=0.0)

        # Use order prices to value the portfolio *before* trading; then compute the notional
        # adjustment required to reach the target weights.
        equity_before = cash + float(np.dot(positions, op_eff_safe))
        target_notional = weights_i * equity_before
        current_notional = positions * op_eff_safe
        delta_notional = target_notional - current_notional

        # When weights are 0/NaN we interpret that as a "close" target. Closing orders are allowed
        # even if they're below the minimum notional threshold.
        closing_mask = np.isnan(weights_raw) | (weights_raw == 0.0)

        # If we don't have a tradable order price, skip opening/rebalancing; closing is still
        # allowed using the carried-forward order price.
        untradable_open_mask = np.isnan(op_raw) | np.isnan(op_eff)
        delta_notional = np.where(untradable_open_mask & ~closing_mask, 0.0, delta_notional)

        # Skip small non-closing orders to reduce churn / unrealistic fills.
        small_mask = np.abs(delta_notional) < min_notional
        delta_notional = np.where(small_mask & ~closing_mask, 0.0, delta_notional)

        buys = delta_notional > 0.0
        sells = delta_notional < 0.0
        exec_prices = op_eff_safe * (1.0 + slip * buys - slip * sells)

        traded_units = np.zeros(n_assets, dtype=float)
        nonzero_mask = delta_notional != 0.0
        traded_units[nonzero_mask] = delta_notional[nonzero_mask] / exec_prices[nonzero_mask]

        # Slippage is treated as an adverse execution penalty (always against the trader).
        slippage_cost = np.zeros(n_assets, dtype=float)
        if slip > 0 and np.any(nonzero_mask):
            buy_mask = buys & nonzero_mask
            sell_mask = sells & nonzero_mask
            slippage_cost[buy_mask] = delta_notional[buy_mask] * slip / (1.0 + slip)
            slippage_cost[sell_mask] = np.abs(delta_notional[sell_mask]) * slip / (1.0 - slip)
        slippage_paid[i] = float(slippage_cost.sum())

        order_count_period[i] = int(nonzero_mask.sum())
        order_notional_sum_period[i] = float(np.abs(delta_notional[nonzero_mask]).sum())

        if first_order_date is None and np.any(delta_notional != 0.0):
            first_order_date = inputs.weights.index[i]

        fee = np.abs(delta_notional) * fee_rate

        # Cash decreases by the signed notional (buys spend cash; sells receive cash) and fees.
        cash -= float(delta_notional.sum() + fee.sum())
        positions += traded_units

        fees_paid[i] = float(fee.sum())
        # Turnover is measured one-sided (industry convention) to avoid double-counting buys+sells.
        buys_notional = float(np.sum(delta_notional[delta_notional > 0.0]))
        sells_notional = float(-np.sum(delta_notional[delta_notional < 0.0]))
        denom_turnover = abs(equity_before)
        turnover_ratio[i] = (
            min(buys_notional, sells_notional) / denom_turnover if denom_turnover != 0 else 0.0
        )

        close_notional = positions * cp_eff_safe
        gross_exposure = float(np.abs(close_notional).sum())
        net_exposure = float(close_notional.sum())

        # Funding uses close notional. Positive rates mean longs pay and shorts earn.
        fr_eff = np.nan_to_num(fr_raw, nan=0.0)
        fr_eff = np.where(np.isnan(cp_raw), 0.0, fr_eff)
        funding_payment = -float(np.sum(fr_eff * close_notional))
        cash += funding_payment

        funding_earned[i] = funding_payment
        equity[i] = cash + float(close_notional.sum())
        denom_equity = abs(equity[i])
        gross_exposure_ratio[i] = gross_exposure / denom_equity if denom_equity != 0 else 0.0
        net_exposure_ratio[i] = net_exposure / denom_equity if denom_equity != 0 else 0.0
        positions_hist[i] = positions

    return _RunOutputs(
        equity=equity,
        fees_paid=fees_paid,
        funding_earned=funding_earned,
        turnover_ratio=turnover_ratio,
        gross_exposure_ratio=gross_exposure_ratio,
        net_exposure_ratio=net_exposure_ratio,
        order_count_period=order_count_period,
        order_notional_sum_period=order_notional_sum_period,
        slippage_paid=slippage_paid,
        positions_hist=positions_hist,
        first_order_date=first_order_date,
    )


def simulate(
    *,
    weights: pd.DataFrame | pd.Series,
    market: MarketData,
    config: SimConfig | None = None,
) -> SimulationResult:
    """
    Simulate a (vectorized) perpetual futures portfolio from target weights.

    This implements a simple rebalancing/perp accounting model:

    - Value the portfolio at the start of each period using the **order** price.
    - Convert target weights into target notionals (`target_notional = weights * equity_before`).
    - Trade the notional difference at the order price (with optional slippage), paying fees on
      absolute order notional.
    - Mark-to-market the resulting positions using the **close** price to produce end-of-period
      equity and returns.
    - Apply per-period funding to close notional (positive rates mean longs pay and shorts earn).

    Missing data is handled to keep the simulation robust to temporary gaps:

    - `weights` NaNs are treated as 0 targets.
    - If an asset's `order_prices` is NaN for a period, opening/rebalancing that asset is skipped
      for that period; however, positions can still be closed using the last observed order price.
    - If an asset's `close_prices` is NaN for a period, PnL and funding are forced to 0 for that
      asset in that period (positions remain valued using the last observed close for equity
      continuity).

    The returned `metrics` table also includes helpful series in `metrics.attrs` that downstream
    utilities (e.g. `tearsheet()`) can use for richer reporting.

    Args:
        weights: Target percentage portfolio weights. Positive=long, negative=short.
            Weights are in decimal units (1.0=100%). NaN weights are treated as 0.0 targets.
            Must have same index and columns as market.close_prices.
        market: Market data inputs (close/order prices and optional funding).
            All DataFrames must have matching index and columns.
        config: Simulation configuration (costs, annualization, benchmark, init cash).

    Returns:
        A SimulationResult with returns and metrics:

        - `returns`: Period returns as a pandas Series aligned to the input index.
        - `metrics`: Summary statistics as a pandas DataFrame with `Value` and `Note` columns.
          Extra artifacts are attached via `metrics.attrs`, including:

          - `metrics.attrs["returns"]`: The returned `returns` series.
          - `metrics.attrs["equity"]`: The simulated equity curve (same index as `returns`).
          - `metrics.attrs["init_cash"]`: The initial cash used for the simulation.
          - `metrics.attrs["benchmark_equity"]`: Benchmark equity curve (only when
            `benchmark_asset` is provided and present in `close_prices`).

    Raises:
        ValueError: If inputs are not properly aligned or do not meet validation requirements.
    """

    cfg = config or SimConfig()

    inputs = _normalize_inputs(
        weights=weights,
        close_prices=market.close_prices,
        order_prices=market.order_prices,
        funding_rates=market.funding_rates,
    )

    run = _run_simulation(
        inputs=inputs,
        init_cash=cfg.init_cash,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        order_notional_min=cfg.order_notional_min,
    )

    equity_series = pd.Series(run.equity, index=inputs.weights.index, name="equity")
    returns = equity_series.pct_change().fillna(0.0)
    returns.name = "returns"

    metrics = _metrics(
        weights=inputs.weights,
        close_prices=inputs.close_prices,
        returns=returns,
        equity=equity_series,
        init_cash=cfg.init_cash,
        fee_pct=cfg.fee_rate,
        slippage_pct=cfg.slippage_rate,
        freq_rule=cfg.freq_rule,
        trading_days_year=cfg.trading_days_year,
        risk_free_rate=cfg.risk_free_rate,
        benchmark_asset=cfg.benchmark_asset,
        first_order_date=run.first_order_date,
        fees_paid=run.fees_paid,
        funding_earned=run.funding_earned,
        turnover_ratio=run.turnover_ratio,
        gross_exposure_ratio=run.gross_exposure_ratio,
        net_exposure_ratio=run.net_exposure_ratio,
        order_count_period=run.order_count_period,
        order_notional_sum_period=run.order_notional_sum_period,
        slippage_paid=run.slippage_paid,
        positions_hist=run.positions_hist,
    )

    metrics.attrs["returns"] = returns
    metrics.attrs["equity"] = equity_series
    metrics.attrs["init_cash"] = float(cfg.init_cash)

    if cfg.benchmark_asset is not None and cfg.benchmark_asset in inputs.close_prices.columns:
        bench_prices = inputs.close_prices[cfg.benchmark_asset].copy().ffill().bfill()
        if bench_prices.notna().any():
            first = float(bench_prices.iloc[0])
            if np.isfinite(first) and first != 0.0:
                metrics.attrs["benchmark_equity"] = (cfg.init_cash * bench_prices / first).rename(
                    "benchmark_equity"
                )

    return SimulationResult(returns=returns, metrics=metrics)
