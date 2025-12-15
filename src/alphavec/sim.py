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


def _to_frame(x: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name or name)
    return x


def _normalize_inputs(
    *,
    weights: pd.DataFrame | pd.Series,
    close_prices: pd.DataFrame | pd.Series,
    order_prices: pd.DataFrame | pd.Series,
    funding_rates: pd.DataFrame | pd.Series | None,
) -> _Inputs:
    w = _to_frame(weights, "weights").astype(float)
    cp = _to_frame(close_prices, "close_prices").astype(float)
    op = _to_frame(order_prices, "order_prices").astype(float)
    fr = (
        _to_frame(funding_rates, "funding_rates").astype(float)
        if funding_rates is not None
        else None
    )

    index = w.index
    columns = w.columns

    def _align(df: pd.DataFrame, label: str) -> pd.DataFrame:
        missing_cols = columns.difference(df.columns)
        if len(missing_cols) > 0:
            raise ValueError(f"{label} is missing columns: {list(missing_cols)}")
        missing_index = index.difference(df.index)
        if len(missing_index) > 0:
            raise ValueError(f"{label} is missing index values for weights.")
        return df.reindex(index=index, columns=columns)

    cp = _align(cp, "close_prices")
    op = _align(op, "order_prices")
    if fr is None:
        fr = pd.DataFrame(0.0, index=index, columns=columns)
    else:
        fr = _align(fr, "funding_rates")

    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("weights must use a DatetimeIndex.")

    return _Inputs(weights=w, close_prices=cp, order_prices=op, funding_rates=fr)


def _run_simulation(
    *,
    inputs: _Inputs,
    init_cash: float,
    fee_pct: float,
    slippage_pct: float,
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

    slip = float(slippage_pct)
    fee_rate = float(fee_pct)
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
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate a perpetual futures portfolio from target weights.

    Args:
        weights: Target percentage portfolio weights. Positive=long, negative=short.
            Weights are in decimal units (1.0=100%). NaN weights are treated as 0.0 targets.
        close_prices: Close prices used for period PnL. NaNs indicate an asset is not tradable;
            the last close price is carried forward for valuation and PnL is 0 for that period.
        order_prices: Order execution prices (before slippage). NaNs indicate an asset is not tradable
            for opening/rebalancing; the last order price is carried forward to allow closing.
        funding_rates: Per-period signed funding rates; +ve means longs pay, shorts earn. NaNs are
            treated as 0, and funding is always 0 when close price is NaN.
        benchmark_asset: Optional asset column name to compute alpha and beta against.
        order_notional_min: Skip non-closing orders below this notional.
        fee_pct: Fee percentage on order notional.
        slippage_pct: Slippage percentage applied against the trader.
        init_cash: Starting capital.
        freq_rule: Pandas frequency rule for the data periodicity.
        trading_days_year: Trading days per year for annualization.
        risk_free_rate: Risk free rate for Sharpe.

    Returns:
        Portfolio period returns as a pandas Series.
        Metrics as a pandas DataFrame with Value and Note columns.
    """

    inputs = _normalize_inputs(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
    )

    run = _run_simulation(
        inputs=inputs,
        init_cash=init_cash,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        order_notional_min=order_notional_min,
    )

    equity_series = pd.Series(run.equity, index=inputs.weights.index, name="equity")
    returns = equity_series.pct_change().fillna(0.0)
    returns.name = "returns"

    metrics = _metrics(
        weights=inputs.weights,
        close_prices=inputs.close_prices,
        returns=returns,
        equity=equity_series,
        init_cash=init_cash,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        freq_rule=freq_rule,
        trading_days_year=trading_days_year,
        risk_free_rate=risk_free_rate,
        benchmark_asset=benchmark_asset,
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
    metrics.attrs["init_cash"] = float(init_cash)

    if benchmark_asset is not None and benchmark_asset in inputs.close_prices.columns:
        bench_prices = inputs.close_prices[benchmark_asset].copy().ffill().bfill()
        if bench_prices.notna().any():
            first = float(bench_prices.iloc[0])
            if np.isfinite(first) and first != 0.0:
                metrics.attrs["benchmark_equity"] = (init_cash * bench_prices / first).rename(
                    "benchmark_equity"
                )

    return returns, metrics
