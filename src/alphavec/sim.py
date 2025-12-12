"""
Vector-based backtest simulation for perpetual futures strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Inputs:
    weights: pd.DataFrame
    close_prices: pd.DataFrame
    order_prices: pd.DataFrame
    funding_rates: pd.DataFrame


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
    fr = _to_frame(funding_rates, "funding_rates").astype(float) if funding_rates is not None else None

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


def _max_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    return float(dd.min())


def _max_drawdown_pnl(pnl_curve: pd.Series) -> float:
    running_max = pnl_curve.cummax()
    denom = running_max.replace(0.0, np.nan)
    dd = (pnl_curve - running_max) / denom
    return float(dd.min(skipna=True) if dd.notna().any() else 0.0)


def _annualization_factor(freq_rule: str, trading_days_year: int) -> float:
    try:
        offset = pd.tseries.frequencies.to_offset(freq_rule)
        delta = pd.Timedelta(offset)
        if delta is pd.NaT:
            return float(trading_days_year)
        seconds = delta.total_seconds()
        if seconds <= 0:
            return float(trading_days_year)
        periods_per_day = (24.0 * 60.0 * 60.0) / seconds
        return float(trading_days_year) * periods_per_day
    except Exception:
        return float(trading_days_year)


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
) -> tuple[pd.Series, pd.Series]:
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
        benchmark_asset: Optional asset column name to compute alpha and beta against. Benchmark
            is a buy-and-hold of that asset using close prices with the same NaN carry-forward rules.
        order_notional_min: Skip non-closing orders below this notional.
        fee_pct: Fee percentage on order notional.
        slippage_pct: Slippage percentage applied against the trader.
        init_cash: Starting capital.
        freq_rule: Pandas frequency rule for the data periodicity.
        trading_days_year: Trading days per year for annualization.
        risk_free_rate: Risk free rate for Sharpe.

    Returns:
        Portfolio period returns as a pandas Series.
        Tearsheet metrics as a pandas Series.
    """

    inputs = _normalize_inputs(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
    )

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
    order_count_period = np.empty(n_periods, dtype=int)
    order_notional_sum_period = np.empty(n_periods, dtype=float)

    first_order_date: pd.Timestamp | None = None

    slip: Final[float] = float(slippage_pct)
    fee_rate: Final[float] = float(fee_pct)
    min_notional: Final[float] = float(order_notional_min)

    last_op = np.full(n_assets, np.nan, dtype=float)
    last_cp = np.full(n_assets, np.nan, dtype=float)

    for i in range(n_periods):
        weights_raw = w[i]
        weights_i = np.nan_to_num(weights_raw, nan=0.0)
        op_raw = op[i]
        cp_raw = cp[i]
        fr_raw = fr[i]

        op_eff = np.where(np.isnan(op_raw), last_op, op_raw)
        cp_eff = np.where(np.isnan(cp_raw), last_cp, cp_raw)

        last_op = np.where(np.isnan(op_raw), last_op, op_raw)
        last_cp = np.where(np.isnan(cp_raw), last_cp, cp_raw)

        op_eff_safe = np.nan_to_num(op_eff, nan=0.0)
        cp_eff_safe = np.nan_to_num(cp_eff, nan=0.0)

        equity_before = cash + float(np.dot(positions, op_eff_safe))
        target_notional = weights_i * equity_before
        current_notional = positions * op_eff_safe
        delta_notional = target_notional - current_notional

        closing_mask = np.isnan(weights_raw) | (weights_raw == 0.0)
        untradable_open_mask = np.isnan(op_raw) | np.isnan(op_eff)
        delta_notional = np.where(untradable_open_mask & ~closing_mask, 0.0, delta_notional)
        small_mask = np.abs(delta_notional) < min_notional
        delta_notional = np.where(small_mask & ~closing_mask, 0.0, delta_notional)

        buys = delta_notional > 0.0
        sells = delta_notional < 0.0
        exec_prices = op_eff_safe * (1.0 + slip * buys - slip * sells)
        traded_units = np.zeros(n_assets, dtype=float)
        nonzero_mask = delta_notional != 0.0
        traded_units[nonzero_mask] = delta_notional[nonzero_mask] / exec_prices[nonzero_mask]

        order_count_period[i] = int(nonzero_mask.sum())
        order_notional_sum_period[i] = float(np.abs(delta_notional[nonzero_mask]).sum())

        if first_order_date is None and np.any(delta_notional != 0.0):
            first_order_date = inputs.weights.index[i]

        fee = np.abs(delta_notional) * fee_rate
        cash -= float(delta_notional.sum() + fee.sum())
        positions += traded_units

        fees_paid[i] = float(fee.sum())
        turnover_ratio[i] = (
            float(np.abs(delta_notional).sum() / equity_before) if equity_before != 0 else 0.0
        )

        close_notional = positions * cp_eff_safe
        gross_exposure = float(np.abs(close_notional).sum())
        fr_eff = np.nan_to_num(fr_raw, nan=0.0)
        fr_eff = np.where(np.isnan(cp_raw), 0.0, fr_eff)
        funding_payment = -float(np.sum(fr_eff * close_notional))
        cash += funding_payment

        funding_earned[i] = funding_payment
        equity[i] = cash + float(close_notional.sum())
        denom_equity = abs(equity[i])
        gross_exposure_ratio[i] = gross_exposure / denom_equity if denom_equity != 0 else 0.0

    equity_series = pd.Series(equity, index=inputs.weights.index, name="equity")
    returns = equity_series.pct_change().fillna(0.0)
    returns.name = "returns"

    pnl_curve = equity_series - init_cash
    total_return_pct = float(equity_series.iloc[-1] / init_cash - 1.0)

    dd_equity = _max_drawdown(equity_series)
    dd_pnl = _max_drawdown_pnl(pnl_curve)

    annual_factor = _annualization_factor(freq_rule, trading_days_year)
    if n_periods > 0:
        annual_return = float((1.0 + returns).prod() ** (annual_factor / n_periods) - 1.0)
        annual_vol = float(returns.std(ddof=0) * np.sqrt(annual_factor))
        annual_sharpe = float(
            (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
        )
        annual_turnover = float(turnover_ratio.mean() * annual_factor)
    else:
        annual_return = 0.0
        annual_vol = 0.0
        annual_sharpe = np.nan
        annual_turnover = 0.0

    total_order_count = int(order_count_period.sum())
    avg_order_notional = (
        float(order_notional_sum_period.sum() / total_order_count)
        if total_order_count > 0
        else 0.0
    )
    max_gross_exposure_pct = float(np.nanmax(gross_exposure_ratio) * 100.0)
    avg_gross_exposure_pct = float(np.nanmean(gross_exposure_ratio) * 100.0)

    alpha: float = np.nan
    beta: float = np.nan
    if benchmark_asset is not None:
        if benchmark_asset not in inputs.close_prices.columns:
            raise ValueError(f"benchmark_asset '{benchmark_asset}' not in weights columns.")
        bench_prices = inputs.close_prices[benchmark_asset].copy()
        bench_prices = bench_prices.ffill().bfill()
        if bench_prices.isna().all():
            alpha = np.nan
            beta = np.nan
        else:
            bench_returns = bench_prices.pct_change().fillna(0.0)
            rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annual_factor) - 1.0
            y = returns - rf_per_period
            x = bench_returns - rf_per_period
            var_x = float(np.var(x, ddof=0))
            if var_x > 0:
                cov_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
                beta = cov_xy / var_x
                alpha_per_period = float(y.mean() - beta * x.mean())
                alpha = alpha_per_period * annual_factor
            else:
                alpha = np.nan
                beta = np.nan

    tearsheet = pd.Series(
        {
            "simulation start date": inputs.weights.index.min(),
            "simulation end date": inputs.weights.index.max(),
            "first order date": first_order_date,
            "total return %": total_return_pct * 100.0,
            "max drawdown (equity) %": dd_equity * 100.0,
            "max drawdown (PnL) %": dd_pnl * 100.0,
            "funding earnings": float(funding_earned.sum()),
            "fees": float(fees_paid.sum()),
            "annualized return": annual_return,
            "annualized volatility": annual_vol,
            "annualized sharpe": annual_sharpe,
            "annual turnover": annual_turnover,
            "total order count": total_order_count,
            "average order notional": avg_order_notional,
            "max gross exposure %": max_gross_exposure_pct,
            "average gross exposure %": avg_gross_exposure_pct,
            "alpha": alpha,
            "beta": beta,
        }
    )

    return returns, tearsheet
