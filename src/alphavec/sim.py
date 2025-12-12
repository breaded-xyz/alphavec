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
        Tearsheet metrics as a pandas DataFrame with Value and Note columns.
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
    net_exposure_ratio = np.empty(n_periods, dtype=float)
    order_count_period = np.empty(n_periods, dtype=int)
    order_notional_sum_period = np.empty(n_periods, dtype=float)
    slippage_paid = np.empty(n_periods, dtype=float)

    first_order_date: pd.Timestamp | None = None

    slip: Final[float] = float(slippage_pct)
    fee_rate: Final[float] = float(fee_pct)
    min_notional: Final[float] = float(order_notional_min)

    last_op = np.full(n_assets, np.nan, dtype=float)
    last_cp = np.full(n_assets, np.nan, dtype=float)
    positions_hist = np.empty((n_periods, n_assets), dtype=float)

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
        cash -= float(delta_notional.sum() + fee.sum())
        positions += traded_units

        fees_paid[i] = float(fee.sum())
        turnover_ratio[i] = (
            float(np.abs(delta_notional).sum() / equity_before) if equity_before != 0 else 0.0
        )

        close_notional = positions * cp_eff_safe
        gross_exposure = float(np.abs(close_notional).sum())
        net_exposure = float(close_notional.sum())
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
        float(order_notional_sum_period.sum() / total_order_count) if total_order_count > 0 else 0.0
    )
    max_gross_exposure_pct = float(np.nanmax(gross_exposure_ratio) * 100.0)
    avg_gross_exposure_pct = float(np.nanmean(gross_exposure_ratio) * 100.0)

    calmar_ratio = (
        annual_return / abs(dd_equity) if dd_equity != 0 and np.isfinite(dd_equity) else np.nan
    )
    skewness = float(returns.skew())
    kurtosis = float(returns.kurtosis())
    best_period_return = float(returns.max())
    worst_period_return = float(returns.min())

    nonzero_returns = returns[returns != 0.0]
    wins = nonzero_returns[nonzero_returns > 0.0]
    losses = nonzero_returns[nonzero_returns < 0.0]
    win_count = int(wins.count())
    loss_count = int(losses.count())
    hit_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else np.nan
    avg_win = float(wins.mean()) if win_count > 0 else 0.0
    avg_loss = float(losses.mean()) if loss_count > 0 else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    drawdown = equity_series / equity_series.cummax() - 1.0
    underwater = drawdown < 0.0
    dd_groups = (underwater != underwater.shift()).cumsum()
    dd_durations = underwater.groupby(dd_groups).sum()
    max_drawdown_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    trough_ts = drawdown.idxmin()
    peak_value = float(equity_series.cummax().loc[trough_ts])
    peak_ts_candidates = equity_series.loc[:trough_ts]
    peak_ts = peak_ts_candidates[peak_ts_candidates == peak_value].index[-1]
    post_trough = equity_series.loc[trough_ts:]
    recovered = post_trough[post_trough >= peak_value]
    if len(recovered) > 0:
        recovery_ts = recovered.index[0]
        time_to_recovery = int((equity_series.loc[peak_ts:recovery_ts]).shape[0] - 1)
    else:
        time_to_recovery = np.nan

    net_exposure_mean_pct = float(np.nanmean(net_exposure_ratio) * 100.0)
    net_exposure_median_pct = float(np.nanmedian(net_exposure_ratio) * 100.0)
    net_exposure_max_pct = float(np.nanmax(np.abs(net_exposure_ratio)) * 100.0)

    gross_exposure_median_pct = float(np.nanmedian(gross_exposure_ratio) * 100.0)

    holding_lengths: list[int] = []
    signs = np.sign(positions_hist)
    for asset_idx in range(n_assets):
        s = signs[:, asset_idx]
        start_idx: int | None = None
        current_sign = 0.0
        for t, val in enumerate(s):
            if val != 0.0 and start_idx is None:
                start_idx = t
                current_sign = val
            elif start_idx is not None and (val == 0.0 or val != current_sign):
                holding_lengths.append(t - start_idx)
                if val != 0.0:
                    start_idx = t
                    current_sign = val
                else:
                    start_idx = None
        if start_idx is not None:
            holding_lengths.append(n_periods - start_idx)
    average_holding_period = float(np.mean(holding_lengths)) if holding_lengths else 0.0

    total_fees = float(fees_paid.sum())
    total_slippage = float(slippage_paid.sum())
    net_pnl = float(equity_series.iloc[-1] - init_cash)
    gross_pnl = net_pnl + total_fees + total_slippage
    costs_pct_gross_pnl = (
        (total_fees + total_slippage) / abs(gross_pnl) * 100.0 if gross_pnl != 0 else np.nan
    )

    funding_total = float(funding_earned.sum())
    funding_pct_total_pnl = funding_total / net_pnl * 100.0 if net_pnl != 0 else np.nan
    average_funding_settled = float(np.nanmean(funding_earned))

    abs_weights = np.abs(w)
    max_abs_weight = float(np.nanmax(abs_weights))
    mean_abs_weight = float(np.nanmean(abs_weights))

    alpha: float = np.nan
    beta: float = np.nan
    benchmark_annual_return: float = np.nan
    tracking_error: float = np.nan
    active_annual_return: float = np.nan
    information_ratio: float = np.nan
    r2: float = np.nan
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
            if n_periods > 0:
                benchmark_annual_return = float(
                    (1.0 + bench_returns).prod() ** (annual_factor / n_periods) - 1.0
                )
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
            active_returns = returns - bench_returns
            tracking_error = float(active_returns.std(ddof=0) * np.sqrt(annual_factor))
            active_annual_return = float(active_returns.mean() * annual_factor)
            information_ratio = (
                active_annual_return / tracking_error if tracking_error > 0 else np.nan
            )
            corr = float(returns.corr(bench_returns))
            r2 = corr * corr if np.isfinite(corr) else np.nan

    metrics_meta = {
        "Period frequency": freq_rule,
        "Simulation start date": inputs.weights.index.min(),
        "Simulation end date": inputs.weights.index.max(),
        "First transaction date": first_order_date,
    }
    metrics_performance = {
        "Annualized return %": annual_return,
        "Annualized volatility": annual_vol,
        "Annualized Sharpe": annual_sharpe,
        "Max drawdown (equity) %": dd_equity * 100.0,
        "Max drawdown (PnL) %": dd_pnl * 100.0,
        "Total return %": total_return_pct * 100.0,
    }
    metrics_costs_and_trading = {
        "Funding earnings": funding_total,
        "Fees": total_fees,
        "Annual turnover": annual_turnover,
        "Total order count": total_order_count,
        "Average order notional": avg_order_notional,
    }
    metrics_exposure = {
        "Gross exposure mean %": avg_gross_exposure_pct,
        "Gross exposure median %": gross_exposure_median_pct,
        "Gross exposure max %": max_gross_exposure_pct,
        "Net exposure mean %": net_exposure_mean_pct,
        "Net exposure median %": net_exposure_median_pct,
        "Net exposure max %": net_exposure_max_pct,
    }
    metrics_benchmark = {
        "Alpha": alpha,
        "Beta": beta,
        "Benchmark annualized return %": benchmark_annual_return * 100.0,
        "Active annual return %": active_annual_return * 100.0,
        "Tracking error": tracking_error,
        "Information ratio": information_ratio,
        "R2 vs benchmark": r2,
    }
    metrics_distribution = {
        "Calmar ratio": calmar_ratio,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Best period return": best_period_return,
        "Worst period return": worst_period_return,
        "Hit rate": hit_rate,
        "Avg win": avg_win,
        "Avg loss": avg_loss,
        "Profit factor": profit_factor,
        "Max drawdown duration (periods)": max_drawdown_duration,
        "Time to recovery (periods)": time_to_recovery,
    }
    metrics_portfolio = {
        "Average holding period": average_holding_period,
        "Costs % gross pnl": costs_pct_gross_pnl,
        "Funding % total pnl": funding_pct_total_pnl,
        "Average funding settled": average_funding_settled,
        "Max abs weight": max_abs_weight,
        "Mean abs weight": mean_abs_weight,
    }

    metrics = (
        metrics_meta
        | metrics_performance
        | metrics_costs_and_trading
        | metrics_exposure
        | metrics_benchmark
        | metrics_distribution
        | metrics_portfolio
    )
    metric_notes = {
        "Period frequency": "Sampling frequency used for annualization. Smaller periods are generally more granular (but can be noisier).",
        "Simulation start date": "First timestamp in the simulation index. Earlier start dates generally make estimates more statistically stable.",
        "Simulation end date": "Last timestamp in the simulation index. More recent end dates generally better reflect current market conditions.",
        "First transaction date": "First timestamp with any executed trade. Earlier is generally better (less time inactive), depending on the strategy.",
        "Annualized return %": "Geometric mean return annualized (decimal units). Higher is generally better, but interpret alongside risk and drawdowns.",
        "Annualized volatility": "Standard deviation of returns annualized (decimal units). Lower is generally better for a given return level.",
        "Annualized Sharpe": "Annualized excess return divided by annualized volatility. Higher is generally better (rule of thumb: >1 is good, >2 is strong).",
        "Max drawdown (equity) %": "Worst peak-to-trough % decline in equity. Less negative (closer to 0) is generally better.",
        "Max drawdown (PnL) %": "Worst drawdown of cumulative PnL relative to prior PnL peak. Less negative (closer to 0) is generally better.",
        "Total return %": "Ending equity / initial cash minus 1, expressed in percent. Higher is generally better.",
        "Funding earnings": "Sum of funding payments (positive means net earned). Higher is generally better; negative values mean funding cost.",
        "Fees": "Sum of trading fees paid. Lower is generally better.",
        "Annual turnover": "Average per-period turnover annualized (not percent). Lower is generally better (less trading/costs), unless the strategy requires frequent rebalancing.",
        "Total order count": "Count of non-zero notional orders executed. Lower generally means less trading (and costs), but too low can indicate inactivity.",
        "Average order notional": "Mean absolute notional per executed order. Good depends on liquidity and constraints; too large can be hard to execute.",
        "Gross exposure mean %": "Average sum(|positions|) as % of equity. Lower generally means less leverage; values above 100% indicate leveraged exposure.",
        "Gross exposure median %": "Median sum(|positions|) as % of equity. Lower generally means less leverage; values above 100% indicate leveraged exposure.",
        "Gross exposure max %": "Maximum sum(|positions|) as % of equity. Lower generally means tighter leverage control; very high peaks imply occasional high leverage.",
        "Net exposure mean %": "Average signed exposure as % of equity. Closer to 0 is generally more market-neutral; positive means net long, negative net short.",
        "Net exposure median %": "Median signed exposure as % of equity. Closer to 0 is generally more market-neutral; positive means net long, negative net short.",
        "Net exposure max %": "Max absolute signed exposure as % of equity. Lower absolute values generally mean better exposure control.",
        "Alpha": "Annualized intercept vs benchmark excess returns (CAPM-style). Higher is generally better; near 0 implies little outperformance after adjusting for beta.",
        "Beta": "Slope vs benchmark excess returns (CAPM-style). Values near 1 behave like the benchmark; values near 0 have low benchmark sensitivity.",
        "Tracking error": "Std dev of active returns annualized (decimal units). Lower means closer to the benchmark; higher means more active risk.",
        "Information ratio": "Active annual return divided by tracking error. Higher is generally better (rule of thumb: >0.5 is decent, >1 is strong).",
        "R2 vs benchmark": "Squared correlation of returns vs benchmark returns. Higher means the benchmark explains more of the returns; lower implies more idiosyncratic behavior.",
        "Benchmark annualized return %": "Benchmark geometric mean return annualized (percent units). Higher is generally better, but depends on your benchmark choice and sample.",
        "Active annual return %": "Arithmetic mean of (strategy - benchmark) period returns annualized (percent units). Higher is generally better; negative means underperformance vs the benchmark.",
        "Calmar ratio": "Annualized return divided by absolute max equity drawdown. Higher is generally better (more return per unit of drawdown).",
        "Skewness": "Skewness of period returns distribution. More positive skewness is often preferred (more upside tail), all else equal.",
        "Kurtosis": "Kurtosis of period returns distribution. Lower generally means fewer extreme tail events; high kurtosis suggests fat tails.",
        "Best period return": "Maximum single-period return. Higher is generally better, but interpret alongside worst-period and drawdowns.",
        "Worst period return": "Minimum single-period return. Less negative (closer to 0) is generally better (smaller tail losses).",
        "Hit rate": "Fraction of non-zero return periods that are positive. Higher is generally better, but can be low for strategies with infrequent large wins.",
        "Avg win": "Mean return of positive-return periods. Higher is generally better.",
        "Avg loss": "Mean return of negative-return periods. Less negative (closer to 0) is generally better.",
        "Profit factor": "Sum of wins divided by absolute sum of losses. Higher is generally better; values >1 mean wins outweigh losses.",
        "Max drawdown duration (periods)": "Longest consecutive underwater duration in periods. Shorter is generally better (capital recovers faster).",
        "Time to recovery (periods)": "Periods from drawdown peak to recovering the prior peak. Shorter is generally better.",
        "Average holding period": "Average consecutive periods with a non-zero position per asset. Good depends on the strategy; shorter implies more trading, longer implies lower turnover.",
        "Costs % gross pnl": "Fees+slippage as % of gross PnL (before costs). Lower is generally better; near 0 means costs are small relative to edge.",
        "Funding % total pnl": "Funding as % of net PnL. Lower absolute values are generally better; large magnitudes mean funding dominates PnL.",
        "Average funding settled": "Average funding payment per period. Positive is generally better; negative means funding paid on average.",
        "Max abs weight": "Maximum absolute target weight across assets/periods. Lower is generally better (less concentration/leverage), given the strategy's intent.",
        "Mean abs weight": "Mean absolute target weight across assets/periods. Lower is generally better (less aggregate risk), given the strategy's intent.",
    }
    tearsheet = pd.DataFrame(
        [{"Value": value, "Note": metric_notes.get(metric, "")} for metric, value in metrics.items()],
        index=pd.Index(metrics.keys(), name="Metric"),
        columns=["Value", "Note"],
    )

    return returns, tearsheet
