"""
Metrics calculation utilities for alphavec simulations.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

TEARSHEET_NOTES: Final[dict[str, str]] = {
    "Period frequency": "Sampling frequency used for annualization. Smaller periods are generally more granular (but can be noisier).",
    "Benchmark Asset": "Column name of the benchmark asset used for alpha/beta and benchmark charts (if provided).",
    "Fee %": "Trading fee rate applied to order notional (decimal units; e.g. 0.001 = 10 bps).",
    "Slippage %": "Slippage applied against the trader on execution prices (decimal units; e.g. 0.001 = 10 bps).",
    "Init Cash": "Initial cash (starting equity) used for the simulation.",
    "Trading Days Year": "Trading days per year used for annualization.",
    "Risk Free Rate": "Annual risk-free rate used for Sharpe/Sortino (decimal units).",
    "Simulation start date": "First timestamp in the simulation index. Earlier start dates generally make estimates more statistically stable.",
    "Simulation end date": "Last timestamp in the simulation index. More recent end dates generally better reflect current market conditions.",
    "First transaction date": "First timestamp with any executed trade. Earlier is generally better (less time inactive), depending on the strategy.",
    "Annualized return %": "Geometric mean return annualized (decimal units). Higher is generally better, but interpret alongside risk and drawdowns.",
    "Annualized volatility": "Sample standard deviation of returns annualized (decimal units). Lower is generally better for a given return level. Uses Bessel's correction (ddof=1) per industry standard.",
    "Annualized Sharpe": "Annualized excess return divided by annualized volatility (sample statistics). Higher is generally better (rule of thumb: >1 is good, >2 is strong).",
    "Max drawdown (equity) %": "Worst peak-to-trough % decline in equity. Less negative (closer to 0) is generally better.",
    "Max drawdown (PnL) %": "Worst drawdown of cumulative PnL relative to prior PnL peak. Less negative (closer to 0) is generally better.",
    "Total return %": "Ending equity / initial cash minus 1, expressed in percent. Higher is generally better.",
    "Funding earnings": "Sum of funding payments (positive means net earned). Higher is generally better; negative values mean funding cost.",
    "Fees": "Sum of trading fees paid. Lower is generally better.",
    "Annual turnover": "Average per-period one-sided turnover annualized (not percent), computed as min(total buys, total sells) / equity before trading. Lower is generally better (less trading/costs), unless the strategy requires frequent rebalancing.",
    "Total order count": "Count of non-zero notional orders executed. Lower generally means less trading (and costs), but too low can indicate inactivity.",
    "Average order notional": "Mean absolute notional per executed order. Good depends on liquidity and constraints; too large can be hard to execute.",
    "Gross exposure mean %": "Average sum(|positions|) as % of equity. Lower generally means less leverage; values above 100% indicate leveraged exposure.",
    "Gross exposure median %": "Median sum(|positions|) as % of equity. Lower generally means less leverage; values above 100% indicate leveraged exposure.",
    "Gross exposure max %": "Maximum sum(|positions|) as % of equity. Lower generally means tighter leverage control; very high peaks imply occasional high leverage.",
    "Net exposure mean %": "Average signed exposure as % of equity. Closer to 0 is generally more market-neutral; positive means net long, negative net short.",
    "Net exposure median %": "Median signed exposure as % of equity. Closer to 0 is generally more market-neutral; positive means net long, negative net short.",
    "Net exposure max %": "Max absolute signed exposure as % of equity. Lower absolute values generally mean better exposure control.",
    "Alpha": "Annualized intercept vs benchmark excess returns (CAPM-style, sample statistics). Higher is generally better; near 0 implies little outperformance after adjusting for beta.",
    "Beta": "Slope vs benchmark excess returns (CAPM-style, sample covariance/variance). Values near 1 behave like the benchmark; values near 0 have low benchmark sensitivity.",
    "Tracking error": "Sample std dev of active returns annualized (decimal units). Lower means closer to the benchmark; higher means more active risk.",
    "Information ratio": "Active annual return divided by tracking error. Higher is generally better (rule of thumb: >0.5 is decent, >1 is strong).",
    "R2 vs benchmark": "Squared correlation of returns vs benchmark returns. Higher means the benchmark explains more of the returns; lower implies more idiosyncratic behavior.",
    "Benchmark annualized return %": "Benchmark geometric mean return annualized (percent units). Higher is generally better, but depends on your benchmark choice and sample.",
    "Active annual return %": "Arithmetic mean of (strategy - benchmark) period returns annualized (percent units). Uses arithmetic (not geometric) mean to match tracking error calculation. Higher is generally better; negative means underperformance vs the benchmark.",
    "Calmar ratio": "Annualized return divided by absolute max equity drawdown. Higher is generally better (more return per unit of drawdown).",
    "Skewness": "Skewness of period returns distribution. More positive skewness is often preferred (more upside tail), all else equal.",
    "Kurtosis": "Excess kurtosis of period returns distribution (normal distribution = 0). Higher values indicate fatter tails; lower (negative) values indicate thinner tails.",
    "Best period return": "Maximum single-period return. Higher is generally better, but interpret alongside worst-period and drawdowns.",
    "Worst period return": "Minimum single-period return. Less negative (closer to 0) is generally better.",
    "Hit rate": "Fraction of non-zero return periods that are positive. Higher is generally better.",
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
    "Annualized Sortino": "Annualized excess return divided by annualized downside deviation. Higher is generally better; focuses on downside risk unlike Sharpe which penalizes upside volatility.",
    "Downside deviation": "Sample std dev of negative returns annualized (decimal units). Lower is generally better; measures downside risk only.",
    "VaR 95%": "Value at Risk at 95% confidence level (5th percentile of returns). Less negative (closer to 0) is generally better; worst expected loss in 19 out of 20 periods.",
    "CVaR 95%": "Conditional Value at Risk at 95% confidence level (mean of returns below VaR). Less negative (closer to 0) is generally better; average loss when VaR is exceeded.",
    "Omega Ratio": "Probability-weighted ratio of gains above threshold vs losses below threshold (uses 0 as threshold). Higher is generally better; values >1 mean gains outweigh losses.",
    "Gain-to-Pain Ratio": "Sum of returns divided by sum of absolute returns. Higher is generally better; measures return per unit of total volatility.",
    "Ulcer Index": "RMS (root mean square) of drawdowns, annualized. Lower is generally better; alternative drawdown-based risk measure that penalizes depth and duration.",
    "Weight IC mean (next)": "Time-average cross-sectional correlation between weights at t and next-period asset returns (close-to-close), computed over the active universe (non-zero weights) each period.",
    "Weight IC t-stat (next)": "t-stat of the time series of per-period weight IC values. Higher absolute values suggest more statistically reliable alignment (not a guarantee).",
    "Weight Rank IC mean (next)": "Time-average Spearman-style (rank) IC between weights and next-period asset returns, computed over the active universe (non-zero weights) each period.",
    "Weight Rank IC t-stat (next)": "t-stat of the time series of per-period weight rank IC values.",
    "Top-bottom decile spread mean (next)": "Time-average next-period return spread between the top and bottom weight deciles within the active universe (assets with non-zero weights) each period.",
    "Top-bottom decile spread t-stat (next)": "t-stat of the time series of top-minus-bottom decile spreads.",
    "Weighted long hit rate mean (next)": "Average fraction of long gross weight placed in assets that have positive next-period returns (weights within each period). Higher is generally better.",
    "Weighted short hit rate mean (next)": "Average fraction of short gross weight placed in assets that have negative next-period returns (weights within each period). Higher is generally better.",
    "Forward return per gross mean (next)": "Average of (Σ w_t,i r_{t+1,i}) / (Σ |w_t,i|) each period. Normalizes for varying leverage and compares return per unit of gross weight.",
    "Forward return selection per gross mean (next)": "Average of the cross-sectional selection component of Σ w_t,i r_{t+1,i}, normalized by gross weight (active universe, next-period). Higher is generally better.",
    "Forward return selection per gross t-stat (next)": "t-stat of the time series of per-period selection-per-gross values.",
    "Forward return directional per gross mean (next)": "Average of the directional component (net weight × mean next return of the active universe), normalized by gross weight (next-period). Magnitude near 0 indicates little directional dependence.",
    "Forward return directional per gross t-stat (next)": "t-stat of the time series of per-period directional-per-gross values.",
    "Gross weight mean": "Average gross weight (Σ |w_t,i|) across periods with available next returns. Higher implies more leverage/total exposure in the signal.",
    "Directionality mean": "Average net-to-gross ratio (Σ w_t,i) / (Σ |w_t,i|). Values near 0 indicate market-neutral; positive is net long; negative net short.",
}


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


def _t_stat(series: pd.Series) -> float:
    s = series.dropna()
    n = int(s.shape[0])
    if n < 2:
        return np.nan
    std = float(s.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return np.nan
    return float(s.mean() / (std / np.sqrt(n)))


def _weight_forward_diagnostics(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    w = weights.fillna(0.0).astype(float)
    fwd = close_prices.shift(-1).divide(close_prices).subtract(1.0)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)

    n_periods = int(len(w.index))
    spread = np.full(n_periods, np.nan, dtype=float)
    ic = np.full(n_periods, np.nan, dtype=float)
    rank_ic = np.full(n_periods, np.nan, dtype=float)

    port_fwd = np.full(n_periods, np.nan, dtype=float)
    port_fwd_per_gross = np.full(n_periods, np.nan, dtype=float)
    gross_w = np.full(n_periods, np.nan, dtype=float)
    net_w = np.full(n_periods, np.nan, dtype=float)
    directionality = np.full(n_periods, np.nan, dtype=float)

    sel = np.full(n_periods, np.nan, dtype=float)
    dirn = np.full(n_periods, np.nan, dtype=float)
    sel_per_gross = np.full(n_periods, np.nan, dtype=float)
    dirn_per_gross = np.full(n_periods, np.nan, dtype=float)

    long_hit_w = np.full(n_periods, np.nan, dtype=float)
    short_hit_w = np.full(n_periods, np.nan, dtype=float)

    long_gross_w = np.full(n_periods, np.nan, dtype=float)
    short_gross_w = np.full(n_periods, np.nan, dtype=float)
    long_ret_per_gross = np.full(n_periods, np.nan, dtype=float)
    short_ret_per_gross = np.full(n_periods, np.nan, dtype=float)

    dec_sum = np.zeros(10, dtype=float)
    dec_count = np.zeros(10, dtype=int)

    w_np = w.to_numpy(dtype=float)
    fwd_np = fwd.to_numpy(dtype=float)

    for t in range(n_periods):
        w_row = w_np[t]
        r_row = fwd_np[t]
        mask = np.isfinite(w_row) & np.isfinite(r_row) & (w_row != 0.0)
        if int(np.sum(mask)) < 2:
            continue

        w_m = w_row[mask]
        r_m = r_row[mask]

        if np.std(w_m) > 0.0 and np.std(r_m) > 0.0:
            ic[t] = float(np.corrcoef(w_m, r_m)[0, 1])

        w_ranks = pd.Series(w_m).rank(method="average").to_numpy(dtype=float)
        r_ranks = pd.Series(r_m).rank(method="average").to_numpy(dtype=float)
        if np.std(w_ranks) > 0.0 and np.std(r_ranks) > 0.0:
            rank_ic[t] = float(np.corrcoef(w_ranks, r_ranks)[0, 1])

        gross = float(np.sum(np.abs(w_m)))
        net = float(np.sum(w_m))
        gross_w[t] = gross
        net_w[t] = net
        directionality[t] = net / gross if gross != 0.0 else np.nan

        pfwd = float(np.sum(w_m * r_m))
        port_fwd[t] = pfwd
        port_fwd_per_gross[t] = pfwd / gross if gross != 0.0 else np.nan

        mean_w = float(np.mean(w_m))
        mean_r = float(np.mean(r_m))
        dir_component = net * mean_r
        sel_component = float(np.sum((w_m - mean_w) * (r_m - mean_r)))
        dirn[t] = dir_component
        sel[t] = sel_component
        dirn_per_gross[t] = dir_component / gross if gross != 0.0 else np.nan
        sel_per_gross[t] = sel_component / gross if gross != 0.0 else np.nan

        long_mask = w_m > 0.0
        if np.any(long_mask):
            long_w_abs = np.abs(w_m[long_mask])
            long_denom = float(long_w_abs.sum())
            long_gross_w[t] = long_denom
            if long_denom > 0.0:
                long_hit_w[t] = float(long_w_abs[r_m[long_mask] > 0.0].sum() / long_denom)
                long_ret_per_gross[t] = float(np.sum(w_m[long_mask] * r_m[long_mask]) / long_denom)

        short_mask = w_m < 0.0
        if np.any(short_mask):
            short_w_abs = np.abs(w_m[short_mask])
            short_denom = float(short_w_abs.sum())
            short_gross_w[t] = short_denom
            if short_denom > 0.0:
                short_hit_w[t] = float(short_w_abs[r_m[short_mask] < 0.0].sum() / short_denom)
                short_ret_per_gross[t] = float(np.sum(w_m[short_mask] * r_m[short_mask]) / short_denom)

        n = int(w_m.shape[0])
        if n >= 10:
            order = np.argsort(w_m, kind="mergesort")
            r_sorted = r_m[order]
            n = int(r_sorted.shape[0])
            dec = (np.arange(n) * 10) // n
            bottom = r_sorted[dec == 0]
            top = r_sorted[dec == 9]
            if bottom.size > 0 and top.size > 0:
                spread[t] = float(np.mean(top) - np.mean(bottom))

            for d in range(10):
                vals = r_sorted[dec == d]
                if vals.size == 0:
                    continue
                dec_sum[d] += float(np.sum(vals))
                dec_count[d] += int(vals.size)

    wf = pd.DataFrame(
        {
            "ic": pd.Series(ic, index=w.index),
            "rank_ic": pd.Series(rank_ic, index=w.index),
            "top_bottom_spread": pd.Series(spread, index=w.index),
            "forward_return": pd.Series(port_fwd, index=w.index),
            "forward_return_per_gross": pd.Series(port_fwd_per_gross, index=w.index),
            "forward_return_selection": pd.Series(sel, index=w.index),
            "forward_return_directional": pd.Series(dirn, index=w.index),
            "forward_return_selection_per_gross": pd.Series(sel_per_gross, index=w.index),
            "forward_return_directional_per_gross": pd.Series(dirn_per_gross, index=w.index),
            "gross_weight": pd.Series(gross_w, index=w.index),
            "net_weight": pd.Series(net_w, index=w.index),
            "directionality": pd.Series(directionality, index=w.index),
            "long_hit_weighted": pd.Series(long_hit_w, index=w.index),
            "short_hit_weighted": pd.Series(short_hit_w, index=w.index),
            "long_gross_weight": pd.Series(long_gross_w, index=w.index),
            "short_gross_weight": pd.Series(short_gross_w, index=w.index),
            "long_forward_return_per_gross": pd.Series(long_ret_per_gross, index=w.index),
            "short_forward_return_per_gross": pd.Series(short_ret_per_gross, index=w.index),
        },
        index=w.index,
    )

    denom = np.where(dec_count > 0, dec_count.astype(float), np.nan)
    decile_means = dec_sum / denom
    decile_curve = pd.Series(
        decile_means,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Mean next return",
    )
    return wf, decile_curve


def _metrics(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    returns: pd.Series,
    equity: pd.Series,
    init_cash: float,
    fee_pct: float,
    slippage_pct: float,
    freq_rule: str,
    trading_days_year: int,
    risk_free_rate: float,
    benchmark_asset: str | None,
    first_order_date: pd.Timestamp | None,
    fees_paid: np.ndarray,
    funding_earned: np.ndarray,
    turnover_ratio: np.ndarray,
    gross_exposure_ratio: np.ndarray,
    net_exposure_ratio: np.ndarray,
    order_count_period: np.ndarray,
    order_notional_sum_period: np.ndarray,
    slippage_paid: np.ndarray,
    positions_hist: np.ndarray,
) -> pd.DataFrame:
    """
    Build a metrics DataFrame from simulation outputs.
    """

    n_periods = int(len(returns))
    n_assets = int(weights.shape[1])

    wf, wf_deciles = _weight_forward_diagnostics(weights=weights, close_prices=close_prices)

    pnl_curve = equity - init_cash
    total_return_pct = float(equity.iloc[-1] / init_cash - 1.0)

    dd_equity = _max_drawdown(equity)
    dd_pnl = _max_drawdown_pnl(pnl_curve)

    annual_factor = _annualization_factor(freq_rule, trading_days_year)
    if n_periods > 0:
        annual_return = float((1.0 + returns).prod() ** (annual_factor / n_periods) - 1.0)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(annual_factor))
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

    drawdown = equity / equity.cummax() - 1.0
    underwater = drawdown < 0.0
    dd_groups = (underwater != underwater.shift()).cumsum()
    dd_durations = underwater.groupby(dd_groups).sum()
    max_drawdown_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    trough_ts = drawdown.idxmin()
    peak_value = float(equity.cummax().loc[trough_ts])
    peak_ts_candidates = equity.loc[:trough_ts]
    peak_ts = peak_ts_candidates[peak_ts_candidates == peak_value].index[-1]
    post_trough = equity.loc[trough_ts:]
    recovered = post_trough[post_trough >= peak_value]
    if len(recovered) > 0:
        recovery_ts = recovered.index[0]
        time_to_recovery = int((equity.loc[peak_ts:recovery_ts]).shape[0] - 1)
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
    net_pnl = float(equity.iloc[-1] - init_cash)
    gross_pnl = net_pnl + total_fees + total_slippage
    costs_pct_gross_pnl = (
        (total_fees + total_slippage) / abs(gross_pnl) * 100.0 if gross_pnl != 0 else np.nan
    )

    funding_total = float(funding_earned.sum())
    funding_pct_total_pnl = funding_total / net_pnl * 100.0 if net_pnl != 0 else np.nan
    average_funding_settled = float(np.nanmean(funding_earned))

    abs_weights = np.abs(weights.to_numpy(dtype=float))
    max_abs_weight = float(np.nanmax(abs_weights))
    mean_abs_weight = float(np.nanmean(abs_weights))

    # Additional risk metrics
    # Sortino Ratio - uses downside deviation instead of total volatility
    downside_returns = returns[returns < 0.0]
    if len(downside_returns) > 1:
        downside_deviation = float(downside_returns.std(ddof=1) * np.sqrt(annual_factor))
        annual_sortino = (
            (annual_return - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else np.nan
        )
    else:
        downside_deviation = 0.0
        annual_sortino = np.nan

    # VaR (Value at Risk) - 5th percentile (95% confidence)
    var_95 = float(returns.quantile(0.05))

    # CVaR (Conditional VaR) - mean of returns below VaR
    returns_below_var = returns[returns <= var_95]
    cvar_95 = float(returns_below_var.mean()) if len(returns_below_var) > 0 else var_95

    # Omega Ratio - probability-weighted gains/losses ratio (using 0 as threshold)
    threshold = 0.0
    gains = returns[returns > threshold]
    losses = returns[returns < threshold]
    gains_sum = float(gains.sum()) if len(gains) > 0 else 0.0
    losses_sum = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    omega_ratio = gains_sum / losses_sum if losses_sum > 0 else np.nan

    # Gain-to-Pain Ratio
    sum_returns = float(returns.sum())
    sum_abs_returns = float(returns.abs().sum())
    gain_to_pain = sum_returns / sum_abs_returns if sum_abs_returns > 0 else np.nan

    # Ulcer Index - RMS of drawdowns
    drawdown_pct = drawdown * 100.0  # Convert to percentage
    ulcer_index = float(
        np.sqrt(np.mean(drawdown_pct**2)) * np.sqrt(annual_factor / n_periods)
        if n_periods > 0
        else 0.0
    )

    alpha: float = np.nan
    beta: float = np.nan
    benchmark_annual_return: float = np.nan
    tracking_error: float = np.nan
    active_annual_return: float = np.nan
    information_ratio: float = np.nan
    r2: float = np.nan
    if benchmark_asset is not None:
        if benchmark_asset not in close_prices.columns:
            raise ValueError(f"benchmark_asset '{benchmark_asset}' not in weights columns.")
        bench_prices = close_prices[benchmark_asset].copy()
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
            var_x = float(np.var(x, ddof=1))
            if var_x > 0:
                cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
                beta = cov_xy / var_x
                alpha_per_period = float(y.mean() - beta * x.mean())
                alpha = alpha_per_period * annual_factor
            else:
                alpha = np.nan
                beta = np.nan
            active_returns = returns - bench_returns
            tracking_error = float(active_returns.std(ddof=1) * np.sqrt(annual_factor))
            active_annual_return = float(active_returns.mean() * annual_factor)
            information_ratio = (
                active_annual_return / tracking_error if tracking_error > 0 else np.nan
            )
            corr = float(returns.corr(bench_returns))
            r2 = corr * corr if np.isfinite(corr) else np.nan

    metrics_meta = {
        "Period frequency": freq_rule,
        "Benchmark Asset": benchmark_asset,
        "Fee %": float(fee_pct),
        "Slippage %": float(slippage_pct),
        "Init Cash": float(init_cash),
        "Trading Days Year": int(trading_days_year),
        "Risk Free Rate": float(risk_free_rate),
        "Simulation start date": weights.index.min(),
        "Simulation end date": weights.index.max(),
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
    metrics_risk = {
        "Annualized Sortino": annual_sortino,
        "Downside deviation": downside_deviation,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Omega Ratio": omega_ratio,
        "Gain-to-Pain Ratio": gain_to_pain,
        "Ulcer Index": ulcer_index,
    }
    metrics_weight_vs_next = {
        "Weight IC mean (next)": float(wf["ic"].mean(skipna=True)),
        "Weight IC t-stat (next)": _t_stat(wf["ic"]),
        "Weight Rank IC mean (next)": float(wf["rank_ic"].mean(skipna=True)),
        "Weight Rank IC t-stat (next)": _t_stat(wf["rank_ic"]),
        "Top-bottom decile spread mean (next)": float(wf["top_bottom_spread"].mean(skipna=True)),
        "Top-bottom decile spread t-stat (next)": _t_stat(wf["top_bottom_spread"]),
        "Weighted long hit rate mean (next)": float(wf["long_hit_weighted"].mean(skipna=True)),
        "Weighted short hit rate mean (next)": float(wf["short_hit_weighted"].mean(skipna=True)),
        "Forward return per gross mean (next)": float(wf["forward_return_per_gross"].mean(skipna=True)),
        "Forward return selection per gross mean (next)": float(
            wf["forward_return_selection_per_gross"].mean(skipna=True)
        ),
        "Forward return selection per gross t-stat (next)": _t_stat(
            wf["forward_return_selection_per_gross"]
        ),
        "Forward return directional per gross mean (next)": float(
            wf["forward_return_directional_per_gross"].mean(skipna=True)
        ),
        "Forward return directional per gross t-stat (next)": _t_stat(
            wf["forward_return_directional_per_gross"]
        ),
        "Gross weight mean": float(wf["gross_weight"].mean(skipna=True)),
        "Directionality mean": float(wf["directionality"].mean(skipna=True)),
    }

    # Build complete metrics with categories
    all_metrics = [
        ("Meta", metrics_meta),
        ("Performance", metrics_performance),
        ("Costs", metrics_costs_and_trading),
        ("Exposure", metrics_exposure),
        ("Benchmark", metrics_benchmark),
        ("Distribution", metrics_distribution),
        ("Portfolio", metrics_portfolio),
        ("Risk", metrics_risk),
        ("Signal", metrics_weight_vs_next),
    ]

    rows = []
    for category, metrics_dict in all_metrics:
        for metric_name, value in metrics_dict.items():
            rows.append(
                {
                    "Category": category,
                    "Value": value,
                    "Note": TEARSHEET_NOTES.get(metric_name, ""),
                }
            )

    df = pd.DataFrame(
        rows,
        index=pd.Index([m for _, md in all_metrics for m in md.keys()], name="Metric"),
        columns=["Category", "Value", "Note"],
    )
    df.attrs["weight_forward"] = wf
    df.attrs["weight_forward_deciles"] = wf_deciles
    return df
