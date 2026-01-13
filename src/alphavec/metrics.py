"""
Metrics calculation utilities for alphavec simulations.
"""

from __future__ import annotations

import re
from typing import Final, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class MetricsAccessor(Protocol):
    """
    Protocol for unified metric extraction across result types.

    All result types (SimulationResult, GridSearchResults, WalkForwardResult)
    implement this protocol for consistent metric access.

    Example:
        >>> def print_sharpe(result: MetricsAccessor) -> None:
        ...     sharpe = result.metric_value(MetricKey.ANNUALIZED_SHARPE)
        ...     print(f"Sharpe: {sharpe}")
    """

    def metric_value(self, metric: str, *, default: object = None) -> object:
        """
        Get the value of a specific metric.

        Args:
            metric: The metric key (use MetricKey constants for type safety).
            default: Value to return if metric is not found or is NaN.

        Returns:
            The metric value, or default if not found.
        """
        ...

    def available_metrics(self, category: str | None = None) -> list[str]:
        """
        Return available metric keys.

        Args:
            category: Optional category filter (e.g., "Performance", "Risk").

        Returns:
            List of metric key strings.
        """
        ...

    def metrics_dict(self, category: str | None = None) -> dict[str, object]:
        """
        Return metrics as a dictionary.

        Args:
            category: Optional category filter.

        Returns:
            Dictionary mapping metric keys to values.
        """
        ...


class MetricKey:
    """
    Constants for all available metric keys.

    Use these constants instead of hardcoding strings when extracting metrics
    from any result type that implements `MetricsAccessor` (SimulationResult,
    GridSearchResults, WalkForwardResult).

    Example:
        >>> # Works with any MetricsAccessor
        >>> sharpe = result.metric_value(MetricKey.ANNUALIZED_SHARPE)
        >>> max_dd = result.metric_value(MetricKey.MAX_DRAWDOWN_EQUITY_PCT)
        >>> perf = result.metrics_dict("Performance")

    Discovery:
        >>> MetricKey.all_keys()           # List all metric keys
        >>> MetricKey.keys_by_category()   # Keys grouped by category
        >>> result.available_metrics()     # Keys available in result
        >>> result.available_metrics("Risk")  # Filter by category

    Categories:
        Meta, Performance, Costs, Exposure, Benchmark, Distribution,
        Portfolio, Risk, Signal
    """

    # --- Meta ---
    PERIOD_FREQUENCY: Final[str] = "Period frequency"
    BENCHMARK_ASSET: Final[str] = "Benchmark Asset"
    FEE_PCT: Final[str] = "Fee %"
    SLIPPAGE_PCT: Final[str] = "Slippage %"
    INIT_CASH: Final[str] = "Init Cash"
    TRADING_DAYS_YEAR: Final[str] = "Trading Days Year"
    RISK_FREE_RATE: Final[str] = "Risk Free Rate"
    SIMULATION_START_DATE: Final[str] = "Simulation start date"
    SIMULATION_END_DATE: Final[str] = "Simulation end date"
    FIRST_TRANSACTION_DATE: Final[str] = "First transaction date"

    # --- Performance ---
    ANNUALIZED_RETURN_PCT: Final[str] = "Annualized return %"
    ANNUALIZED_VOLATILITY: Final[str] = "Annualized volatility"
    ANNUALIZED_SHARPE: Final[str] = "Annualized Sharpe"
    MAX_DRAWDOWN_EQUITY_PCT: Final[str] = "Max drawdown (equity) %"
    TOTAL_RETURN_PCT: Final[str] = "Total return %"
    TOTAL_RETURN: Final[str] = "Total return"

    # --- Costs ---
    FUNDING_EARNINGS: Final[str] = "Funding earnings"
    FUNDING_PCT_TOTAL_RETURN: Final[str] = "Funding % total return"
    FEES: Final[str] = "Fees"
    DAILY_TURNOVER_1WAY: Final[str] = "Daily Turnover (1-way)"
    ANNUAL_TURNOVER_1WAY: Final[str] = "Annual Turnover (1-way)"
    DAILY_WEIGHT_TURNOVER: Final[str] = "Daily Weight-based Turnover"
    ANNUAL_WEIGHT_TURNOVER: Final[str] = "Annual Weight-based Turnover"
    TOTAL_ORDER_COUNT: Final[str] = "Total order count"
    AVERAGE_ORDER_NOTIONAL: Final[str] = "Average order notional"

    # --- Exposure ---
    GROSS_EXPOSURE_MEAN_PCT: Final[str] = "Gross exposure mean %"
    GROSS_EXPOSURE_MAX_PCT: Final[str] = "Gross exposure max %"
    NET_EXPOSURE_MEAN_PCT: Final[str] = "Net exposure mean %"
    NET_EXPOSURE_MAX_PCT: Final[str] = "Net exposure max %"

    # --- Benchmark ---
    ALPHA: Final[str] = "Alpha"
    BETA: Final[str] = "Beta"
    BENCHMARK_ANNUALIZED_RETURN_PCT: Final[str] = "Benchmark annualized return %"
    ACTIVE_ANNUAL_RETURN_PCT: Final[str] = "Active annual return %"
    TRACKING_ERROR: Final[str] = "Tracking error"
    INFORMATION_RATIO: Final[str] = "Information ratio"
    R2_VS_BENCHMARK: Final[str] = "R2 vs benchmark"

    # --- Distribution ---
    CALMAR_RATIO: Final[str] = "Calmar ratio"
    SKEWNESS: Final[str] = "Skewness"
    KURTOSIS: Final[str] = "Kurtosis"
    BEST_PERIOD_RETURN: Final[str] = "Best period return"
    WORST_PERIOD_RETURN: Final[str] = "Worst period return"
    HIT_RATE: Final[str] = "Hit rate"
    PROFIT_FACTOR: Final[str] = "Profit factor"
    MAX_DRAWDOWN_DURATION_PERIODS: Final[str] = "Max drawdown duration (periods)"
    TIME_TO_RECOVERY_PERIODS: Final[str] = "Time to recovery (periods)"

    # --- Portfolio ---
    AVERAGE_HOLDING_PERIOD: Final[str] = "Average holding period"
    COSTS_PCT_GROSS_PNL: Final[str] = "Costs % gross pnl"
    FUNDING_PCT_TOTAL_PNL: Final[str] = "Funding % total pnl"
    AVERAGE_FUNDING_SETTLED: Final[str] = "Average funding settled"
    MAX_ABS_WEIGHT: Final[str] = "Max abs weight"

    # --- Risk ---
    ANNUALIZED_SORTINO: Final[str] = "Annualized Sortino"
    DOWNSIDE_DEVIATION: Final[str] = "Downside deviation"
    VAR_95_PCT: Final[str] = "VaR 95%"
    CVAR_95_PCT: Final[str] = "CVaR 95%"
    OMEGA_RATIO: Final[str] = "Omega Ratio"
    ULCER_INDEX: Final[str] = "Ulcer Index"

    # --- Signal ---
    WEIGHT_IC_MEAN_NEXT: Final[str] = "Weight IC mean (next)"
    WEIGHT_IC_TSTAT_NEXT: Final[str] = "Weight IC t-stat (next)"
    WEIGHT_RANK_IC_MEAN_NEXT: Final[str] = "Weight Rank IC mean (next)"
    WEIGHT_RANK_IC_TSTAT_NEXT: Final[str] = "Weight Rank IC t-stat (next)"
    TOP_BOTTOM_DECILE_SPREAD_MEAN_NEXT: Final[str] = "Top-bottom decile spread mean (next)"
    TOP_BOTTOM_DECILE_SPREAD_TSTAT_NEXT: Final[str] = "Top-bottom decile spread t-stat (next)"
    WEIGHTED_LONG_HIT_RATE_MEAN_NEXT: Final[str] = "Weighted long hit rate mean (next)"
    WEIGHTED_SHORT_HIT_RATE_MEAN_NEXT: Final[str] = "Weighted short hit rate mean (next)"
    FORWARD_RETURN_PER_GROSS_MEAN_NEXT: Final[str] = "Forward return per gross mean (next)"
    FORWARD_RETURN_SELECTION_PER_GROSS_MEAN_NEXT: Final[str] = (
        "Forward return selection per gross mean (next)"
    )
    FORWARD_RETURN_SELECTION_PER_GROSS_TSTAT_NEXT: Final[str] = (
        "Forward return selection per gross t-stat (next)"
    )
    GROSS_WEIGHT_MEAN: Final[str] = "Gross weight mean"
    DIRECTIONALITY_MEAN: Final[str] = "Directionality mean"
    # Turnover-adjusted signal metrics
    TURNOVER_ADJUSTED_IC_MEAN_NEXT: Final[str] = "Turnover-adjusted IC mean (next)"
    NET_FORWARD_RETURN_PER_GROSS_MEAN_NEXT: Final[str] = "Net forward return per gross mean (next)"

    @classmethod
    def all_keys(cls) -> list[str]:
        """Return a list of all available metric keys."""
        return [
            v
            for k, v in vars(cls).items()
            if isinstance(v, str) and not k.startswith("_") and k.isupper()
        ]

    @classmethod
    def keys_by_category(cls) -> dict[str, list[str]]:
        """Return metric keys organized by category."""
        return {
            "Meta": [
                cls.PERIOD_FREQUENCY,
                cls.BENCHMARK_ASSET,
                cls.FEE_PCT,
                cls.SLIPPAGE_PCT,
                cls.INIT_CASH,
                cls.TRADING_DAYS_YEAR,
                cls.RISK_FREE_RATE,
                cls.SIMULATION_START_DATE,
                cls.SIMULATION_END_DATE,
                cls.FIRST_TRANSACTION_DATE,
            ],
            "Performance": [
                cls.ANNUALIZED_RETURN_PCT,
                cls.ANNUALIZED_VOLATILITY,
                cls.ANNUALIZED_SHARPE,
                cls.MAX_DRAWDOWN_EQUITY_PCT,
                cls.TOTAL_RETURN_PCT,
                cls.TOTAL_RETURN,
            ],
            "Costs": [
                cls.FUNDING_EARNINGS,
                cls.FUNDING_PCT_TOTAL_RETURN,
                cls.FEES,
                cls.DAILY_TURNOVER_1WAY,
                cls.ANNUAL_TURNOVER_1WAY,
                cls.DAILY_WEIGHT_TURNOVER,
                cls.ANNUAL_WEIGHT_TURNOVER,
                cls.TOTAL_ORDER_COUNT,
                cls.AVERAGE_ORDER_NOTIONAL,
            ],
            "Exposure": [
                cls.GROSS_EXPOSURE_MEAN_PCT,
                cls.GROSS_EXPOSURE_MAX_PCT,
                cls.NET_EXPOSURE_MEAN_PCT,
                cls.NET_EXPOSURE_MAX_PCT,
            ],
            "Benchmark": [
                cls.ALPHA,
                cls.BETA,
                cls.BENCHMARK_ANNUALIZED_RETURN_PCT,
                cls.ACTIVE_ANNUAL_RETURN_PCT,
                cls.TRACKING_ERROR,
                cls.INFORMATION_RATIO,
                cls.R2_VS_BENCHMARK,
            ],
            "Distribution": [
                cls.CALMAR_RATIO,
                cls.SKEWNESS,
                cls.KURTOSIS,
                cls.BEST_PERIOD_RETURN,
                cls.WORST_PERIOD_RETURN,
                cls.HIT_RATE,
                cls.PROFIT_FACTOR,
                cls.MAX_DRAWDOWN_DURATION_PERIODS,
                cls.TIME_TO_RECOVERY_PERIODS,
            ],
            "Portfolio": [
                cls.AVERAGE_HOLDING_PERIOD,
                cls.COSTS_PCT_GROSS_PNL,
                cls.FUNDING_PCT_TOTAL_PNL,
                cls.AVERAGE_FUNDING_SETTLED,
                cls.MAX_ABS_WEIGHT,
            ],
            "Risk": [
                cls.ANNUALIZED_SORTINO,
                cls.DOWNSIDE_DEVIATION,
                cls.VAR_95_PCT,
                cls.CVAR_95_PCT,
                cls.OMEGA_RATIO,
                cls.ULCER_INDEX,
            ],
            "Signal": [
                cls.WEIGHT_IC_MEAN_NEXT,
                cls.WEIGHT_IC_TSTAT_NEXT,
                cls.WEIGHT_RANK_IC_MEAN_NEXT,
                cls.WEIGHT_RANK_IC_TSTAT_NEXT,
                cls.TOP_BOTTOM_DECILE_SPREAD_MEAN_NEXT,
                cls.TOP_BOTTOM_DECILE_SPREAD_TSTAT_NEXT,
                cls.WEIGHTED_LONG_HIT_RATE_MEAN_NEXT,
                cls.WEIGHTED_SHORT_HIT_RATE_MEAN_NEXT,
                cls.FORWARD_RETURN_PER_GROSS_MEAN_NEXT,
                cls.FORWARD_RETURN_SELECTION_PER_GROSS_MEAN_NEXT,
                cls.FORWARD_RETURN_SELECTION_PER_GROSS_TSTAT_NEXT,
                cls.GROSS_WEIGHT_MEAN,
                cls.DIRECTIONALITY_MEAN,
                cls.TURNOVER_ADJUSTED_IC_MEAN_NEXT,
                cls.NET_FORWARD_RETURN_PER_GROSS_MEAN_NEXT,
            ],
        }


class SignalArtifacts:
    """
    Signal-related artifacts derived from weights vs next returns.
    """

    def __init__(self, attrs: dict[str, object]):
        self._attrs = attrs

    @property
    def weight_forward(self) -> pd.DataFrame | None:
        return self._attrs.get("weight_forward")

    @property
    def weight_forward_deciles(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles")

    @property
    def weight_forward_deciles_median(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_median")

    @property
    def weight_forward_deciles_std(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_std")

    @property
    def weight_forward_deciles_count(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_count")

    @property
    def weight_forward_deciles_contrib(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_contrib")

    @property
    def weight_forward_deciles_contrib_long(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_contrib_long")

    @property
    def weight_forward_deciles_contrib_short(self) -> pd.Series | None:
        return self._attrs.get("weight_forward_deciles_contrib_short")

    @property
    def alpha_decay_next_return_by_type(self) -> pd.DataFrame | None:
        return self._attrs.get("alpha_decay_next_return_by_type")

    @property
    def decay_horizons(self) -> pd.DataFrame | None:
        """Multi-horizon signal decay (IC, Rank IC, decile spread by forward horizon)."""
        return self._attrs.get("signal_decay_horizons")

    @property
    def by_asset(self) -> pd.DataFrame | None:
        """Per-asset signal breakdown (IC, hit rate, contribution per asset)."""
        return self._attrs.get("signal_by_asset")

    # ── Convenience time series accessors ────────────────────────────────────

    @property
    def ic_series(self) -> pd.Series | None:
        """Per-period Information Coefficient (Pearson correlation)."""
        wf = self.weight_forward
        if wf is None or "ic" not in wf.columns:
            return None
        return wf["ic"].rename("ic")

    @property
    def rank_ic_series(self) -> pd.Series | None:
        """Per-period Spearman Rank IC."""
        wf = self.weight_forward
        if wf is None or "rank_ic" not in wf.columns:
            return None
        return wf["rank_ic"].rename("rank_ic")

    @property
    def decile_spread_series(self) -> pd.Series | None:
        """Per-period top-bottom decile spread."""
        wf = self.weight_forward
        if wf is None or "top_bottom_spread" not in wf.columns:
            return None
        return wf["top_bottom_spread"].rename("decile_spread")

    @property
    def directionality_series(self) -> pd.Series | None:
        """Per-period net/gross ratio (market neutrality measure)."""
        wf = self.weight_forward
        if wf is None or "directionality" not in wf.columns:
            return None
        return wf["directionality"].rename("directionality")

    @property
    def forward_return_per_gross_series(self) -> pd.Series | None:
        """Per-period forward return normalized by gross weight."""
        wf = self.weight_forward
        if wf is None or "forward_return_per_gross" not in wf.columns:
            return None
        return wf["forward_return_per_gross"].rename("forward_return_per_gross")

    @property
    def long_hit_rate_series(self) -> pd.Series | None:
        """Per-period weighted long hit rate."""
        wf = self.weight_forward
        if wf is None or "long_hit_weighted" not in wf.columns:
            return None
        return wf["long_hit_weighted"].rename("long_hit_rate")

    @property
    def short_hit_rate_series(self) -> pd.Series | None:
        """Per-period weighted short hit rate."""
        wf = self.weight_forward
        if wf is None or "short_hit_weighted" not in wf.columns:
            return None
        return wf["short_hit_weighted"].rename("short_hit_rate")

    @property
    def gross_weight_series(self) -> pd.Series | None:
        """Per-period total gross weight."""
        wf = self.weight_forward
        if wf is None or "gross_weight" not in wf.columns:
            return None
        return wf["gross_weight"].rename("gross_weight")

    def rolling(
        self,
        window: int = 90,
        min_periods: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Compute rolling-window signal metrics.

        This is computed lazily on each call (not stored in attrs) since
        users may want different window sizes.

        Args:
            window: Rolling window size in periods (default: 90)
            min_periods: Minimum observations required (default: window // 3)

        Returns:
            DataFrame with columns:
            - ic_mean: Rolling mean IC
            - ic_tstat: Rolling t-stat of IC
            - rank_ic_mean: Rolling mean Rank IC
            - rank_ic_tstat: Rolling t-stat of Rank IC
            - decile_spread_mean: Rolling mean decile spread
            - decile_spread_tstat: Rolling t-stat of decile spread

        Example:
            >>> rolling = result.artifacts.signal.rolling(window=252)
            >>> rolling["ic_mean"].plot(title="Rolling 1Y IC")
        """
        wf = self.weight_forward
        if wf is None or len(wf) < window:
            return None

        if min_periods is None:
            min_periods = max(3, window // 3)

        result_cols: dict[str, pd.Series] = {}

        for col, prefix in [
            ("ic", "ic"),
            ("rank_ic", "rank_ic"),
            ("top_bottom_spread", "decile_spread"),
        ]:
            if col not in wf.columns:
                result_cols[f"{prefix}_mean"] = pd.Series(np.nan, index=wf.index)
                result_cols[f"{prefix}_tstat"] = pd.Series(np.nan, index=wf.index)
                continue

            series = wf[col]
            rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
            rolling_count = series.rolling(window=window, min_periods=min_periods).count()

            # t-stat = mean / (std / sqrt(n))
            rolling_tstat = rolling_mean / (rolling_std / np.sqrt(rolling_count))

            result_cols[f"{prefix}_mean"] = rolling_mean
            result_cols[f"{prefix}_tstat"] = rolling_tstat

        return pd.DataFrame(result_cols, index=wf.index)

    def get(self, key: str, default: object | None = None) -> object:
        return self._attrs.get(key, default)


class MetricsArtifacts:
    """
    Typed accessors for metrics.attrs artifacts.
    """

    def __init__(self, attrs: dict[str, object]):
        self._attrs = attrs
        self._signal = SignalArtifacts(attrs)

    @property
    def returns(self) -> pd.Series | None:
        return self._attrs.get("returns")

    @property
    def returns_pct(self) -> pd.Series | None:
        return self._attrs.get("returns_pct")

    @property
    def equity(self) -> pd.Series | None:
        return self._attrs.get("equity")

    @property
    def equity_pct(self) -> pd.Series | None:
        return self._attrs.get("equity_pct")

    @property
    def drawdown_pct(self) -> pd.Series | None:
        return self._attrs.get("drawdown_pct")

    @property
    def init_cash(self) -> float | None:
        return self._attrs.get("init_cash")

    @property
    def benchmark_equity(self) -> pd.Series | None:
        return self._attrs.get("benchmark_equity")

    @property
    def benchmark_equity_pct(self) -> pd.Series | None:
        return self._attrs.get("benchmark_equity_pct")

    @property
    def transaction_costs(self) -> pd.Series | None:
        return self._attrs.get("transaction_costs")

    @property
    def turnover(self) -> pd.Series | None:
        """One-way turnover ratio per period."""
        return self._attrs.get("turnover")

    @property
    def weight_turnover(self) -> pd.Series | None:
        """Weight-based turnover per period (0.5 * sum of abs weight changes)."""
        return self._attrs.get("weight_turnover")

    @property
    def n_positions(self) -> pd.Series | None:
        return self._attrs.get("n_positions")

    @property
    def net_exposure(self) -> pd.Series | None:
        return self._attrs.get("net_exposure")

    @property
    def gross_exposure(self) -> pd.Series | None:
        return self._attrs.get("gross_exposure")

    @property
    def long_exposure(self) -> pd.Series | None:
        return self._attrs.get("long_exposure")

    @property
    def short_exposure(self) -> pd.Series | None:
        return self._attrs.get("short_exposure")

    @property
    def concentration(self) -> pd.Series | None:
        return self._attrs.get("concentration")

    @property
    def signal(self) -> SignalArtifacts:
        return self._signal

    def rolling_sharpe(self, window: int = 30) -> pd.Series | None:
        return self._attrs.get(f"rolling_sharpe_{window}")

    def get(self, key: str, default: object | None = None) -> object:
        return self._attrs.get(key, default)


def metrics_artifacts(metrics: pd.DataFrame) -> MetricsArtifacts:
    """
    Return a typed accessor for metrics.attrs artifacts.
    """
    return MetricsArtifacts(metrics.attrs)


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
    "Total return %": "Ending equity / initial cash minus 1, expressed in percent. Higher is generally better.",
    "Total return": "Ending equity minus initial cash, expressed in absolute currency. Higher is generally better; this is the net profit or loss.",
    "Funding earnings": "Sum of funding payments (positive means net earned). Higher is generally better; negative values mean funding cost.",
    "Fees": "Sum of trading fees paid. Lower is generally better.",
    "Daily Turnover (1-way)": "Mean daily one-sided turnover ratio, computed as min(total buys, total sells) / equity before trading. Lower is generally better (less trading/costs).",
    "Annual Turnover (1-way)": "Annualized one-sided turnover ratio. Lower is generally better (less trading/costs), unless the strategy requires frequent rebalancing.",
    "Daily Weight-based Turnover": "Mean daily weight-based turnover: 0.5 × Σ|w_t - w_{t-1}|, where w = notional/equity. Measures fraction of portfolio rebalanced per period.",
    "Annual Weight-based Turnover": "Annualized weight-based turnover. For a 2× gross market-neutral portfolio fully rotating, this yields ~2.0 (200% of equity traded).",
    "Total order count": "Count of non-zero notional orders executed. Lower generally means less trading (and costs), but too low can indicate inactivity.",
    "Average order notional": "Mean absolute notional per executed order. Good depends on liquidity and constraints; too large can be hard to execute.",
    "Gross exposure mean %": "Average sum(|positions|) as % of equity. Lower generally means less leverage; values above 100% indicate leveraged exposure.",
    "Gross exposure max %": "Maximum sum(|positions|) as % of equity. Lower generally means tighter leverage control; very high peaks imply occasional high leverage.",
    "Net exposure mean %": "Average signed exposure as % of equity. Closer to 0 is generally more market-neutral; positive means net long, negative net short.",
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
    "Profit factor": "Sum of wins divided by absolute sum of losses. Higher is generally better; values >1 mean wins outweigh losses.",
    "Max drawdown duration (periods)": "Longest consecutive underwater duration in periods. Shorter is generally better (capital recovers faster).",
    "Time to recovery (periods)": "Periods from drawdown peak to recovering the prior peak. Shorter is generally better.",
    "Average holding period": "Average consecutive periods with a non-zero position per asset. Good depends on the strategy; shorter implies more trading, longer implies lower turnover.",
    "Costs % gross pnl": "Fees+slippage as % of gross PnL (before costs). Lower is generally better; near 0 means costs are small relative to edge.",
    "Funding % total pnl": "Funding as % of net PnL. Lower absolute values are generally better; large magnitudes mean funding dominates PnL.",
    "Funding % total return": "Funding earnings as % of total return. Shows funding's contribution to the net result; values above 100% indicate trading/costs detracted from funding gains; negative values indicate funding costs were offset by trading gains.",
    "Average funding settled": "Average funding payment per period. Positive is generally better; negative means funding paid on average.",
    "Max abs weight": "Maximum absolute target weight across assets/periods. Lower is generally better (less concentration/leverage), given the strategy's intent.",
    "Annualized Sortino": "Annualized excess return divided by annualized downside deviation. Higher is generally better; focuses on downside risk unlike Sharpe which penalizes upside volatility.",
    "Downside deviation": "Sample std dev of negative returns annualized (decimal units). Lower is generally better; measures downside risk only.",
    "VaR 95%": "Value at Risk at 95% confidence level (5th percentile of returns). Less negative (closer to 0) is generally better; worst expected loss in 19 out of 20 periods.",
    "CVaR 95%": "Conditional Value at Risk at 95% confidence level (mean of returns below VaR). Less negative (closer to 0) is generally better; average loss when VaR is exceeded.",
    "Omega Ratio": "Probability-weighted ratio of gains above threshold vs losses below threshold (uses 0 as threshold). Higher is generally better; values >1 mean gains outweigh losses.",
    "Ulcer Index": "RMS (root mean square) of drawdowns, annualized. Lower is generally better; alternative drawdown-based risk measure that penalizes depth and duration.",
    "Weight IC mean (next)": "Time-average cross-sectional Pearson correlation between weights at t and next-period asset returns (close-to-close), computed over the active universe (non-zero weights) each period.",
    "Weight IC t-stat (next)": "t-stat of the time series of per-period weight IC values. Higher absolute values suggest more statistically reliable alignment (not a guarantee).",
    "Weight Rank IC mean (next)": "Time-average cross-sectional Spearman rank correlation between weights at t and next-period asset returns. More robust to outliers than Pearson IC.",
    "Weight Rank IC t-stat (next)": "t-stat of the time series of per-period Rank IC values. Higher absolute values suggest more statistically reliable ranking alignment.",
    "Top-bottom decile spread mean (next)": "Time-average next-period return spread between the top and bottom weight deciles within the active universe (assets with non-zero weights) each period.",
    "Top-bottom decile spread t-stat (next)": "t-stat of the time series of top-minus-bottom decile spreads.",
    "Weighted long hit rate mean (next)": "Average fraction of long gross weight placed in assets that have positive next-period returns (weights within each period). Higher is generally better.",
    "Weighted short hit rate mean (next)": "Average fraction of short gross weight placed in assets that have negative next-period returns (weights within each period). Higher is generally better.",
    "Forward return per gross mean (next)": "Average of (Σ w_t,i r_{t+1,i}) / (Σ |w_t,i|) each period. Normalizes for varying leverage and compares return per unit of gross weight.",
    "Forward return selection per gross mean (next)": "Average of the cross-sectional selection component of Σ w_t,i r_{t+1,i}, normalized by gross weight (active universe, next-period). Higher is generally better.",
    "Forward return selection per gross t-stat (next)": "t-stat of the time series of per-period selection-per-gross values.",
    "Gross weight mean": "Average gross weight (Σ |w_t,i|) across periods with available next returns. Higher implies more leverage/total exposure in the signal.",
    "Directionality mean": "Average net-to-gross ratio (Σ w_t,i) / (Σ |w_t,i|). Values near 0 indicate market-neutral; positive is net long; negative net short.",
    "Turnover-adjusted IC mean (next)": "IC adjusted for trading costs: IC × (1 - turnover × cost_per_turnover). Lower values indicate signal quality erodes significantly after costs.",
    "Net forward return per gross mean (next)": "Forward return per gross minus estimated trading costs (turnover × cost_per_turnover). Negative values indicate costs exceed gross returns.",
}

_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


def _format_tearsheet_note(note: str, *, include_definition: bool = True) -> str:
    s = str(note or "").strip()
    if not s:
        return ""
    if not include_definition:
        return f"Interpretation: {s}"
    parts = _SENTENCE_SPLIT_RE.split(s, maxsplit=1)
    definition = parts[0].strip()
    interpretation = parts[1].strip() if len(parts) > 1 else "See definition."
    return f"Definition: {definition}\nInterpretation: {interpretation}"


def _max_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    return float(dd.min())


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


def _alpha_decay_next_return_by_type(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Compute a simple "alpha decay" curve by lagging the weights and evaluating them against
    next-period close-to-close returns.

    The metric is the mean next-period return per unit gross weight, decomposed into:
    - total: sum(w * r) / gross
    - selection: (sum(w * r) - sum(w) * mean(r)) / gross
    - directional: (sum(w) * mean(r)) / gross

    This treats "decay" as the loss of edge when acting on the signal with a delay (lag).

    Note: Forward returns are computed as close[t+1]/close[t] - 1. For 24/7 markets (e.g. crypto),
    this accurately approximates the return from execution at open[t+1] to close[t+1], since
    open[t+1] ≈ close[t] in continuous trading. For markets with overnight gaps, this may
    overstate or understate signal quality depending on gap direction relative to the signal.
    """

    w = weights.fillna(0.0).astype(float)
    fwd = close_prices.shift(-1).divide(close_prices).subtract(1.0)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)

    n_periods = int(len(w.index))
    max_lag = int(max_lag)
    if max_lag < 0:
        max_lag = 0
    max_lag = min(max_lag, max(0, n_periods - 2))

    rows: list[dict[str, float]] = []
    for lag in range(max_lag + 1):
        w_lag = w.shift(lag).fillna(0.0)
        active = (w_lag != 0.0) & fwd.notna()

        w_act = w_lag.where(active, 0.0)
        r_act = fwd.where(active, 0.0)

        n_active = active.sum(axis=1).astype(float)
        gross = w_act.abs().sum(axis=1)
        s_w = w_act.sum(axis=1)
        s_r = r_act.sum(axis=1)
        s_wr = (w_act * r_act).sum(axis=1)

        n_safe = n_active.where(n_active > 0, np.nan)
        mean_r = s_r / n_safe
        directional = s_w * mean_r
        selection = s_wr - directional

        valid = (n_active >= 2) & (gross > 0.0)
        total_per_gross = (s_wr / gross).where(valid)
        selection_per_gross = (selection / gross).where(valid)
        directional_per_gross = (directional / gross).where(valid)

        rows.append(
            {
                "lag": float(lag),
                "total_per_gross_mean": float(total_per_gross.mean(skipna=True)),
                "total_per_gross_t_stat": _t_stat(total_per_gross),
                "selection_per_gross_mean": float(selection_per_gross.mean(skipna=True)),
                "selection_per_gross_t_stat": _t_stat(selection_per_gross),
                "directional_per_gross_mean": float(directional_per_gross.mean(skipna=True)),
                "directional_per_gross_t_stat": _t_stat(directional_per_gross),
            }
        )

    df = pd.DataFrame(rows).set_index(pd.Index([int(r["lag"]) for r in rows], name="Lag"))
    df = df.drop(columns=["lag"], errors="ignore")
    return df


def _signal_decay_multi_horizon(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    horizons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """
    Compute IC, Rank IC, and decile spread at multiple forward horizons.

    Unlike `_alpha_decay_next_return_by_type` which lags weights to measure execution delay decay,
    this function measures how predictive current weights are of returns at different forward horizons.

    Args:
        weights: Target weight DataFrame (index=time, columns=assets)
        close_prices: Close prices DataFrame
        horizons: Forward periods to compute (default: (1, 3, 5, 10, 21))

    Returns:
        DataFrame with index=horizon and columns:
        - ic_mean, ic_tstat (Pearson correlation)
        - rank_ic_mean, rank_ic_tstat (Spearman rank correlation)
        - decile_spread_mean, decile_spread_tstat (top-bottom spread)
    """
    if horizons is None:
        horizons = (1, 3, 5, 10, 21)

    w = weights.fillna(0.0).astype(float)
    w_np = w.to_numpy(dtype=float, copy=True)
    n_periods = len(w.index)

    rows: list[dict[str, object]] = []

    for horizon in sorted(set(horizons)):
        if horizon < 1 or horizon >= n_periods:
            continue

        # Forward returns at this horizon: close[t+h] / close[t] - 1
        fwd = close_prices.shift(-horizon).divide(close_prices).subtract(1.0)
        fwd = fwd.replace([np.inf, -np.inf], np.nan)
        fwd_np = fwd.to_numpy(dtype=float, copy=True)

        ic_values: list[float] = []
        rank_ic_values: list[float] = []
        spread_values: list[float] = []

        for t in range(n_periods - horizon):
            w_row = w_np[t]
            r_row = fwd_np[t]
            mask = np.isfinite(w_row) & np.isfinite(r_row) & (w_row != 0.0)
            n_active = int(np.sum(mask))

            if n_active < 2:
                continue

            w_m = w_row[mask]
            r_m = r_row[mask]

            # IC (Pearson)
            if np.std(w_m) > 0.0 and np.std(r_m) > 0.0:
                ic_values.append(float(np.corrcoef(w_m, r_m)[0, 1]))

            # Rank IC (Spearman)
            w_ranks = pd.Series(w_m).rank(method="average").to_numpy(dtype=float)
            r_ranks = pd.Series(r_m).rank(method="average").to_numpy(dtype=float)
            if np.std(w_ranks) > 0.0 and np.std(r_ranks) > 0.0:
                rank_ic_values.append(float(np.corrcoef(w_ranks, r_ranks)[0, 1]))

            # Decile spread
            n = int(w_m.shape[0])
            if n >= 10:
                order = np.argsort(w_m, kind="mergesort")
                r_sorted = r_m[order]
                dec = (np.arange(n) * 10) // n
                bottom = r_sorted[dec == 0]
                top = r_sorted[dec == 9]
                if bottom.size > 0 and top.size > 0:
                    spread_values.append(float(np.mean(top) - np.mean(bottom)))

        # Use empty arrays cautiously to avoid warnings
        ic_arr = np.array(ic_values, dtype=float) if ic_values else np.array([])
        rank_ic_arr = np.array(rank_ic_values, dtype=float) if rank_ic_values else np.array([])
        spread_arr = np.array(spread_values, dtype=float) if spread_values else np.array([])

        rows.append(
            {
                "horizon": horizon,
                "ic_mean": float(np.mean(ic_arr)) if len(ic_arr) > 0 else np.nan,
                "ic_tstat": _t_stat(pd.Series(ic_arr)) if len(ic_arr) > 0 else np.nan,
                "rank_ic_mean": float(np.mean(rank_ic_arr)) if len(rank_ic_arr) > 0 else np.nan,
                "rank_ic_tstat": _t_stat(pd.Series(rank_ic_arr)) if len(rank_ic_arr) > 0 else np.nan,
                "decile_spread_mean": float(np.mean(spread_arr)) if len(spread_arr) > 0 else np.nan,
                "decile_spread_tstat": _t_stat(pd.Series(spread_arr)) if len(spread_arr) > 0 else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ic_mean",
                "ic_tstat",
                "rank_ic_mean",
                "rank_ic_tstat",
                "decile_spread_mean",
                "decile_spread_tstat",
            ]
        )

    df = pd.DataFrame(rows).set_index("horizon")
    df.index.name = "Horizon"
    return df


def _signal_by_asset(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-asset signal diagnostics.

    For each asset, computes time-series correlation between that asset's weight
    and its next-period return, along with hit rate and contribution metrics.

    Returns:
        DataFrame with index=asset and columns:
        - ic_mean: Time-series correlation between asset weight and next-period return
        - hit_rate: Fraction of periods where weight direction matches return direction
        - avg_abs_weight: Average absolute weight for this asset
        - alpha_contribution: Sum of (weight × return) for this asset
        - alpha_contribution_pct: Percentage contribution to total portfolio alpha
    """
    w = weights.fillna(0.0).astype(float)
    fwd = close_prices.shift(-1).divide(close_prices).subtract(1.0)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)

    assets = list(w.columns)
    results: list[dict[str, object]] = []
    total_contribution = 0.0

    # First pass: compute all metrics and total contribution
    for asset in assets:
        w_asset = w[asset].values
        r_asset = fwd[asset].values if asset in fwd.columns else np.full(len(w_asset), np.nan)

        # Valid periods: both weight non-zero and return available
        valid = np.isfinite(w_asset) & np.isfinite(r_asset) & (w_asset != 0.0)
        n_valid = int(np.sum(valid))

        if n_valid < 3:
            results.append(
                {
                    "asset": asset,
                    "ic_mean": np.nan,
                    "hit_rate": np.nan,
                    "avg_abs_weight": np.nan,
                    "alpha_contribution": np.nan,
                }
            )
            continue

        w_v = w_asset[valid]
        r_v = r_asset[valid]

        # Per-asset IC (time series correlation)
        if np.std(w_v) > 0.0 and np.std(r_v) > 0.0:
            ic_mean = float(np.corrcoef(w_v, r_v)[0, 1])
        else:
            ic_mean = np.nan

        # Hit rate: weight direction matches return direction
        correct = np.sign(w_v) == np.sign(r_v)
        hit_rate = float(np.mean(correct))

        # Average absolute weight
        avg_abs_weight = float(np.mean(np.abs(w_v)))

        # Alpha contribution: sum(w × r) for this asset
        contrib = float(np.sum(w_v * r_v))
        total_contribution += abs(contrib)

        results.append(
            {
                "asset": asset,
                "ic_mean": ic_mean,
                "hit_rate": hit_rate,
                "avg_abs_weight": avg_abs_weight,
                "alpha_contribution": contrib,
            }
        )

    # Second pass: compute percentage contribution
    df = pd.DataFrame(results).set_index("asset")
    if total_contribution > 0:
        df["alpha_contribution_pct"] = (
            df["alpha_contribution"].abs() / total_contribution * 100.0
        ) * np.sign(df["alpha_contribution"])
    else:
        df["alpha_contribution_pct"] = np.nan

    return df


def _weight_forward_diagnostics(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Compute signal diagnostics by comparing weights at time t to next-period returns.

    Forward returns are computed as close[t+1]/close[t] - 1. For 24/7 markets (e.g. crypto),
    this accurately approximates the return from execution at open[t+1] to close[t+1], since
    open[t+1] ≈ close[t] in continuous trading. For markets with overnight gaps, this may
    overstate or understate signal quality depending on gap direction relative to the signal.
    """
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

    dec_values: list[list[np.ndarray]] = [[] for _ in range(10)]
    dec_contrib_sum = np.zeros(10, dtype=float)
    dec_contrib_long_sum = np.zeros(10, dtype=float)
    dec_contrib_short_sum = np.zeros(10, dtype=float)

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
                short_ret_per_gross[t] = float(
                    np.sum(w_m[short_mask] * r_m[short_mask]) / short_denom
                )

        n = int(w_m.shape[0])
        if n >= 10:
            order = np.argsort(w_m, kind="mergesort")
            w_sorted = w_m[order]
            r_sorted = r_m[order]
            n = int(r_sorted.shape[0])
            dec = (np.arange(n) * 10) // n
            bottom = r_sorted[dec == 0]
            top = r_sorted[dec == 9]
            if bottom.size > 0 and top.size > 0:
                spread[t] = float(np.mean(top) - np.mean(bottom))

            for d in range(10):
                d_mask = dec == d
                vals_r = r_sorted[d_mask]
                if vals_r.size == 0:
                    continue
                vals_w = w_sorted[d_mask]
                dec_values[d].append(vals_r.astype(float, copy=True))

                contrib = vals_w * vals_r
                dec_contrib_sum[d] += float(np.sum(contrib))
                dec_contrib_long_sum[d] += float(np.sum(contrib[vals_w > 0.0]))
                dec_contrib_short_sum[d] += float(np.sum(contrib[vals_w < 0.0]))

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

    decile_mean = np.full(10, np.nan, dtype=float)
    decile_median = np.full(10, np.nan, dtype=float)
    decile_std = np.full(10, np.nan, dtype=float)
    decile_count = np.zeros(10, dtype=float)
    for d in range(10):
        if len(dec_values[d]) == 0:
            continue
        vals = np.concatenate(dec_values[d], axis=0)
        if vals.size == 0:
            continue
        decile_count[d] = float(vals.size)
        decile_mean[d] = float(np.mean(vals))
        decile_median[d] = float(np.median(vals))
        decile_std[d] = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan

    denom = np.where(decile_count > 0, decile_count, np.nan)
    decile_contrib_mean = dec_contrib_sum / denom
    decile_contrib_long = dec_contrib_long_sum / denom
    decile_contrib_short = dec_contrib_short_sum / denom

    decile_mean_curve = pd.Series(
        decile_mean,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Mean next return",
    )
    decile_median_curve = pd.Series(
        decile_median,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Median next return",
    )
    decile_std_curve = pd.Series(
        decile_std,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Std next return",
    )
    decile_count_curve = pd.Series(
        decile_count,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Count",
    )
    decile_contrib_curve = pd.Series(
        decile_contrib_mean,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Mean next return contribution",
    )
    decile_contrib_long_curve = pd.Series(
        decile_contrib_long,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Mean next return contribution (long)",
    )
    decile_contrib_short_curve = pd.Series(
        decile_contrib_short,
        index=pd.Index(range(1, 11), name="Decile"),
        name="Mean next return contribution (short)",
    )
    return (
        wf,
        decile_mean_curve,
        decile_median_curve,
        decile_std_curve,
        decile_count_curve,
        decile_contrib_curve,
        decile_contrib_long_curve,
        decile_contrib_short_curve,
    )


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
    weight_turnover_ratio: np.ndarray,
    gross_exposure_ratio: np.ndarray,
    net_exposure_ratio: np.ndarray,
    order_count_period: np.ndarray,
    order_notional_sum_period: np.ndarray,
    slippage_paid: np.ndarray,
    positions_hist: np.ndarray,
    compute_signal_diagnostics: bool = True,
    signal_decay_horizons: tuple[int, ...] | None = None,
    signal_cost_per_turnover: float = 0.001,
) -> pd.DataFrame:
    """
    Build a metrics DataFrame from simulation outputs.

    Args:
        compute_signal_diagnostics: If True, compute expensive signal diagnostics
            (weight IC, rank IC, decile analysis). Set to False for faster grid search
            when only basic metrics (Sharpe, returns, etc.) are needed.
        signal_decay_horizons: Forward horizons for multi-horizon signal decay analysis.
            If None, uses default (1, 3, 5, 10, 21).
        signal_cost_per_turnover: Cost per unit turnover for turnover-adjusted metrics
            (default: 0.001 = 10bps).
    """

    n_periods = int(len(returns))
    n_assets = int(weights.shape[1])

    if compute_signal_diagnostics:
        (
            wf,
            wf_deciles,
            wf_deciles_median,
            wf_deciles_std,
            wf_deciles_count,
            wf_deciles_contrib,
            wf_deciles_contrib_long,
            wf_deciles_contrib_short,
        ) = _weight_forward_diagnostics(weights=weights, close_prices=close_prices)
    else:
        # Skip expensive diagnostics - return empty placeholders
        wf = pd.DataFrame()
        wf_deciles = pd.DataFrame()
        wf_deciles_median = pd.Series(dtype=float)
        wf_deciles_std = pd.Series(dtype=float)
        wf_deciles_count = pd.Series(dtype=float)
        wf_deciles_contrib = pd.Series(dtype=float)
        wf_deciles_contrib_long = pd.Series(dtype=float)
        wf_deciles_contrib_short = pd.Series(dtype=float)

    total_return_pct = float(equity.iloc[-1] / init_cash - 1.0)

    dd_equity = _max_drawdown(equity)

    annual_factor = _annualization_factor(freq_rule, trading_days_year)
    if n_periods > 0:
        annual_return = float((1.0 + returns).prod() ** (annual_factor / n_periods) - 1.0)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(annual_factor))
        annual_sharpe = float(
            (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
        )
        daily_turnover_1way = float(turnover_ratio.mean())
        annual_turnover_1way = float(turnover_ratio.mean() * annual_factor)
        daily_weight_turnover = float(weight_turnover_ratio.mean())
        annual_weight_turnover = float(weight_turnover_ratio.mean() * annual_factor)
    else:
        annual_return = 0.0
        annual_vol = 0.0
        annual_sharpe = np.nan
        daily_turnover_1way = 0.0
        annual_turnover_1way = 0.0
        daily_weight_turnover = 0.0
        annual_weight_turnover = 0.0

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
    net_exposure_max_pct = float(np.nanmax(np.abs(net_exposure_ratio)) * 100.0)

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
        "Total return %": total_return_pct * 100.0,
        "Total return": net_pnl,
    }
    metrics_costs_and_trading = {
        "Funding earnings": funding_total,
        "Funding % total return": funding_pct_total_pnl,
        "Fees": total_fees,
        "Daily Turnover (1-way)": daily_turnover_1way,
        "Annual Turnover (1-way)": annual_turnover_1way,
        "Daily Weight-based Turnover": daily_weight_turnover,
        "Annual Weight-based Turnover": annual_weight_turnover,
        "Total order count": total_order_count,
        "Average order notional": avg_order_notional,
    }
    metrics_exposure = {
        "Gross exposure mean %": avg_gross_exposure_pct,
        "Gross exposure max %": max_gross_exposure_pct,
        "Net exposure mean %": net_exposure_mean_pct,
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
    }
    metrics_risk = {
        "Annualized Sortino": annual_sortino,
        "Downside deviation": downside_deviation,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Omega Ratio": omega_ratio,
        "Ulcer Index": ulcer_index,
    }
    if compute_signal_diagnostics and not wf.empty:
        # Compute turnover-adjusted metrics
        # weight_turnover_ratio is per-period turnover aligned with wf index
        wt_series = pd.Series(weight_turnover_ratio, index=weights.index)
        # Align turnover with wf index (wf may have fewer periods due to forward return requirement)
        wt_aligned = wt_series.reindex(wf.index)
        # Turnover-adjusted IC: IC × (1 - turnover × cost)
        turnover_adj_ic = wf["ic"] * (1.0 - wt_aligned * signal_cost_per_turnover)
        # Net forward return per gross: forward_return - (turnover × cost)
        net_fwd_per_gross = wf["forward_return_per_gross"] - wt_aligned * signal_cost_per_turnover

        metrics_weight_vs_next = {
            "Weight IC mean (next)": float(wf["ic"].mean(skipna=True)),
            "Weight IC t-stat (next)": _t_stat(wf["ic"]),
            "Weight Rank IC mean (next)": float(wf["rank_ic"].mean(skipna=True)),
            "Weight Rank IC t-stat (next)": _t_stat(wf["rank_ic"]),
            "Top-bottom decile spread mean (next)": float(wf["top_bottom_spread"].mean(skipna=True)),
            "Top-bottom decile spread t-stat (next)": _t_stat(wf["top_bottom_spread"]),
            "Weighted long hit rate mean (next)": float(wf["long_hit_weighted"].mean(skipna=True)),
            "Weighted short hit rate mean (next)": float(wf["short_hit_weighted"].mean(skipna=True)),
            "Forward return per gross mean (next)": float(
                wf["forward_return_per_gross"].mean(skipna=True)
            ),
            "Forward return selection per gross mean (next)": float(
                wf["forward_return_selection_per_gross"].mean(skipna=True)
            ),
            "Forward return selection per gross t-stat (next)": _t_stat(
                wf["forward_return_selection_per_gross"]
            ),
            "Gross weight mean": float(wf["gross_weight"].mean(skipna=True)),
            "Directionality mean": float(wf["directionality"].mean(skipna=True)),
            "Turnover-adjusted IC mean (next)": float(turnover_adj_ic.mean(skipna=True)),
            "Net forward return per gross mean (next)": float(net_fwd_per_gross.mean(skipna=True)),
        }
    else:
        # Signal diagnostics skipped - return NaN placeholders
        metrics_weight_vs_next = {
            "Weight IC mean (next)": np.nan,
            "Weight IC t-stat (next)": np.nan,
            "Weight Rank IC mean (next)": np.nan,
            "Weight Rank IC t-stat (next)": np.nan,
            "Top-bottom decile spread mean (next)": np.nan,
            "Top-bottom decile spread t-stat (next)": np.nan,
            "Weighted long hit rate mean (next)": np.nan,
            "Weighted short hit rate mean (next)": np.nan,
            "Forward return per gross mean (next)": np.nan,
            "Forward return selection per gross mean (next)": np.nan,
            "Forward return selection per gross t-stat (next)": np.nan,
            "Gross weight mean": np.nan,
            "Directionality mean": np.nan,
            "Turnover-adjusted IC mean (next)": np.nan,
            "Net forward return per gross mean (next)": np.nan,
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
                    "Note": _format_tearsheet_note(
                        TEARSHEET_NOTES.get(metric_name, ""),
                        include_definition=(category != "Meta"),
                    ),
                }
            )

    df = pd.DataFrame(
        rows,
        index=pd.Index([m for _, md in all_metrics for m in md.keys()], name="Metric"),
        columns=["Category", "Value", "Note"],
    )
    df.attrs["weight_forward"] = wf
    df.attrs["weight_forward_deciles"] = wf_deciles
    df.attrs["weight_forward_deciles_median"] = wf_deciles_median
    df.attrs["weight_forward_deciles_std"] = wf_deciles_std
    df.attrs["weight_forward_deciles_count"] = wf_deciles_count
    df.attrs["weight_forward_deciles_contrib"] = wf_deciles_contrib
    df.attrs["weight_forward_deciles_contrib_long"] = wf_deciles_contrib_long
    df.attrs["weight_forward_deciles_contrib_short"] = wf_deciles_contrib_short
    if compute_signal_diagnostics:
        df.attrs["alpha_decay_next_return_by_type"] = _alpha_decay_next_return_by_type(
            weights=weights,
            close_prices=close_prices,
            max_lag=10,
        )
        df.attrs["signal_decay_horizons"] = _signal_decay_multi_horizon(
            weights=weights,
            close_prices=close_prices,
            horizons=signal_decay_horizons,
        )
        df.attrs["signal_by_asset"] = _signal_by_asset(
            weights=weights,
            close_prices=close_prices,
        )
    else:
        df.attrs["alpha_decay_next_return_by_type"] = pd.DataFrame()
        df.attrs["signal_decay_horizons"] = pd.DataFrame()
        df.attrs["signal_by_asset"] = pd.DataFrame()

    # Trading Activity & Costs time series
    df.attrs["turnover"] = pd.Series(turnover_ratio, index=weights.index, name="turnover")
    df.attrs["weight_turnover"] = pd.Series(
        weight_turnover_ratio, index=weights.index, name="weight_turnover"
    )
    df.attrs["transaction_costs"] = pd.Series(
        (fees_paid + slippage_paid) / equity.values, index=weights.index, name="transaction_costs"
    )
    # Count non-zero, non-nan weights per period (the strategy's active universe at each time)
    weights_active = weights.fillna(0.0)
    n_positions_per_period = (weights_active.abs() > 1e-8).sum(axis=1).values
    df.attrs["n_positions"] = pd.Series(
        n_positions_per_period, index=weights.index, name="n_positions"
    )

    # Exposure & Risk Management time series
    df.attrs["net_exposure"] = pd.Series(
        net_exposure_ratio, index=weights.index, name="net_exposure"
    )
    df.attrs["gross_exposure"] = pd.Series(
        gross_exposure_ratio, index=weights.index, name="gross_exposure"
    )

    # Compute long and short exposure from positions_hist (actual positions held, not weights)
    # positions_hist is in dollar terms, so divide by equity to get % of portfolio
    long_positions = np.clip(positions_hist, 0, None)
    short_positions = np.clip(positions_hist, None, 0)
    long_exp = long_positions.sum(axis=1) / equity.values
    short_exp = short_positions.sum(axis=1) / equity.values
    df.attrs["long_exposure"] = pd.Series(long_exp, index=weights.index, name="long_exposure")
    df.attrs["short_exposure"] = pd.Series(short_exp, index=weights.index, name="short_exposure")

    # Concentration (Herfindahl index): sum of squared position weights as fraction of portfolio
    # Use absolute positions to measure concentration regardless of direction
    abs_positions = np.abs(positions_hist)
    total_abs_positions = abs_positions.sum(axis=1, keepdims=True)
    # Avoid division by zero
    total_abs_positions = np.where(total_abs_positions == 0, 1.0, total_abs_positions)
    position_fractions = abs_positions / total_abs_positions
    concentration = (position_fractions**2).sum(axis=1)
    df.attrs["concentration"] = pd.Series(concentration, index=weights.index, name="concentration")

    return df
