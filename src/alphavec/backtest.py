"""Backtest module for evaluating trading strategies."""

import logging
import time
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numpy.random import MT19937, RandomState, SeedSequence

pd.options.mode.chained_assignment = "raise"  # fail-fast on accidental views

logger = logging.getLogger(__name__)

DEFAULT_TRADING_DAYS_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02

EPSILON = 1e-8


def zero_commission(
    weights: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Zero trading commission.

    Args:
        weights: Weights of the assets in the portfolio.

    Returns:
        Dataframe or series with zero commission for each trade.
    """
    fees = weights.copy()
    fees[:] = 0.0
    return fees


def pct_commission(
    weights: pd.DataFrame | pd.Series,
    fee_pct: float,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
) -> pd.DataFrame | pd.Series:
    """
    Commission charged as a **percentage of traded notional**.

    A trade is recognised only when  |Δw|  exceeds

        max(|w_prev| · rel_tol, abs_tol),

    which filters out optimiser or float-rounding noise.
    """
    w = weights.fillna(0)
    delta = w.diff().abs()

    tol = np.maximum(w.abs() * rel_tol, abs_tol)
    traded = delta > tol

    commission = (delta * fee_pct).where(traded, 0.0)
    commission[weights.isna()] = np.nan
    return commission


def equity_curve(
    rets: pd.DataFrame | pd.Series, initial: float = 1
) -> pd.DataFrame | pd.Series:
    """Calculate the compounded equity curve from arithmetic returns."""
    growth_factors = 1 + rets
    return initial * growth_factors.cumprod()


def _arith_rets(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Generate arithmetic (simple) returns from price data."""
    return data.pct_change(fill_method=None)


class BacktestResult(NamedTuple):
    perf: pd.DataFrame
    perf_curve: pd.DataFrame
    perf_roll_sr: pd.DataFrame
    port_perf: pd.DataFrame
    port_rets: pd.Series


CommissionFunc = Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series]


def backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    freq_day: int = 1,
    trading_days_year: int = DEFAULT_TRADING_DAYS_YEAR,
    lags: int = 1,
    commission_func: CommissionFunc = zero_commission,
    spread_pct: float = 0,
    ann_borrow_rate: float = 0,
    is_perp_funding: bool = False,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    bootstrap_n: int = 1000,
) -> BacktestResult:
    """Backtest a trading strategy.

    Strategy is simulated using the given weights, prices, and cost parameters.
    Zero costs are calculated by default: no commission, spread or borrowing.

    Args:
        weights: Weights of the assets at each period.
        prices: Prices of the assets at each period.
        freq_day: Number of periods in a trading day. Defaults to 1.
        trading_days_year: Number of trading days in a year. Defaults to 252.
        lags: Positive integer for n periods to shift returns relative to weights. Defaults to 1.
        commission_func: Function to calculate commission cost. Defaults to zero_commission.
        spread_pct: Spread cost as a decimal percentage. Defaults to 0.
        ann_borrow_rate: Annualized borrowing rate. Defaults to 0.
        is_perp_funding: Simulate perpetual funding. Defaults to False.
        ann_risk_free_rate: Annualized risk-free rate. Defaults to 0.02.
        bootstrap_n: Number of bootstrap iterations. Defaults to 1000.
    Returns:
        A tuple:
            1. Asset-wise performance table
            2. Asset-wise equity curves
            3. Asset-wise rolling annualized Sharpes
            4. Portfolio performance table
            5. Portfolio (arithmetic) returns
    """
    if weights.shape != prices.shape:
        raise ValueError("Weights and prices must have the same shape")

    if weights.index.tolist() != prices.index.tolist():
        raise ValueError("Index of weights and prices must match")

    if weights.columns.tolist() != prices.columns.tolist():
        raise ValueError("Columns of weights and prices must match")

    row_n, col_n = weights.shape
    ts_start = time.time()
    logging.info(
        f"Executing backtest for {col_n} assets over {row_n} periods with {bootstrap_n} bootstrap iterations..."
    )

    freq_year = freq_day * trading_days_year

    asset_rets = _arith_rets(prices).shift(-lags)  # align to weights
    if lags > 0:
        asset_rets = asset_rets.iloc[:-lags]  # drop trailing NaNs

    asset_curve = equity_curve(asset_rets)

    asset_perf = pd.concat(
        [
            _ann_sharpe(
                asset_rets, freq_year=freq_year, ann_risk_free_rate=ann_risk_free_rate
            ),
            _ann_vol(asset_rets, freq_year=freq_year),
            _cagr(asset_rets, freq_year=freq_year),
            _max_drawdown(asset_rets),
        ],
        keys=["annual_sharpe", "annual_volatility", "cagr", "max_drawdown"],
        axis=1,
    )

    strat_rets = weights * _arith_rets(prices).shift(-lags)
    strat_rets = strat_rets.iloc[:-lags] if lags > 0 else strat_rets

    cmn_costs = commission_func(weights)
    borrow_costs = _borrow(weights, ann_borrow_rate, freq_year, is_perp_funding)
    spread_costs = _spread(weights, spread_pct)
    costs = cmn_costs + borrow_costs + spread_costs
    costs_pct = costs.clip(lower=0, upper=1 - EPSILON)

    strat_rets = strat_rets - costs_pct
    strat_rets[weights.isna()] = np.nan

    strat_curve = equity_curve(strat_rets)

    strat_perf = pd.concat(
        [
            _ann_sharpe(
                strat_rets, freq_year=freq_year, ann_risk_free_rate=ann_risk_free_rate
            ),
            _ann_vol(strat_rets, freq_year=freq_year),
            _cagr(strat_rets, freq_year=freq_year),
            _max_drawdown(strat_rets),
            _ann_turnover(weights, freq_year=freq_year),
        ],
        keys=[
            "annual_sharpe",
            "annual_volatility",
            "cagr",
            "max_drawdown",
            "annual_turnover",
        ],
        axis=1,
    )

    # -----------------------------------------------------------------
    #  PORTFOLIO AGGREGATION  – include residual cash earning risk-free
    # -----------------------------------------------------------------
    aligned_w = weights.iloc[:-lags] if lags > 0 else weights
    cash_weight = 1.0 - aligned_w.sum(axis=1)
    period_rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)

    # strat_rets holds asset-level *net* returns after costs
    asset_leg = strat_rets.sum(axis=1)
    cash_leg = cash_weight * period_rfr

    port_rets = asset_leg + cash_leg
    port_curve = equity_curve(port_rets)

    # --------- Standard portfolio-turnover: 0.5 Σ|Δw|, annualised -----------
    port_turn_ts = 0.5 * aligned_w.diff().abs().sum(axis=1) * freq_year
    port_turn_ts.iloc[0] = np.nan  # undefined for the first period
    port_ann_turnover = port_turn_ts.mean()  # average annual turnover

    perf = pd.concat([asset_perf, strat_perf], keys=["asset", "strategy"], axis=1)

    perf_curve = pd.concat(
        [port_curve, asset_curve, strat_curve],
        keys=["portfolio", "asset", "strategy"],
        axis=1,
    ).rename(columns={0: "equity_curve"})

    perf_roll_sr = pd.concat(
        [
            _ann_roll_sharpe(
                port_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                asset_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                strat_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
        ],
        keys=["portfolio", "asset", "strategy"],
        axis=1,
    ).rename(columns={0: "sharpe"})

    def calc_port_metrics(
        port_rets: pd.Series,
        port_ann_turnover: float,
        freq_year: int,
    ):
        return pd.DataFrame(
            {
                "annual_sharpe": _ann_sharpe(
                    port_rets,
                    freq_year=freq_year,
                    ann_risk_free_rate=ann_risk_free_rate,
                ).squeeze(),
                "annual_volatility": _ann_vol(port_rets, freq_year=freq_year).squeeze(),
                "cagr": _cagr(port_rets, freq_year=freq_year).squeeze(),
                "max_drawdown": _max_drawdown(port_rets).squeeze(),
                "annual_turnover": port_ann_turnover,
            },
            index=["observed"],
        )

    port_perf = calc_port_metrics(port_rets, port_ann_turnover, freq_year)

    if bootstrap_n > 0:
        bs_sampled = _bootstrap_sampling(
            port_rets, n=bootstrap_n, stationary_method=True
        )
        sampled_perf = pd.concat(
            [
                calc_port_metrics(sampled_rets, port_ann_turnover, freq_year)
                for sampled_rets in bs_sampled
            ]
        )

        def describe(x: pd.Series) -> pd.Series:
            return pd.Series(
                {
                    "mean": x.mean(),
                    "std": x.std(),
                    "median": x.median(),
                    "ucl.95": np.percentile(x, 97.5),
                    "lcl.95": np.percentile(x, 2.5),
                }
            )

        port_perf = pd.concat([port_perf, sampled_perf.apply(describe)]).round(4)

    ts_end = time.time()
    logging.info(f"Backtest complete in {ts_end - ts_start:.2f} seconds.")
    return BacktestResult(perf, perf_curve, perf_roll_sr, port_perf, port_rets)


def _bootstrap_sampling(
    x: pd.Series,
    n: int = 1000,
    seed: int = 1,
    stationary_method: bool = False,
) -> list[pd.Series]:
    """Bootstrap sampling of a time series, optionally using a stationary (block) method."""
    rs = RandomState(MT19937(SeedSequence(seed)))
    samples: list[pd.Series] = []

    if stationary_method:
        block_size = optimal_block_length(x.dropna())["stationary"].squeeze()
        bs = StationaryBootstrap(block_size, x.dropna().values, seed=rs)  # type: ignore

        # --- inside _bootstrap_sampling(), in the stationary_method branch -------------
        for data_pack, _ in bs.bootstrap(n):
            #    └────── tuple of resampled positional datasets
            # Extract the resampled array for *our* single series (index 0),
            #   then flatten to 1-D and rebuild the Series.
            resampled = np.asarray(data_pack[0]).reshape(-1)  # ← was data.ravel()
            sample = (
                pd.Series(resampled, index=x.dropna().index).reindex_like(
                    x
                )  # keep time-stamps  # re-insert NaNs
            )
            samples.append(sample)
        # -------------------------------------------------------------------------------
    else:
        for _ in range(n):
            sample_vals = rs.choice(x.dropna(), size=x.shape, replace=True)
            sample = pd.Series(sample_vals, index=x.index)
            sample[x.isna()] = np.nan
            samples.append(sample)

    return samples


def _ann_to_period_rate(ann_rate: float, periods_year: int) -> float:
    """Calculate the annualized compounding rate for the given period frequency."""
    return (1 + ann_rate) ** (1 / periods_year) - 1


def _ann_sharpe(
    rets: pd.DataFrame | pd.Series,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series | float:
    """
    Annualised Sharpe ratio with a robust zero-σ guard.

    • Works for Series (returns a scalar) and for DataFrames
      (returns a Series per column).
    """
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.mean()  # scalar or Series
    sigma = rets.std()  # scalar or Series

    # ----- zero-volatility guard ------------------------------------
    if isinstance(sigma, (pd.Series, pd.DataFrame)):
        sigma = sigma.replace(0, np.nan)
    else:  # scalar path
        sigma = np.nan if np.isclose(sigma, 0.0) else sigma
    # ----------------------------------------------------------------

    sharpe = (mu - rfr) / sigma
    return sharpe * np.sqrt(freq_year)


def _ann_roll_sharpe(
    rets: pd.DataFrame | pd.Series,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    window: int = DEFAULT_TRADING_DAYS_YEAR,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.DataFrame | pd.Series:
    """
    Rolling (window) annualised Sharpe ratio with zero-σ guard.
    """
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()

    # Guard: replace exact zeros with NaN
    sigma = sigma.where(sigma != 0, np.nan)

    roll_sharpe = (mu - rfr) / sigma
    return roll_sharpe * np.sqrt(freq_year)


def _ann_vol(
    rets: pd.DataFrame | pd.Series, freq_year: int = DEFAULT_TRADING_DAYS_YEAR
) -> pd.Series:
    """Calculate annualized volatility."""
    return pd.Series(rets.std() * np.sqrt(freq_year))


def _cagr(
    rets: pd.DataFrame | pd.Series,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
    init_equity: float = 1.0,
) -> pd.Series | float:
    """
    Compound annual growth rate (CAGR).

    • Works for Series (returns a scalar) and DataFrame (returns Series).
    • Uses the exact sample length so that partial years are handled correctly.
    """
    if len(rets) == 0:
        return (
            np.nan
            if isinstance(rets, pd.Series)
            else pd.Series(np.nan, index=rets.columns)
        )

    equity = equity_curve(rets, init_equity)
    final = equity.iloc[-1]  # scalar or Series

    years = len(rets) / freq_year
    cagr = (final / init_equity) ** (1.0 / years) - 1.0
    return cagr


def _max_drawdown(
    rets: pd.DataFrame | pd.Series,
) -> pd.Series:
    """Calculate the max drawdown in pct."""
    curve = equity_curve(rets)
    hwm = curve.cummax()
    return pd.Series(((curve - hwm) / hwm).min())


def _ann_turnover(
    weights: pd.DataFrame | pd.Series,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series:
    """
    Annualised portfolio turnover.

    For each rebalancing step t the (non-annualised) turnover is
        0.5 * Σ_i | w_{t,i} - w_{t-1,i} |.
    Multiplying by `freq_year` expresses the figure on a per-year basis.

    Parameters
    ----------
    weights : pd.DataFrame or pd.Series
        Time series of portfolio weights indexed by rebalance date.
        • DataFrame → columns are assets.
        • Series    → single-asset “portfolio”.
    freq_year : int, default 252
        Number of rebalancing periods in a calendar year
        (252 for daily, 12 for monthly, etc.).

    Returns
    -------
    pd.Series
        Annualised turnover per date (first entry is NaN because there
        is no previous weight vector to compare with).
    """
    if weights.empty:
        return pd.Series(dtype=float, index=weights.index)

    weights = weights.sort_index()
    delta = weights.diff().abs().fillna(0)

    if isinstance(delta, pd.DataFrame):
        # per-asset turnover, avg across time
        asset_turn = 0.5 * delta * freq_year
        return asset_turn.mean()  # index → assets
    else:
        # single-asset portfolio ⇒ scalar
        step_turn = 0.5 * delta * freq_year
        return pd.Series(step_turn.mean(), index=[weights.name])


def _spread(
    weights: pd.DataFrame | pd.Series,
    spread_pct: float = 0,
) -> pd.DataFrame | pd.Series:
    """Calculate the spread costs for each trade."""
    size = weights.fillna(0).diff().abs()
    return size * (spread_pct * 0.5)


def _borrow(
    weights: pd.DataFrame | pd.Series,
    ann_borrow_rate: float = 0.0,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
    is_perp_funding: bool = False,
) -> pd.DataFrame | pd.Series:
    """
    Borrowing (or funding) cost paid on the *absolute* size of
    • short exposure, and
    • synthetic long leverage above +1 × (or 0 × on perps).
    """
    # Decide whether the supplied rate is already per-period
    period_rate_threshold = 1 / freq_year + EPSILON
    if ann_borrow_rate <= period_rate_threshold:
        rate = ann_borrow_rate
    else:
        rate = _ann_to_period_rate(ann_borrow_rate, freq_year)

    # Threshold weight where borrowing begins: 1 × for spot, 0 × for perpetuals
    lev_threshold_weight = 0 if is_perp_funding else 1

    wts = weights.fillna(0)

    # **NEW:** treat long leverage and shorts separately
    short_size = (-wts).clip(lower=0)  # |w| for w < 0
    long_leverage = (wts - lev_threshold_weight).clip(lower=0)

    leveraged_size = short_size + long_leverage  # always ≥ 0
    costs = leveraged_size * rate
    costs[weights.isna()] = np.nan  # preserve missing data

    return costs
