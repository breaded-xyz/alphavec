"""Backtest module for evaluating trading strategies."""

import logging
import time
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numpy.random import MT19937, RandomState, SeedSequence

logger = logging.getLogger(__name__)

DEFAULT_TRADING_DAYS_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02

EPSILON = 1e-8  # Renamed from EPSILION


def zero_commission(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Zero trading commission.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.

    Returns:
        Dataframe or series with zero commission for each trade.
    """
    fees = weights.copy()
    fees[:] = 0.0
    return fees


def flat_commission(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    fee_amount: float,
) -> pd.DataFrame | pd.Series:
    """Flat commission applies a fixed fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee_amount: Fixed fee per trade in units of currency.

    Returns:
        Dataframe or series with the commission amount for each trade.
    """
    diff = (
        weights.fillna(0).astype(float).diff().abs() > EPSILON  # Updated constant name
    )  # Avoid fees on floating point errors
    tx = diff.astype(int)
    commission = tx * fee_amount
    return commission


def pct_commission(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    fee_pct: float,
) -> pd.DataFrame | pd.Series:
    """Percentage commission applies a percentage fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee_pct: Percentage fee per trade in decimal.

    Returns:
        Dataframe or series with the commission amount for each trade.
    """
    # Portfolio weights already express traded notional as Δw (fraction of equity)
    size = weights.fillna(0).diff().abs()
    commission = size * fee_pct
    return commission


def equity_curve(
    log_rets: pd.DataFrame | pd.Series, initial: float = 1
) -> pd.DataFrame | pd.Series:
    """Calculate the compounded equity curve from arithmetic returns."""
    growth_factors = (
        1 + log_rets
    )  # `log_rets` var retained to minimise downstream edits
    return initial * growth_factors.cumprod()


def _arith_rets(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Generate arithmetic (simple) returns from price data."""
    return data.pct_change()


class BacktestResult(NamedTuple):
    perf: pd.DataFrame
    perf_curve: pd.DataFrame
    perf_roll_sr: pd.DataFrame
    port_perf: pd.DataFrame
    port_rets: pd.Series


CommissionFunc = Callable[
    [pd.DataFrame | pd.Series, pd.DataFrame | pd.Series], pd.DataFrame | pd.Series
]


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

    asset_rets = _arith_rets(prices)
    asset_rets = asset_rets.iloc[:-lags] if lags > 0 else asset_rets
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

    prices_shifted = prices.shift(-lags)
    cmn_costs = commission_func(weights, prices_shifted)
    borrow_costs = _borrow(
        weights, prices_shifted, ann_borrow_rate, freq_year, is_perp_funding
    )
    spread_costs = _spread(weights, prices_shifted, spread_pct)

    costs = cmn_costs + borrow_costs + spread_costs

    # Costs are already expressed as fractions of equity (return terms).
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
            _ann_turnover(weights, strat_rets, freq_year=freq_year),
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

    port_rets = strat_rets.sum(axis=1)
    port_curve = equity_curve(port_rets)
    port_ann_turnover = (strat_perf["annual_turnover"] * weights.mean().abs()).sum()

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
    """Bootstrap sampling of a time series, optionally using a stationary method."""
    samples = []
    rs = RandomState(MT19937(SeedSequence(seed)))

    if stationary_method:
        block_size = optimal_block_length(x.dropna())["stationary"].squeeze()
        bs = StationaryBootstrap(block_size, x.dropna().values, seed=rs)  # type: ignore
        for sample in bs.bootstrap(n):
            sample = pd.Series(sample[0][0], index=x.index)  # type: ignore
            sample[x.isna()] = np.nan
            samples.append(sample)
    else:
        for _ in range(n):
            sample_vals = rs.choice(x.dropna(), size=x.shape, replace=True)
            sample = pd.Series(sample_vals, index=x.index)
            sample[x.isna()] = np.nan
            samples.append(sample)

    return samples


def _log_rets(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Generate log returns from data."""
    return data.transform(lambda x: np.log(x / x.shift(1)))


def _ann_to_period_rate(ann_rate: float, periods_year: int) -> float:
    """Calculate the annualized compounding rate for the given period frequency."""
    return (1 + ann_rate) ** (1 / periods_year) - 1


def _ann_sharpe(
    rets: pd.DataFrame | pd.Series,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series:
    """Calculate annualized Sharpe ratio."""
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.mean()
    sigma = rets.std()
    sr = (mu - rfr) / sigma
    return pd.Series(sr * np.sqrt(freq_year))


def _ann_roll_sharpe(
    rets: pd.DataFrame | pd.Series,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    window: int = DEFAULT_TRADING_DAYS_YEAR,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.DataFrame | pd.Series:
    """Calculate rolling annualized Sharpe ratio."""
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()
    sr = (mu - rfr) / sigma
    return sr * np.sqrt(freq_year)


def _ann_vol(
    rets: pd.DataFrame | pd.Series, freq_year: int = DEFAULT_TRADING_DAYS_YEAR
) -> pd.Series:
    """Calculate annualized volatility."""
    return pd.Series(rets.std() * np.sqrt(freq_year))


def _cagr(
    rets: pd.DataFrame | pd.Series,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series:
    """Calculate CAGR."""
    n_years = rets.count() / freq_year
    final = equity_curve(rets, initial=1).iloc[-1] - 1
    return pd.Series((1 + final) ** (1 / n_years) - 1)


def _max_drawdown(
    rets: pd.DataFrame | pd.Series,
) -> pd.Series:
    """Calculate the max drawdown in pct."""
    curve = equity_curve(rets)
    hwm = curve.cummax()
    return pd.Series(((curve - hwm) / hwm).min())


def _ann_turnover(
    weights: pd.DataFrame | pd.Series,
    rets: pd.DataFrame | pd.Series,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series:
    """Calculate the annualized turnover of the strategy."""

    # Calculate period-to-period changes in weights and convert them into notional volumes
    diff = weights.fillna(0).diff().fillna(0)
    buy_volume = diff.where(lambda x: x.gt(0), 0).abs().sum()
    sell_volume = diff.where(lambda x: x.lt(0), 0).abs().sum()

    # Calculate the minimum of buy and sell volumes
    trade_volume = pd.concat(
        [pd.Series(buy_volume), pd.Series(sell_volume)], axis=1
    ).min(axis=1)

    # Calculate turnover with safe division
    equity_avg = equity_curve(rets).mean()
    periods = rets.count()
    turnover = trade_volume.div(equity_avg, fill_value=0)
    ann_factor = periods / freq_year
    ann_turnover = turnover.div(ann_factor, fill_value=0)

    return pd.Series(ann_turnover)


def _spread(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    spread_pct: float = 0,
) -> pd.DataFrame | pd.Series:
    """Calculate the spread costs for each trade."""
    size = weights.fillna(0).diff().abs()
    return size * (spread_pct * 0.5)


def _borrow(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    ann_borrow_rate: float = 0,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
    is_perp_funding: bool = False,
) -> pd.DataFrame | pd.Series:
    """Calculate borrowing costs for short and leveraged positions.

    Accepts either an *annual* borrow rate (common case) or a
    pre‑computed *per‑period* rate.  Rates ≤ 1/freq_year are assumed
    to be per‑period and are used as‑is; larger values are treated
    as annual and converted to per‑period compound rates.
    """
    # If the supplied rate already looks like a single‑period rate
    # (i.e. smaller than one period of 100 % per‑year), use it directly.
    # Otherwise convert the annual rate to a per‑period rate.
    period_rate_threshold = 1 / freq_year + EPSILON
    if ann_borrow_rate <= period_rate_threshold:
        rate = ann_borrow_rate
    else:
        rate = _ann_to_period_rate(ann_borrow_rate, freq_year)

    lev_threshold_weight = 1 if not is_perp_funding else 0
    wts = weights.copy()
    wts[wts < 0] = wts.abs().add(lev_threshold_weight)

    leveraged_size = wts.fillna(0).sub(lev_threshold_weight).clip(lower=0)
    costs = leveraged_size * rate

    return costs
