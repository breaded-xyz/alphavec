import os
import sys
from pathlib import PurePath

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import alphavec.backtest as av

workspace_root = str(PurePath(os.getcwd()))
sys.path.append(workspace_root)


def truncated_normal_array(shape, mean=0.0, std=0.3, low=-1.0, high=1.0, seed=42):
    """Return an ndarray of the requested shape filled with
    N(mean, std) draws truncated to [low, high]."""
    rng, size = np.random.default_rng(seed), int(np.prod(shape))
    samples = []
    while len(samples) < size:  # rejection sampling
        draw = rng.normal(loc=mean, scale=std, size=size * 2)
        samples.extend(draw[(draw >= low) & (draw <= high)][: size - len(samples)])
    return np.array(samples).reshape(shape)


def weights_like(df, mean=0.0, std=0.3, low=-1.0, high=1.0, seed=42):
    """DataFrame of weights with **same shape, index, and columns**
    as `df`, reproducibly drawn from a truncated Gaussian."""
    arr = truncated_normal_array(df.shape, mean, std, low, high, seed)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def ohlcv_from_csv(filename):
    return pd.read_csv(
        filename,
        index_col=["symbol", "dt"],
        parse_dates=["dt"],
        dtype={
            "o": np.float64,
            "h": np.float64,
            "l": np.float64,
            "c": np.float64,
            "v": np.float64,
        },
        dayfirst=True,
    )


def load_close_prices(symbols: list):
    prices_filename = f"{workspace_root}/tests/testdata/binance-margin-1d.csv"
    market = ohlcv_from_csv(prices_filename)
    market = market[~market.index.duplicated()]
    market = market.unstack(level=0).sort_index(axis=1).stack()
    prices = pd.DataFrame(
        market.loc[:, ["c"]].unstack(level=1).droplevel(level=0, axis=1)
    )[symbols]
    return prices


def test_backtest_fixed_weights():
    """Assert that asset performance is equal to strategy performance when using fixed weights."""

    prices = load_close_prices(["BTCUSDT"])
    weights = prices.copy()
    weights[:] = 1

    perf, _, _, _, _ = av.backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        lags=0,
    )
    assert (
        perf.loc["BTCUSDT", ("asset", "annual_sharpe")]
        == perf.loc["BTCUSDT", ("strategy", "annual_sharpe")]
    )


def test_backtest_changing_weights():
    """Assert that portfolio performance is equal to a known external source (portfolioslab.com)."""
    prices = load_close_prices(["ETHUSDT", "BTCUSDT"])
    weights = weights_like(prices)

    perf, _, _, port_perf, _ = av.backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        lags=1,
    )
    assert perf.shape == (len(prices.columns), 9)

    assert port_perf.loc["observed", "annual_sharpe"] == pytest.approx(
        -0.4708, abs=0.01
    )
    assert port_perf.loc["observed", "annual_volatility"]
    assert port_perf.loc["observed", "cagr"]
    assert port_perf.loc["observed", "max_drawdown"]
    assert port_perf.loc["observed", "annual_turnover"]


def test_pct_commission():
    weights = pd.Series([0, np.nan, 0, 1, -2.5])
    fee_pct = 0.10

    act = av.pct_commission(weights, fee_pct)

    # Expected: |Δw| * fee_pct  (weights are portfolio‑equity fractions)
    expected = weights.fillna(0).diff().abs() * fee_pct

    # Compare element‑wise, treating NaN == NaN
    assert_allclose(act.fillna(0), expected.fillna(0))


def test_ann_turnover():
    """Turnover should be zero when there are no trades and strictly non‑negative otherwise."""
    # No trades case
    weights = pd.Series([0.5, 0.5, 0.5])
    assert av._ann_turnover(weights).eq(0).all()

    # ------------------------------------------------------------------ #
    # 1️⃣  Correctness check (daily data, 252 periods per year)
    # ------------------------------------------------------------------ #
    # Weights for three consecutive trading days
    #   t0 -> 50 %
    #   t1 -> 60 %  (Δ = +10 %)
    #   t2 -> 40 %  (Δ = –20 %)
    weights = pd.Series([0.50, 0.60, 0.40], name="w")

    # Average annual turnover:
    #   Δw = [0.00, 0.10, 0.20]               (abs diff, NaN→0)
    #   mean(|Δw|) = 0.10 / 3 = 0.10
    #   turnover = 0.5 × mean(|Δw|) × 252
    expected_scalar = 0.5 * weights.diff().abs().fillna(0).mean() * 252
    expected = pd.Series([expected_scalar], index=["w"], name="w")

    pd.testing.assert_series_equal(
        av._ann_turnover(weights),
        expected,
        check_names=False,  # we don’t care about the Series name
    )


def test_spread():
    weights = pd.Series([np.nan, 0.5, -2.5])
    spread_pct = 0.02

    act = av._spread(weights, spread_pct)

    expected = weights.fillna(0).diff().abs() * (spread_pct * 0.5)
    assert_allclose(act.fillna(0), expected.fillna(0))


def test_borrow():
    """
    Borrow cost should equal (|w| adj − 1)*period_rate whenever absolute
    exposure exceeds the 1× equity threshold.

    period_rate = (1 + annual_rate) ** (1 / periods) − 1
    """
    ann_rate = 0.10
    periods = 11
    period_rate = (1 + ann_rate) ** (1 / periods) - 1  # ≈ 0.008702

    # ----- Case 1: no borrowing required (|w| <= 1) -----
    weights = pd.Series([0.50, 0.30])
    act = av._borrow(weights, period_rate, periods)
    assert act.eq(0).all()

    # ----- Case 2: short side borrowing (|w| < 0) -----
    weights = pd.Series([0.5, -0.1])
    act = av._borrow(weights, period_rate, periods)
    expected = pd.Series([0.0, 0.1 * period_rate])
    assert_allclose(act, expected)

    # ----- Case 3: leveraged long (w > 1) -----
    weights = pd.Series([2.0, 0.5])
    act = av._borrow(weights, period_rate, periods)
    expected = pd.Series([(2.0 - 1) * period_rate, 0.0])
    assert_allclose(act, expected)

    # ----- Case 4: dataframe handling -----
    weights_df = pd.DataFrame({0: [0.5, 0.0], 1: [-0.3, 0.0]})
    act_df = av._borrow(weights_df, period_rate, periods)
    expected_df = (
        pd.DataFrame(
            {0: [0.0, 0.0], 1: [0.3 + 1 - 1, 0.0]}  # only short position taxed
        )
        * period_rate
    )
    assert_allclose(act_df.fillna(0), expected_df.fillna(0))

    # ----- Case 5: perp funding (cost applies to the full notional) -----
    weights = pd.Series([0.5, -0.3])
    act = av._borrow(weights, period_rate, periods, is_perp_funding=True)
    expected = pd.Series([0.5 * period_rate, 0.3 * period_rate])
    assert_allclose(act, expected)
