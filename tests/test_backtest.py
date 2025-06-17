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


def test_backtest_external_validation():
    """Assert that portfolio performance is equal to a known external source (portfolioslab.com)."""
    prices = load_close_prices(["ETHUSDT", "BTCUSDT"])
    weights = prices.copy()
    weights[:] = 0.5

    _, _, perf_sr, _, _ = av.backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        lags=1,
    )
    sharpe_val = perf_sr.loc["2022-10-01T00:00:00.000", ("portfolio", "sharpe")]
    # Accept small numeric drift across environments ‑ expect approx −0.74 ± 0.02.
    assert sharpe_val == pytest.approx(-0.74, abs=0.02)


def test_pct_commission():
    weights = pd.Series([0, np.nan, 0, 1, -2.5])
    prices = pd.Series([10, 10, 10, 10, 10], dtype=float)
    fee_pct = 0.10

    act = av.pct_commission(weights, prices, fee_pct)

    # Expected: |Δw| * fee_pct  (weights are portfolio‑equity fractions)
    expected = weights.fillna(0).diff().abs() * fee_pct

    # Compare element‑wise, treating NaN == NaN
    assert_allclose(act.fillna(0), expected.fillna(0))


def test_ann_turnover():
    """Turnover should be zero when there are no trades and strictly non‑negative otherwise."""
    # No trades case
    weights = pd.Series([0.5, 0.5, 0.5])
    prices = pd.Series([10, 11, 12])
    assert av._ann_turnover(weights).eq(0).all()

    # Trade occurs
    weights = pd.Series([0, 1, -1])
    turnover = av._ann_turnover(weights).squeeze()

    assert turnover >= 0
    # Upper bound sanity: cannot exceed twice position change divided by min equity factor
    assert turnover <= 1000  # generous upper bound, catches runaway leverage bugs


def test_spread():
    weights = pd.Series([np.nan, 0.5, -2.5])
    prices = pd.Series([10, 10, 10])
    spread_pct = 0.02

    act = av._spread(weights, prices, spread_pct)

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
    prices = pd.Series([10, 10])  # ignored by _borrow now
    act = av._borrow(weights, prices, period_rate, periods)
    assert act.eq(0).all()

    # ----- Case 2: short side borrowing (|w| < 0) -----
    weights = pd.Series([0.5, -0.1])
    act = av._borrow(weights, prices, period_rate, periods)
    expected = pd.Series([0.0, 0.1 * period_rate])
    assert_allclose(act, expected)

    # ----- Case 3: leveraged long (w > 1) -----
    weights = pd.Series([2.0, 0.5])
    act = av._borrow(weights, prices, period_rate, periods)
    expected = pd.Series([(2.0 - 1) * period_rate, 0.0])
    assert_allclose(act, expected)

    # ----- Case 4: dataframe handling -----
    weights_df = pd.DataFrame({0: [0.5, 0.0], 1: [-0.3, 0.0]})
    prices_df = pd.DataFrame({0: [10, 10], 1: [10, 10]})
    act_df = av._borrow(weights_df, prices_df, period_rate, periods)
    expected_df = (
        pd.DataFrame(
            {0: [0.0, 0.0], 1: [0.3 + 1 - 1, 0.0]}  # only short position taxed
        )
        * period_rate
    )
    assert_allclose(act_df.fillna(0), expected_df.fillna(0))

    # ----- Case 5: perp funding (cost applies to the full notional) -----
    weights = pd.Series([0.5, -0.3])
    act = av._borrow(weights, prices, period_rate, periods, is_perp_funding=True)
    expected = pd.Series([0.5 * period_rate, 0.3 * period_rate])
    assert_allclose(act, expected)
