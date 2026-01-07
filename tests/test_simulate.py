import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from alphavec import MarketData, SimConfig, metrics_artifacts, simulate, tearsheet


def _sim(
    *,
    weights: pd.DataFrame | pd.Series,
    close_prices: pd.DataFrame | pd.Series,
    exec_prices: pd.DataFrame | pd.Series,
    funding_rates: pd.DataFrame | pd.Series | None,
    **config_overrides: object,
):
    market = MarketData(
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
    )
    config = SimConfig(**config_overrides)
    return simulate(weights=weights, market=market, config=config)


def _simulate_reference(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    exec_prices: pd.DataFrame,
    funding_rates: pd.DataFrame | None,
    order_notional_min: float,
    fee_pct: float,
    slippage_pct: float,
    init_cash: float,
    trading_days_year: int,
) -> tuple[pd.Series, pd.DataFrame]:
    columns = list(weights.columns)
    positions: dict[str, float] = {c: 0.0 for c in columns}
    cash = float(init_cash)

    equities: list[float] = []
    fees: list[float] = []
    funding: list[float] = []
    first_order_date: pd.Timestamp | None = None

    for ts in weights.index:
        w_row = weights.loc[ts]
        op_row = exec_prices.loc[ts]
        cp_row = close_prices.loc[ts]
        fr_row = (
            funding_rates.loc[ts]
            if funding_rates is not None
            else pd.Series(0.0, index=columns)
        )

        equity_before = cash + sum(positions[c] * float(op_row[c]) for c in columns)

        deltas: dict[str, float] = {}
        closing: dict[str, bool] = {}
        for c in columns:
            w_c = w_row[c]
            weight_c = float(w_c) if pd.notna(w_c) else 0.0
            target_notional = weight_c * equity_before
            current_notional = positions[c] * float(op_row[c])
            delta = target_notional - current_notional
            deltas[c] = delta
            closing[c] = pd.isna(w_c) or float(w_c) == 0.0

        period_fee = 0.0
        for c in columns:
            delta = deltas[c]
            if abs(delta) < order_notional_min and not closing[c]:
                delta = 0.0
            if delta == 0.0:
                continue

            exec_price = float(op_row[c]) * (
                1.0 + slippage_pct if delta > 0.0 else 1.0 - slippage_pct
            )
            traded_units = delta / exec_price
            fee = abs(delta) * fee_pct

            cash -= delta + fee
            positions[c] += traded_units
            period_fee += fee

            if first_order_date is None:
                first_order_date = ts

        close_notional_sum = 0.0
        funding_payment = 0.0
        for c in columns:
            close_notional = positions[c] * float(cp_row[c])
            close_notional_sum += close_notional
            funding_payment += -float(fr_row[c]) * close_notional

        cash += funding_payment

        equities.append(cash + close_notional_sum)
        fees.append(period_fee)
        funding.append(funding_payment)

    equity_series = pd.Series(equities, index=weights.index, name="equity")
    returns = equity_series.pct_change().fillna(0.0)

    pnl_curve = equity_series - init_cash
    max_dd_equity = float((equity_series / equity_series.cummax() - 1.0).min())

    running_max_pnl = pnl_curve.cummax().replace(0.0, np.nan)
    dd_pnl = ((pnl_curve - pnl_curve.cummax()) / running_max_pnl).min(skipna=True)
    max_dd_pnl = float(dd_pnl) if np.isfinite(dd_pnl) else 0.0

    annual_factor = float(trading_days_year)
    annual_return = float((1.0 + returns).prod() ** (annual_factor / len(returns)) - 1.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(annual_factor))
    annual_sharpe = float(annual_return / annual_vol) if annual_vol > 0 else np.nan

    metrics = {
        "Simulation start date": weights.index.min(),
        "Simulation end date": weights.index.max(),
        "First transaction date": first_order_date,
        "Total return %": float(equity_series.iloc[-1] / init_cash - 1.0) * 100.0,
        "Max drawdown (equity) %": max_dd_equity * 100.0,
        "Max drawdown (PnL) %": max_dd_pnl * 100.0,
        "Funding earnings": float(sum(funding)),
        "Fees": float(sum(fees)),
        "Annualized Sharpe": annual_sharpe,
    }
    tearsheet = pd.DataFrame(
        [{"Value": v, "Note": ""} for v in metrics.values()],
        index=pd.Index(metrics.keys(), name="Metric"),
        columns=["Value", "Note"],
    )

    return returns, tearsheet


def test_simulate_oracle():
    # Case: Closed-form single-asset with fees (no slippage or funding).
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 110.0, 120.0]}, index=dates)
    exec_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [0.0, 1.0, 0.0]}, index=dates)

    init_cash = 1000.0
    fee_pct = 0.001

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=init_cash,
        fee_rate=fee_pct,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = pd.Series(
        [1000.0, 999.0, 1088.8181818182],
        index=dates,
        name="equity",
    )
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(
        float((expected_equity.iloc[-1] / init_cash - 1.0) * 100.0)
    )
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(2.0909090909)
    assert float(metrics.loc["Funding earnings", "Value"]) == pytest.approx(0.0)
    assert float(metrics.loc["Max drawdown (equity) %", "Value"]) == pytest.approx(-0.1)
    assert float(metrics.loc["Max drawdown (PnL) %", "Value"]) == pytest.approx(0.0)
    assert int(metrics.loc["Total order count", "Value"]) == 2
    assert float(metrics.loc["Average order notional", "Value"]) == pytest.approx(1045.4545454545)
    assert float(metrics.loc["Gross exposure max %", "Value"]) == pytest.approx(100.1001001001)
    assert float(metrics.loc["Gross exposure mean %", "Value"]) == pytest.approx(33.3667000334)
    assert pd.isna(metrics.loc["Benchmark Asset", "Value"])


def test_simulate_reference():
    # Case: Differential vs independent scalar reference (two assets with costs).
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame(
        {
            "BTC": [100.0, 110.0, 105.0, 120.0, 118.0, 125.0],
            "ETH": [10.0, 9.0, 11.0, 10.0, 12.0, 11.0],
        },
        index=dates,
    )
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame(
        {
            "BTC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ETH": [0.5, 0.5, -0.5, -0.5, 0.0, 0.5],
        },
        index=dates,
    )
    funding_rates = pd.DataFrame(
        {
            "BTC": [0.0001] * 6,
            "ETH": [-0.0002, 0.0003, 0.0003, -0.0001, 0.0, 0.0002],
        },
        index=dates,
    )

    init_cash = 1000.0
    fee_pct = 0.001
    slippage_pct = 0.0005
    trading_days_year = 365

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
        init_cash=init_cash,
        fee_rate=fee_pct,
        slippage_rate=slippage_pct,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=trading_days_year,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    ref_returns, ref_tearsheet = _simulate_reference(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
        order_notional_min=0.0,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        init_cash=init_cash,
        trading_days_year=trading_days_year,
    )

    assert np.allclose(returns.to_numpy(), ref_returns.to_numpy())
    for key in [
        "Total return %",
        "Max drawdown (equity) %",
        "Max drawdown (PnL) %",
        "Funding earnings",
        "Fees",
        "Annualized Sharpe",
    ]:
        assert float(metrics.loc[key, "Value"]) == pytest.approx(float(ref_tearsheet.loc[key, "Value"]))
    assert pd.isna(metrics.loc["Benchmark Asset", "Value"])


def test_simulate_series_inputs():
    # Case: Series inputs are accepted and behave like single-column frames.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close = pd.Series([100.0, 110.0, 120.0], index=dates, name="BTC")
    order = close.shift(1).fillna(close.iloc[0])
    weights = pd.Series([1.0, 1.0, 1.0], index=dates, name="BTC")

    result = _sim(
        weights=weights,
        close_prices=close,
        exec_prices=order,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = 1000.0 * close / close.iloc[0]
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Funding earnings", "Value"]) == pytest.approx(0.0)
    assert pd.isna(metrics.loc["Benchmark Asset", "Value"])


def test_simulate_period_slicing():
    # Case: Datetime string slicing uses loc semantics.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
        index=dates,
    )
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0] * len(dates)}, index=dates)

    config = {
        "init_cash": 1000.0,
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
        "order_notional_min": 0.0,
        "freq_rule": "1D",
        "trading_days_year": 365,
        "risk_free_rate": 0.0,
    }

    result_loc = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        start_period="2024-01-02",
        end_period="2024-01-05",
        **config,
    )
    expected_loc = _sim(
        weights=weights.loc["2024-01-02":"2024-01-05"],
        close_prices=close_prices.loc["2024-01-02":"2024-01-05"],
        exec_prices=exec_prices.loc["2024-01-02":"2024-01-05"],
        funding_rates=None,
        **config,
    )

    assert result_loc.returns.index.equals(expected_loc.returns.index)
    assert np.allclose(result_loc.returns.to_numpy(), expected_loc.returns.to_numpy())
    assert result_loc.metrics.loc["Simulation start date", "Value"] == expected_loc.metrics.loc[
        "Simulation start date", "Value"
    ]
    assert result_loc.metrics.loc["Simulation end date", "Value"] == expected_loc.metrics.loc[
        "Simulation end date", "Value"
    ]

    # Case: iloc slicing uses positional semantics.
    result_iloc = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        start_period=1,
        end_period=4,
        **config,
    )
    expected_iloc = _sim(
        weights=weights.iloc[1:4],
        close_prices=close_prices.iloc[1:4],
        exec_prices=exec_prices.iloc[1:4],
        funding_rates=None,
        **config,
    )

    assert result_iloc.returns.index.equals(expected_iloc.returns.index)
    assert np.allclose(result_iloc.returns.to_numpy(), expected_iloc.returns.to_numpy())
    assert result_iloc.metrics.loc["Simulation start date", "Value"] == expected_iloc.metrics.loc[
        "Simulation start date", "Value"
    ]
    assert result_iloc.metrics.loc["Simulation end date", "Value"] == expected_iloc.metrics.loc[
        "Simulation end date", "Value"
    ]


def test_simulate_trim_warmup_basic():
    # Case: trim_warmup slices to first row with finite weights.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
        index=dates,
    )
    exec_prices = close_prices.copy()
    # First 2 rows are NaN (warm-up period)
    weights = pd.DataFrame(
        {"BTC": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0]},
        index=dates,
    )

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        trim_warmup=True,
    )

    # Result should start at 2024-01-03 (index 2), the first non-NaN row
    assert result.returns.index[0] == pd.Timestamp("2024-01-03")
    assert len(result.returns) == 4


def test_simulate_trim_warmup_with_later_start_period():
    # Case: start_period is later than warmup end, so start_period wins.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
        index=dates,
    )
    exec_prices = close_prices.copy()
    # First 2 rows are NaN (warm-up ends at index 2 = 2024-01-03)
    weights = pd.DataFrame(
        {"BTC": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0]},
        index=dates,
    )

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        trim_warmup=True,
        start_period="2024-01-04",  # Later than warmup end
    )

    # start_period is later, so it takes precedence
    assert result.returns.index[0] == pd.Timestamp("2024-01-04")
    assert len(result.returns) == 3


def test_simulate_trim_warmup_with_earlier_start_period():
    # Case: start_period is earlier than warmup end, so warmup end wins.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
        index=dates,
    )
    exec_prices = close_prices.copy()
    # First 2 rows are NaN (warm-up ends at index 2 = 2024-01-03)
    weights = pd.DataFrame(
        {"BTC": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0]},
        index=dates,
    )

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        trim_warmup=True,
        start_period="2024-01-02",  # Earlier than warmup end
    )

    # warmup end (2024-01-03) is later, so it takes precedence
    assert result.returns.index[0] == pd.Timestamp("2024-01-03")
    assert len(result.returns) == 4


def test_simulate_order_notional_min_skips_small_rebalance():
    # Case: Small rebalance below min notional is skipped.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 100.0, 100.0]}, index=dates)
    exec_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [1.0, 1.01, 1.01]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.01,
        slippage_rate=0.0,
        order_notional_min=1.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = pd.Series([990.0, 990.0, 990.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(10.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(-1.0)


def test_simulate_closing_ignores_order_notional_min():
    # Case: Closing trades execute even below min notional.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 50.0, 50.0]}, index=dates)
    exec_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [1.0, 0.0, 0.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.01,
        slippage_rate=0.0,
        order_notional_min=750.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = pd.Series([990.0, 485.0, 485.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(15.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(-51.5)


def test_simulate_funding_sign_convention():
    # Case: Positive funding rate means longs pay, shorts earn.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 100.0, 100.0]}, index=dates)
    exec_prices = close_prices.copy()
    funding_rates = pd.DataFrame({"BTC": [0.01, 0.01, 0.01]}, index=dates)

    # Case: long pays.
    weights_long = pd.DataFrame({"BTC": [1.0, 1.0, 1.0]}, index=dates)
    metrics_long = _sim(
        weights=weights_long,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    ).metrics
    assert float(metrics_long.loc["Funding earnings", "Value"]) == pytest.approx(-29.701)
    assert float(metrics_long.loc["Total return %", "Value"]) == pytest.approx(-2.9701)

    # Case: short earns.
    weights_short = pd.DataFrame({"BTC": [-1.0, -1.0, -1.0]}, index=dates)
    metrics_short = _sim(
        weights=weights_short,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    ).metrics
    assert float(metrics_short.loc["Funding earnings", "Value"]) == pytest.approx(30.301)
    assert float(metrics_short.loc["Total return %", "Value"]) == pytest.approx(3.0301)


def test_simulate_mismatched_inputs_raises():
    # Case: Prices missing weight columns raise.
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    weights = pd.DataFrame({"BTC": [1.0, 1.0]}, index=dates)
    close_prices = pd.DataFrame({"ETH": [10.0, 10.0]}, index=dates)
    exec_prices = close_prices.copy()

    with pytest.raises(ValueError):
        simulate(
            weights=weights,
            market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
        )


def test_simulate_nan_order_skips_open_allows_close():
    # Case: NaN order price skips opening/rebalancing but still allows closing via carry-forward.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 100.0, 100.0]}, index=dates)
    exec_prices = pd.DataFrame({"BTC": [100.0, np.nan, np.nan]}, index=dates)
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 0.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = pd.Series([1000.0, 1000.0, 1000.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(0.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(0.0)


def test_simulate_nan_close_carries_forward_and_zero_funding():
    # Case: NaN close carries forward last close for valuation and implies zero funding.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, np.nan, 100.0]}, index=dates)
    exec_prices = pd.DataFrame({"BTC": [100.0, 100.0, 100.0]}, index=dates)
    funding_rates = pd.DataFrame({"BTC": [0.0, 0.01, 0.0]}, index=dates)
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 1.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    expected_equity = pd.Series([1000.0, 1000.0, 1000.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Funding earnings", "Value"]) == pytest.approx(0.0)


def test_simulate_alpha_beta_benchmark():
    # Case: Portfolio identical to benchmark has beta~1 and alpha~0.
    dates = pd.date_range("2024-01-01", periods=5, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 110.0, 105.0, 120.0, 115.0]}, index=dates)
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0] * len(dates)}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        benchmark_asset="BTC",
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    returns = result.returns
    metrics = result.metrics

    bench_returns = close_prices["BTC"].pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), bench_returns.to_numpy())
    assert float(metrics.loc["Beta", "Value"]) == pytest.approx(1.0, abs=1e-9)
    assert float(metrics.loc["Alpha", "Value"]) == pytest.approx(0.0, abs=1e-9)
    assert metrics.loc["Benchmark Asset", "Value"] == "BTC"


def test_tearsheet_renders_html(tmp_path: Path):
    # Case: HTML is produced and can be written to disk.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 110.0, 120.0]}, index=dates)
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 1.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        benchmark_asset="BTC",
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    out = tmp_path / "tearsheet.html"
    html = tearsheet(sim_result=result, output_path=out, smooth_periods=1)
    assert "<title>Tearsheet</title>" in html
    assert "<h1>Tearsheet</h1>" in html
    assert "Equity Curve (Cumulative Return %)" in html
    assert "Drawdown (%)" in html
    assert "Rolling Sharpe" in html
    assert "Returns Distribution (Per-Period, %)" in html
    assert "Returns Q-Q Plot" in html
    assert "Signal: Next-Period Return by Weight Decile (Mean/Median)" in html
    assert "Signal: Next-Period Return Contribution by Weight Decile (Long/Short)" in html
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_tearsheet_does_not_store_smoothed_series(tmp_path: Path):
    # Case: Smoothed series are not stored; rolling Sharpe is still stored per period.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates)
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.001,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    out = tmp_path / "smooth.html"
    tearsheet(sim_result=result, output_path=out, smooth_periods=3)

    metrics = result.metrics
    assert "rolling_sharpe_3" in metrics.attrs
    series = metrics.attrs["rolling_sharpe_3"]
    assert isinstance(series, pd.Series)
    assert series.index.equals(result.returns.index)
    assert not any("_smooth_" in key for key in metrics.attrs)

    # Case: No smoothing also avoids smoothed series keys.
    result_no_smooth = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.001,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    out = tmp_path / "no_smooth.html"
    tearsheet(sim_result=result_no_smooth, output_path=out, smooth_periods=0)
    assert not any("_smooth_" in key for key in result_no_smooth.metrics.attrs)


def test_metrics_artifacts():
    # Case: SimulationResult.artifacts exposes per-period series.
    dates = pd.date_range("2024-01-01", periods=6, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates)
    exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)

    result = _sim(
        weights=weights,
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    artifacts = result.artifacts
    assert artifacts.returns is result.metrics.attrs["returns"]
    assert artifacts.equity is result.metrics.attrs["equity"]
    assert artifacts.drawdown_pct is result.metrics.attrs["drawdown_pct"]
    assert artifacts.rolling_sharpe(30) is result.metrics.attrs["rolling_sharpe_30"]
    assert artifacts.signal.weight_forward is result.metrics.attrs["weight_forward"]

    # Case: metrics_artifacts provides the same access from the metrics table.
    artifacts_from_metrics = metrics_artifacts(result.metrics)
    assert artifacts_from_metrics.returns is artifacts.returns
    assert artifacts_from_metrics.signal.weight_forward is artifacts.signal.weight_forward
