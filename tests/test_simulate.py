import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from alphavec import simulate, tearsheet


def _simulate_reference(
    *,
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    order_prices: pd.DataFrame,
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
        op_row = order_prices.loc[ts]
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
    order_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [0.0, 1.0, 0.0]}, index=dates)

    init_cash = 1000.0
    fee_pct = 0.001

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        init_cash=init_cash,
        fee_pct=fee_pct,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

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
    order_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
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

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
        init_cash=init_cash,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=trading_days_year,
        risk_free_rate=0.0,
    )

    ref_returns, ref_tearsheet = _simulate_reference(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
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
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    close = pd.Series([100.0, 110.0], index=dates, name="BTC")
    order = close.shift(1).fillna(close.iloc[0])
    weights = pd.Series([1.0, 1.0], index=dates, name="BTC")

    returns, metrics = simulate(
        weights=weights,
        close_prices=close,
        order_prices=order,
        funding_rates=None,
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    expected_equity = 1000.0 * close / close.iloc[0]
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Funding earnings", "Value"]) == pytest.approx(0.0)
    assert pd.isna(metrics.loc["Benchmark Asset", "Value"])


def test_simulate_order_notional_min_skips_small_rebalance():
    # Case: Small rebalance below min notional is skipped.
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 100.0]}, index=dates)
    order_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [1.0, 1.01]}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_pct=0.01,
        slippage_pct=0.0,
        order_notional_min=1.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    expected_equity = pd.Series([990.0, 990.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(10.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(-1.0)


def test_simulate_closing_ignores_order_notional_min():
    # Case: Closing trades execute even below min notional.
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 50.0]}, index=dates)
    order_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": [1.0, 0.0]}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_pct=0.01,
        slippage_pct=0.0,
        order_notional_min=750.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    expected_equity = pd.Series([990.0, 485.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)

    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(15.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(-51.5)


def test_simulate_funding_sign_convention():
    # Case: Positive funding rate means longs pay, shorts earn.
    dates = pd.date_range("2024-01-01", periods=1, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0]}, index=dates)
    order_prices = close_prices.copy()
    funding_rates = pd.DataFrame({"BTC": [0.01]}, index=dates)

    # Case: long pays.
    weights_long = pd.DataFrame({"BTC": [1.0]}, index=dates)
    _, metrics_long = simulate(
        weights=weights_long,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    assert float(metrics_long.loc["Funding earnings", "Value"]) == pytest.approx(-10.0)
    assert float(metrics_long.loc["Total return %", "Value"]) == pytest.approx(-1.0)

    # Case: short earns.
    weights_short = pd.DataFrame({"BTC": [-1.0]}, index=dates)
    _, metrics_short = simulate(
        weights=weights_short,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    assert float(metrics_short.loc["Funding earnings", "Value"]) == pytest.approx(10.0)
    assert float(metrics_short.loc["Total return %", "Value"]) == pytest.approx(1.0)


def test_simulate_mismatched_inputs_raises():
    # Case: Prices missing weight columns raise.
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    weights = pd.DataFrame({"BTC": [1.0, 1.0]}, index=dates)
    close_prices = pd.DataFrame({"ETH": [10.0, 10.0]}, index=dates)
    order_prices = close_prices.copy()

    with pytest.raises(ValueError):
        simulate(
            weights=weights,
            close_prices=close_prices,
            order_prices=order_prices,
            funding_rates=None,
        )


def test_simulate_nan_order_skips_open_allows_close():
    # Case: NaN order price skips opening/rebalancing but still allows closing via carry-forward.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 100.0, 100.0]}, index=dates)
    order_prices = pd.DataFrame({"BTC": [100.0, np.nan, np.nan]}, index=dates)
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 0.0]}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    expected_equity = pd.Series([1000.0, 1000.0, 1000.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Fees", "Value"]) == pytest.approx(0.0)
    assert float(metrics.loc["Total return %", "Value"]) == pytest.approx(0.0)


def test_simulate_nan_close_carries_forward_and_zero_funding():
    # Case: NaN close carries forward last close for valuation and implies zero funding.
    dates = pd.date_range("2024-01-01", periods=2, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, np.nan]}, index=dates)
    order_prices = pd.DataFrame({"BTC": [100.0, 100.0]}, index=dates)
    funding_rates = pd.DataFrame({"BTC": [0.0, 0.01]}, index=dates)
    weights = pd.DataFrame({"BTC": [1.0, 1.0]}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=funding_rates,
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    expected_equity = pd.Series([1000.0, 1000.0], index=dates)
    expected_returns = expected_equity.pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), expected_returns.to_numpy())
    assert float(metrics.loc["Funding earnings", "Value"]) == pytest.approx(0.0)


def test_simulate_alpha_beta_benchmark():
    # Case: Portfolio identical to benchmark has beta~1 and alpha~0.
    dates = pd.date_range("2024-01-01", periods=5, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 110.0, 105.0, 120.0, 115.0]}, index=dates)
    order_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0] * len(dates)}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        benchmark_asset="BTC",
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    bench_returns = close_prices["BTC"].pct_change().fillna(0.0)
    assert np.allclose(returns.to_numpy(), bench_returns.to_numpy())
    assert float(metrics.loc["Beta", "Value"]) == pytest.approx(1.0, abs=1e-9)
    assert float(metrics.loc["Alpha", "Value"]) == pytest.approx(0.0, abs=1e-9)
    assert metrics.loc["Benchmark Asset", "Value"] == "BTC"


def test_tearsheet_renders_html(tmp_path: Path):
    # Case: HTML is produced and can be written to disk.
    dates = pd.date_range("2024-01-01", periods=3, freq="1D")
    close_prices = pd.DataFrame({"BTC": [100.0, 110.0, 120.0]}, index=dates)
    order_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    weights = pd.DataFrame({"BTC": [1.0, 1.0, 1.0]}, index=dates)

    returns, metrics = simulate(
        weights=weights,
        close_prices=close_prices,
        order_prices=order_prices,
        funding_rates=None,
        benchmark_asset="BTC",
        init_cash=1000.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        order_notional_min=0.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    out = tmp_path / "tearsheet.html"
    html = tearsheet(metrics=metrics, returns=returns, output_path=out, signal_smooth_window=1)
    assert "<title>Tearsheet</title>" in html
    assert "<h1>Tearsheet</h1>" in html
    assert "Equity Curve" in html
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
