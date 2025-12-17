import numpy as np
import pandas as pd
import pytest

from alphavec import MarketData, SimConfig, grid_search, simulate


def test_search_and_simulate_matches_simulate_for_one_point():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="1D")
    assets = ["BTC", "ETH", "SOL"]

    rets = pd.DataFrame(rng.normal(0.0005, 0.02, size=(len(dates), len(assets))), index=dates, columns=assets)
    close_prices = (100.0 * (1.0 + rets).cumprod()).astype(float)
    exec_prices = close_prices.copy()

    def generate_weights(params: dict) -> pd.DataFrame:
        lookback = int(params["lookback"])
        power = float(params["power"])
        mom = close_prices.pct_change(lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = np.sign(mom) * (np.abs(mom) ** power)
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
        return raw.div(denom, axis=0).fillna(0.0)

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[
            {"lookback": [2, 5], "power": [0.5, 1.0]},
        ],
        objective_metric="Annualized Sharpe",
        max_workers=2,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
        config=SimConfig(
            init_cash=1000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            order_notional_min=0.0,
            freq_rule="1D",
            trading_days_year=365,
            risk_free_rate=0.0,
        ),
    )

    assert results.objective_metric == "Annualized Sharpe"
    assert results.table.shape[0] == 4
    assert results.best is not None

    grid = results.pivot(grid_index=0)
    assert list(grid.index) == [2, 5]
    assert list(grid.columns) == [0.5, 1.0]

    weights = generate_weights({"lookback": 2, "power": 1.0})
    metrics = simulate(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
        config=SimConfig(
            init_cash=1000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            order_notional_min=0.0,
            freq_rule="1D",
            trading_days_year=365,
            risk_free_rate=0.0,
        ),
    ).metrics
    expected = float(metrics.loc["Annualized Sharpe", "Value"])

    row = results.table[
        (results.table["param1_value"] == 2) & (results.table["param2_value"] == 1.0)
    ].iloc[0]
    assert float(row["objective_value"]) == pytest.approx(expected)

    best_obj = pd.to_numeric(results.table["objective_value"], errors="coerce").max()
    assert float(results.best.objective_value) == pytest.approx(float(best_obj))
    assert float(results.best.result.metrics.loc["Annualized Sharpe", "Value"]) == pytest.approx(
        float(results.best.objective_value)
    )
