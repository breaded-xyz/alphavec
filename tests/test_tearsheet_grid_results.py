import numpy as np
import pandas as pd

from alphavec import Grid2D, MarketData, SimConfig, grid_search, tearsheet


def test_tearsheet_renders_grid_heatmaps():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=80, freq="1D")
    assets = ["BTC", "ETH", "SOL"]

    rets = pd.DataFrame(
        rng.normal(0.0004, 0.02, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    close_prices = (100.0 * (1.0 + rets).cumprod()).astype(float)
    order_prices = close_prices.copy()

    def generate_weights(params: dict) -> pd.DataFrame:
        lookback = int(params["lookback"])
        leverage = float(params["leverage"])
        mom = close_prices.pct_change(lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = np.sign(mom) * np.sqrt(np.abs(mom))
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
        w = raw.div(denom, axis=0).fillna(0.0)
        return w * leverage

    results = grid_search(
        generate_weights=generate_weights,
        base_params={},
        param_grids=[
            Grid2D(
                param1_name="lookback",
                param1_values=[5, 10],
                param2_name="leverage",
                param2_values=[0.5, 1.0],
            )
        ],
        objective_metric="Annualized Sharpe",
        max_workers=2,
        market=MarketData(close_prices=close_prices, order_prices=order_prices, funding_rates=None),
        config=SimConfig(),
    )
    assert results.best is not None

    html = tearsheet(grid_results=results)
    assert "<h2>Parameter Search</h2>" in html
    assert results.param_grids[0].label() in html
