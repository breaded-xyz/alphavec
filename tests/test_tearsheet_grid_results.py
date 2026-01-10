import numpy as np
import pandas as pd

from alphavec import MarketData, SimConfig, grid_search, tearsheet


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
    exec_prices = close_prices.copy()

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
            {"lookback": [5, 10], "leverage": [0.5, 1.0]},
        ],
        objective_metric="Annualized Sharpe",
        max_workers=2,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
        config=SimConfig(),
    )
    assert results.best is not None

    html = tearsheet(grid_results=results)
    assert "<h2>Parameter Search</h2>" in html
    assert "Signal: Alpha Decay by Lag (Next-Period Return per Gross)" in html
    assert results.param_grids[0].label() in html


def test_tearsheet_renders_1d_grid_line_plots():
    """Test that tearsheet renders line plots for 1D grids."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=80, freq="1D")
    assets = ["BTC", "ETH", "SOL"]

    rets = pd.DataFrame(
        rng.normal(0.0004, 0.02, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    close_prices = (100.0 * (1.0 + rets).cumprod()).astype(float)
    exec_prices = close_prices.copy()

    def generate_weights(params: dict) -> pd.DataFrame:
        lookback = int(params["lookback"])
        mom = close_prices.pct_change(lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = np.sign(mom) * np.sqrt(np.abs(mom))
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
        w = raw.div(denom, axis=0).fillna(0.0)
        return w

    # 1D grid - single parameter
    results = grid_search(
        generate_weights=generate_weights,
        base_params={},
        param_grids=[
            {"lookback": [3, 5, 7, 10, 15]},  # 1D grid
        ],
        objective_metric="Annualized Sharpe",
        max_workers=2,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
        config=SimConfig(),
    )
    assert results.best is not None
    assert results.param_grids[0].is_1d()

    html = tearsheet(grid_results=results)
    assert "<h2>Parameter Search</h2>" in html
    # Should contain line plot title format (not heatmap)
    assert f"{results.objective_metric} vs lookback" in html
    # Should contain the 1D-specific note
    assert "1D parameter grid" in html
