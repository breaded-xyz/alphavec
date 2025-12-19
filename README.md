# Alphavec

[![CI](https://github.com/breaded-xyz/alphavec/actions/workflows/ci.yml/badge.svg)](https://github.com/breaded-xyz/alphavec/actions/workflows/ci.yml)
[![Upload Python Package](https://github.com/breaded-xyz/alphavec/actions/workflows/python-publish.yml/badge.svg)](https://github.com/breaded-xyz/alphavec/actions/workflows/python-publish.yml)

>**Disclaimer**
>
>The content provided in this project is for informational purposes only and does not constitute financial advice. This information should not be construed as professional financial advice, and it is recommended to consult with a qualified financial advisor before making any financial decisions.
>
>No liability is accepted for any losses or damages incurred as a result of acting or refraining from action based on the information provided in this project. Use this information at your own risk.
>

```
          $$\           $$\                                               
          $$ |          $$ |                                              
 $$$$$$\  $$ | $$$$$$\  $$$$$$$\   $$$$$$\ $$\    $$\  $$$$$$\   $$$$$$$\ 
 \____$$\ $$ |$$  __$$\ $$  __$$\  \____$$\\$$\  $$  |$$  __$$\ $$  _____|
 $$$$$$$ |$$ |$$ /  $$ |$$ |  $$ | $$$$$$$ |\$$\$$  / $$$$$$$$ |$$ /      
$$  __$$ |$$ |$$ |  $$ |$$ |  $$ |$$  __$$ | \$$$  /  $$   ____|$$ |      
\$$$$$$$ |$$ |$$$$$$$  |$$ |  $$ |\$$$$$$$ |  \$  /   \$$$$$$$\ \$$$$$$$\ 
 \_______|\__|$$  ____/ \__|  \__| \_______|   \_/     \_______| \_______|
              $$ |                                                        
              $$ |                                                        
              \__|                                                                                                         
```

Alphavec is a lightning fast, minimalist, cost-aware vectorized backtest engine inspired by the guys at RobotWealth.

The backtest input is the natural output of a typical quant research process - a time series of portfolio weights. You simply provide a dataframe of weights and a dataframe of close prices and order prices, along with some optional cost parameters and the backtest returns a streamlined performance report with insight into the key metrics.

`alphavec` has first-class support for simulating leveraged perptual futures strategies using a small, fast, verifiable simulation core.

## Rationale

Alphavec is an antidote to the various bloated and complex backtest frameworks.

To validate ideas all you really need is...

```python

weights * returns.shift(-1)
```

The goal was to add just enough extra complexity to this basic formula to support sound development of cost-aware systematic trading strategies.

## Install

Requires Python `>=3.10`

`pip install alphavec`

- From source:
  - `python3 -m venv .venv`
  - `./.venv/bin/pip install -e .`
- For development:
  - `./.venv/bin/pip install -e ".[dev]"`

## Usage

### Notes

- Simulates cross‑margin (cash pooling) with unlimited leverage and borrowing (no liquidations or margin calls).
- Orders execute at `exec_prices` plus slippage and fees.
- Funding applies per period using signed `funding_rates`, +ve rate shorts earn, longs pay, and vice versa for a -ve rate.
- NaNs in `exec_prices` or `close_prices` imply the asset is not tradable that period.
- NaNs in `funding_rates` are treated as 0, and funding is always 0 when `close_prices` is NaN.
- Positions will always be closed if target weight is zero, regardless of minimum notional filter.

### Simulation

`simulate()` runs a cross‑margin perpetual futures backtest from target portfolio weights.

Key inputs:
- `weights`: pandas `DataFrame` with a `DatetimeIndex` and columns for each asset.
  Values are decimal percentage target weights (1.0 = 100% equity invested).
  Positive = long, negative = short. Weights may sum greater than 1 at a time period for leverage.
- `close_prices`, `exec_prices`, `funding_rates` (optional): same shape/index/columns as `weights`.

Returns:
- `returns`: period returns as a pandas `Series`.
- `metrics`: key performance metrics as a pandas `DataFrame` with `Value` and `Note` columns.

Example:

See `examples/simulate.ipynb`

```python
import pandas as pd
from alphavec import MarketData, SimConfig, simulate, tearsheet

weights = pd.DataFrame({"BTC": [1, 1, 1]}, index=pd.date_range("2024-01-01", periods=3, freq="1D"))
close_prices = pd.DataFrame({"BTC": [100, 105, 110]}, index=weights.index)
exec_prices = close_prices.shift(1).fillna(close_prices.iloc[0])

result = simulate(
    weights=weights,
    market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
    config=SimConfig(
        benchmark_asset="BTC",
        order_notional_min=10,
        fee_rate=0.00025,       # 2.5 bps per trade
        slippage_rate=0.001,    # 10 bps per trade
        init_cash=10_000,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.03,
    ),
)
html_str = tearsheet(
    sim_result=result,
    output_path="tearsheet.html",
    signal_smooth_window=30,
    rolling_sharpe_window=30,
)
```

### Parameter Search

`grid_search()` wraps `simulate()` and runs grid search over 1D or 2D parameter spaces where the objective is a simulation metric.

See `examples/search.ipynb`

```python
from alphavec import grid_search, Metrics, MarketData, SimConfig

# Simple 1D search with dict syntax (most concise)
results = grid_search(
    generate_weights=generate_weights,  # def generate_weights(params: Mapping) -> pd.DataFrame
    objective_metric=Metrics.SHARPE,  # Type-safe metric names
    param_grid={"lookback": [5, 10, 20, 40]},  # Dict syntax - no Grid2D needed!
    market=MarketData(close_prices=close_prices, exec_prices=exec_prices, funding_rates=None),
    config=SimConfig(),
)

# 2D search with multiple grids
results = grid_search(
    generate_weights=generate_weights,
    param_grids=[
        {"lookback": [5, 10, 20], "leverage": [0.5, 1.0, 2.0]},  # 2D dict
        {"lookback": [5, 10, 20], "smooth_span": [1, 5, 10]},   # Another 2D search
    ],
    market=market,
    config=config,
)

# Minimize risk metrics
results = grid_search(
    generate_weights=generate_weights,
    objective_metric=Metrics.MAX_DRAWDOWN,
    maximize=False,  # Minimize drawdown instead of maximize
    param_grid={"lookback": [10, 20, 40]},
    market=market,
)

# Access results
results.table          # Full results DataFrame
results.best.params    # Best parameters
results.best.metrics   # Best run's metrics
results.top(5)         # Top 5 parameter combinations
results.summary()      # Summary statistics by grid
results.plot()         # Smart plot: line for 1D, heatmap for 2D
```

## Metrics

`simulate()` returns a `metrics` DataFrame with `Category`, `Value`, and `Note` columns. Metrics are grouped into categories:

### Categories

1. **Meta**: Simulation metadata and configuration
2. **Performance**: Returns, volatility, Sharpe ratio, drawdowns
3. **Costs**: Fees, funding, turnover, order statistics
4. **Exposure**: Gross/net leverage metrics
5. **Benchmark**: Alpha, beta, tracking error, information ratio (CAPM)
6. **Distribution**: Win/loss stats, skewness, kurtosis, drawdown duration
7. **Portfolio**: Holding periods, weights, cost ratios
8. **Risk**: Sortino, VaR, CVaR, Omega, downside deviation, Ulcer Index
9. **Signal**: ICs, decile spreads, hit-rates, and selection vs directional decomposition (vs next-period returns)

### Additional Diagnostics (for charting)

Some richer time series / grouped diagnostics are attached as `metrics.attrs[...]` for use in the tearsheet:

- `metrics.attrs["equity"]`: equity curve (Series)
- `metrics.attrs["benchmark_equity"]`: benchmark equity curve when `benchmark_asset` is provided (Series)
- `metrics.attrs["weight_forward"]`: per-period signal diagnostics vs next returns (DataFrame), including:
  - `ic`, `rank_ic`
  - `top_bottom_spread`
  - selection vs directional attribution (and per-gross variants)
- `metrics.attrs["weight_forward_deciles"]`: mean next return by weight decile (Series)
- `metrics.attrs["weight_forward_deciles_median"]`: median next return by weight decile (Series)
- `metrics.attrs["weight_forward_deciles_std"]`: std dev of next return by weight decile (Series)
- `metrics.attrs["weight_forward_deciles_count"]`: observation count `n` by weight decile (Series)
- `metrics.attrs["alpha_decay_next_return_by_type"]`: alpha decay curve as a DataFrame indexed by lag (periods), with mean next return per gross and t-stats for:
  - `total_per_gross_*`, `selection_per_gross_*`, `directional_per_gross_*`

`n` is the number of (asset, timestamp) observations that fell into that decile with a non-zero weight and a finite next-period return.

### Statistical Methodology

Alphavec follows **industry-standard statistical practices** for backtesting:

- **Sample statistics** (Bessel's correction, `ddof=1`) for all variance/standard deviation calculations
  - Rationale: Backtests are samples from possible market outcomes, not complete populations
  - Aligns with quantstats, empyrical, pyfolio, and academic finance literature
- **Geometric mean** for total returns (compounds properly over time)
- **Arithmetic mean** for active returns (matches tracking error calculation for Information Ratio)
- **Sample covariance** for beta calculation (CAPM-consistent)
- **Excess kurtosis** (normal distribution = 0, not 3)

This ensures alphavec metrics are directly comparable to industry benchmarks and professional analytics platforms.

## Tearsheet

The built-in `tearsheet()` renderer produces a self-contained HTML report (static charts + metrics table), including:

- Equity curve (portfolio and optional benchmark)
- Drawdown
- Rolling Sharpe (configurable via `rolling_sharpe_window`, default `30`)
- Returns distribution
- Signal diagnostics (when available): directionality and IC/rank-IC vs next-period return scatters with linear fits, alpha decay by lag (total/selection/directional), and decile charts (mean/median, long/short contribution)

## Tearsheet Example

![Tearsheet](examples/tearsheet.png)
