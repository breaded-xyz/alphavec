# alphavec

Fast minimalist vector-based backtesting for perpetual futures portfolios.

## Purpose

`alphavec` provides a small, verifiable simulation core for running cross‑margin perpetual
futures backtests from target portfolio weights, producing both:

- `returns`: period-by-period portfolio returns
- `metrics`: a compact table of performance, risk, trading, and exposure metrics

## Rationale

- **Vector-first**: the simulator works directly on aligned pandas objects and uses NumPy for
  tight loops and aggregation.
- **Backtest clarity**: explicit treatment of execution prices (`order_prices`) vs valuation
  prices (`close_prices`) makes assumptions easier to audit.
- **Practical constraints**: supports fees, slippage, funding, and minimum notional filters while
  keeping the model intentionally simple (cross‑margin, unlimited borrowing).

## Install

Requires Python `>=3.12`.

- From source:
  - `python -m venv .venv`
  - `./.venv/bin/pip install -e .`
- For development:
  - `./.venv/bin/pip install -e ".[dev]"`

## Usage

### simulate

`simulate()` runs a cross‑margin perpetual futures backtest from target portfolio weights.

Inputs:
- `weights`: pandas `DataFrame` or `Series` with a `DatetimeIndex`. Values are decimal target weights
  (1.0 = 100% notional). Positive = long, negative = short. Weights may sum above 1 for leverage.
- `close_prices`, `order_prices`, `funding_rates` (optional): same shape/index/columns as `weights`.

Returns:
- `returns`: period returns as a pandas `Series`.
- `metrics`: key performance metrics as a pandas `DataFrame` with `Value` and `Note` columns.

Metrics includes the core metrics from the spec plus:
- total order count and average order notional
- gross and net exposure summary (% of equity)
- drawdown duration and recovery time
- return distribution stats (best/worst, hit rate, profit factor, skew/kurtosis)
- alpha/beta and tracking stats versus an optional benchmark buy‑and‑hold

Example:

```python
import pandas as pd
from alphavec import simulate, tearsheet

weights = pd.DataFrame({"BTC": [1, 1, 1]}, index=pd.date_range("2024-01-01", periods=3, freq="1D"))
close = pd.DataFrame({"BTC": [100, 105, 110]}, index=weights.index)
order = close.shift(1).fillna(close.iloc[0])

returns, metrics = simulate(
    weights=weights,
    close_prices=close,
    order_prices=order,
    funding_rates=None,
    benchmark_asset="BTC",  # optional
)

html = tearsheet(metrics=metrics, returns=returns, output_path="tearsheet.html")
```

Assumptions:
- Cross‑margin with unlimited leverage and borrowing.
- Orders execute at `order_prices` plus slippage and fees.
- Funding applies per period using signed `funding_rates` and close notional.
- NaNs in `order_prices`/`close_prices` mean the asset is not tradable that period; prices are carried forward for valuation, and closing uses carried‑forward `order_prices`.
- NaNs in `funding_rates` are treated as 0, and funding is always 0 when `close_prices` is NaN.
