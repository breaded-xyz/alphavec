# alphavec

Fast minimalist vector-based backtesting for perpetual futures portfolios.

## simulate

`simulate()` runs a cross‑margin perpetual futures backtest from target portfolio weights.

Inputs:
- `weights`: pandas `DataFrame` or `Series` with a `DatetimeIndex`. Values are decimal target weights
  (1.0 = 100% notional). Positive = long, negative = short. Weights may sum above 1 for leverage.
- `close_prices`, `order_prices`, `funding_rates` (optional): same shape/index/columns as `weights`.

Returns:
- `returns`: period returns as a pandas `Series`.
- `tearsheet`: key performance metrics as a pandas `Series`.

Tearsheet includes the core metrics from the spec plus:
- total order count and average order notional
- max and average gross exposure (% of equity)
- alpha (annualized) and beta versus an optional benchmark buy‑and‑hold

Example:

```python
import pandas as pd
from alphavec import simulate

weights = pd.DataFrame({"BTC": [1, 1, 1]}, index=pd.date_range("2024-01-01", periods=3, freq="1D"))
close = pd.DataFrame({"BTC": [100, 105, 110]}, index=weights.index)
order = close.shift(1).fillna(close.iloc[0])

returns, tearsheet = simulate(
    weights=weights,
    close_prices=close,
    order_prices=order,
    funding_rates=None,
    benchmark_asset="BTC",  # optional
)
```

Assumptions:
- Cross‑margin with unlimited leverage and borrowing.
- Orders execute at `order_prices` plus slippage and fees.
- Funding applies per period using signed `funding_rates` and close notional.
- NaNs in `order_prices`/`close_prices` mean the asset is not tradable that period; prices are carried forward for valuation, and closing uses carried‑forward `order_prices`.
- NaNs in `funding_rates` are treated as 0, and funding is always 0 when `close_prices` is NaN.
