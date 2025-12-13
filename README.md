# Alphavec

[![CI](https://github.com/breaded-xyz/alphavec/actions/workflows/ci.yml/badge.svg)](https://github.com/breaded-xyz/alphavec/actions/workflows/ci.yml)

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

Requires Python `>=3.12`

`pip install alphavec`

- From source:
  - `python3 -m venv .venv`
  - `./.venv/bin/pip install -e .`
- For development:
  - `./.venv/bin/pip install -e ".[dev]"`

## Usage

### Notes

- Simulates cross‑margin (cash pooling) with unlimited leverage and borrowing (no liquidations or margin calls).
- Orders execute at `order_prices` plus slippage and fees.
- Funding applies per period using signed `funding_rates`, +ve rate shorts earn, longs pay, and vice versa for a -ve rate.
- NaNs in `order_prices` or `close_prices` imply the asset is not tradable that period.
- NaNs in `funding_rates` are treated as 0, and funding is always 0 when `close_prices` is NaN.
- Positions will always be closed if target weight is zero, regardless of minimum notional filter.

### Simulation

`simulate()` runs a cross‑margin perpetual futures backtest from target portfolio weights.

Key inputs:
- `weights`: pandas `DataFrame` with a `DatetimeIndex` and columns for each asset.
  Values are decimal percentage target weights (1.0 = 100% equity invested).
  Positive = long, negative = short. Weights may sum greater than 1 at a time period for leverage.
- `close_prices`, `order_prices`, `funding_rates` (optional): same shape/index/columns as `weights`.

Returns:
- `returns`: period returns as a pandas `Series`.
- `metrics`: key performance metrics as a pandas `DataFrame` with `Value` and `Note` columns.

Example:

See `examples/example.ipynb`

```python
import pandas as pd
from alphavec import simulate, tearsheet

weights = pd.DataFrame({"BTC": [1, 1, 1]}, index=pd.date_range("2024-01-01", periods=3, freq="1D"))
close_prices = pd.DataFrame({"BTC": [100, 105, 110]}, index=weights.index)
order_prices = close.shift(1).fillna(close.iloc[0])

returns, metrics= simulate(
    weights=weights,
    close_prices=close_prices,
    order_prices=order_prices,
    funding_rates=funding_rates,
    benchmark_asset="BTC",
    order_notional_min=10,
    fee_pct=0.00025,       # 2.5 bps per trade
    slippage_pct=0.001,  # 10 bps per trade
    init_cash=10_000,
    freq_rule="1D",
    trading_days_year=365,
    risk_free_rate=0.03,
)
html_str = tearsheet(metrics=metrics, returns=returns, output_path="tearsheet.html")
```

## Tearsheet Example

![Tearsheet](examples/tearsheet.png)
