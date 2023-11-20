import sys
import os
from pathlib import PurePath
from functools import partial

import numpy as np
import pandas as pd

from backtest import backtest, _pct_commission

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
    )


def test_backtest():
    prices_filename = f"{workspace_root}/tests/testdata/binance-margin-1d.csv"
    market = ohlcv_from_csv(prices_filename)
    market = market[~market.index.duplicated()]
    market = market.unstack(level=0).sort_index(axis=1).stack()

    marks = pd.DataFrame(
        market.loc[:, ["c"]].unstack(level=1).droplevel(level=0, axis=1)
    )

    weights = marks.copy()
    weights[:] = 1

    perf, perf_cum, perf_sr, port_perf, port_cum = backtest(
        weights,
        marks,
        leverage=2,
        freq_day=1,
        commission_func=partial(_pct_commission, fee=0.001),
        ann_borrow_pct=0.05,
        spread_pct=0.001,
    )

    assert perf.loc["BTCUSDT", ("asset", "sharpe")].round(2) == 0.75
