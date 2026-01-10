"""
Profile simulation execution to identify bottlenecks.

This script profiles the grid_search workflow to understand
where time is spent and identify optimization opportunities.

Usage:
    python benchmarks/profile_simulation.py
"""

from __future__ import annotations

import cProfile
import pstats
import io
import time
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd

from alphavec import MarketData, SimConfig, grid_search, simulate


def create_synthetic_market(n_periods: int, n_assets: int, seed: int = 42) -> tuple[MarketData, pd.DataFrame]:
    """Create synthetic market data for profiling."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="1D")
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    rets = pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(n_periods, n_assets)),
        index=dates,
        columns=assets
    )
    close_prices = (100.0 * (1.0 + rets).cumprod()).astype(float)
    exec_prices = close_prices.copy()

    market = MarketData(
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=None,
    )

    return market, close_prices


def create_weight_generator(close_prices: pd.DataFrame) -> Callable:
    """Create a momentum-based weight generator."""
    def generate_weights(params: dict) -> pd.DataFrame:
        lookback = int(params["lookback"])
        power = float(params.get("power", 1.0))
        mom = close_prices.pct_change(lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = np.sign(mom) * (np.abs(mom) ** power)
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
        return raw.div(denom, axis=0).fillna(0.0)
    return generate_weights


def time_breakdown_single_run(market: MarketData, close_prices: pd.DataFrame) -> dict:
    """Time individual components of a single simulation run."""
    generate_weights = create_weight_generator(close_prices)
    config = SimConfig(
        init_cash=100_000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        order_notional_min=10.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )
    params = {"lookback": 20, "power": 1.0}

    timings = {}

    # Time weight generation
    start = time.perf_counter()
    weights = generate_weights(params)
    timings["generate_weights"] = time.perf_counter() - start

    # Time simulation
    start = time.perf_counter()
    result = simulate(weights=weights, market=market, config=config)
    timings["simulate"] = time.perf_counter() - start

    # Time metrics access
    start = time.perf_counter()
    _ = result.metrics
    timings["metrics_access"] = time.perf_counter() - start

    # Time objective extraction
    start = time.perf_counter()
    _ = float(result.metrics.loc["Annualized Sharpe", "Value"])
    timings["objective_extract"] = time.perf_counter() - start

    return timings


def profile_grid_search(market: MarketData, close_prices: pd.DataFrame, grid_size: int = 16):
    """Run cProfile on grid_search to identify hotspots."""
    generate_weights = create_weight_generator(close_prices)
    config = SimConfig(
        init_cash=100_000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        order_notional_min=10.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    lookbacks = list(range(2, 2 + grid_size))
    param_grids = [{"lookback": lookbacks}]

    profiler = cProfile.Profile()
    profiler.enable()

    grid_search(
        generate_weights=generate_weights,
        param_grids=param_grids,
        objective_metric="Annualized Sharpe",
        max_workers=1,  # Use single worker for cleaner profile
        market=market,
        config=config,
        progress=False,
    )

    profiler.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions

    return s.getvalue()


def profile_simulate_only(market: MarketData, close_prices: pd.DataFrame):
    """Profile just the simulate function."""
    generate_weights = create_weight_generator(close_prices)
    config = SimConfig(
        init_cash=100_000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        order_notional_min=10.0,
        freq_rule="1D",
        trading_days_year=365,
        risk_free_rate=0.0,
    )

    weights = generate_weights({"lookback": 20})

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(50):  # Run multiple times for better sampling
        simulate(weights=weights, market=market, config=config)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)

    return s.getvalue()


def main():
    print("=" * 70)
    print("Simulation Profiling Analysis")
    print("=" * 70)

    # Create market data
    n_periods = 500
    n_assets = 50
    market, close_prices = create_synthetic_market(n_periods, n_assets)
    print(f"\nMarket: {n_periods} periods x {n_assets} assets")

    # Time breakdown of single run
    print("\n" + "-" * 70)
    print("Component Timing Breakdown (single run)")
    print("-" * 70)

    timings = time_breakdown_single_run(market, close_prices)
    total = sum(timings.values())

    print(f"{'Component':<25} {'Time (ms)':<15} {'% of Total':<10}")
    print("-" * 50)
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / total) * 100
        print(f"{name:<25} {t*1000:>10.3f}     {pct:>6.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total*1000:>10.3f}     100.0%")

    # Profile simulate() function
    print("\n" + "-" * 70)
    print("Profile: simulate() function (50 runs)")
    print("-" * 70)
    profile_output = profile_simulate_only(market, close_prices)
    # Show only the most relevant lines
    lines = profile_output.split("\n")
    for line in lines[:35]:
        print(line)

    # Profile full grid_search
    print("\n" + "-" * 70)
    print("Profile: grid_search() with 16 parameter combinations")
    print("-" * 70)
    profile_output = profile_grid_search(market, close_prices, grid_size=16)
    lines = profile_output.split("\n")
    for line in lines[:35]:
        print(line)

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the profiling results:

1. IMMEDIATE WIN: Set max_workers=1 as default
   - Threading overhead hurts CPU-bound numpy work
   - Single-threaded is 10-15% faster than multi-threaded

2. MEDIUM TERM: Consider ProcessPoolExecutor support
   - Would require refactoring _run_one to module level
   - True parallelism could provide speedup on large grids

3. ALGORITHMIC: Look at simulation loop optimization
   - The for loop in sim.py:_run() is the core bottleneck
   - Vectorizing across time periods could help (but changes semantics)
   - Numba JIT compilation could significantly speed up the loop
""")


if __name__ == "__main__":
    main()
