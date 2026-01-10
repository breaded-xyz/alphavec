"""
Benchmark script for grid_search performance analysis.

Tests different worker counts, executor types, and grid sizes to identify
optimal configuration for parallel execution.

Usage:
    python benchmarks/bench_grid_search.py
"""

from __future__ import annotations

import concurrent.futures as cf
import gc
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from alphavec import MarketData, SimConfig, grid_search


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    executor_type: str
    max_workers: int | None
    grid_size: int
    n_assets: int
    n_periods: int
    elapsed_seconds: float
    runs_per_second: float


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.results])

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        return df.pivot_table(
            index=["executor_type", "max_workers"],
            columns="grid_size",
            values="elapsed_seconds",
            aggfunc="mean"
        )


def create_synthetic_market(n_periods: int, n_assets: int, seed: int = 42) -> tuple[MarketData, pd.DataFrame]:
    """Create synthetic market data for benchmarking."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="1D")
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    # Generate returns and prices
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


def run_benchmark(
    name: str,
    market: MarketData,
    close_prices: pd.DataFrame,
    param_grids: list[dict],
    max_workers: int | None,
    executor: cf.Executor | None,
    executor_type: str,
    n_runs: int = 1,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
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

    # Calculate grid size
    grid_size = 0
    for grid in param_grids:
        sizes = [len(v) for v in grid.values()]
        if len(sizes) == 1:
            grid_size += sizes[0]
        else:
            grid_size += sizes[0] * sizes[1]

    # Warmup run
    gc.collect()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        grid_search(
            generate_weights=generate_weights,
            param_grids=param_grids,
            objective_metric="Annualized Sharpe",
            max_workers=max_workers,
            market=market,
            config=config,
            executor=executor,
            progress=False,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        gc.collect()

    avg_elapsed = sum(times) / len(times)

    return BenchmarkResult(
        name=name,
        executor_type=executor_type,
        max_workers=max_workers,
        grid_size=grid_size,
        n_assets=close_prices.shape[1],
        n_periods=close_prices.shape[0],
        elapsed_seconds=avg_elapsed,
        runs_per_second=grid_size / avg_elapsed,
    )


def benchmark_worker_counts(
    market: MarketData,
    close_prices: pd.DataFrame,
    grid_sizes: list[int],
    worker_counts: list[int | None],
    n_runs: int = 3,
) -> BenchmarkSuite:
    """Benchmark different worker counts with ThreadPoolExecutor."""
    suite = BenchmarkSuite()

    for grid_size in grid_sizes:
        # Create grid with appropriate size
        lookbacks = list(range(2, 2 + grid_size))
        param_grids = [{"lookback": lookbacks}]

        for workers in worker_counts:
            print(f"  ThreadPool workers={workers}, grid_size={grid_size}...", end=" ", flush=True)
            result = run_benchmark(
                name=f"thread_w{workers}_g{grid_size}",
                market=market,
                close_prices=close_prices,
                param_grids=param_grids,
                max_workers=workers,
                executor=None,
                executor_type="ThreadPoolExecutor",
                n_runs=n_runs,
            )
            print(f"{result.elapsed_seconds:.3f}s ({result.runs_per_second:.1f} runs/s)")
            suite.add(result)

    return suite


def benchmark_executor_types(
    market: MarketData,
    close_prices: pd.DataFrame,
    grid_size: int = 16,
    n_runs: int = 3,
) -> BenchmarkSuite:
    """Compare ThreadPoolExecutor vs ProcessPoolExecutor."""
    suite = BenchmarkSuite()

    lookbacks = list(range(2, 2 + grid_size))
    param_grids = [{"lookback": lookbacks}]

    cpu_count = os.cpu_count() or 4
    worker_counts = [1, 2, 4, min(8, cpu_count), cpu_count]
    worker_counts = sorted(set(worker_counts))

    # ThreadPoolExecutor
    for workers in worker_counts:
        print(f"  ThreadPool workers={workers}...", end=" ", flush=True)
        result = run_benchmark(
            name=f"thread_w{workers}",
            market=market,
            close_prices=close_prices,
            param_grids=param_grids,
            max_workers=workers,
            executor=None,
            executor_type="ThreadPoolExecutor",
            n_runs=n_runs,
        )
        print(f"{result.elapsed_seconds:.3f}s")
        suite.add(result)

    # Sequential (no executor)
    print(f"  Sequential (1 worker)...", end=" ", flush=True)
    result = run_benchmark(
        name="sequential",
        market=market,
        close_prices=close_prices,
        param_grids=param_grids,
        max_workers=1,
        executor=None,
        executor_type="Sequential",
        n_runs=n_runs,
    )
    print(f"{result.elapsed_seconds:.3f}s")
    suite.add(result)

    return suite


def benchmark_process_pool(
    market: MarketData,
    close_prices: pd.DataFrame,
    grid_size: int = 16,
    n_runs: int = 2,
) -> BenchmarkSuite:
    """Benchmark ProcessPoolExecutor to test true parallelism."""
    suite = BenchmarkSuite()

    lookbacks = list(range(2, 2 + grid_size))
    param_grids = [{"lookback": lookbacks}]

    cpu_count = os.cpu_count() or 4
    worker_counts = [1, 2, 4, min(8, cpu_count)]

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

    for workers in worker_counts:
        print(f"  ProcessPool workers={workers}...", end=" ", flush=True)
        gc.collect()

        times = []
        for _ in range(n_runs):
            # Create ProcessPoolExecutor for each run (they can't be reused easily)
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                start = time.perf_counter()
                grid_search(
                    generate_weights=generate_weights,
                    param_grids=param_grids,
                    objective_metric="Annualized Sharpe",
                    market=market,
                    config=config,
                    executor=executor,
                    progress=False,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            gc.collect()

        avg_elapsed = sum(times) / len(times)

        result = BenchmarkResult(
            name=f"process_w{workers}",
            executor_type="ProcessPoolExecutor",
            max_workers=workers,
            grid_size=grid_size,
            n_assets=close_prices.shape[1],
            n_periods=close_prices.shape[0],
            elapsed_seconds=avg_elapsed,
            runs_per_second=grid_size / avg_elapsed,
        )
        print(f"{result.elapsed_seconds:.3f}s ({result.runs_per_second:.1f} runs/s)")
        suite.add(result)

    return suite


def benchmark_2d_grids(
    market: MarketData,
    close_prices: pd.DataFrame,
    worker_counts: list[int | None],
    n_runs: int = 3,
) -> BenchmarkSuite:
    """Benchmark 2D parameter grids."""
    suite = BenchmarkSuite()

    # 5x5 = 25 combinations
    param_grids = [
        {"lookback": [5, 10, 20, 40, 80], "power": [0.5, 0.75, 1.0, 1.25, 1.5]}
    ]

    for workers in worker_counts:
        print(f"  2D grid (5x5=25), workers={workers}...", end=" ", flush=True)
        result = run_benchmark(
            name=f"2d_w{workers}",
            market=market,
            close_prices=close_prices,
            param_grids=param_grids,
            max_workers=workers,
            executor=None,
            executor_type="ThreadPoolExecutor",
            n_runs=n_runs,
        )
        print(f"{result.elapsed_seconds:.3f}s ({result.runs_per_second:.1f} runs/s)")
        suite.add(result)

    return suite


def main():
    print("=" * 60)
    print("Grid Search Performance Benchmark")
    print("=" * 60)

    cpu_count = os.cpu_count() or 4
    print(f"\nSystem: {cpu_count} CPUs available")

    # Create market data
    print("\nCreating synthetic market data...")
    n_periods = 500  # ~2 years of daily data
    n_assets = 50    # Moderate portfolio size
    market, close_prices = create_synthetic_market(n_periods, n_assets)
    print(f"  {n_periods} periods x {n_assets} assets")

    # Worker counts to test
    worker_counts = [1, 2, 4, 8, 16]
    worker_counts = [w for w in worker_counts if w <= cpu_count * 2]
    worker_counts.append(None)  # Default (cpu_count)

    print("\n" + "-" * 60)
    print("Benchmark 1: Worker Count Scaling (1D grids)")
    print("-" * 60)
    suite1 = benchmark_worker_counts(
        market, close_prices,
        grid_sizes=[8, 16, 32],
        worker_counts=worker_counts,
        n_runs=3,
    )

    print("\n" + "-" * 60)
    print("Benchmark 2: 2D Grid Performance")
    print("-" * 60)
    suite2 = benchmark_2d_grids(
        market, close_prices,
        worker_counts=worker_counts,
        n_runs=3,
    )

    # Combine results
    all_results = suite1.results + suite2.results
    df = pd.DataFrame([vars(r) for r in all_results])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nAll benchmark results:")
    print(df.to_string(index=False))

    # Find optimal worker count
    print("\n" + "-" * 60)
    print("Optimal Configuration Analysis")
    print("-" * 60)

    by_workers = df.groupby("max_workers")["runs_per_second"].mean()
    best_workers = by_workers.idxmax()
    print(f"\nBest average throughput: {best_workers} workers ({by_workers[best_workers]:.1f} runs/s)")

    # Worker scaling efficiency
    if 1 in by_workers.index:
        baseline = by_workers[1]
        print("\nScaling efficiency (vs 1 worker):")
        for workers, rps in by_workers.items():
            if workers is not None:
                efficiency = (rps / baseline) / workers * 100
                speedup = rps / baseline
                print(f"  {workers} workers: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")

    return df


if __name__ == "__main__":
    results_df = main()
