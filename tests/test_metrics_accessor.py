"""Tests for the unified MetricsAccessor interface."""

import numpy as np
import pandas as pd
import pytest

from alphavec import (
    MarketData,
    MetricKey,
    MetricsAccessor,
    SimConfig,
    SimulationResult,
    GridSearchResults,
    GridSearchBest,
    WalkForwardResult,
    FoldConfig,
    grid_search,
    simulate,
    walk_forward,
)


def _create_test_data(n_periods: int = 100, seed: int = 42):
    """Create synthetic test data."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="1D")
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100.0 * np.cumprod(1 + returns)
    close_prices = pd.DataFrame({"BTC": prices}, index=dates)
    exec_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": np.sign(np.random.randn(n_periods)) * 0.5}, index=dates)
    return weights, close_prices, exec_prices


# --- Protocol compliance tests ---


def test_simulation_result_is_metrics_accessor():
    """SimulationResult satisfies MetricsAccessor protocol."""
    weights, close_prices, exec_prices = _create_test_data()
    result = simulate(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )
    assert isinstance(result, MetricsAccessor)


def test_grid_search_results_is_metrics_accessor():
    """GridSearchResults satisfies MetricsAccessor protocol."""
    weights, close_prices, exec_prices = _create_test_data()

    def generate_weights(params):
        return weights * params["scale"]

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [0.5, 1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )
    assert isinstance(results, MetricsAccessor)


def test_walk_forward_result_is_metrics_accessor():
    """WalkForwardResult satisfies MetricsAccessor protocol."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )
    assert isinstance(result, MetricsAccessor)


# --- GridSearchResults tests ---


def test_grid_search_results_metric_value():
    """GridSearchResults.metric_value delegates to best.result."""
    weights, close_prices, exec_prices = _create_test_data()

    def generate_weights(params):
        return weights * params["scale"]

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [0.5, 1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )

    sharpe = results.metric_value(MetricKey.ANNUALIZED_SHARPE)
    assert sharpe == results.best.result.metric_value(MetricKey.ANNUALIZED_SHARPE)


def test_grid_search_results_available_metrics():
    """GridSearchResults.available_metrics delegates to best.result."""
    weights, close_prices, exec_prices = _create_test_data()

    def generate_weights(params):
        return weights * params["scale"]

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [0.5, 1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )

    all_metrics = results.available_metrics()
    assert all_metrics == results.best.result.available_metrics()

    perf_metrics = results.available_metrics("Performance")
    assert MetricKey.ANNUALIZED_SHARPE in perf_metrics


def test_grid_search_results_metrics_dict():
    """GridSearchResults.metrics_dict delegates to best.result."""
    weights, close_prices, exec_prices = _create_test_data()

    def generate_weights(params):
        return weights * params["scale"]

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [0.5, 1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )

    metrics = results.metrics_dict("Performance")
    assert MetricKey.ANNUALIZED_SHARPE in metrics


def test_grid_search_results_best_none_with_default():
    """GridSearchResults.metric_value returns default when best is None."""
    from alphavec.search import _Grid

    results = GridSearchResults(
        table=pd.DataFrame(),
        param_grids=(),
        objective_metric="Annualized Sharpe",
        best=None,
    )

    assert results.metric_value(MetricKey.ANNUALIZED_SHARPE, default=-999) == -999


def test_grid_search_results_best_none_raises():
    """GridSearchResults methods raise ValueError when best is None."""
    results = GridSearchResults(
        table=pd.DataFrame(),
        param_grids=(),
        objective_metric="Annualized Sharpe",
        best=None,
    )

    with pytest.raises(ValueError, match="no valid results exist"):
        results.metric_value(MetricKey.ANNUALIZED_SHARPE)

    with pytest.raises(ValueError, match="no valid results exist"):
        results.available_metrics()

    with pytest.raises(ValueError, match="no valid results exist"):
        results.metrics_dict()


# --- WalkForwardResult tests ---


def test_walk_forward_result_metric_value():
    """WalkForwardResult.metric_value delegates to full_result."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )

    sharpe = result.metric_value(MetricKey.ANNUALIZED_SHARPE)
    assert sharpe == result.full_result.metric_value(MetricKey.ANNUALIZED_SHARPE)


def test_walk_forward_result_available_metrics():
    """WalkForwardResult.available_metrics delegates to full_result."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )

    all_metrics = result.available_metrics()
    assert all_metrics == result.full_result.available_metrics()

    perf_metrics = result.available_metrics("Performance")
    assert MetricKey.ANNUALIZED_SHARPE in perf_metrics


def test_walk_forward_result_metrics_dict():
    """WalkForwardResult.metrics_dict delegates to full_result."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )

    metrics = result.metrics_dict("Performance")
    assert MetricKey.ANNUALIZED_SHARPE in metrics


def test_walk_forward_result_full_result_none_with_default():
    """WalkForwardResult.metric_value returns default when full_result is None."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=False,
    )

    assert result.metric_value(MetricKey.ANNUALIZED_SHARPE, default=-999) == -999


def test_walk_forward_result_full_result_none_raises():
    """WalkForwardResult methods raise ValueError when full_result is None."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=False,
    )

    with pytest.raises(ValueError, match="full_result is None"):
        result.metric_value(MetricKey.ANNUALIZED_SHARPE)

    with pytest.raises(ValueError, match="full_result is None"):
        result.available_metrics()

    with pytest.raises(ValueError, match="full_result is None"):
        result.metrics_dict()


# --- Generic function tests ---


def _extract_sharpe(accessor: MetricsAccessor) -> float:
    """Example function that works with any MetricsAccessor."""
    return accessor.metric_value(MetricKey.ANNUALIZED_SHARPE)


def test_generic_function_with_simulation_result():
    """Generic function works with SimulationResult."""
    weights, close_prices, exec_prices = _create_test_data()
    result = simulate(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )
    sharpe = _extract_sharpe(result)
    assert isinstance(sharpe, (int, float))


def test_generic_function_with_grid_search_results():
    """Generic function works with GridSearchResults."""
    weights, close_prices, exec_prices = _create_test_data()

    def generate_weights(params):
        return weights * params["scale"]

    results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [0.5, 1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )
    sharpe = _extract_sharpe(results)
    assert isinstance(sharpe, (int, float))


def test_generic_function_with_walk_forward_result():
    """Generic function works with WalkForwardResult."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)
    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )
    sharpe = _extract_sharpe(result)
    assert isinstance(sharpe, (int, float))


# --- Unified API comparison test ---


def test_unified_api_returns_same_values():
    """Verify all three result types return consistent values for the same data."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200, seed=123)

    # Simulate
    sim_result = simulate(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )

    # Grid search with single param that matches original
    def generate_weights(params):
        return weights * params["scale"]

    grid_results = grid_search(
        generate_weights=generate_weights,
        param_grids=[{"scale": [1.0]}],
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
    )

    # Walk forward with full_result
    wf_result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )

    # Simulate and grid search with scale=1.0 should match
    sim_sharpe = sim_result.metric_value(MetricKey.ANNUALIZED_SHARPE)
    grid_sharpe = grid_results.metric_value(MetricKey.ANNUALIZED_SHARPE)
    wf_sharpe = wf_result.metric_value(MetricKey.ANNUALIZED_SHARPE)

    # sim and grid should be exactly equal (same underlying simulation)
    assert sim_sharpe == grid_sharpe

    # wf_sharpe uses the same data but different time handling, should be close
    assert isinstance(wf_sharpe, (int, float))

    # Verify available_metrics returns same set
    sim_metrics = set(sim_result.available_metrics())
    grid_metrics = set(grid_results.available_metrics())
    wf_metrics = set(wf_result.available_metrics())

    assert sim_metrics == grid_metrics == wf_metrics
