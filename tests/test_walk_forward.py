import numpy as np
import pandas as pd
import pytest

from alphavec import (
    MarketData,
    MetricKey,
    SimConfig,
    walk_forward,
    FoldConfig,
    FoldResult,
    FoldAggregation,
    WalkForwardResult,
    DEFAULT_AGGREGATE_METRICS,
)


def _create_test_data(n_periods: int = 365, seed: int = 42):
    """Create synthetic test data for walk-forward testing."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="1D")

    # Create price series with slight upward drift
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100.0 * np.cumprod(1 + returns)

    close_prices = pd.DataFrame({"BTC": prices}, index=dates)
    exec_prices = close_prices.copy()

    # Create simple momentum weights
    weights = pd.DataFrame({"BTC": np.sign(np.random.randn(n_periods)) * 0.5}, index=dates)

    return weights, close_prices, exec_prices


def test_walk_forward_basic_3m_folds():
    """Basic test with 3-month folds."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=365)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="3ME"),
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) >= 3  # At least 3 full quarters in 365 days
    assert result.full_result is not None

    # Verify fold structure
    for i, fold in enumerate(result.folds):
        assert isinstance(fold, FoldResult)
        assert fold.fold_index == i
        assert fold.n_periods > 0
        assert fold.start_period <= fold.end_period


def test_walk_forward_6m_folds():
    """Test with 6-month folds."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=400)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="6ME"),
    )

    assert len(result.folds) >= 2  # At least 2 half-year folds


def test_walk_forward_min_periods_filtering():
    """Test that folds with < min_periods are discarded."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=100)

    # With high min_periods, fewer folds should be generated
    result_high = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME", min_periods=30),
    )

    result_low = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME", min_periods=3),
    )

    # Higher min_periods should result in same or fewer folds
    assert len(result_high.folds) <= len(result_low.folds)

    # All folds in high min_periods result should have >= 30 periods
    for fold in result_high.folds:
        assert fold.n_periods >= 30


def test_walk_forward_aggregations():
    """Test that metric aggregations are correct."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=365)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="3ME"),
    )

    # Check Sharpe aggregation
    sharpe_agg = result.metric_aggregation(MetricKey.ANNUALIZED_SHARPE)
    assert sharpe_agg is not None
    assert isinstance(sharpe_agg, FoldAggregation)
    assert sharpe_agg.count == len(result.folds)

    # Verify aggregation math
    sharpe_values = [fold.result.metric_value(MetricKey.ANNUALIZED_SHARPE) for fold in result.folds]
    valid_values = [v for v in sharpe_values if isinstance(v, (int, float)) and np.isfinite(v)]

    if len(valid_values) > 0:
        assert np.isclose(sharpe_agg.median, np.median(valid_values), rtol=1e-10)
        assert np.isclose(sharpe_agg.mean, np.mean(valid_values), rtol=1e-10)
        assert np.isclose(sharpe_agg.min, np.min(valid_values), rtol=1e-10)
        assert np.isclose(sharpe_agg.max, np.max(valid_values), rtol=1e-10)


def test_walk_forward_summary_dataframe():
    """Test summary DataFrame structure and content."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    summary = result.summary
    assert isinstance(summary, pd.DataFrame)
    assert "fold_index" in summary.columns
    assert "start_period" in summary.columns
    assert "end_period" in summary.columns
    assert "n_periods" in summary.columns
    assert len(summary) == len(result.folds)


def test_walk_forward_fold_metric_values():
    """Test accessing metrics through fold_metric_values method."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    sharpe_values = result.fold_metric_values(MetricKey.ANNUALIZED_SHARPE)
    assert isinstance(sharpe_values, pd.Series)
    assert len(sharpe_values) == len(result.folds)


def test_walk_forward_summary_stats():
    """Test summary_stats method."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    stats = result.summary_stats([MetricKey.ANNUALIZED_SHARPE, MetricKey.MAX_DRAWDOWN_EQUITY_PCT])
    assert isinstance(stats, pd.DataFrame)
    assert "Metric" in stats.columns
    assert "Median" in stats.columns
    assert "Mean" in stats.columns
    assert "Std" in stats.columns
    assert "Min" in stats.columns
    assert "Max" in stats.columns
    assert "Count" in stats.columns


def test_walk_forward_full_result_included():
    """Test full_result is populated when include_full_result=True."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=True,
    )

    assert result.full_result is not None
    assert hasattr(result.full_result, "returns")
    assert hasattr(result.full_result, "metrics")


def test_walk_forward_full_result_excluded():
    """Test full_result is None when include_full_result=False."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        include_full_result=False,
    )

    assert result.full_result is None


def test_walk_forward_custom_metrics_list():
    """Test custom metrics_to_aggregate list."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    custom_metrics = [MetricKey.ANNUALIZED_SHARPE, MetricKey.TOTAL_RETURN_PCT]

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
        metrics_to_aggregate=custom_metrics,
    )

    # Only custom metrics should be aggregated
    assert set(result.aggregations.keys()) == set(custom_metrics)


def test_walk_forward_insufficient_data_raises():
    """Test ValueError when no valid folds can be generated."""
    # Only 10 days of data with 1-year folds
    weights, close_prices, exec_prices = _create_test_data(n_periods=10)

    with pytest.raises(ValueError, match="No valid folds generated"):
        walk_forward(
            weights=weights,
            market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
            fold_config=FoldConfig(fold_period="1YE"),
        )


def test_walk_forward_with_sim_config():
    """Test that SimConfig parameters are passed through to folds."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    config = SimConfig(
        init_cash=10000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
    )

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        config=config,
        fold_config=FoldConfig(fold_period="1ME"),
    )

    # Verify config is stored
    assert result.sim_config.init_cash == 10000.0
    assert result.sim_config.fee_rate == 0.001

    # Verify folds have costs applied (fees > 0)
    total_fees = sum(
        fold.result.metric_value(MetricKey.FEES) or 0 for fold in result.folds
    )
    assert total_fees > 0


def test_walk_forward_tz_aware_index():
    """Test with timezone-aware DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")

    prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, 200))
    close_prices = pd.DataFrame({"BTC": prices}, index=dates)
    exec_prices = close_prices.copy()
    weights = pd.DataFrame({"BTC": np.ones(200) * 0.5}, index=dates)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    assert len(result.folds) > 0
    # Verify timestamps preserve timezone
    for fold in result.folds:
        assert fold.start_period.tz is not None or fold.start_period.tzinfo is not None


def test_walk_forward_series_inputs():
    """Test with Series inputs (single asset)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="1D")

    prices = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, 200)), index=dates, name="BTC")
    weights = pd.Series(np.ones(200) * 0.5, index=dates, name="BTC")

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=prices, exec_prices=prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    assert len(result.folds) > 0


def test_walk_forward_align_start_false():
    """Test raw fold boundaries from data start (align_start=False)."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result_aligned = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME", align_start=True),
    )

    result_raw = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME", align_start=False),
    )

    # Both should produce results
    assert len(result_aligned.folds) > 0
    assert len(result_raw.folds) > 0

    # With align_start=False, boundaries start from data start using the frequency
    # Both configurations should produce valid folds with the same period structure
    assert result_aligned.folds[0].start_period <= result_raw.folds[0].start_period


def test_default_aggregate_metrics():
    """Test that DEFAULT_AGGREGATE_METRICS is properly defined."""
    assert len(DEFAULT_AGGREGATE_METRICS) > 0
    assert MetricKey.ANNUALIZED_SHARPE in DEFAULT_AGGREGATE_METRICS
    assert MetricKey.MAX_DRAWDOWN_EQUITY_PCT in DEFAULT_AGGREGATE_METRICS
    assert MetricKey.TOTAL_RETURN_PCT in DEFAULT_AGGREGATE_METRICS


def test_fold_config_immutable():
    """Test that FoldConfig is immutable (frozen dataclass)."""
    config = FoldConfig(fold_period="3ME")

    with pytest.raises(Exception):  # FrozenInstanceError
        config.fold_period = "6ME"


def test_walk_forward_result_immutable():
    """Test that WalkForwardResult is immutable (frozen dataclass)."""
    weights, close_prices, exec_prices = _create_test_data(n_periods=200)

    result = walk_forward(
        weights=weights,
        market=MarketData(close_prices=close_prices, exec_prices=exec_prices),
        fold_config=FoldConfig(fold_period="1ME"),
    )

    with pytest.raises(Exception):  # FrozenInstanceError
        result.folds = ()
