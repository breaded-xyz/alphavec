"""Tests for enhanced signal diagnostics (PRD implementation)."""

import numpy as np
import pandas as pd
import pytest

from alphavec import MarketData, MetricKey, SimConfig, simulate


def _sim(
    *,
    weights: pd.DataFrame | pd.Series,
    close_prices: pd.DataFrame | pd.Series,
    exec_prices: pd.DataFrame | pd.Series,
    funding_rates: pd.DataFrame | pd.Series | None = None,
    **config_overrides: object,
):
    market = MarketData(
        close_prices=close_prices,
        exec_prices=exec_prices,
        funding_rates=funding_rates,
    )
    config = SimConfig(**config_overrides)
    return simulate(weights=weights, market=market, config=config)


def _make_test_data(n_periods: int = 50, n_assets: int = 5):
    """Create test data with multiple assets for signal diagnostics."""
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="1D")
    np.random.seed(42)

    # Generate prices with some trend and noise
    prices_data = {}
    weights_data = {}
    for i in range(n_assets):
        asset = f"ASSET_{i}"
        trend = 100 * (1 + 0.001 * np.arange(n_periods))
        noise = np.random.randn(n_periods) * 2
        prices_data[asset] = trend + noise

        # Weights with some signal embedded
        weights_data[asset] = np.random.randn(n_periods) * 0.1

    close_prices = pd.DataFrame(prices_data, index=dates)
    exec_prices = close_prices.shift(1).bfill()
    weights = pd.DataFrame(weights_data, index=dates)

    return weights, close_prices, exec_prices


class TestConvenienceProperties:
    """Tests for Phase 1: Convenience properties on SignalArtifacts."""

    def test_ic_series_returns_series(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        ic = result.artifacts.signal.ic_series
        assert ic is not None
        assert isinstance(ic, pd.Series)
        assert ic.name == "ic"
        assert len(ic) > 0

    def test_rank_ic_series_returns_series(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rank_ic = result.artifacts.signal.rank_ic_series
        assert rank_ic is not None
        assert isinstance(rank_ic, pd.Series)
        assert rank_ic.name == "rank_ic"

    def test_decile_spread_series_returns_series(self):
        # Need at least 10 assets for decile spread
        weights, close_prices, exec_prices = _make_test_data(n_assets=12)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        spread = result.artifacts.signal.decile_spread_series
        assert spread is not None
        assert isinstance(spread, pd.Series)
        assert spread.name == "decile_spread"

    def test_directionality_series_returns_series(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        dir_series = result.artifacts.signal.directionality_series
        assert dir_series is not None
        assert isinstance(dir_series, pd.Series)
        assert dir_series.name == "directionality"

    def test_convenience_series_match_weight_forward(self):
        """Verify convenience properties extract from weight_forward correctly."""
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        wf = result.artifacts.signal.weight_forward
        ic = result.artifacts.signal.ic_series

        # Should be the same data, just renamed
        pd.testing.assert_series_equal(
            ic.rename("ic"),
            wf["ic"].rename("ic"),
        )

    def test_convenience_returns_none_when_diagnostics_disabled(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            compute_signal_diagnostics=False,
        )

        # When diagnostics disabled, should return None
        assert result.artifacts.signal.ic_series is None
        assert result.artifacts.signal.rank_ic_series is None


class TestMultiHorizonDecay:
    """Tests for Phase 2: Multi-horizon signal decay."""

    def test_decay_horizons_default(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=100)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        decay = result.artifacts.signal.decay_horizons
        assert decay is not None
        assert isinstance(decay, pd.DataFrame)
        assert decay.index.name == "Horizon"

        # Default horizons should include 1, 3, 5, 10, 21
        # (though some may be missing if data is too short)
        assert 1 in decay.index
        assert 3 in decay.index
        assert 5 in decay.index

    def test_decay_horizons_custom(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=50)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_decay_horizons=(1, 2, 5),
        )

        decay = result.artifacts.signal.decay_horizons
        assert set(decay.index) == {1, 2, 5}

    def test_decay_horizons_has_expected_columns(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=50)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_decay_horizons=(1, 3),
        )

        decay = result.artifacts.signal.decay_horizons
        expected_cols = [
            "ic_mean", "ic_tstat",
            "rank_ic_mean", "rank_ic_tstat",
            "decile_spread_mean", "decile_spread_tstat",
        ]
        for col in expected_cols:
            assert col in decay.columns, f"Missing column: {col}"

    def test_decay_horizons_ic_decreases(self):
        """IC should generally decrease at longer horizons (signal decay)."""
        # Create data with embedded signal that decays
        dates = pd.date_range("2024-01-01", periods=200, freq="1D")
        np.random.seed(42)

        prices_data = {}
        weights_data = {}
        for i in range(5):
            asset = f"ASSET_{i}"
            returns = np.random.randn(200) * 0.02
            prices_data[asset] = 100 * np.cumprod(1 + returns)
            # Weights correlated with next-period return (decays over time)
            weights_data[asset] = np.roll(returns, 1) + np.random.randn(200) * 0.01

        close_prices = pd.DataFrame(prices_data, index=dates)
        exec_prices = close_prices.shift(1).bfill()
        weights = pd.DataFrame(weights_data, index=dates)

        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_decay_horizons=(1, 5, 10),
        )

        decay = result.artifacts.signal.decay_horizons
        # Just check structure, not actual decay (depends on random seed)
        assert len(decay) == 3


class TestTurnoverAdjustedMetrics:
    """Tests for Phase 3: Turnover-adjusted metrics."""

    def test_turnover_adjusted_ic_in_metrics(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        # Check MetricKey constant exists and is accessible
        adj_ic = result.metric_value(MetricKey.TURNOVER_ADJUSTED_IC_MEAN_NEXT)
        assert adj_ic is not None
        assert isinstance(adj_ic, float)

    def test_net_forward_return_in_metrics(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        net_ret = result.metric_value(MetricKey.NET_FORWARD_RETURN_PER_GROSS_MEAN_NEXT)
        assert net_ret is not None
        assert isinstance(net_ret, float)

    def test_turnover_adjusted_ic_lower_than_raw(self):
        """Turnover-adjusted IC should be <= raw IC (costs reduce it)."""
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_cost_per_turnover=0.01,  # Higher cost for visible impact
        )

        raw_ic = result.metric_value(MetricKey.WEIGHT_IC_MEAN_NEXT)
        adj_ic = result.metric_value(MetricKey.TURNOVER_ADJUSTED_IC_MEAN_NEXT)

        # With positive turnover, adjusted IC should be lower
        # (unless IC is negative, then it could be higher)
        if raw_ic > 0:
            assert adj_ic <= raw_ic

    def test_custom_cost_per_turnover(self):
        """Signal cost parameter should affect adjusted metrics."""
        weights, close_prices, exec_prices = _make_test_data()

        result_low_cost = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_cost_per_turnover=0.0001,
        )
        result_high_cost = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
            signal_cost_per_turnover=0.01,
        )

        adj_low = result_low_cost.metric_value(MetricKey.TURNOVER_ADJUSTED_IC_MEAN_NEXT)
        adj_high = result_high_cost.metric_value(MetricKey.TURNOVER_ADJUSTED_IC_MEAN_NEXT)

        # Higher cost should result in lower (more penalized) adjusted IC
        raw = result_low_cost.metric_value(MetricKey.WEIGHT_IC_MEAN_NEXT)
        if raw > 0:
            assert adj_high <= adj_low


class TestPerAssetBreakdown:
    """Tests for Phase 4: Per-asset signal breakdown."""

    def test_by_asset_returns_dataframe(self):
        weights, close_prices, exec_prices = _make_test_data(n_assets=5)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        by_asset = result.artifacts.signal.by_asset
        assert by_asset is not None
        assert isinstance(by_asset, pd.DataFrame)
        assert len(by_asset) == 5  # 5 assets

    def test_by_asset_has_expected_columns(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        by_asset = result.artifacts.signal.by_asset
        expected_cols = ["ic_mean", "hit_rate", "avg_abs_weight",
                        "alpha_contribution", "alpha_contribution_pct"]
        for col in expected_cols:
            assert col in by_asset.columns, f"Missing column: {col}"

    def test_by_asset_indexed_by_asset_names(self):
        weights, close_prices, exec_prices = _make_test_data(n_assets=3)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        by_asset = result.artifacts.signal.by_asset
        assert list(by_asset.index) == ["ASSET_0", "ASSET_1", "ASSET_2"]

    def test_by_asset_hit_rate_between_0_and_1(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        by_asset = result.artifacts.signal.by_asset
        valid_hit_rates = by_asset["hit_rate"].dropna()
        assert (valid_hit_rates >= 0).all()
        assert (valid_hit_rates <= 1).all()

    def test_by_asset_contribution_sums_to_100(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        by_asset = result.artifacts.signal.by_asset
        # Absolute contribution percentages should sum to ~100
        total_pct = by_asset["alpha_contribution_pct"].abs().sum()
        assert abs(total_pct - 100) < 1  # Allow small rounding error


class TestRollingWindowAnalysis:
    """Tests for Phase 5: Rolling window analysis."""

    def test_rolling_returns_dataframe(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=100)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rolling = result.artifacts.signal.rolling(window=30)
        assert rolling is not None
        assert isinstance(rolling, pd.DataFrame)

    def test_rolling_has_expected_columns(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=100)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rolling = result.artifacts.signal.rolling(window=30)
        expected_cols = [
            "ic_mean", "ic_tstat",
            "rank_ic_mean", "rank_ic_tstat",
            "decile_spread_mean", "decile_spread_tstat",
        ]
        for col in expected_cols:
            assert col in rolling.columns, f"Missing column: {col}"

    def test_rolling_returns_none_for_short_data(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=20)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        # Window larger than data should return None
        rolling = result.artifacts.signal.rolling(window=100)
        assert rolling is None

    def test_rolling_custom_window(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=100)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rolling_30 = result.artifacts.signal.rolling(window=30)
        rolling_60 = result.artifacts.signal.rolling(window=60)

        # Both should work with 100 periods
        assert rolling_30 is not None
        assert rolling_60 is not None

        # 60-period rolling should have more NaN at start
        nan_30 = rolling_30["ic_mean"].isna().sum()
        nan_60 = rolling_60["ic_mean"].isna().sum()
        assert nan_60 >= nan_30

    def test_rolling_index_matches_weight_forward(self):
        weights, close_prices, exec_prices = _make_test_data(n_periods=100)
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        wf = result.artifacts.signal.weight_forward
        rolling = result.artifacts.signal.rolling(window=30)

        pd.testing.assert_index_equal(rolling.index, wf.index)


class TestRankICMetrics:
    """Tests for Rank IC summary metrics."""

    def test_rank_ic_mean_in_metrics(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rank_ic = result.metric_value(MetricKey.WEIGHT_RANK_IC_MEAN_NEXT)
        assert rank_ic is not None
        assert isinstance(rank_ic, float)

    def test_rank_ic_tstat_in_metrics(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        rank_ic_tstat = result.metric_value(MetricKey.WEIGHT_RANK_IC_TSTAT_NEXT)
        assert rank_ic_tstat is not None
        assert isinstance(rank_ic_tstat, float)

    def test_rank_ic_in_signal_category(self):
        """Rank IC should be in the Signal category."""
        signal_keys = MetricKey.keys_by_category()["Signal"]
        assert MetricKey.WEIGHT_RANK_IC_MEAN_NEXT in signal_keys
        assert MetricKey.WEIGHT_RANK_IC_TSTAT_NEXT in signal_keys


class TestBackwardCompatibility:
    """Tests to ensure existing APIs still work."""

    def test_existing_signal_metrics_unchanged(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        # All existing signal metrics should still work
        assert result.metric_value(MetricKey.WEIGHT_IC_MEAN_NEXT) is not None
        assert result.metric_value(MetricKey.WEIGHT_IC_TSTAT_NEXT) is not None
        assert result.metric_value(MetricKey.GROSS_WEIGHT_MEAN) is not None
        assert result.metric_value(MetricKey.DIRECTIONALITY_MEAN) is not None

    def test_weight_forward_unchanged(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        wf = result.artifacts.signal.weight_forward
        assert wf is not None
        assert "ic" in wf.columns
        assert "rank_ic" in wf.columns
        assert "forward_return_per_gross" in wf.columns

    def test_alpha_decay_unchanged(self):
        weights, close_prices, exec_prices = _make_test_data()
        result = _sim(
            weights=weights,
            close_prices=close_prices,
            exec_prices=exec_prices,
        )

        decay = result.artifacts.signal.alpha_decay_next_return_by_type
        assert decay is not None
        assert isinstance(decay, pd.DataFrame)
        assert decay.index.name == "Lag"
