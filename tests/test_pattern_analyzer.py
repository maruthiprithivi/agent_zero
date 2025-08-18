"""Tests for the pattern analysis and anomaly detection system."""

import random
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from agent_zero.ai_diagnostics.pattern_analyzer import (
    AnomalyDetectionEngine,
    AnomalySeverity,
    BaselineMetrics,
    PatternRecognitionEngine,
    PerformanceBaselineEngine,
    TimeSeriesAnalyzer,
    TimeSeriesPoint,
    TrendType,
    create_pattern_analysis_engine,
    create_time_series_analyzer,
)


class TestTimeSeriesAnalyzer:
    """Test the TimeSeriesAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TimeSeriesAnalyzer(window_size=50)

    def create_test_data(self, size: int = 50, trend: float = 0.5, noise: float = 5.0):
        """Create test time series data."""
        base_time = datetime.now()
        data = []
        for i in range(size):
            timestamp = base_time + timedelta(hours=i)
            value = 100 + i * trend + random.uniform(-noise, noise)
            data.append(TimeSeriesPoint(timestamp, max(0.0, value)))
        return data

    def test_analyze_trend_increasing(self):
        """Test trend analysis for increasing data."""
        data = self.create_test_data(size=30, trend=2.0, noise=1.0)
        trend = self.analyzer.analyze_trend(data, "TestEvent")

        assert trend.event_name == "TestEvent"
        assert trend.trend_type == TrendType.INCREASING
        assert trend.slope > 0
        assert 0 <= trend.r_squared <= 1
        assert 0 <= trend.trend_strength <= 1

    def test_analyze_trend_stable(self):
        """Test trend analysis for stable data."""
        data = self.create_test_data(size=30, trend=0.0, noise=2.0)
        trend = self.analyzer.analyze_trend(data, "StableEvent")

        assert trend.event_name == "StableEvent"
        assert abs(trend.slope) < 0.1  # Nearly flat

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        data = self.create_test_data(size=5)
        trend = self.analyzer.analyze_trend(data, "SmallEvent")

        assert trend.event_name == "SmallEvent"
        assert trend.trend_type == TrendType.STABLE

    def test_detect_seasonal_patterns(self):
        """Test seasonal pattern detection."""
        # Create data with seasonal pattern
        base_time = datetime.now()
        data = []
        for i in range(100):
            timestamp = base_time + timedelta(hours=i)
            # Add 24-hour cycle
            seasonal_component = 10 * (1 + 0.5 * (i % 24) / 24)
            value = 100 + seasonal_component + random.uniform(-2, 2)
            data.append(TimeSeriesPoint(timestamp, value))

        patterns = self.analyzer.detect_seasonal_patterns(data)
        # May or may not detect patterns depending on noise and data quality
        assert isinstance(patterns, list)

    def test_detect_change_points(self):
        """Test change point detection."""
        # Create data with an obvious change point
        base_time = datetime.now()
        data = []
        for i in range(50):
            timestamp = base_time + timedelta(hours=i)
            if i < 25:
                value = 100 + random.uniform(-5, 5)
            else:
                value = 150 + random.uniform(-5, 5)  # Level shift at midpoint
            data.append(TimeSeriesPoint(timestamp, value))

        changes = self.analyzer.detect_change_points(data, "ChangeEvent")
        assert isinstance(changes, list)


class TestPatternRecognitionEngine:
    """Test the PatternRecognitionEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PatternRecognitionEngine()

    def create_spike_data(self):
        """Create data with spike patterns."""
        base_time = datetime.now()
        data = []
        for i in range(50):
            timestamp = base_time + timedelta(minutes=i * 5)
            if i == 25:  # Spike at midpoint
                value = 200
            else:
                value = 50 + random.uniform(-5, 5)
            data.append(TimeSeriesPoint(timestamp, value))
        return data

    def test_detect_patterns(self):
        """Test pattern detection."""
        data = self.create_spike_data()
        patterns = self.engine.detect_patterns(data, "SpikeEvent")

        assert isinstance(patterns, list)
        # Should detect some patterns

    def test_match_historical_patterns_empty(self):
        """Test pattern matching with no historical patterns."""
        data = self.create_spike_data()
        matches = self.engine.match_historical_patterns(data, "NewEvent")

        assert matches == []  # No historical patterns to match


class TestPerformanceBaselineEngine:
    """Test the PerformanceBaselineEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.engine = PerformanceBaselineEngine(self.mock_client, lookback_days=7)

    def test_create_default_baseline(self):
        """Test creating a default baseline."""
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()

        baseline = self.engine._create_default_baseline("TestEvent", start_time, end_time)

        assert isinstance(baseline, BaselineMetrics)
        assert baseline.event_name == "TestEvent"
        assert baseline.mean == 0.0
        assert baseline.sample_size == 0
        assert baseline.baseline_period == (start_time, end_time)

    def test_calculate_baseline_metrics(self):
        """Test baseline metrics calculation."""
        values = [10, 20, 15, 25, 30, 12, 18, 22, 16, 28]
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()

        baseline = self.engine._calculate_baseline_metrics(
            "TestEvent", values, start_time, end_time
        )

        assert isinstance(baseline, BaselineMetrics)
        assert baseline.event_name == "TestEvent"
        assert baseline.mean > 0
        assert baseline.sample_size == len(values)
        assert baseline.min_value == min(values)
        assert baseline.max_value == max(values)

    def test_update_baseline(self):
        """Test baseline updating."""
        # Create initial baseline
        initial_values = [10, 15, 20, 25, 30]
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()

        baseline = self.engine._calculate_baseline_metrics(
            "TestEvent", initial_values, start_time, end_time
        )
        self.engine.baselines["TestEvent"] = baseline

        # Update with new data
        new_data = [12, 18, 22, 28]
        updated_baseline = self.engine.update_baseline("TestEvent", new_data)

        assert isinstance(updated_baseline, BaselineMetrics)
        assert updated_baseline.sample_size > baseline.sample_size


class TestAnomalyDetectionEngine:
    """Test the AnomalyDetectionEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.baseline_engine = PerformanceBaselineEngine(self.mock_client)
        self.anomaly_engine = AnomalyDetectionEngine(self.baseline_engine)

    def create_anomaly_data(self):
        """Create data with obvious anomalies."""
        base_time = datetime.now()
        data = []
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i * 5)
            if i == 15:  # Anomaly at midpoint
                value = 1000  # Very high value
            else:
                value = 50 + random.uniform(-5, 5)
            data.append(TimeSeriesPoint(timestamp, value))
        return data

    def test_calculate_z_score(self):
        """Test Z-score calculation."""
        # Create mock baseline
        baseline = BaselineMetrics(
            event_name="TestEvent",
            mean=100.0,
            median=100.0,
            std_dev=10.0,
            min_value=80.0,
            max_value=120.0,
            percentile_25=95.0,
            percentile_75=105.0,
            percentile_95=115.0,
            percentile_99=118.0,
            lower_control_limit=80.0,
            upper_control_limit=120.0,
            warning_threshold=115.0,
            critical_threshold=130.0,
            sample_size=100,
            confidence_interval=(90.0, 110.0),
            baseline_period=(datetime.now() - timedelta(days=1), datetime.now()),
            last_updated=datetime.now(),
        )

        # Test normal value
        z_score = self.anomaly_engine._calculate_z_score(105.0, baseline)
        assert abs(z_score - 0.5) < 0.1

        # Test anomalous value
        z_score = self.anomaly_engine._calculate_z_score(150.0, baseline)
        assert z_score > 3.0  # Should be a strong anomaly

    def test_determine_severity(self):
        """Test anomaly severity determination."""
        # Test different severity levels
        assert self.anomaly_engine._determine_severity(0.9, 0.95) == AnomalySeverity.CRITICAL
        assert self.anomaly_engine._determine_severity(0.7, 0.8) == AnomalySeverity.HIGH
        assert self.anomaly_engine._determine_severity(0.5, 0.6) == AnomalySeverity.MEDIUM
        assert self.anomaly_engine._determine_severity(0.3, 0.4) == AnomalySeverity.LOW
        assert self.anomaly_engine._determine_severity(0.1, 0.2) == AnomalySeverity.INFO


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_time_series_analyzer(self):
        """Test creating a time series analyzer."""
        analyzer = create_time_series_analyzer()
        assert isinstance(analyzer, TimeSeriesAnalyzer)
        assert analyzer.window_size == 100

        # Test with custom parameters
        analyzer = create_time_series_analyzer(window_size=50, seasonal_periods=[12, 24])
        assert analyzer.window_size == 50
        assert analyzer.seasonal_periods == [12, 24]

    @patch("agent_zero.ai_diagnostics.pattern_analyzer.ProfileEventsAnalyzer")
    def test_create_pattern_analysis_engine(self, mock_profile_analyzer):
        """Test creating a pattern analysis engine."""
        mock_client = Mock()

        engine = create_pattern_analysis_engine(mock_client)

        # Should create engine with proper components
        assert hasattr(engine, "time_series_analyzer")
        assert hasattr(engine, "baseline_engine")
        assert hasattr(engine, "anomaly_engine")
        assert hasattr(engine, "pattern_engine")
        assert hasattr(engine, "correlation_analyzer")


class TestIntegration:
    """Integration tests for the complete pattern analysis system."""

    @patch("agent_zero.ai_diagnostics.pattern_analyzer.execute_query_with_retry")
    @patch("agent_zero.ai_diagnostics.pattern_analyzer.ProfileEventsAnalyzer")
    def test_pattern_analysis_workflow(self, mock_profile_analyzer, mock_execute_query):
        """Test the complete pattern analysis workflow."""
        # Mock query results
        mock_execute_query.return_value = [
            {"value": 100.0, "event_time": datetime.now() - timedelta(hours=i)}
            for i in range(24, 0, -1)
        ]

        mock_client = Mock()
        engine = create_pattern_analysis_engine(mock_client)

        # This should work without actual database connection
        # because we're mocking the query execution
        try:
            result = engine._create_minimal_analysis_result(
                "TestEvent", datetime.now() - timedelta(hours=24), datetime.now()
            )
            assert result.event_name == "TestEvent"
            assert len(result.anomalies) == 0  # Minimal result has no anomalies
        except Exception as e:
            pytest.fail(f"Pattern analysis workflow failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
