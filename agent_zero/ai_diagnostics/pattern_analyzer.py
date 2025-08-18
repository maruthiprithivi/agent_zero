"""Advanced pattern analysis and anomaly detection system for ClickHouse ProfileEvents.

This module provides comprehensive statistical and machine learning capabilities for:
- Baseline establishment from historical ProfileEvents data
- Multi-layered anomaly detection using statistical methods
- Long-term performance trend identification and forecasting
- Correlation analysis between different ProfileEvents
- Seasonal pattern recognition and change point detection
- Pattern matching for recurring performance issues

The pattern analyzer serves as the statistical foundation for all AI-powered features,
enabling intelligent trend analysis, anomaly detection, and pattern recognition.
"""

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, NamedTuple

from clickhouse_connect.driver.client import Client

from agent_zero.monitoring.profile_events_core import (
    ProfileEventsAnalyzer,
)
from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    CORRELATION_ANOMALY = "correlation_anomaly"
    CHANGE_POINT = "change_point"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    CRITICAL = "critical"  # 90-100% confidence, major deviation
    HIGH = "high"  # 75-90% confidence, significant deviation
    MEDIUM = "medium"  # 50-75% confidence, moderate deviation
    LOW = "low"  # 25-50% confidence, minor deviation
    INFO = "info"  # < 25% confidence, informational


class TrendType(Enum):
    """Types of trends that can be identified."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"


class ChangePointType(Enum):
    """Types of change points that can be detected."""

    LEVEL_SHIFT = "level_shift"  # Sudden change in mean
    TREND_CHANGE = "trend_change"  # Change in trend direction
    VARIANCE_CHANGE = "variance_change"  # Change in volatility
    DISTRIBUTION_CHANGE = "distribution_change"  # Change in data distribution


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""

    timestamp: datetime
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineMetrics:
    """Statistical baseline metrics for a ProfileEvent."""

    event_name: str
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float

    # Dynamic thresholds
    lower_control_limit: float
    upper_control_limit: float
    warning_threshold: float
    critical_threshold: float

    # Baseline metadata
    sample_size: int
    confidence_interval: tuple[float, float]
    baseline_period: tuple[datetime, datetime]
    last_updated: datetime


@dataclass
class AnomalyScore:
    """Comprehensive anomaly scoring for a data point."""

    timestamp: datetime
    event_name: str
    value: float

    # Individual scoring components
    z_score: float
    modified_z_score: float
    iqr_score: float
    percentile_rank: float

    # Composite scores
    statistical_score: float
    pattern_score: float
    trend_score: float

    # Final anomaly assessment
    overall_score: float
    severity: AnomalySeverity
    anomaly_types: list[AnomalyType]
    confidence: float

    # Context information
    baseline_deviation: float
    seasonal_adjustment: float
    correlation_context: dict[str, float] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """A detected pattern match in time series data."""

    pattern_id: str
    event_name: str
    start_time: datetime
    end_time: datetime
    similarity_score: float
    pattern_type: str

    # Pattern characteristics
    duration: timedelta
    amplitude: float
    frequency: float | None

    # Context
    historical_occurrences: int
    typical_severity: AnomalySeverity
    associated_events: list[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis results."""

    event_name: str
    analysis_period: tuple[datetime, datetime]
    trend_type: TrendType

    # Statistical measures
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: tuple[float, float]

    # Trend characteristics
    trend_strength: float  # 0-1 scale
    volatility: float
    persistence: float  # How consistent the trend is

    # Forecasting
    forecast_values: list[float] = field(default_factory=list)
    forecast_timestamps: list[datetime] = field(default_factory=list)
    forecast_confidence: list[float] = field(default_factory=list)


@dataclass
class CorrelationAnalysis:
    """Correlation analysis between ProfileEvents."""

    primary_event: str
    secondary_event: str
    correlation_coefficient: float
    p_value: float

    # Relationship characteristics
    relationship_type: str  # "linear", "exponential", "logarithmic", etc.

    # Stability metrics
    correlation_stability: float
    recent_correlation: float
    historical_correlation: float

    # Fields with defaults must come after non-default fields
    lag_correlation: dict[int, float] = field(default_factory=dict)

    # Causality indicators
    granger_causality_p: float | None = None
    mutual_information: float | None = None


@dataclass
class ChangePoint:
    """A detected change point in time series data."""

    event_name: str
    timestamp: datetime
    change_type: ChangePointType

    # Change characteristics
    magnitude: float
    confidence: float

    # Before/after statistics
    before_mean: float
    after_mean: float
    before_std: float
    after_std: float

    # Context
    potential_causes: list[str] = field(default_factory=list)
    impact_assessment: str = ""


class SeasonalPattern(NamedTuple):
    """A detected seasonal pattern."""

    period: int  # Period in time units
    amplitude: float  # Strength of seasonal component
    phase: float  # Phase offset
    confidence: float  # Confidence in pattern detection


@dataclass
class PatternAnalysisResult:
    """Comprehensive pattern analysis results."""

    event_name: str
    analysis_period: tuple[datetime, datetime]

    # Baseline and anomalies
    baseline_metrics: BaselineMetrics
    anomalies: list[AnomalyScore]

    # Patterns and trends
    detected_patterns: list[PatternMatch]
    trend_analysis: TrendAnalysis
    seasonal_patterns: list[SeasonalPattern]
    change_points: list[ChangePoint]

    # Correlations
    correlations: list[CorrelationAnalysis]

    # Summary statistics
    anomaly_rate: float
    pattern_coverage: float  # Percentage of data explained by patterns
    predictability_score: float  # How predictable the time series is
    stability_score: float  # How stable the behavior is


class TimeSeriesAnalyzer:
    """Advanced time-series analysis and forecasting engine."""

    def __init__(self, window_size: int = 100, seasonal_periods: list[int] = None):
        """Initialize the time series analyzer.

        Args:
            window_size: Size of the rolling window for analysis
            seasonal_periods: List of potential seasonal periods to detect
        """
        self.window_size = window_size
        self.seasonal_periods = seasonal_periods or [24, 168, 720]  # hourly, daily, weekly, monthly
        self.trend_cache: dict[str, TrendAnalysis] = {}

    def analyze_trend(self, data: list[TimeSeriesPoint], event_name: str) -> TrendAnalysis:
        """Perform comprehensive trend analysis on time series data.

        Args:
            data: Time series data points
            event_name: Name of the ProfileEvent

        Returns:
            Comprehensive trend analysis results
        """
        if len(data) < 10:
            # Not enough data for reliable trend analysis
            return self._create_minimal_trend_analysis(data, event_name)

        # Extract values and timestamps
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        # Convert timestamps to numeric values for regression
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]

        # Linear regression for trend
        slope, intercept, r_squared, p_value = self._linear_regression(time_numeric, values)

        # Determine trend type
        trend_type = self._classify_trend(slope, r_squared, values)

        # Calculate trend strength and volatility
        trend_strength = min(abs(r_squared), 1.0)
        volatility = statistics.stdev(values) / (statistics.mean(values) + 1e-10)

        # Calculate persistence (trend consistency)
        persistence = self._calculate_persistence(values, self.window_size // 2)

        # Generate forecast
        forecast_values, forecast_timestamps, forecast_confidence = self._generate_forecast(
            data, slope, intercept, trend_type
        )

        return TrendAnalysis(
            event_name=event_name,
            analysis_period=(timestamps[0], timestamps[-1]),
            trend_type=trend_type,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=self._calculate_confidence_interval(slope, values),
            trend_strength=trend_strength,
            volatility=volatility,
            persistence=persistence,
            forecast_values=forecast_values,
            forecast_timestamps=forecast_timestamps,
            forecast_confidence=forecast_confidence,
        )

    def detect_seasonal_patterns(self, data: list[TimeSeriesPoint]) -> list[SeasonalPattern]:
        """Detect seasonal patterns in time series data.

        Args:
            data: Time series data points

        Returns:
            List of detected seasonal patterns
        """
        if len(data) < max(self.seasonal_periods) * 2:
            return []

        values = [point.value for point in data]
        patterns = []

        for period in self.seasonal_periods:
            if len(values) < period * 2:
                continue

            # Detect seasonal pattern using autocorrelation
            pattern = self._detect_seasonal_pattern(values, period)
            if pattern and pattern.confidence > 0.3:
                patterns.append(pattern)

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def detect_change_points(
        self, data: list[TimeSeriesPoint], event_name: str
    ) -> list[ChangePoint]:
        """Detect change points in time series data.

        Args:
            data: Time series data points
            event_name: Name of the ProfileEvent

        Returns:
            List of detected change points
        """
        if len(data) < 20:
            return []

        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]
        change_points = []

        # Use CUSUM (Cumulative Sum) algorithm for change point detection
        change_indices = self._cusum_change_detection(values)

        for idx in change_indices:
            if 5 <= idx <= len(data) - 5:  # Ensure enough data on both sides
                change_point = self._analyze_change_point(data, idx, event_name)
                if change_point.confidence > 0.5:
                    change_points.append(change_point)

        return change_points

    def _create_minimal_trend_analysis(
        self, data: list[TimeSeriesPoint], event_name: str
    ) -> TrendAnalysis:
        """Create minimal trend analysis for insufficient data."""
        if not data:
            values = [0.0]
            timestamps = [datetime.now()]
        else:
            values = [point.value for point in data]
            timestamps = [point.timestamp for point in data]

        return TrendAnalysis(
            event_name=event_name,
            analysis_period=(timestamps[0], timestamps[-1]),
            trend_type=TrendType.STABLE,
            slope=0.0,
            r_squared=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            trend_strength=0.0,
            volatility=0.0,
            persistence=0.0,
        )

    def _linear_regression(
        self, x: list[float], y: list[float]
    ) -> tuple[float, float, float, float]:
        """Perform linear regression and return slope, intercept, r-squared, and p-value."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 0.0, 0.0, 1.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
        sum_x_squared = sum(xi * xi for xi in x)
        sum_y_squared = sum(yi * yi for yi in y)

        # Calculate slope and intercept
        denominator = n * sum_x_squared - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, statistics.mean(y), 0.0, 1.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y, strict=False))

        if ss_tot < 1e-10:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        # Simplified p-value calculation
        # In practice, you'd use proper statistical methods
        p_value = max(0.01, 1.0 - abs(r_squared))

        return slope, intercept, r_squared, p_value

    def _classify_trend(self, slope: float, r_squared: float, values: list[float]) -> TrendType:
        """Classify the type of trend based on slope and R-squared."""
        if r_squared < 0.1:
            # Low correlation suggests volatile or stable data
            cv = statistics.stdev(values) / (statistics.mean(values) + 1e-10)
            return TrendType.VOLATILE if cv > 0.5 else TrendType.STABLE

        if abs(slope) < 1e-6:
            return TrendType.STABLE
        elif slope > 0:
            return TrendType.INCREASING
        else:
            return TrendType.DECREASING

    def _calculate_persistence(self, values: list[float], window: int) -> float:
        """Calculate trend persistence using moving windows."""
        if len(values) < window * 2:
            return 0.0

        # Calculate trend direction in overlapping windows
        trends = []
        for i in range(len(values) - window + 1):
            window_values = values[i : i + window]
            x = list(range(window))
            slope, _, _, _ = self._linear_regression(x, window_values)
            trends.append(1 if slope > 0 else -1 if slope < 0 else 0)

        if not trends:
            return 0.0

        # Calculate consistency
        mode_trend = max(set(trends), key=trends.count)
        consistency = trends.count(mode_trend) / len(trends)

        return consistency

    def _calculate_confidence_interval(
        self, slope: float, values: list[float]
    ) -> tuple[float, float]:
        """Calculate confidence interval for the slope estimate."""
        if len(values) < 3:
            return (slope, slope)

        std_error = statistics.stdev(values) / math.sqrt(len(values))
        margin = 1.96 * std_error  # 95% confidence interval

        return (slope - margin, slope + margin)

    def _generate_forecast(
        self, data: list[TimeSeriesPoint], slope: float, intercept: float, trend_type: TrendType
    ) -> tuple[list[float], list[datetime], list[float]]:
        """Generate forecasts based on trend analysis."""
        if len(data) < 2:
            return [], [], []

        last_timestamp = data[-1].timestamp
        forecast_points = 10
        forecast_values = []
        forecast_timestamps = []
        forecast_confidence = []

        # Calculate time delta between points
        if len(data) >= 2:
            time_delta = (data[-1].timestamp - data[-2].timestamp).total_seconds()
        else:
            time_delta = 3600  # Default to 1 hour

        for i in range(1, forecast_points + 1):
            future_timestamp = last_timestamp + timedelta(seconds=time_delta * i)

            # Simple linear extrapolation
            future_time_numeric = (future_timestamp - data[0].timestamp).total_seconds()
            forecast_value = slope * future_time_numeric + intercept

            # Add some uncertainty
            confidence = max(0.1, 1.0 - (i * 0.1))  # Decreasing confidence over time

            forecast_values.append(max(0.0, forecast_value))  # Ensure non-negative
            forecast_timestamps.append(future_timestamp)
            forecast_confidence.append(confidence)

        return forecast_values, forecast_timestamps, forecast_confidence

    def _detect_seasonal_pattern(self, values: list[float], period: int) -> SeasonalPattern | None:
        """Detect seasonal pattern with given period using autocorrelation."""
        if len(values) < period * 2:
            return None

        # Calculate autocorrelation at the specified lag
        autocorr = self._autocorrelation(values, period)

        if autocorr < 0.3:  # Minimum correlation threshold
            return None

        # Estimate amplitude and phase
        amplitude = self._estimate_seasonal_amplitude(values, period)
        phase = self._estimate_seasonal_phase(values, period)

        return SeasonalPattern(period=period, amplitude=amplitude, phase=phase, confidence=autocorr)

    def _autocorrelation(self, values: list[float], lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(values) <= lag:
            return 0.0

        n = len(values) - lag
        if n <= 1:
            return 0.0

        mean_val = statistics.mean(values)

        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
        denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _estimate_seasonal_amplitude(self, values: list[float], period: int) -> float:
        """Estimate the amplitude of seasonal variation."""
        if len(values) < period * 2:
            return 0.0

        # Group values by position in period
        period_groups = defaultdict(list)
        for i, value in enumerate(values):
            period_groups[i % period].append(value)

        # Calculate range of period averages
        period_means = [statistics.mean(group) for group in period_groups.values() if group]

        if len(period_means) < 2:
            return 0.0

        return max(period_means) - min(period_means)

    def _estimate_seasonal_phase(self, values: list[float], period: int) -> float:
        """Estimate the phase offset of seasonal pattern."""
        if len(values) < period:
            return 0.0

        # Find the position of maximum in the first complete period
        first_period = values[:period]
        max_index = first_period.index(max(first_period))

        return max_index / period  # Normalized phase (0-1)

    def _cusum_change_detection(self, values: list[float], threshold: float = 5.0) -> list[int]:
        """Detect change points using CUSUM algorithm."""
        if len(values) < 10:
            return []

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if std_val < 1e-10:
            return []

        # Normalize values
        normalized = [(v - mean_val) / std_val for v in values]

        # CUSUM for upward changes
        cusum_pos = [0.0]
        cusum_neg = [0.0]

        for val in normalized:
            cusum_pos.append(max(0, cusum_pos[-1] + val - 0.5))
            cusum_neg.append(min(0, cusum_neg[-1] + val + 0.5))

        # Find change points
        change_points = []
        for i in range(1, len(cusum_pos)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                change_points.append(i)

        # Remove consecutive change points
        filtered_points = []
        for point in change_points:
            if not filtered_points or point - filtered_points[-1] > 5:
                filtered_points.append(point)

        return filtered_points

    def _analyze_change_point(
        self, data: list[TimeSeriesPoint], index: int, event_name: str
    ) -> ChangePoint:
        """Analyze a detected change point."""
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        # Calculate before/after statistics
        before_values = values[:index]
        after_values = values[index:]

        before_mean = statistics.mean(before_values) if before_values else 0.0
        after_mean = statistics.mean(after_values) if after_values else 0.0
        before_std = statistics.stdev(before_values) if len(before_values) > 1 else 0.0
        after_std = statistics.stdev(after_values) if len(after_values) > 1 else 0.0

        # Determine change type and magnitude
        mean_change = abs(after_mean - before_mean)
        std_change = abs(after_std - before_std)

        if mean_change > std_change:
            change_type = ChangePointType.LEVEL_SHIFT
            magnitude = mean_change
        else:
            change_type = ChangePointType.VARIANCE_CHANGE
            magnitude = std_change

        # Calculate confidence based on effect size
        pooled_std = math.sqrt((before_std**2 + after_std**2) / 2)
        confidence = min(1.0, magnitude / (pooled_std + 1e-10) / 3.0)  # Cohen's d scaled

        return ChangePoint(
            event_name=event_name,
            timestamp=timestamps[index],
            change_type=change_type,
            magnitude=magnitude,
            confidence=confidence,
            before_mean=before_mean,
            after_mean=after_mean,
            before_std=before_std,
            after_std=after_std,
        )


class PerformanceBaselineEngine:
    """Engine for establishing and managing performance baselines."""

    def __init__(self, client: Client, lookback_days: int = 30):
        """Initialize the baseline engine.

        Args:
            client: ClickHouse client instance
            lookback_days: Number of days to look back for baseline calculation
        """
        self.client = client
        self.lookback_days = lookback_days
        self.baselines: dict[str, BaselineMetrics] = {}

    def establish_baseline(self, event_name: str, force_refresh: bool = False) -> BaselineMetrics:
        """Establish or refresh baseline for a ProfileEvent.

        Args:
            event_name: Name of the ProfileEvent
            force_refresh: Whether to force refresh existing baseline

        Returns:
            Established baseline metrics
        """
        # Check if we have a recent baseline
        if not force_refresh and event_name in self.baselines:
            baseline = self.baselines[event_name]
            if datetime.now() - baseline.last_updated < timedelta(hours=6):
                return baseline

        # Query historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.lookback_days)

        query = f"""
        SELECT
            ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event_name}')] as value,
            event_time
        FROM system.query_log
        WHERE event_time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
          AND event_time <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
          AND has(ProfileEvents.Names, '{event_name}')
          AND ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event_name}')] > 0
        ORDER BY event_time
        """

        try:
            result = execute_query_with_retry(self.client, query)
            if not result:
                return self._create_default_baseline(event_name, start_time, end_time)

            values = [row["value"] for row in result]
            if len(values) < 10:
                return self._create_default_baseline(event_name, start_time, end_time)

            # Calculate baseline statistics
            baseline = self._calculate_baseline_metrics(event_name, values, start_time, end_time)
            self.baselines[event_name] = baseline

            return baseline

        except Exception as e:
            logger.error(f"Error establishing baseline for {event_name}: {e}")
            return self._create_default_baseline(event_name, start_time, end_time)

    def update_baseline(self, event_name: str, new_data: list[float]) -> BaselineMetrics:
        """Update existing baseline with new data using exponential smoothing.

        Args:
            event_name: Name of the ProfileEvent
            new_data: New data points to incorporate

        Returns:
            Updated baseline metrics
        """
        current_baseline = self.baselines.get(event_name)
        if not current_baseline:
            # No existing baseline, establish new one
            return self.establish_baseline(event_name)

        if not new_data:
            return current_baseline

        # Exponential smoothing parameters
        alpha = 0.1  # Smoothing factor

        # Update statistics with exponential smoothing
        new_mean = statistics.mean(new_data)
        new_std = statistics.stdev(new_data) if len(new_data) > 1 else current_baseline.std_dev

        updated_mean = alpha * new_mean + (1 - alpha) * current_baseline.mean
        updated_std = alpha * new_std + (1 - alpha) * current_baseline.std_dev

        # Update percentiles (simplified approach)
        combined_data = new_data[-100:]  # Use recent data for percentile calculation

        updated_baseline = BaselineMetrics(
            event_name=event_name,
            mean=updated_mean,
            median=statistics.median(combined_data) if combined_data else current_baseline.median,
            std_dev=updated_std,
            min_value=min(min(new_data), current_baseline.min_value),
            max_value=max(max(new_data), current_baseline.max_value),
            percentile_25=(
                self._percentile(combined_data, 25)
                if combined_data
                else current_baseline.percentile_25
            ),
            percentile_75=(
                self._percentile(combined_data, 75)
                if combined_data
                else current_baseline.percentile_75
            ),
            percentile_95=(
                self._percentile(combined_data, 95)
                if combined_data
                else current_baseline.percentile_95
            ),
            percentile_99=(
                self._percentile(combined_data, 99)
                if combined_data
                else current_baseline.percentile_99
            ),
            lower_control_limit=updated_mean - 2 * updated_std,
            upper_control_limit=updated_mean + 2 * updated_std,
            warning_threshold=updated_mean + 1.5 * updated_std,
            critical_threshold=updated_mean + 3 * updated_std,
            sample_size=current_baseline.sample_size + len(new_data),
            confidence_interval=(
                updated_mean - 1.96 * updated_std,
                updated_mean + 1.96 * updated_std,
            ),
            baseline_period=current_baseline.baseline_period,
            last_updated=datetime.now(),
        )

        self.baselines[event_name] = updated_baseline
        return updated_baseline

    def get_baseline(self, event_name: str) -> BaselineMetrics | None:
        """Get existing baseline for a ProfileEvent.

        Args:
            event_name: Name of the ProfileEvent

        Returns:
            Baseline metrics if available, None otherwise
        """
        return self.baselines.get(event_name)

    def _calculate_baseline_metrics(
        self, event_name: str, values: list[float], start_time: datetime, end_time: datetime
    ) -> BaselineMetrics:
        """Calculate comprehensive baseline metrics from historical data."""
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)

        # Calculate percentiles
        p25 = self._percentile(values, 25)
        p75 = self._percentile(values, 75)
        p95 = self._percentile(values, 95)
        p99 = self._percentile(values, 99)

        # Calculate control limits
        lower_control = mean_val - 2 * std_val
        upper_control = mean_val + 2 * std_val
        warning_threshold = mean_val + 1.5 * std_val
        critical_threshold = mean_val + 3 * std_val

        return BaselineMetrics(
            event_name=event_name,
            mean=mean_val,
            median=median_val,
            std_dev=std_val,
            min_value=min_val,
            max_value=max_val,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            percentile_99=p99,
            lower_control_limit=lower_control,
            upper_control_limit=upper_control,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            sample_size=len(values),
            confidence_interval=(mean_val - 1.96 * std_val, mean_val + 1.96 * std_val),
            baseline_period=(start_time, end_time),
            last_updated=datetime.now(),
        )

    def _create_default_baseline(
        self, event_name: str, start_time: datetime, end_time: datetime
    ) -> BaselineMetrics:
        """Create a default baseline when insufficient data is available."""
        return BaselineMetrics(
            event_name=event_name,
            mean=0.0,
            median=0.0,
            std_dev=1.0,
            min_value=0.0,
            max_value=0.0,
            percentile_25=0.0,
            percentile_75=0.0,
            percentile_95=0.0,
            percentile_99=0.0,
            lower_control_limit=0.0,
            upper_control_limit=10.0,
            warning_threshold=5.0,
            critical_threshold=10.0,
            sample_size=0,
            confidence_interval=(0.0, 0.0),
            baseline_period=(start_time, end_time),
            last_updated=datetime.now(),
        )

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate the specified percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)

        return d0 + d1


class AnomalyDetectionEngine:
    """Multi-method anomaly detection engine."""

    def __init__(self, baseline_engine: PerformanceBaselineEngine):
        """Initialize the anomaly detection engine.

        Args:
            baseline_engine: Baseline engine for reference metrics
        """
        self.baseline_engine = baseline_engine
        self.anomaly_cache: dict[str, list[AnomalyScore]] = {}

    def detect_anomalies(
        self,
        data: list[TimeSeriesPoint],
        event_name: str,
        seasonal_patterns: list[SeasonalPattern] = None,
    ) -> list[AnomalyScore]:
        """Detect anomalies in time series data using multiple methods.

        Args:
            data: Time series data points
            event_name: Name of the ProfileEvent
            seasonal_patterns: Known seasonal patterns for adjustment

        Returns:
            List of anomaly scores for each data point
        """
        if len(data) < 3:
            return []

        # Get or establish baseline
        baseline = self.baseline_engine.establish_baseline(event_name)

        anomalies = []
        values = [point.value for point in data]

        for i, point in enumerate(data):
            anomaly_score = self._calculate_anomaly_score(
                point, baseline, values, i, seasonal_patterns
            )

            if anomaly_score.overall_score > 0.25:  # Minimum threshold
                anomalies.append(anomaly_score)

        # Cache results
        self.anomaly_cache[event_name] = anomalies

        return anomalies

    def _calculate_anomaly_score(
        self,
        point: TimeSeriesPoint,
        baseline: BaselineMetrics,
        all_values: list[float],
        index: int,
        seasonal_patterns: list[SeasonalPattern] = None,
    ) -> AnomalyScore:
        """Calculate comprehensive anomaly score for a single point."""
        value = point.value

        # Statistical scoring methods
        z_score = self._calculate_z_score(value, baseline)
        modified_z_score = self._calculate_modified_z_score(value, all_values, index)
        iqr_score = self._calculate_iqr_score(value, baseline)
        percentile_rank = self._calculate_percentile_rank(value, baseline)

        # Composite statistical score
        statistical_score = max(abs(z_score) / 3.0, abs(modified_z_score) / 3.5, iqr_score)
        statistical_score = min(1.0, statistical_score)

        # Pattern-based scoring
        pattern_score = self._calculate_pattern_score(point, all_values, index)

        # Trend-based scoring
        trend_score = self._calculate_trend_score(all_values, index)

        # Seasonal adjustment
        seasonal_adjustment = (
            self._calculate_seasonal_adjustment(point, seasonal_patterns)
            if seasonal_patterns
            else 0.0
        )

        # Overall anomaly score (weighted combination)
        overall_score = (
            0.4 * statistical_score
            + 0.3 * pattern_score
            + 0.2 * trend_score
            + 0.1 * abs(seasonal_adjustment)
        )

        # Determine severity and anomaly types
        severity = self._determine_severity(overall_score, statistical_score)
        anomaly_types = self._identify_anomaly_types(
            z_score, pattern_score, trend_score, seasonal_adjustment
        )

        # Calculate confidence
        confidence = min(1.0, overall_score * 1.2)

        return AnomalyScore(
            timestamp=point.timestamp,
            event_name=baseline.event_name,
            value=value,
            z_score=z_score,
            modified_z_score=modified_z_score,
            iqr_score=iqr_score,
            percentile_rank=percentile_rank,
            statistical_score=statistical_score,
            pattern_score=pattern_score,
            trend_score=trend_score,
            overall_score=overall_score,
            severity=severity,
            anomaly_types=anomaly_types,
            confidence=confidence,
            baseline_deviation=abs(value - baseline.mean) / (baseline.std_dev + 1e-10),
            seasonal_adjustment=seasonal_adjustment,
        )

    def _calculate_z_score(self, value: float, baseline: BaselineMetrics) -> float:
        """Calculate standard Z-score."""
        if baseline.std_dev < 1e-10:
            return 0.0
        return (value - baseline.mean) / baseline.std_dev

    def _calculate_modified_z_score(
        self, value: float, all_values: list[float], index: int
    ) -> float:
        """Calculate modified Z-score using median and MAD."""
        if len(all_values) < 3:
            return 0.0

        # Use a window around the current point
        window_start = max(0, index - 25)
        window_end = min(len(all_values), index + 25)
        window_values = all_values[window_start:window_end]

        median_val = statistics.median(window_values)

        # Calculate Median Absolute Deviation (MAD)
        mad = statistics.median([abs(v - median_val) for v in window_values])

        if mad < 1e-10:
            return 0.0

        return 0.6745 * (value - median_val) / mad

    def _calculate_iqr_score(self, value: float, baseline: BaselineMetrics) -> float:
        """Calculate IQR-based anomaly score."""
        iqr = baseline.percentile_75 - baseline.percentile_25
        if iqr < 1e-10:
            return 0.0

        # Check if value is outside IQR bounds
        lower_bound = baseline.percentile_25 - 1.5 * iqr
        upper_bound = baseline.percentile_75 + 1.5 * iqr

        if value < lower_bound:
            return (lower_bound - value) / iqr
        elif value > upper_bound:
            return (value - upper_bound) / iqr
        else:
            return 0.0

    def _calculate_percentile_rank(self, value: float, baseline: BaselineMetrics) -> float:
        """Calculate percentile rank of the value."""
        # Simplified percentile rank calculation
        if value <= baseline.percentile_25:
            return 0.25
        elif value <= baseline.median:
            return 0.5
        elif value <= baseline.percentile_75:
            return 0.75
        elif value <= baseline.percentile_95:
            return 0.95
        else:
            return 0.99

    def _calculate_pattern_score(
        self, point: TimeSeriesPoint, all_values: list[float], index: int
    ) -> float:
        """Calculate pattern-based anomaly score."""
        if index < 5 or index >= len(all_values) - 5:
            return 0.0

        # Look at local pattern around the point
        before_window = all_values[index - 5 : index]
        after_window = all_values[index + 1 : index + 6]

        if not before_window or not after_window:
            return 0.0

        # Calculate local trend before and after
        before_trend = (before_window[-1] - before_window[0]) / len(before_window)
        after_trend = (after_window[-1] - after_window[0]) / len(after_window)

        # Check for sudden changes in trend
        trend_change = abs(after_trend - before_trend)

        # Check for spikes relative to local context
        local_mean = statistics.mean(before_window + after_window)
        local_std = (
            statistics.stdev(before_window + after_window)
            if len(before_window + after_window) > 1
            else 1.0
        )

        spike_score = abs(point.value - local_mean) / (local_std + 1e-10)

        return min(1.0, (trend_change + spike_score) / 6.0)

    def _calculate_trend_score(self, all_values: list[float], index: int) -> float:
        """Calculate trend-based anomaly score."""
        if index < 10:
            return 0.0

        # Compare recent trend with historical trend
        recent_window = all_values[max(0, index - 10) : index + 1]
        historical_window = all_values[max(0, index - 30) : index - 10]

        if len(recent_window) < 3 or len(historical_window) < 3:
            return 0.0

        # Calculate trend slopes
        recent_trend = self._calculate_slope(recent_window)
        historical_trend = self._calculate_slope(historical_window)

        # Score based on trend deviation
        trend_deviation = abs(recent_trend - historical_trend)

        return min(1.0, trend_deviation / 5.0)

    def _calculate_slope(self, values: list[float]) -> float:
        """Calculate slope of values using linear regression."""
        if len(values) < 2:
            return 0.0

        x = list(range(len(values)))
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x_squared = sum(x[i] * x[i] for i in range(n))

        denominator = n * sum_x_squared - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_seasonal_adjustment(
        self, point: TimeSeriesPoint, seasonal_patterns: list[SeasonalPattern]
    ) -> float:
        """Calculate seasonal adjustment factor."""
        if not seasonal_patterns:
            return 0.0

        # For simplicity, use the strongest seasonal pattern
        strongest_pattern = max(seasonal_patterns, key=lambda p: p.confidence)

        # Calculate expected seasonal component
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        hour_of_day = point.timestamp.hour
        seasonal_factor = math.sin(2 * math.pi * hour_of_day / 24) * strongest_pattern.amplitude

        return seasonal_factor

    def _determine_severity(
        self, overall_score: float, statistical_score: float
    ) -> AnomalySeverity:
        """Determine anomaly severity based on scores."""
        if overall_score > 0.8 or statistical_score > 0.9:
            return AnomalySeverity.CRITICAL
        elif overall_score > 0.6 or statistical_score > 0.7:
            return AnomalySeverity.HIGH
        elif overall_score > 0.4 or statistical_score > 0.5:
            return AnomalySeverity.MEDIUM
        elif overall_score > 0.25:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    def _identify_anomaly_types(
        self, z_score: float, pattern_score: float, trend_score: float, seasonal_adjustment: float
    ) -> list[AnomalyType]:
        """Identify the types of anomalies present."""
        anomaly_types = []

        if abs(z_score) > 2.5:
            anomaly_types.append(AnomalyType.STATISTICAL_OUTLIER)

        if pattern_score > 0.5:
            anomaly_types.append(AnomalyType.PATTERN_DEVIATION)

        if trend_score > 0.4:
            anomaly_types.append(AnomalyType.TREND_ANOMALY)

        if abs(seasonal_adjustment) > 0.3:
            anomaly_types.append(AnomalyType.SEASONAL_ANOMALY)

        return anomaly_types


class PatternRecognitionEngine:
    """Engine for recognizing and matching patterns in time series data."""

    def __init__(self):
        """Initialize the pattern recognition engine."""
        self.known_patterns: dict[str, list[PatternMatch]] = {}
        self.pattern_templates: dict[str, list[float]] = {}

    def detect_patterns(self, data: list[TimeSeriesPoint], event_name: str) -> list[PatternMatch]:
        """Detect patterns in time series data.

        Args:
            data: Time series data points
            event_name: Name of the ProfileEvent

        Returns:
            List of detected patterns
        """
        if len(data) < 10:
            return []

        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        patterns = []

        # Detect spike patterns
        spike_patterns = self._detect_spike_patterns(data, event_name)
        patterns.extend(spike_patterns)

        # Detect periodic patterns
        periodic_patterns = self._detect_periodic_patterns(data, event_name)
        patterns.extend(periodic_patterns)

        # Detect gradual increase/decrease patterns
        trend_patterns = self._detect_trend_patterns(data, event_name)
        patterns.extend(trend_patterns)

        # Store detected patterns for future reference
        self.known_patterns[event_name] = patterns

        return patterns

    def match_historical_patterns(
        self, current_data: list[TimeSeriesPoint], event_name: str
    ) -> list[PatternMatch]:
        """Match current data against historical patterns.

        Args:
            current_data: Current time series data
            event_name: Name of the ProfileEvent

        Returns:
            List of pattern matches
        """
        if event_name not in self.known_patterns or len(current_data) < 5:
            return []

        historical_patterns = self.known_patterns[event_name]
        matches = []

        current_values = [point.value for point in current_data]

        for historical_pattern in historical_patterns:
            similarity = self._calculate_pattern_similarity(
                current_values, historical_pattern, event_name
            )

            if similarity > 0.6:  # Similarity threshold
                match = PatternMatch(
                    pattern_id=f"{historical_pattern.pattern_type}_{len(matches)}",
                    event_name=event_name,
                    start_time=current_data[0].timestamp,
                    end_time=current_data[-1].timestamp,
                    similarity_score=similarity,
                    pattern_type=historical_pattern.pattern_type,
                    duration=current_data[-1].timestamp - current_data[0].timestamp,
                    amplitude=max(current_values) - min(current_values),
                    frequency=None,
                    historical_occurrences=len(
                        [
                            p
                            for p in historical_patterns
                            if p.pattern_type == historical_pattern.pattern_type
                        ]
                    ),
                    typical_severity=historical_pattern.typical_severity,
                )
                matches.append(match)

        return matches

    def _detect_spike_patterns(
        self, data: list[TimeSeriesPoint], event_name: str
    ) -> list[PatternMatch]:
        """Detect spike patterns (sudden increases followed by decreases)."""
        patterns = []
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        if len(values) < 5:
            return patterns

        # Calculate rolling statistics
        window_size = min(10, len(values) // 3)

        for i in range(window_size, len(values) - window_size):
            # Check for spike pattern
            before_window = values[i - window_size : i]
            spike_value = values[i]
            after_window = values[i + 1 : i + window_size + 1]

            before_mean = statistics.mean(before_window)
            after_mean = statistics.mean(after_window)
            before_std = statistics.stdev(before_window) if len(before_window) > 1 else 1.0

            # Check if current value is significantly higher than before and after
            if (
                spike_value > before_mean + 2 * before_std
                and spike_value > after_mean + 2 * before_std
                and spike_value > max(before_mean, after_mean) * 1.5
            ):
                # Find pattern boundaries
                start_idx = max(0, i - 2)
                end_idx = min(len(data), i + 3)

                pattern = PatternMatch(
                    pattern_id=f"spike_{i}",
                    event_name=event_name,
                    start_time=timestamps[start_idx],
                    end_time=timestamps[end_idx - 1],
                    similarity_score=0.8,  # High confidence for detected spikes
                    pattern_type="spike",
                    duration=timestamps[end_idx - 1] - timestamps[start_idx],
                    amplitude=spike_value - before_mean,
                    frequency=None,
                    historical_occurrences=0,
                    typical_severity=AnomalySeverity.HIGH,
                )
                patterns.append(pattern)

        return patterns

    def _detect_periodic_patterns(
        self, data: list[TimeSeriesPoint], event_name: str
    ) -> list[PatternMatch]:
        """Detect periodic/cyclical patterns."""
        patterns = []
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        if len(values) < 20:
            return patterns

        # Look for periodic patterns using autocorrelation
        for period in [5, 10, 24, 168]:  # Common periods (5min, 10min, hourly, weekly)
            if len(values) >= period * 2:
                autocorr = self._calculate_autocorrelation(values, period)

                if autocorr > 0.5:  # Strong periodic pattern
                    pattern = PatternMatch(
                        pattern_id=f"periodic_{period}",
                        event_name=event_name,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        similarity_score=autocorr,
                        pattern_type="periodic",
                        duration=timestamps[-1] - timestamps[0],
                        amplitude=max(values) - min(values),
                        frequency=1.0 / period,
                        historical_occurrences=0,
                        typical_severity=AnomalySeverity.INFO,
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_trend_patterns(
        self, data: list[TimeSeriesPoint], event_name: str
    ) -> list[PatternMatch]:
        """Detect gradual trend patterns (increases/decreases)."""
        patterns = []
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]

        if len(values) < 10:
            return patterns

        # Calculate overall trend
        x = list(range(len(values)))
        slope = self._calculate_slope(x, values)

        # Determine if trend is significant
        mean_val = statistics.mean(values)
        relative_slope = abs(slope) / (mean_val + 1e-10)

        if relative_slope > 0.1:  # Significant trend
            trend_type = "increasing_trend" if slope > 0 else "decreasing_trend"
            severity = AnomalySeverity.MEDIUM if relative_slope > 0.5 else AnomalySeverity.LOW

            pattern = PatternMatch(
                pattern_id=f"trend_{trend_type}",
                event_name=event_name,
                start_time=timestamps[0],
                end_time=timestamps[-1],
                similarity_score=0.7,
                pattern_type=trend_type,
                duration=timestamps[-1] - timestamps[0],
                amplitude=abs(values[-1] - values[0]),
                frequency=None,
                historical_occurrences=0,
                typical_severity=severity,
            )
            patterns.append(pattern)

        return patterns

    def _calculate_autocorrelation(self, values: list[float], lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(values) <= lag:
            return 0.0

        n = len(values) - lag
        if n <= 1:
            return 0.0

        mean_val = statistics.mean(values)

        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
        denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _calculate_slope(self, x: list[float], y: list[float]) -> float:
        """Calculate slope using linear regression."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x_squared = sum(x[i] * x[i] for i in range(n))

        denominator = n * sum_x_squared - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_pattern_similarity(
        self, current_values: list[float], historical_pattern: PatternMatch, event_name: str
    ) -> float:
        """Calculate similarity between current data and historical pattern."""
        # This is a simplified similarity calculation
        # In practice, you'd use more sophisticated methods like DTW (Dynamic Time Warping)

        if not hasattr(historical_pattern, "template_values"):
            return 0.0

        template_values = getattr(historical_pattern, "template_values", [])

        if not template_values or len(current_values) != len(template_values):
            return 0.0

        # Normalize both series
        current_normalized = self._normalize_series(current_values)
        template_normalized = self._normalize_series(template_values)

        # Calculate correlation coefficient
        if len(current_normalized) != len(template_normalized):
            return 0.0

        try:
            correlation = statistics.correlation(current_normalized, template_normalized)
            return max(0.0, correlation)
        except statistics.StatisticsError:
            return 0.0

    def _normalize_series(self, values: list[float]) -> list[float]:
        """Normalize a time series to zero mean and unit variance."""
        if len(values) < 2:
            return values

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if std_val < 1e-10:
            return [0.0] * len(values)

        return [(v - mean_val) / std_val for v in values]


class CorrelationAnalyzer:
    """Engine for analyzing correlations between different ProfileEvents."""

    def __init__(self, client: Client):
        """Initialize the correlation analyzer.

        Args:
            client: ClickHouse client instance
        """
        self.client = client
        self.correlation_cache: dict[tuple[str, str], CorrelationAnalysis] = {}

    def analyze_correlations(
        self, event_pairs: list[tuple[str, str]], lookback_hours: int = 24
    ) -> list[CorrelationAnalysis]:
        """Analyze correlations between pairs of ProfileEvents.

        Args:
            event_pairs: List of (event1, event2) tuples to analyze
            lookback_hours: How many hours back to analyze

        Returns:
            List of correlation analyses
        """
        correlations = []

        for event1, event2 in event_pairs:
            # Check cache first
            cache_key = (event1, event2)
            if cache_key in self.correlation_cache:
                cached_result = self.correlation_cache[cache_key]
                if datetime.now() - cached_result.recent_correlation < timedelta(hours=1):
                    correlations.append(cached_result)
                    continue

            # Calculate correlation
            correlation = self._calculate_correlation(event1, event2, lookback_hours)
            if correlation:
                correlations.append(correlation)
                self.correlation_cache[cache_key] = correlation

        return correlations

    def discover_correlations(
        self, events: list[str], lookback_hours: int = 24, min_correlation: float = 0.3
    ) -> list[CorrelationAnalysis]:
        """Discover correlations among a list of ProfileEvents.

        Args:
            events: List of ProfileEvent names to analyze
            lookback_hours: How many hours back to analyze
            min_correlation: Minimum correlation threshold

        Returns:
            List of discovered correlations above threshold
        """
        correlations = []

        # Generate all pairs
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                correlation = self._calculate_correlation(events[i], events[j], lookback_hours)
                if correlation and abs(correlation.correlation_coefficient) >= min_correlation:
                    correlations.append(correlation)

        return sorted(correlations, key=lambda c: abs(c.correlation_coefficient), reverse=True)

    def _calculate_correlation(
        self, event1: str, event2: str, lookback_hours: int
    ) -> CorrelationAnalysis | None:
        """Calculate correlation between two ProfileEvents."""
        # Query data for both events
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        query = f"""
        WITH event1_data AS (
            SELECT
                toUnixTimestamp(event_time) as ts,
                ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event1}')] as value1
            FROM system.query_log
            WHERE event_time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND event_time <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND has(ProfileEvents.Names, '{event1}')
              AND ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event1}')] > 0
        ),
        event2_data AS (
            SELECT
                toUnixTimestamp(event_time) as ts,
                ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event2}')] as value2
            FROM system.query_log
            WHERE event_time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND event_time <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND has(ProfileEvents.Names, '{event2}')
              AND ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event2}')] > 0
        )
        SELECT
            e1.ts,
            e1.value1,
            e2.value2
        FROM event1_data e1
        INNER JOIN event2_data e2 ON abs(e1.ts - e2.ts) <= 60  -- 1 minute tolerance
        ORDER BY e1.ts
        """

        try:
            result = execute_query_with_retry(self.client, query)
            if len(result) < 10:  # Need minimum data points
                return None

            values1 = [row["value1"] for row in result]
            values2 = [row["value2"] for row in result]

            # Calculate Pearson correlation
            correlation_coeff = self._pearson_correlation(values1, values2)

            if abs(correlation_coeff) < 0.1:  # Too weak to be meaningful
                return None

            # Calculate statistical significance (simplified)
            n = len(values1)
            t_stat = correlation_coeff * math.sqrt((n - 2) / (1 - correlation_coeff**2))
            p_value = 2 * (1 - abs(t_stat) / math.sqrt(n))  # Simplified p-value

            # Analyze relationship type
            relationship_type = self._analyze_relationship_type(values1, values2)

            # Calculate lag correlations
            lag_correlations = self._calculate_lag_correlations(values1, values2)

            # Calculate stability metrics
            correlation_stability = self._calculate_correlation_stability(values1, values2)

            return CorrelationAnalysis(
                primary_event=event1,
                secondary_event=event2,
                correlation_coefficient=correlation_coeff,
                p_value=p_value,
                relationship_type=relationship_type,
                lag_correlation=lag_correlations,
                correlation_stability=correlation_stability,
                recent_correlation=correlation_coeff,  # Simplified
                historical_correlation=correlation_coeff,  # Simplified
            )

        except Exception as e:
            logger.error(f"Error calculating correlation between {event1} and {event2}: {e}")
            return None

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x_squared = sum(x[i] * x[i] for i in range(n))
        sum_y_squared = sum(y[i] * y[i] for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x_squared - sum_x**2) * (n * sum_y_squared - sum_y**2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _analyze_relationship_type(self, x: list[float], y: list[float]) -> str:
        """Analyze the type of relationship (linear, exponential, etc.)."""
        # Simple heuristic - in practice you'd use more sophisticated methods
        correlation_linear = abs(self._pearson_correlation(x, y))

        # Try log transformation
        try:
            log_x = [math.log(max(1e-10, val)) for val in x]
            correlation_log = abs(self._pearson_correlation(log_x, y))
        except (ValueError, OverflowError):
            correlation_log = 0.0

        if correlation_log > correlation_linear * 1.1:
            return "logarithmic"
        else:
            return "linear"

    def _calculate_lag_correlations(self, x: list[float], y: list[float]) -> dict[int, float]:
        """Calculate correlations at different lags."""
        lag_correlations = {}

        for lag in range(1, min(10, len(x) // 4)):  # Check up to 10 lags
            if len(x) > lag:
                lagged_x = x[:-lag]
                lagged_y = y[lag:]

                if len(lagged_x) == len(lagged_y) and len(lagged_x) > 1:
                    correlation = self._pearson_correlation(lagged_x, lagged_y)
                    lag_correlations[lag] = correlation

        return lag_correlations

    def _calculate_correlation_stability(self, x: list[float], y: list[float]) -> float:
        """Calculate how stable the correlation is over time."""
        if len(x) < 20:
            return 0.0

        # Calculate correlation in different time windows
        window_size = len(x) // 4
        correlations = []

        for i in range(0, len(x) - window_size, window_size // 2):
            window_x = x[i : i + window_size]
            window_y = y[i : i + window_size]

            if len(window_x) == len(window_y) and len(window_x) > 1:
                correlation = self._pearson_correlation(window_x, window_y)
                correlations.append(correlation)

        if len(correlations) < 2:
            return 0.0

        # Stability is inverse of standard deviation of correlations
        correlation_std = statistics.stdev(correlations)
        return max(0.0, 1.0 - correlation_std)


class PatternAnalysisEngine:
    """Main coordinator for all pattern analysis capabilities."""

    def __init__(self, client: Client):
        """Initialize the pattern analysis engine.

        Args:
            client: ClickHouse client instance
        """
        self.client = client
        self.profile_events_analyzer = ProfileEventsAnalyzer(client)

        # Initialize sub-engines
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.baseline_engine = PerformanceBaselineEngine(client)
        self.anomaly_engine = AnomalyDetectionEngine(self.baseline_engine)
        self.pattern_engine = PatternRecognitionEngine()
        self.correlation_analyzer = CorrelationAnalyzer(client)

        # Analysis cache
        self.analysis_cache: dict[str, PatternAnalysisResult] = {}

    @log_execution_time
    def analyze_patterns(
        self, event_name: str, lookback_hours: int = 24, force_refresh: bool = False
    ) -> PatternAnalysisResult:
        """Perform comprehensive pattern analysis for a ProfileEvent.

        Args:
            event_name: Name of the ProfileEvent to analyze
            lookback_hours: How many hours back to analyze
            force_refresh: Whether to force refresh cached results

        Returns:
            Comprehensive pattern analysis results
        """
        # Check cache
        cache_key = f"{event_name}_{lookback_hours}"
        if not force_refresh and cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            cache_age = datetime.now() - cached_result.analysis_period[1]
            if cache_age < timedelta(hours=1):
                return cached_result

        # Get time series data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        time_series_data = self._get_time_series_data(event_name, start_time, end_time)

        if len(time_series_data) < 5:
            return self._create_minimal_analysis_result(event_name, start_time, end_time)

        # Establish baseline
        baseline_metrics = self.baseline_engine.establish_baseline(event_name)

        # Trend analysis
        trend_analysis = self.time_series_analyzer.analyze_trend(time_series_data, event_name)

        # Seasonal pattern detection
        seasonal_patterns = self.time_series_analyzer.detect_seasonal_patterns(time_series_data)

        # Change point detection
        change_points = self.time_series_analyzer.detect_change_points(time_series_data, event_name)

        # Anomaly detection
        anomalies = self.anomaly_engine.detect_anomalies(
            time_series_data, event_name, seasonal_patterns
        )

        # Pattern detection
        detected_patterns = self.pattern_engine.detect_patterns(time_series_data, event_name)

        # Pattern matching
        pattern_matches = self.pattern_engine.match_historical_patterns(
            time_series_data, event_name
        )
        detected_patterns.extend(pattern_matches)

        # Correlation analysis (with related events)
        related_events = self._get_related_events(event_name)
        correlations = []
        if related_events:
            event_pairs = [(event_name, related_event) for related_event in related_events[:5]]
            correlations = self.correlation_analyzer.analyze_correlations(
                event_pairs, lookback_hours
            )

        # Calculate summary statistics
        anomaly_rate = len(anomalies) / len(time_series_data) if time_series_data else 0.0
        pattern_coverage = self._calculate_pattern_coverage(detected_patterns, time_series_data)
        predictability_score = self._calculate_predictability_score(
            trend_analysis, seasonal_patterns
        )
        stability_score = self._calculate_stability_score(
            anomalies, change_points, time_series_data
        )

        # Create comprehensive result
        result = PatternAnalysisResult(
            event_name=event_name,
            analysis_period=(start_time, end_time),
            baseline_metrics=baseline_metrics,
            anomalies=anomalies,
            detected_patterns=detected_patterns,
            trend_analysis=trend_analysis,
            seasonal_patterns=seasonal_patterns,
            change_points=change_points,
            correlations=correlations,
            anomaly_rate=anomaly_rate,
            pattern_coverage=pattern_coverage,
            predictability_score=predictability_score,
            stability_score=stability_score,
        )

        # Cache result
        self.analysis_cache[cache_key] = result

        return result

    def analyze_multiple_events(
        self, event_names: list[str], lookback_hours: int = 24
    ) -> dict[str, PatternAnalysisResult]:
        """Analyze patterns for multiple ProfileEvents.

        Args:
            event_names: List of ProfileEvent names to analyze
            lookback_hours: How many hours back to analyze

        Returns:
            Dictionary mapping event names to analysis results
        """
        results = {}

        for event_name in event_names:
            try:
                result = self.analyze_patterns(event_name, lookback_hours)
                results[event_name] = result
            except Exception as e:
                logger.error(f"Error analyzing patterns for {event_name}: {e}")
                continue

        return results

    def get_anomaly_summary(self, lookback_hours: int = 24) -> dict[str, Any]:
        """Get a summary of anomalies across all ProfileEvents.

        Args:
            lookback_hours: How many hours back to analyze

        Returns:
            Summary of anomaly information
        """
        # Get top ProfileEvents by activity
        top_events = self._get_top_profile_events(lookback_hours, limit=20)

        anomaly_summary = {
            "total_events_analyzed": 0,
            "events_with_anomalies": 0,
            "total_anomalies": 0,
            "critical_anomalies": 0,
            "high_anomalies": 0,
            "anomalies_by_event": {},
            "most_anomalous_events": [],
            "recent_change_points": [],
        }

        for event_name in top_events:
            try:
                analysis = self.analyze_patterns(event_name, lookback_hours)
                anomaly_summary["total_events_analyzed"] += 1

                if analysis.anomalies:
                    anomaly_summary["events_with_anomalies"] += 1

                anomaly_summary["total_anomalies"] += len(analysis.anomalies)

                critical_count = len(
                    [a for a in analysis.anomalies if a.severity == AnomalySeverity.CRITICAL]
                )
                high_count = len(
                    [a for a in analysis.anomalies if a.severity == AnomalySeverity.HIGH]
                )

                anomaly_summary["critical_anomalies"] += critical_count
                anomaly_summary["high_anomalies"] += high_count

                anomaly_summary["anomalies_by_event"][event_name] = {
                    "total": len(analysis.anomalies),
                    "critical": critical_count,
                    "high": high_count,
                    "anomaly_rate": analysis.anomaly_rate,
                }

                # Collect recent change points
                recent_changes = [
                    cp
                    for cp in analysis.change_points
                    if datetime.now() - cp.timestamp < timedelta(hours=6)
                ]
                anomaly_summary["recent_change_points"].extend(recent_changes)

            except Exception as e:
                logger.error(f"Error getting anomaly summary for {event_name}: {e}")
                continue

        # Find most anomalous events
        events_by_anomaly_rate = sorted(
            anomaly_summary["anomalies_by_event"].items(),
            key=lambda x: x[1]["anomaly_rate"],
            reverse=True,
        )
        anomaly_summary["most_anomalous_events"] = events_by_anomaly_rate[:10]

        return anomaly_summary

    def _get_time_series_data(
        self, event_name: str, start_time: datetime, end_time: datetime
    ) -> list[TimeSeriesPoint]:
        """Get time series data for a ProfileEvent."""
        query = f"""
        SELECT
            ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event_name}')] as value,
            event_time
        FROM system.query_log
        WHERE event_time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
          AND event_time <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
          AND has(ProfileEvents.Names, '{event_name}')
          AND ProfileEvents.Values[indexOf(ProfileEvents.Names, '{event_name}')] > 0
        ORDER BY event_time
        """

        try:
            result = execute_query_with_retry(self.client, query)
            return [
                TimeSeriesPoint(timestamp=row["event_time"], value=float(row["value"]), metadata={})
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error getting time series data for {event_name}: {e}")
            return []

    def _get_related_events(self, event_name: str) -> list[str]:
        """Get ProfileEvents that are potentially related to the given event."""
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        # to identify related events based on semantic similarity, co-occurrence, etc.

        related_events_map = {
            "Query": ["SelectQuery", "InsertQuery", "SelectedRows", "SelectedBytes"],
            "SelectQuery": [
                "Query",
                "SelectedRows",
                "SelectedBytes",
                "ReadBufferFromFileDescriptorRead",
            ],
            "InsertQuery": [
                "Query",
                "InsertedRows",
                "InsertedBytes",
                "WriteBufferFromFileDescriptorWrite",
            ],
            "MemoryTrackingInBackgroundProcessingPool": [
                "BackgroundPoolTask",
                "MemoryTrackingForMerges",
            ],
            "NetworkReceiveElapsedMicroseconds": [
                "NetworkSendElapsedMicroseconds",
                "DistributedConnectionTries",
            ],
        }

        return related_events_map.get(event_name, [])

    def _get_top_profile_events(self, lookback_hours: int, limit: int = 20) -> list[str]:
        """Get the top ProfileEvents by activity volume."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        query = f"""
        SELECT
            event_name,
            count() as occurrences,
            sum(event_value) as total_value
        FROM (
            SELECT
                arrayJoin(ProfileEvents.Names) as event_name,
                arrayJoin(ProfileEvents.Values) as event_value
            FROM system.query_log
            WHERE event_time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND event_time <= '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
              AND length(ProfileEvents.Names) > 0
        )
        WHERE event_value > 0
        GROUP BY event_name
        ORDER BY total_value DESC
        LIMIT {limit}
        """

        try:
            result = execute_query_with_retry(self.client, query)
            return [row["event_name"] for row in result]
        except Exception as e:
            logger.error(f"Error getting top ProfileEvents: {e}")
            return []

    def _create_minimal_analysis_result(
        self, event_name: str, start_time: datetime, end_time: datetime
    ) -> PatternAnalysisResult:
        """Create minimal analysis result when insufficient data is available."""
        minimal_baseline = self.baseline_engine._create_default_baseline(
            event_name, start_time, end_time
        )
        minimal_trend = self.time_series_analyzer._create_minimal_trend_analysis([], event_name)

        return PatternAnalysisResult(
            event_name=event_name,
            analysis_period=(start_time, end_time),
            baseline_metrics=minimal_baseline,
            anomalies=[],
            detected_patterns=[],
            trend_analysis=minimal_trend,
            seasonal_patterns=[],
            change_points=[],
            correlations=[],
            anomaly_rate=0.0,
            pattern_coverage=0.0,
            predictability_score=0.0,
            stability_score=1.0,
        )

    def _calculate_pattern_coverage(
        self, patterns: list[PatternMatch], data: list[TimeSeriesPoint]
    ) -> float:
        """Calculate what percentage of data is explained by detected patterns."""
        if not data or not patterns:
            return 0.0

        total_duration = (data[-1].timestamp - data[0].timestamp).total_seconds()
        if total_duration <= 0:
            return 0.0

        covered_duration = 0.0
        for pattern in patterns:
            pattern_duration = pattern.duration.total_seconds()
            covered_duration += pattern_duration

        return min(1.0, covered_duration / total_duration)

    def _calculate_predictability_score(
        self, trend: TrendAnalysis, seasonal_patterns: list[SeasonalPattern]
    ) -> float:
        """Calculate how predictable the time series is based on trends and patterns."""
        score = 0.0

        # Trend contribution
        if trend.r_squared > 0.5:
            score += 0.4 * trend.r_squared

        # Seasonal pattern contribution
        if seasonal_patterns:
            max_seasonal_confidence = max(p.confidence for p in seasonal_patterns)
            score += 0.6 * max_seasonal_confidence

        return min(1.0, score)

    def _calculate_stability_score(
        self,
        anomalies: list[AnomalyScore],
        change_points: list[ChangePoint],
        data: list[TimeSeriesPoint],
    ) -> float:
        """Calculate stability score based on anomalies and change points."""
        if not data:
            return 1.0

        # Base stability
        stability = 1.0

        # Penalize for anomalies
        anomaly_penalty = len(anomalies) / len(data) * 0.5
        stability -= anomaly_penalty

        # Penalize for change points
        change_point_penalty = len(change_points) * 0.1
        stability -= change_point_penalty

        return max(0.0, stability)


# Utility functions for creating pattern analysis components
def create_pattern_analysis_engine(client: Client) -> PatternAnalysisEngine:
    """Create and return a configured pattern analysis engine.

    Args:
        client: ClickHouse client instance

    Returns:
        Configured PatternAnalysisEngine instance
    """
    return PatternAnalysisEngine(client)


def create_time_series_analyzer(
    window_size: int = 100, seasonal_periods: list[int] = None
) -> TimeSeriesAnalyzer:
    """Create and return a configured time series analyzer.

    Args:
        window_size: Size of the rolling window for analysis
        seasonal_periods: List of potential seasonal periods to detect

    Returns:
        Configured TimeSeriesAnalyzer instance
    """
    return TimeSeriesAnalyzer(window_size, seasonal_periods)


def create_anomaly_detection_engine(
    client: Client, lookback_days: int = 30
) -> AnomalyDetectionEngine:
    """Create and return a configured anomaly detection engine.

    Args:
        client: ClickHouse client instance
        lookback_days: Number of days to look back for baseline calculation

    Returns:
        Configured AnomalyDetectionEngine instance
    """
    baseline_engine = PerformanceBaselineEngine(client, lookback_days)
    return AnomalyDetectionEngine(baseline_engine)
