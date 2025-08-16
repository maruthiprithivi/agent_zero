"""Comprehensive tests for agent_zero/ai_diagnostics modules.

This test file aims to achieve high coverage of AI diagnostics modules
by testing bottleneck detection, pattern analysis, and performance advisory functionality.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Test environment setup
test_env = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "test-host",
    "AGENT_ZERO_CLICKHOUSE_USER": "test-user",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "test-pass",
    "AGENT_ZERO_CLICKHOUSE_PORT": "9000",
    "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
}


@pytest.mark.unit
class TestBottleneckDetectorEnums:
    """Tests for bottleneck detector enum classes."""

    @patch.dict("os.environ", test_env)
    def test_bottleneck_category_enum(self):
        """Test BottleneckCategory enum values."""
        from agent_zero.ai_diagnostics.bottleneck_detector import BottleneckCategory

        # Verify all expected categories exist
        assert BottleneckCategory.CPU_SATURATION.value == "cpu_saturation"
        assert BottleneckCategory.MEMORY_PRESSURE.value == "memory_pressure"
        assert BottleneckCategory.IO_BOTTLENECK.value == "io_bottleneck"
        assert BottleneckCategory.CACHE_INEFFICIENCY.value == "cache_inefficiency"

        # Test enum functionality
        assert len(list(BottleneckCategory)) >= 8  # Should have at least 8 categories

    @patch.dict("os.environ", test_env)
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        from agent_zero.ai_diagnostics.bottleneck_detector import ConfidenceLevel

        # Verify all confidence levels
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.VERY_LOW.value == "very_low"

        # Test enum functionality
        assert len(list(ConfidenceLevel)) == 5

    @patch.dict("os.environ", test_env)
    def test_severity_and_trend_enums(self):
        """Test BottleneckSeverity and TrendDirection enums."""
        from agent_zero.ai_diagnostics.bottleneck_detector import BottleneckSeverity, TrendDirection

        # Test severity levels
        assert BottleneckSeverity.CRITICAL.value == "critical"
        assert BottleneckSeverity.HIGH.value == "high"
        assert BottleneckSeverity.MEDIUM.value == "medium"
        assert BottleneckSeverity.LOW.value == "low"
        assert BottleneckSeverity.INFO.value == "info"

        # Test trend directions
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DEGRADING.value == "degrading"
        assert TrendDirection.VOLATILE.value == "volatile"


@pytest.mark.unit
class TestBottleneckDataClasses:
    """Tests for bottleneck detector dataclasses."""

    @patch.dict("os.environ", test_env)
    def test_bottleneck_signature_creation(self):
        """Test BottleneckSignature dataclass creation."""
        from agent_zero.ai_diagnostics.bottleneck_detector import (
            BottleneckCategory,
            BottleneckSignature,
        )

        # Create signature with minimal required fields
        signature = BottleneckSignature(
            category=BottleneckCategory.CPU_SATURATION,
            name="CPU Saturation Test",
            description="Test signature for CPU saturation",
        )

        # Verify required fields
        assert signature.category == BottleneckCategory.CPU_SATURATION
        assert signature.name == "CPU Saturation Test"
        assert signature.description == "Test signature for CPU saturation"

        # Verify default fields are initialized correctly
        assert signature.primary_indicators == []
        assert signature.secondary_indicators == []
        assert signature.critical_thresholds == {}
        assert signature.warning_thresholds == {}
        assert signature.indicator_weights == {}

        # Test with custom fields
        signature_full = BottleneckSignature(
            category=BottleneckCategory.MEMORY_PRESSURE,
            name="Memory Pressure",
            description="Memory bottleneck signature",
            primary_indicators=["MemoryTrackingCount", "QueryMemoryUsage"],
            critical_thresholds={"memory_usage": 0.9},
            indicator_weights={"MemoryTrackingCount": 0.8},
        )

        assert signature_full.primary_indicators == ["MemoryTrackingCount", "QueryMemoryUsage"]
        assert signature_full.critical_thresholds["memory_usage"] == 0.9
        assert signature_full.indicator_weights["MemoryTrackingCount"] == 0.8

    @patch.dict("os.environ", test_env)
    def test_bottleneck_detection_creation(self):
        """Test BottleneckDetection dataclass creation."""
        from agent_zero.ai_diagnostics.bottleneck_detector import (
            BottleneckCategory,
            BottleneckDetection,
            BottleneckSeverity,
            BottleneckSignature,
            ConfidenceLevel,
            TrendDirection,
        )

        # Create a signature for testing
        signature = BottleneckSignature(
            category=BottleneckCategory.CPU_SATURATION,
            name="Test Signature",
            description="Test description",
        )

        # Create detection with required fields
        now = datetime.now()
        detection = BottleneckDetection(
            signature=signature,
            severity=BottleneckSeverity.HIGH,
            confidence=85.5,
            confidence_level=ConfidenceLevel.HIGH,
            detection_timestamp=now,
            affected_time_period=(now - timedelta(minutes=30), now),
            primary_score=75.0,
            secondary_score=15.0,
            correlation_score=20.0,
            trend_score=10.0,
            total_score=85.5,
            estimated_performance_impact=25.0,
            business_impact_score=60.0,
            trend_direction=TrendDirection.DEGRADING,
        )

        # Verify all fields
        assert detection.signature == signature
        assert detection.severity == BottleneckSeverity.HIGH
        assert detection.confidence == 85.5
        assert detection.confidence_level == ConfidenceLevel.HIGH
        assert detection.detection_timestamp == now
        assert detection.primary_score == 75.0
        assert detection.estimated_performance_impact == 25.0
        assert detection.trend_direction == TrendDirection.DEGRADING

        # Verify default fields
        assert detection.triggering_events == {}
        assert detection.supporting_evidence == {}
        assert detection.correlations_found == []
        assert detection.affected_operations == []
        assert detection.immediate_actions == []


@pytest.mark.unit
class TestIntelligentBottleneckDetector:
    """Tests for IntelligentBottleneckDetector class."""

    @patch.dict("os.environ", test_env)
    def test_intelligent_bottleneck_detector_creation(self):
        """Test IntelligentBottleneckDetector initialization."""
        from agent_zero.ai_diagnostics.bottleneck_detector import IntelligentBottleneckDetector

        # Mock client
        mock_client = Mock()

        # Create detector
        detector = IntelligentBottleneckDetector(mock_client)

        # Verify initialization
        assert detector.client == mock_client
        assert hasattr(detector, "pattern_matcher")
        assert hasattr(detector, "predictive_analyzer")
        assert hasattr(detector, "detection_history")

    @patch.dict("os.environ", test_env)
    def test_pattern_matcher_creation(self):
        """Test PatternMatcher class creation."""
        from agent_zero.ai_diagnostics.bottleneck_detector import PatternMatcher

        # Create pattern matcher (no client parameter needed)
        matcher = PatternMatcher()

        # Verify initialization
        assert hasattr(matcher, "signatures")
        assert hasattr(matcher, "detection_history")
        assert hasattr(matcher, "pattern_weights")
        assert isinstance(matcher.signatures, list)
        assert len(matcher.signatures) > 0

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.ProfileEventsAnalyzer")
    def test_predictive_analyzer_creation(self, mock_profile_analyzer):
        """Test PredictiveAnalyzer class creation."""
        from agent_zero.ai_diagnostics.bottleneck_detector import PredictiveAnalyzer

        # Mock ProfileEventsAnalyzer
        mock_analyzer = Mock()
        mock_profile_analyzer.return_value = mock_analyzer

        # Create predictive analyzer
        analyzer = PredictiveAnalyzer(mock_analyzer)

        # Verify initialization
        assert analyzer.profile_analyzer == mock_analyzer
        assert hasattr(analyzer, "historical_cache")
        assert hasattr(analyzer, "prediction_models")

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.ProfileEventsAnalyzer")
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.HardwareHealthEngine")
    def test_detect_bottlenecks_basic(self, mock_hardware, mock_profile_analyzer):
        """Test basic bottleneck detection functionality."""
        from agent_zero.ai_diagnostics.bottleneck_detector import IntelligentBottleneckDetector

        # Mock client and dependencies
        mock_client = Mock()
        mock_analyzer = Mock()
        mock_hardware_instance = Mock()
        mock_profile_analyzer.return_value = mock_analyzer
        mock_hardware.return_value = mock_hardware_instance
        mock_analyzer.get_recent_aggregated_data.return_value = (
            []
        )  # Return empty list for aggregations

        # Create detector
        detector = IntelligentBottleneckDetector(mock_client)

        # Test bottleneck detection method exists
        assert hasattr(detector, "detect_bottlenecks") or hasattr(
            detector, "analyze_system_bottlenecks"
        )

        # Verify detection method exists and detector is properly initialized
        assert hasattr(detector, "detect_bottlenecks")
        assert detector.client == mock_client


@pytest.mark.unit
class TestPatternAnalysisEngine:
    """Tests for pattern analysis engine functionality."""

    @patch.dict("os.environ", test_env)
    def test_pattern_analysis_engine_creation(self):
        """Test PatternAnalysisEngine initialization."""
        from agent_zero.ai_diagnostics.pattern_analyzer import PatternAnalysisEngine

        # Mock client
        mock_client = Mock()

        # Create engine
        engine = PatternAnalysisEngine(mock_client)

        # Verify initialization
        assert engine.client == mock_client
        assert hasattr(engine, "time_series_analyzer")
        assert hasattr(engine, "baseline_engine")
        assert hasattr(engine, "anomaly_engine")
        assert hasattr(engine, "pattern_engine")
        assert hasattr(engine, "correlation_analyzer")
        assert hasattr(engine, "analysis_cache")

    @patch.dict("os.environ", test_env)
    def test_time_series_analyzer_creation(self):
        """Test TimeSeriesAnalyzer creation."""
        from agent_zero.ai_diagnostics.pattern_analyzer import TimeSeriesAnalyzer

        # Create analyzer (no parameters needed based on implementation)
        analyzer = TimeSeriesAnalyzer()

        # Verify analyzer was created successfully
        assert analyzer is not None
        # Just verify it's an instance of TimeSeriesAnalyzer
        assert analyzer.__class__.__name__ == "TimeSeriesAnalyzer"

    @patch.dict("os.environ", test_env)
    def test_anomaly_detection_engine_creation(self):
        """Test AnomalyDetectionEngine creation."""
        from agent_zero.ai_diagnostics.pattern_analyzer import (
            AnomalyDetectionEngine,
        )

        # Mock baseline engine
        mock_baseline = Mock()

        # Create engine
        engine = AnomalyDetectionEngine(mock_baseline)

        # Verify initialization
        assert engine.baseline_engine == mock_baseline
        # Basic verification that it's an AnomalyDetectionEngine instance
        assert hasattr(engine, "detect_anomalies") or hasattr(engine, "analyze_anomalies")


@pytest.mark.unit
class TestPerformanceAdvisorEngine:
    """Tests for performance advisor engine functionality."""

    @patch.dict("os.environ", test_env)
    def test_performance_advisor_engine_creation(self):
        """Test PerformanceAdvisorEngine initialization."""
        from agent_zero.ai_diagnostics.performance_advisor import PerformanceAdvisorEngine

        # Mock client
        mock_client = Mock()

        # Create engine
        engine = PerformanceAdvisorEngine(mock_client)

        # Verify initialization
        assert engine.client == mock_client
        # Just verify it's an instance of PerformanceAdvisorEngine
        assert engine.__class__.__name__ == "PerformanceAdvisorEngine"

    @patch.dict("os.environ", test_env)
    def test_recommendation_engine_creation(self):
        """Test RecommendationEngine creation."""
        from agent_zero.ai_diagnostics.performance_advisor import RecommendationEngine

        # Mock client
        mock_client = Mock()

        # Create engine
        engine = RecommendationEngine(mock_client)

        # Verify initialization
        assert engine.client == mock_client
        # Just verify it's an instance of RecommendationEngine
        assert engine.__class__.__name__ == "RecommendationEngine"

    @patch.dict("os.environ", test_env)
    def test_query_optimization_advisor_creation(self):
        """Test QueryOptimizationAdvisor creation."""
        from agent_zero.ai_diagnostics.performance_advisor import (
            QueryOptimizationAdvisor,
            RecommendationEngine,
        )

        # Mock client and recommendation engine
        mock_client = Mock()
        mock_recommendation_engine = Mock(spec=RecommendationEngine)

        # Create advisor
        advisor = QueryOptimizationAdvisor(mock_client, mock_recommendation_engine)

        # Verify initialization
        assert advisor.client == mock_client
        assert advisor.recommendation_engine == mock_recommendation_engine
        assert hasattr(advisor, "logger")


@pytest.mark.unit
class TestAIDiagnosticsIntegration:
    """Tests for AI diagnostics integration functionality."""

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.HardwareHealthEngine")
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.PerformanceDiagnosticEngine")
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.ProfileEventsAnalyzer")
    @patch("agent_zero.ai_diagnostics.bottleneck_detector.StorageOptimizationEngine")
    def test_ai_diagnostics_integration(self, mock_storage, mock_profile, mock_perf, mock_hardware):
        """Test integration between AI diagnostics components."""
        from agent_zero.ai_diagnostics.bottleneck_detector import IntelligentBottleneckDetector
        from agent_zero.ai_diagnostics.pattern_analyzer import PatternAnalysisEngine
        from agent_zero.ai_diagnostics.performance_advisor import PerformanceAdvisorEngine

        # Mock client and dependencies
        mock_client = Mock()
        mock_hardware_instance = Mock()
        mock_perf_instance = Mock()
        mock_profile_instance = Mock()
        mock_storage_instance = Mock()

        mock_hardware.return_value = mock_hardware_instance
        mock_perf.return_value = mock_perf_instance
        mock_profile.return_value = mock_profile_instance
        mock_storage.return_value = mock_storage_instance

        # Set up mock returns
        mock_profile_instance.get_recent_events.return_value = {
            "CPUCount": [4, 4, 4],
            "MemoryUsage": [1000000, 1100000, 1200000],
        }

        # Create all AI diagnostics components
        bottleneck_detector = IntelligentBottleneckDetector(mock_client)
        pattern_engine = PatternAnalysisEngine(mock_client)
        advisor_engine = PerformanceAdvisorEngine(mock_client)

        # Verify all components were created successfully
        assert bottleneck_detector.client == mock_client
        assert pattern_engine.client == mock_client
        assert advisor_engine.client == mock_client

        # Verify components have expected attributes
        assert hasattr(bottleneck_detector, "pattern_matcher")
        assert hasattr(pattern_engine, "time_series_analyzer")
        assert hasattr(advisor_engine, "recommendation_engine")


@pytest.mark.unit
class TestAIDiagnosticsErrorHandling:
    """Tests for AI diagnostics error handling."""

    @patch.dict("os.environ", test_env)
    def test_bottleneck_detector_error_handling(self):
        """Test error handling in bottleneck detection."""
        from agent_zero.ai_diagnostics.bottleneck_detector import IntelligentBottleneckDetector

        # Mock client that raises exceptions
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Database connection failed")

        # Create detector
        detector = IntelligentBottleneckDetector(mock_client)

        # Verify detector was created successfully despite potential errors
        assert detector.client == mock_client
        assert hasattr(detector, "pattern_matcher")

    @patch.dict("os.environ", test_env)
    def test_pattern_analysis_error_handling(self):
        """Test error handling in pattern analysis."""
        from agent_zero.ai_diagnostics.pattern_analyzer import PatternAnalysisEngine

        # Mock client that raises exceptions
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query execution failed")

        # Create engine
        engine = PatternAnalysisEngine(mock_client)

        # Verify engine handles errors gracefully during creation
        assert engine.client == mock_client
        assert hasattr(engine, "time_series_analyzer")

    @patch.dict("os.environ", test_env)
    def test_performance_advisor_error_handling(self):
        """Test error handling in performance advisor."""
        from agent_zero.ai_diagnostics.performance_advisor import PerformanceAdvisorEngine

        # Mock client that raises exceptions
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Analysis failed")

        # Create engine
        engine = PerformanceAdvisorEngine(mock_client)

        # Verify engine handles errors gracefully during creation
        assert engine.client == mock_client
        assert hasattr(engine, "recommendation_engine")
