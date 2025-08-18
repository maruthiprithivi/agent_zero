"""Tests for the performance diagnostics module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent_zero.monitoring.performance_diagnostics import (
    CacheAnalyzer,
    IOPerformanceAnalyzer,
    PerformanceBottleneck,
    PerformanceBottleneckType,
    PerformanceDiagnosticEngine,
    PerformanceSeverity,
    QueryExecutionAnalyzer,
)
from agent_zero.monitoring.profile_events_core import (
    ProfileEventAggregation,
    ProfileEventsAnalyzer,
    ProfileEventsCategory,
)


@pytest.fixture
def mock_client():
    """Create a mock ClickHouse client."""
    return MagicMock()


@pytest.fixture
def mock_profile_analyzer(mock_client):
    """Create a mock ProfileEventsAnalyzer."""
    analyzer = MagicMock(spec=ProfileEventsAnalyzer)
    analyzer.client = mock_client
    return analyzer


@pytest.fixture
def sample_aggregation():
    """Create a sample ProfileEventAggregation for testing."""
    return ProfileEventAggregation(
        event_name="TestEvent",
        category=ProfileEventsCategory.QUERY_EXECUTION,
        count=100,
        sum_value=1000.0,
        min_value=5.0,
        max_value=50.0,
        avg_value=10.0,
        p50_value=8.0,
        p90_value=20.0,
        p99_value=45.0,
        stddev_value=5.0,
        time_range_start=datetime(2024, 1, 1, 10, 0, 0),
        time_range_end=datetime(2024, 1, 1, 11, 0, 0),
        sample_queries=["query1", "query2"],
    )


class TestQueryExecutionAnalyzer:
    """Test the QueryExecutionAnalyzer class."""

    def test_init(self, mock_profile_analyzer):
        """Test QueryExecutionAnalyzer initialization."""
        analyzer = QueryExecutionAnalyzer(mock_profile_analyzer)
        assert analyzer.analyzer == mock_profile_analyzer

    def test_analyze_function_performance(self, mock_profile_analyzer, sample_aggregation):
        """Test function performance analysis."""
        # Setup mock data
        function_agg = ProfileEventAggregation(
            event_name="FunctionExecute",
            category=ProfileEventsCategory.QUERY_EXECUTION,
            count=50,
            sum_value=500.0,
            min_value=2.0,
            max_value=25.0,
            avg_value=10.0,
            p50_value=8.0,
            p90_value=18.0,
            p99_value=23.0,
            stddev_value=3.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        mock_profile_analyzer.aggregate_profile_events.return_value = [function_agg]

        analyzer = QueryExecutionAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_function_performance(start_time, end_time)

        assert "FunctionExecute" in result
        assert result["FunctionExecute"]["type"] == "function_execution"
        assert result["FunctionExecute"]["total_executions"] == 500.0
        assert result["FunctionExecute"]["avg_execution_time"] == 10.0
        assert "efficiency_score" in result["FunctionExecute"]
        assert "performance_impact" in result["FunctionExecute"]

    def test_analyze_null_handling_efficiency(self, mock_profile_analyzer):
        """Test NULL handling efficiency analysis."""
        # Setup mock data with no NULL operations
        mock_profile_analyzer.aggregate_profile_events.return_value = []

        analyzer = QueryExecutionAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_null_handling_efficiency(start_time, end_time)

        assert result["status"] == "optimal"
        assert result["total_null_operations"] == 0
        assert result["impact"] == "none"
        assert result["recommendations"] == []

    def test_analyze_memory_allocation_patterns(self, mock_profile_analyzer):
        """Test memory allocation pattern analysis."""
        # Setup mock data
        chunks_agg = ProfileEventAggregation(
            event_name="ArenaAllocChunks",
            category=ProfileEventsCategory.MEMORY_ALLOCATION,
            count=10,
            sum_value=100.0,
            min_value=5.0,
            max_value=15.0,
            avg_value=10.0,
            p50_value=10.0,
            p90_value=14.0,
            p99_value=15.0,
            stddev_value=2.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        bytes_agg = ProfileEventAggregation(
            event_name="ArenaAllocBytes",
            category=ProfileEventsCategory.MEMORY_ALLOCATION,
            count=10,
            sum_value=1024000.0,  # 1MB total
            min_value=50000.0,
            max_value=150000.0,
            avg_value=102400.0,
            p50_value=100000.0,
            p90_value=140000.0,
            p99_value=150000.0,
            stddev_value=20000.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        mock_profile_analyzer.aggregate_profile_events.return_value = [chunks_agg, bytes_agg]

        analyzer = QueryExecutionAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_memory_allocation_patterns(start_time, end_time)

        assert "arena_allocation" in result
        assert result["arena_allocation"]["total_chunks"] == 100.0
        assert result["arena_allocation"]["total_bytes"] == 1024000.0
        assert result["arena_allocation"]["avg_chunk_size"] == 10240.0  # 1MB / 100 chunks
        assert result["arena_allocation"]["efficiency"] in ["good", "fair", "poor"]

    def test_analyze_primary_key_usage_no_data(self, mock_profile_analyzer):
        """Test primary key usage analysis with no data."""
        mock_profile_analyzer.aggregate_profile_events.return_value = []

        analyzer = QueryExecutionAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_primary_key_usage(start_time, end_time)

        assert result["status"] == "no_data"
        assert "Enable primary key usage tracking" in result["recommendations"][0]


class TestIOPerformanceAnalyzer:
    """Test the IOPerformanceAnalyzer class."""

    def test_init(self, mock_profile_analyzer):
        """Test IOPerformanceAnalyzer initialization."""
        analyzer = IOPerformanceAnalyzer(mock_profile_analyzer)
        assert analyzer.analyzer == mock_profile_analyzer

    def test_analyze_file_operations(self, mock_profile_analyzer):
        """Test file operations analysis."""
        # Setup mock data
        file_open_agg = ProfileEventAggregation(
            event_name="FileOpen",
            category=ProfileEventsCategory.FILE_IO,
            count=20,
            sum_value=100.0,
            min_value=2.0,
            max_value=8.0,
            avg_value=5.0,
            p50_value=5.0,
            p90_value=7.0,
            p99_value=8.0,
            stddev_value=1.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        mock_profile_analyzer.aggregate_profile_events.return_value = [file_open_agg]

        analyzer = IOPerformanceAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_file_operations(start_time, end_time)

        assert "file_opens" in result
        assert result["file_opens"]["total_opens"] == 100.0
        assert result["file_opens"]["avg_per_query"] == 5.0
        assert result["file_opens"]["impact"] == "low"  # < 10 average

    def test_analyze_network_performance(self, mock_profile_analyzer):
        """Test network performance analysis."""
        # Setup mock data
        recv_time_agg = ProfileEventAggregation(
            event_name="NetworkReceiveElapsedMicroseconds",
            category=ProfileEventsCategory.NETWORK_IO,
            count=10,
            sum_value=1000000.0,  # 1 second total
            min_value=50000.0,
            max_value=150000.0,
            avg_value=100000.0,
            p50_value=100000.0,
            p90_value=140000.0,
            p99_value=150000.0,
            stddev_value=20000.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        recv_bytes_agg = ProfileEventAggregation(
            event_name="NetworkReceiveBytes",
            category=ProfileEventsCategory.NETWORK_IO,
            count=10,
            sum_value=104857600.0,  # 100MB
            min_value=5242880.0,
            max_value=15728640.0,
            avg_value=10485760.0,
            p50_value=10485760.0,
            p90_value=14680064.0,
            p99_value=15728640.0,
            stddev_value=2097152.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        mock_profile_analyzer.aggregate_profile_events.return_value = [
            recv_time_agg,
            recv_bytes_agg,
        ]

        analyzer = IOPerformanceAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_network_performance(start_time, end_time)

        assert "receive_performance" in result
        assert result["receive_performance"]["total_receive_time"] == 1000000.0
        assert result["receive_performance"]["total_bytes_received"] == 104857600.0
        assert "avg_throughput_mbps" in result["receive_performance"]
        assert "performance" in result["receive_performance"]


class TestCacheAnalyzer:
    """Test the CacheAnalyzer class."""

    def test_init(self, mock_profile_analyzer):
        """Test CacheAnalyzer initialization."""
        analyzer = CacheAnalyzer(mock_profile_analyzer)
        assert analyzer.analyzer == mock_profile_analyzer

    def test_analyze_mark_cache(self, mock_profile_analyzer):
        """Test mark cache analysis."""
        # Setup mock data
        hits_agg = ProfileEventAggregation(
            event_name="MarkCacheHits",
            category=ProfileEventsCategory.MARK_CACHE,
            count=50,
            sum_value=900.0,
            min_value=10.0,
            max_value=30.0,
            avg_value=18.0,
            p50_value=18.0,
            p90_value=28.0,
            p99_value=30.0,
            stddev_value=5.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        misses_agg = ProfileEventAggregation(
            event_name="MarkCacheMisses",
            category=ProfileEventsCategory.MARK_CACHE,
            count=50,
            sum_value=100.0,
            min_value=1.0,
            max_value=5.0,
            avg_value=2.0,
            p50_value=2.0,
            p90_value=4.0,
            p99_value=5.0,
            stddev_value=1.0,
            time_range_start=datetime(2024, 1, 1, 10, 0, 0),
            time_range_end=datetime(2024, 1, 1, 11, 0, 0),
            sample_queries=[],
        )

        mock_profile_analyzer.aggregate_profile_events.return_value = [hits_agg, misses_agg]

        analyzer = CacheAnalyzer(mock_profile_analyzer)
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = analyzer.analyze_mark_cache(start_time, end_time)

        assert result["efficiency"] == "good"  # 90% hit rate should be "good" based on thresholds
        assert result["hit_rate"] == 90.0
        assert result["total_requests"] == 1000.0
        assert result["hits"] == 900.0
        assert result["misses"] == 100.0

    def test_calculate_overall_cache_score(self, mock_profile_analyzer):
        """Test overall cache score calculation."""
        mark_cache_analysis = {"hit_rate": 90.0}
        uncompressed_cache_analysis = {"hit_rate": 80.0}
        query_cache_analysis = {"hit_rate": 50.0}

        analyzer = CacheAnalyzer(mock_profile_analyzer)

        score = analyzer.calculate_overall_cache_score(
            mark_cache_analysis, uncompressed_cache_analysis, query_cache_analysis
        )

        # Should be weighted average: 90*0.5 + 80*0.3 + 50*0.2 = 45 + 24 + 10 = 79
        assert score == 79.0


class TestPerformanceDiagnosticEngine:
    """Test the PerformanceDiagnosticEngine class."""

    def test_init(self, mock_client):
        """Test PerformanceDiagnosticEngine initialization."""
        engine = PerformanceDiagnosticEngine(mock_client)
        assert engine.client == mock_client
        assert engine.profile_analyzer is not None
        assert engine.query_analyzer is not None
        assert engine.io_analyzer is not None
        assert engine.cache_analyzer is not None

    @patch("agent_zero.monitoring.performance_diagnostics.QueryExecutionAnalyzer")
    @patch("agent_zero.monitoring.performance_diagnostics.IOPerformanceAnalyzer")
    @patch("agent_zero.monitoring.performance_diagnostics.CacheAnalyzer")
    def test_generate_comprehensive_report(
        self,
        mock_cache_analyzer_class,
        mock_io_analyzer_class,
        mock_query_analyzer_class,
        mock_client,
    ):
        """Test comprehensive report generation."""
        # Setup mock analyzers
        mock_query_analyzer = MagicMock()
        mock_query_analyzer.analyze_function_performance.return_value = {}
        mock_query_analyzer.analyze_null_handling_efficiency.return_value = {"status": "optimal"}
        mock_query_analyzer.analyze_memory_allocation_patterns.return_value = {}
        mock_query_analyzer.analyze_primary_key_usage.return_value = {"status": "good"}

        mock_io_analyzer = MagicMock()
        mock_io_analyzer.analyze_file_operations.return_value = {}
        mock_io_analyzer.analyze_network_performance.return_value = {}
        mock_io_analyzer.analyze_disk_performance.return_value = {}

        mock_cache_analyzer = MagicMock()
        mock_cache_analyzer.analyze_mark_cache.return_value = {
            "efficiency": "good",
            "hit_rate": 85.0,
        }
        mock_cache_analyzer.analyze_uncompressed_cache.return_value = {
            "efficiency": "good",
            "hit_rate": 80.0,
        }
        mock_cache_analyzer.analyze_query_cache.return_value = {
            "efficiency": "fair",
            "hit_rate": 40.0,
        }
        mock_cache_analyzer.calculate_overall_cache_score.return_value = 75.0

        mock_query_analyzer_class.return_value = mock_query_analyzer
        mock_io_analyzer_class.return_value = mock_io_analyzer
        mock_cache_analyzer_class.return_value = mock_cache_analyzer

        engine = PerformanceDiagnosticEngine(mock_client)

        # Generate report
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        report = engine.generate_comprehensive_report(start_time, end_time)

        assert report.analysis_period_start == start_time
        assert report.analysis_period_end == end_time
        assert report.query_execution_analysis is not None
        assert report.io_performance_analysis is not None
        assert report.cache_analysis is not None
        assert 0 <= report.overall_performance_score <= 100
        assert isinstance(report.critical_bottlenecks, list)
        assert isinstance(report.top_recommendations, list)


class TestPerformanceBottleneck:
    """Test the PerformanceBottleneck class."""

    def test_performance_bottleneck_creation(self):
        """Test PerformanceBottleneck creation."""
        bottleneck = PerformanceBottleneck(
            type=PerformanceBottleneckType.CPU_BOUND,
            severity=PerformanceSeverity.HIGH,
            description="High CPU usage detected",
            impact_score=85.0,
            affected_events=["CPUEvent1", "CPUEvent2"],
            recommendations=["Optimize CPU-intensive queries"],
            metrics={"cpu_usage": 95.0},
        )

        assert bottleneck.type == PerformanceBottleneckType.CPU_BOUND
        assert bottleneck.severity == PerformanceSeverity.HIGH
        assert bottleneck.description == "High CPU usage detected"
        assert bottleneck.impact_score == 85.0
        assert bottleneck.affected_events == ["CPUEvent1", "CPUEvent2"]
        assert bottleneck.recommendations == ["Optimize CPU-intensive queries"]
        assert bottleneck.metrics == {"cpu_usage": 95.0}


def test_performance_bottleneck_types():
    """Test that all performance bottleneck types are defined."""
    expected_types = [
        "CPU_BOUND",
        "IO_BOUND",
        "MEMORY_BOUND",
        "CACHE_MISS",
        "NETWORK_BOUND",
        "DISK_BOUND",
        "FUNCTION_OVERHEAD",
        "QUERY_COMPLEXITY",
        "LOCK_CONTENTION",
        "UNKNOWN",
    ]

    actual_types = [member.name for member in PerformanceBottleneckType]

    for expected_type in expected_types:
        assert expected_type in actual_types


def test_performance_severity_levels():
    """Test that all performance severity levels are defined."""
    expected_severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    actual_severities = [member.name for member in PerformanceSeverity]

    for expected_severity in expected_severities:
        assert expected_severity in actual_severities
