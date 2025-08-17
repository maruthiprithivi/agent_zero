"""Comprehensive tests for storage_cloud_diagnostics module.

This module contains tests for all storage diagnostic components including S3 storage analysis,
Azure blob storage analysis, compression analysis, and the unified storage optimization engine.
Coverage focus on cloud storage performance, cost optimization, and data integrity monitoring.
"""

from unittest.mock import Mock, patch

import pytest

# Import storage diagnostics classes
from agent_zero.monitoring.storage_cloud_diagnostics import (
    StorageIssue,
    StoragePerformanceIssue,
    StorageSeverity,
)

# Test environment configuration
test_env = {
    "CLICKHOUSE_HOST": "test_host",
    "CLICKHOUSE_PORT": "8123",
    "CLICKHOUSE_USER": "test_user",
    "CLICKHOUSE_PASSWORD": "test_password",
    "CLICKHOUSE_DATABASE": "test_db",
    "MCP_TOOLS_ENABLED": "true",
}


@pytest.mark.unit
class TestStorageEnums:
    """Test storage diagnostics enums."""

    def test_storage_performance_issue_enum(self):
        """Test StoragePerformanceIssue enum values."""
        from agent_zero.monitoring.storage_cloud_diagnostics import StoragePerformanceIssue

        # Test enum values exist
        assert StoragePerformanceIssue.HIGH_LATENCY
        assert StoragePerformanceIssue.THROTTLING
        assert StoragePerformanceIssue.HIGH_ERROR_RATE
        assert StoragePerformanceIssue.INEFFICIENT_COMPRESSION
        assert StoragePerformanceIssue.DATA_INTEGRITY_ISSUES
        assert StoragePerformanceIssue.SUBOPTIMAL_REQUEST_PATTERNS
        assert StoragePerformanceIssue.COST_INEFFICIENCY
        assert StoragePerformanceIssue.REGIONAL_LATENCY
        assert StoragePerformanceIssue.CONNECTION_POOLING_ISSUES
        assert StoragePerformanceIssue.MULTIPART_UPLOAD_INEFFICIENCY

    def test_storage_severity_enum(self):
        """Test StorageSeverity enum values."""
        from agent_zero.monitoring.storage_cloud_diagnostics import StorageSeverity

        # Test enum values exist
        assert StorageSeverity.CRITICAL
        assert StorageSeverity.HIGH
        assert StorageSeverity.MEDIUM
        assert StorageSeverity.LOW
        assert StorageSeverity.INFO

    def test_storage_tier_enum(self):
        """Test StorageTier enum values."""
        from agent_zero.monitoring.storage_cloud_diagnostics import StorageTier

        # Test enum values exist
        assert StorageTier.HOT
        assert StorageTier.WARM
        assert StorageTier.COLD
        assert StorageTier.ARCHIVE


@pytest.mark.unit
class TestStorageDataClasses:
    """Test storage diagnostics data classes."""

    def test_storage_issue_dataclass(self):
        """Test StorageIssue dataclass."""
        from agent_zero.monitoring.storage_cloud_diagnostics import (
            StorageIssue,
            StoragePerformanceIssue,
            StorageSeverity,
        )

        issue = StorageIssue(
            type=StoragePerformanceIssue.HIGH_LATENCY,
            severity=StorageSeverity.HIGH,
            description="High S3 read latency detected",
            impact_score=75.0,
            affected_operations=["read_operations", "get_object"],
            recommendations=["Use S3 Transfer Acceleration", "Check network connectivity"],
            metrics={"average_latency_ms": 1500.0, "requests_per_second": 100},
            cost_impact={"estimated_monthly_cost": 500.0, "potential_savings": 150.0},
        )

        assert issue.type == StoragePerformanceIssue.HIGH_LATENCY
        assert issue.severity == StorageSeverity.HIGH
        assert issue.impact_score == 75.0
        assert len(issue.affected_operations) == 2
        assert len(issue.recommendations) == 2
        assert "average_latency_ms" in issue.metrics
        assert "estimated_monthly_cost" in issue.cost_impact

    def test_s3_storage_analysis_dataclass(self):
        """Test S3StorageAnalysis dataclass."""
        from agent_zero.monitoring.storage_cloud_diagnostics import (
            S3StorageAnalysis,
            StorageIssue,
            StoragePerformanceIssue,
            StorageSeverity,
        )

        issue = StorageIssue(
            type=StoragePerformanceIssue.THROTTLING,
            severity=StorageSeverity.MEDIUM,
            description="S3 throttling detected",
            impact_score=60.0,
        )

        analysis = S3StorageAnalysis(
            operation_performance={"read_operations": {"average_latency_ms": 250.0}},
            latency_analysis={"read_latency": {"average_ms": 250.0, "is_high": False}},
            error_analysis={"overall_error_rate": 0.5},
            throttling_analysis={"mitigation_needed": True},
            cost_analysis={"estimated_monthly_costs": {"total": 1500.0}},
            regional_performance={"us-east-1": {"latency": 100.0}},
            multipart_upload_analysis={"completion_rate_percent": 95.0},
            connection_efficiency={"pool_efficiency": {"reuse_rate": 85.0}},
            issues=[issue],
            recommendations=["Implement exponential backoff", "Monitor request patterns"],
        )

        assert "read_operations" in analysis.operation_performance
        assert analysis.latency_analysis["read_latency"]["average_ms"] == 250.0
        assert analysis.error_analysis["overall_error_rate"] == 0.5
        assert analysis.throttling_analysis["mitigation_needed"] is True
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 2

    def test_azure_storage_analysis_dataclass(self):
        """Test AzureStorageAnalysis dataclass."""
        from agent_zero.monitoring.storage_cloud_diagnostics import (
            AzureStorageAnalysis,
            StorageIssue,
            StoragePerformanceIssue,
            StorageSeverity,
        )

        issue = StorageIssue(
            type=StoragePerformanceIssue.HIGH_LATENCY,
            severity=StorageSeverity.HIGH,
            description="High Azure blob latency",
            impact_score=80.0,
        )

        analysis = AzureStorageAnalysis(
            blob_operation_performance={"read_operations": {"throughput_mbps": 150.0}},
            latency_analysis={"read_latency": {"average_ms": 800.0}},
            error_analysis={"read_errors": {"error_rate_percent": 1.5}},
            throttling_analysis={"throttling_detected": False},
            cost_analysis={"estimated_costs": {"monthly": 800.0}},
            tier_optimization={"recommendations": ["Move to cool tier"]},
            issues=[issue],
            recommendations=["Check Azure region", "Consider Azure CDN"],
        )

        assert analysis.blob_operation_performance["read_operations"]["throughput_mbps"] == 150.0
        assert analysis.latency_analysis["read_latency"]["average_ms"] == 800.0
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 2

    def test_compression_analysis_dataclass(self):
        """Test CompressionAnalysis dataclass."""
        from agent_zero.monitoring.storage_cloud_diagnostics import (
            CompressionAnalysis,
            StorageIssue,
            StoragePerformanceIssue,
            StorageSeverity,
        )

        issue = StorageIssue(
            type=StoragePerformanceIssue.INEFFICIENT_COMPRESSION,
            severity=StorageSeverity.MEDIUM,
            description="Low compression ratio",
            impact_score=50.0,
        )

        analysis = CompressionAnalysis(
            compression_efficiency={"efficiency_score": 75.0, "compression_ratio": 2.5},
            compression_ratios={"overall_ratio": 2.5, "effectiveness": "good"},
            decompression_performance={"decompression_speed": {"avg_mbps": 500.0}},
            integrity_analysis={"integrity_score": 98.0, "checksum_failures": 2},
            codec_performance={"lz4": {"ratio": 2.0}, "zstd": {"ratio": 3.5}},
            optimization_opportunities={"cache_optimization": {"needs_optimization": True}},
            issues=[issue],
            recommendations=["Consider zstd codec", "Increase cache size"],
        )

        assert analysis.compression_efficiency["efficiency_score"] == 75.0
        assert analysis.compression_ratios["overall_ratio"] == 2.5
        assert analysis.integrity_analysis["integrity_score"] == 98.0
        assert len(analysis.issues) == 1
        assert len(analysis.recommendations) == 2

    def test_storage_optimization_report_dataclass(self):
        """Test StorageOptimizationReport dataclass."""
        from agent_zero.monitoring.storage_cloud_diagnostics import (
            StorageIssue,
            StorageOptimizationReport,
            StoragePerformanceIssue,
            StorageSeverity,
        )

        issue = StorageIssue(
            type=StoragePerformanceIssue.COST_INEFFICIENCY,
            severity=StorageSeverity.MEDIUM,
            description="High storage costs detected",
            impact_score=60.0,
        )

        report = StorageOptimizationReport(
            s3_analysis=None,  # Could be populated with S3StorageAnalysis
            azure_analysis=None,  # Could be populated with AzureStorageAnalysis
            compression_analysis=None,  # Could be populated with CompressionAnalysis
            overall_score=78.5,
            priority_issues=[issue],
            cost_savings_opportunities=[
                {"type": "tier_optimization", "potential_savings_percent": 25}
            ],
            performance_improvements=[{"type": "latency_reduction", "impact_score": 80.0}],
            recommendations=["Implement lifecycle policies", "Optimize request patterns"],
        )

        assert report.overall_score == 78.5
        assert len(report.priority_issues) == 1
        assert len(report.cost_savings_opportunities) == 1
        assert len(report.performance_improvements) == 1
        assert len(report.recommendations) == 2


@pytest.mark.unit
class TestS3StorageAnalyzer:
    """Test S3StorageAnalyzer class functionality."""

    @patch.dict("os.environ", test_env)
    def test_s3_storage_analyzer_initialization(self):
        """Test S3StorageAnalyzer initialization."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test initialization
            assert analyzer is not None
            assert analyzer.profile_analyzer == mock_profile_analyzer
            assert hasattr(analyzer, "s3_events")
            assert len(analyzer.s3_events) > 20  # Should have comprehensive S3 events list

            # Check for key S3 events
            key_events = [
                "S3ReadMicroseconds",
                "S3WriteMicroseconds",
                "S3ReadBytes",
                "S3WriteBytes",
                "S3ReadRequestsCount",
                "S3WriteRequestsCount",
                "S3ReadRequestsErrors",
                "S3CreateMultipartUpload",
                "S3PutObject",
                "S3GetObject",
            ]
            for event in key_events:
                assert event in analyzer.s3_events

    @patch.dict("os.environ", test_env)
    def test_analyze_s3_performance_method(self):
        """Test S3StorageAnalyzer analyze_s3_performance method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import (
                ProfileEventsAnalyzer,
                ProfileEventsCategory,
            )
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)

            # Mock comprehensive S3 metrics
            mock_s3_metrics = {
                "S3ReadMicroseconds": {"current_value": 500000},  # 500ms
                "S3WriteMicroseconds": {"current_value": 1000000},  # 1s
                "S3ReadBytes": {"current_value": 100 * 1024 * 1024},  # 100MB
                "S3WriteBytes": {"current_value": 50 * 1024 * 1024},  # 50MB
                "S3ReadRequestsCount": {"current_value": 1000},
                "S3WriteRequestsCount": {"current_value": 500},
                "S3ReadRequestsErrors": {"current_value": 10},  # 1% error rate
                "S3WriteRequestsErrors": {"current_value": 5},  # 1% error rate
                "S3ReadRequestsThrottling": {"current_value": 5},  # Throttling detected
                "S3WriteRequestsThrottling": {"current_value": 2},
                "S3GetObject": {"current_value": 800},
                "S3PutObject": {"current_value": 400},
                "S3CreateMultipartUpload": {"current_value": 50},
                "S3CompleteMultipartUpload": {"current_value": 45},  # 90% completion rate
                "S3AbortMultipartUpload": {"current_value": 5},
            }

            mock_profile_analyzer.get_events_by_category.return_value = mock_s3_metrics

            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test method execution
            result = analyzer.analyze_s3_performance(time_range_hours=24, include_historical=True)

            # Should return comprehensive S3 analysis
            assert result is not None
            assert hasattr(result, "operation_performance")
            assert hasattr(result, "latency_analysis")
            assert hasattr(result, "error_analysis")
            assert hasattr(result, "throttling_analysis")
            assert hasattr(result, "cost_analysis")
            assert hasattr(result, "issues")
            assert hasattr(result, "recommendations")

            # Verify ProfileEventsAnalyzer was called
            mock_profile_analyzer.get_events_by_category.assert_called_once_with(
                ProfileEventsCategory.S3_OPERATIONS, time_range_hours=24
            )

            # Check that operations were analyzed
            assert isinstance(result.operation_performance, dict)
            assert isinstance(result.issues, list)
            assert isinstance(result.recommendations, list)

    @patch.dict("os.environ", test_env)
    def test_s3_operations_analysis_methods(self):
        """Test S3StorageAnalyzer operation analysis methods."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test _analyze_s3_operations
            mock_metrics = {
                "S3ReadMicroseconds": {"current_value": 2000000},  # 2s total
                "S3ReadBytes": {"current_value": 200 * 1024 * 1024},  # 200MB
                "S3ReadRequestsCount": {"current_value": 2000},
                "S3GetObject": {"current_value": 1500},
                "S3WriteMicroseconds": {"current_value": 1000000},  # 1s total
                "S3WriteBytes": {"current_value": 100 * 1024 * 1024},  # 100MB
                "S3WriteRequestsCount": {"current_value": 1000},
                "S3PutObject": {"current_value": 800},
            }

            operations = analyzer._analyze_s3_operations(mock_metrics)

            # Should have different operation groups
            assert "read_operations" in operations
            assert "write_operations" in operations
            assert "multipart_operations" in operations
            assert "metadata_operations" in operations

            # Check read operations analysis
            read_ops = operations["read_operations"]
            assert read_ops["total_requests"] == 3500  # ReadRequestsCount + GetObject
            assert read_ops["total_bytes"] == 200 * 1024 * 1024
            assert read_ops["total_time_microseconds"] == 2000000
            assert read_ops["average_latency_ms"] == (2000000 / 1000) / 3500  # Should be ~0.57ms

            # Check write operations analysis
            write_ops = operations["write_operations"]
            assert write_ops["total_requests"] == 1800  # WriteRequestsCount + PutObject
            assert write_ops["total_bytes"] == 100 * 1024 * 1024

    @patch.dict("os.environ", test_env)
    def test_s3_latency_analysis_method(self):
        """Test S3StorageAnalyzer _analyze_s3_latency method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test high latency scenario
            high_latency_metrics = {
                "S3ReadMicroseconds": {"current_value": 1000000},  # 1s total
                "S3ReadRequestsCount": {"current_value": 1000},  # = 1ms avg per request
                "S3WriteMicroseconds": {"current_value": 2000000},  # 2s total
                "S3WriteRequestsCount": {"current_value": 1000},  # = 2ms avg per request
            }

            latency_analysis = analyzer._analyze_s3_latency(high_latency_metrics)

            # Should identify normal read latency (1ms < 500ms threshold)
            assert "read_latency" in latency_analysis
            read_latency = latency_analysis["read_latency"]
            assert read_latency["average_ms"] == 1.0
            assert read_latency["is_high"] is False

            # Should identify normal write latency (2ms < 1000ms threshold)
            assert "write_latency" in latency_analysis
            write_latency = latency_analysis["write_latency"]
            assert write_latency["average_ms"] == 2.0
            assert write_latency["is_high"] is False

            # Test with actually high latency
            very_high_latency_metrics = {
                "S3ReadMicroseconds": {"current_value": 600000000},  # 600s total
                "S3ReadRequestsCount": {"current_value": 1000},  # = 600ms avg (high)
                "S3WriteMicroseconds": {"current_value": 1200000000},  # 1200s total
                "S3WriteRequestsCount": {"current_value": 1000},  # = 1200ms avg (high)
            }

            high_latency_analysis = analyzer._analyze_s3_latency(very_high_latency_metrics)

            read_latency_high = high_latency_analysis["read_latency"]
            assert read_latency_high["average_ms"] == 600.0
            assert read_latency_high["is_high"] is True  # > 500ms threshold

            write_latency_high = high_latency_analysis["write_latency"]
            assert write_latency_high["average_ms"] == 1200.0
            assert write_latency_high["is_high"] is True  # > 1000ms threshold

            # Should identify high latency operations
            high_ops = high_latency_analysis["high_latency_operations"]
            assert len(high_ops) == 2  # Both read and write should be flagged

    @patch.dict("os.environ", test_env)
    def test_s3_error_analysis_method(self):
        """Test S3StorageAnalyzer _analyze_s3_errors method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test high error rate scenario
            high_error_metrics = {
                "S3ReadRequestsErrors": {"current_value": 60},  # 6% error rate (critical)
                "S3ReadRequestsCount": {"current_value": 1000},
                "S3WriteRequestsErrors": {"current_value": 25},  # 2.5% error rate (critical)
                "S3WriteRequestsCount": {"current_value": 1000},
            }

            error_analysis = analyzer._analyze_s3_errors(high_error_metrics)

            # Check read error analysis
            assert "read_errors" in error_analysis
            read_errors = error_analysis["read_errors"]
            assert read_errors["error_count"] == 60
            assert read_errors["total_requests"] == 1000
            assert read_errors["error_rate_percent"] == 6.0
            assert read_errors["is_critical"] is True  # > 5% threshold

            # Check write error analysis
            assert "write_errors" in error_analysis
            write_errors = error_analysis["write_errors"]
            assert write_errors["error_count"] == 25
            assert write_errors["error_rate_percent"] == 2.5
            assert write_errors["is_critical"] is True  # > 2% threshold

            # Check overall error rate
            assert error_analysis["overall_error_rate"] == 4.25  # (60+25)/(1000+1000) = 4.25%

            # Check critical errors identification
            critical_errors = error_analysis["critical_errors"]
            assert len(critical_errors) == 2  # Both read and write should be flagged
            assert any(err["type"] == "high_read_error_rate" for err in critical_errors)
            assert any(err["type"] == "high_write_error_rate" for err in critical_errors)

    @patch.dict("os.environ", test_env)
    def test_s3_cost_analysis_method(self):
        """Test S3StorageAnalyzer _analyze_s3_costs method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test high volume scenario that triggers cost optimizations
            high_volume_metrics = {
                "S3ReadRequestsCount": {
                    "current_value": 50000
                },  # High GET volume (will trigger optimization)
                "S3WriteRequestsCount": {
                    "current_value": 5000
                },  # High PUT volume (will trigger optimization)
                "S3ReadBytes": {"current_value": 10 * 1024**3},  # 10GB read per hour
                "S3WriteBytes": {"current_value": 5 * 1024**3},  # 5GB written per hour
            }

            cost_analysis = analyzer._analyze_s3_costs(high_volume_metrics)

            # Check request costs calculation
            assert "request_costs" in cost_analysis
            request_costs = cost_analysis["request_costs"]

            # Monthly projections (hourly * 24 * 30)
            hours_per_month = 24 * 30
            expected_monthly_gets = 50000 * hours_per_month  # 36M requests
            expected_monthly_puts = 5000 * hours_per_month  # 3.6M requests

            assert request_costs["monthly_get_requests"] == expected_monthly_gets
            assert request_costs["monthly_put_requests"] == expected_monthly_puts

            # Cost calculations based on pricing
            expected_get_cost = (expected_monthly_gets / 1000) * 0.0004  # $14.40
            expected_put_cost = (expected_monthly_puts / 1000) * 0.005  # $18.00

            assert abs(request_costs["estimated_get_cost"] - expected_get_cost) < 0.01
            assert abs(request_costs["estimated_put_cost"] - expected_put_cost) < 0.01

            # Check data transfer costs
            assert "data_transfer_costs" in cost_analysis
            transfer_costs = cost_analysis["data_transfer_costs"]

            # Monthly data transfer projections
            expected_monthly_read_gb = (10 * 1024**3 * hours_per_month) / (1024**3)  # 7200 GB
            expected_transfer_cost = expected_monthly_read_gb * 0.09  # $648

            assert abs(transfer_costs["monthly_read_gb"] - expected_monthly_read_gb) < 0.1
            assert abs(transfer_costs["estimated_transfer_cost"] - expected_transfer_cost) < 1.0

            # Check cost optimization opportunities
            assert "cost_optimization_opportunities" in cost_analysis
            opportunities = cost_analysis["cost_optimization_opportunities"]

            # Should trigger both GET and PUT optimization opportunities
            assert len(opportunities) == 2

            get_opp = next(
                (opp for opp in opportunities if opp["type"] == "high_get_request_volume"), None
            )
            assert get_opp is not None
            assert "CloudFront" in get_opp["description"]
            assert get_opp["potential_savings_percent"] == 30

            put_opp = next(
                (opp for opp in opportunities if opp["type"] == "high_put_request_volume"), None
            )
            assert put_opp is not None
            assert get_opp["potential_savings_percent"] == 15

    @patch.dict("os.environ", test_env)
    def test_s3_issue_detection_method(self):
        """Test S3StorageAnalyzer _detect_s3_issues method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import (
                S3StorageAnalyzer,
                StoragePerformanceIssue,
                StorageSeverity,
            )

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Create problematic analysis results
            operation_perf = {
                "read_operations": {"average_latency_ms": 1500.0},  # High latency
                "write_operations": {"average_latency_ms": 2500.0},  # Very high latency
            }

            latency = {}  # Not used in this method

            errors = {"overall_error_rate": 3.5}  # High error rate

            throttling = {"mitigation_needed": True}

            costs = {}  # Not used in this method

            # Test issue detection
            issues = analyzer._detect_s3_issues(operation_perf, latency, errors, throttling, costs)

            # Should detect multiple issues
            assert len(issues) == 3  # Latency (2) + error + throttling

            # Check latency issues
            latency_issues = [
                issue for issue in issues if issue.type == StoragePerformanceIssue.HIGH_LATENCY
            ]
            assert len(latency_issues) == 2  # Both read and write operations

            read_issue = next(
                (
                    issue
                    for issue in latency_issues
                    if "read_operations" in issue.affected_operations
                ),
                None,
            )
            assert read_issue is not None
            assert read_issue.severity == StorageSeverity.MEDIUM  # 1500ms < 2000ms threshold

            write_issue = next(
                (
                    issue
                    for issue in latency_issues
                    if "write_operations" in issue.affected_operations
                ),
                None,
            )
            assert write_issue is not None
            assert write_issue.severity == StorageSeverity.HIGH  # 2500ms > 2000ms threshold

            # Check error rate issue
            error_issues = [
                issue for issue in issues if issue.type == StoragePerformanceIssue.HIGH_ERROR_RATE
            ]
            assert len(error_issues) == 1
            error_issue = error_issues[0]
            assert error_issue.severity == StorageSeverity.HIGH  # 3.5% < 5% threshold

            # Check throttling issue
            throttling_issues = [
                issue for issue in issues if issue.type == StoragePerformanceIssue.THROTTLING
            ]
            assert len(throttling_issues) == 1
            throttling_issue = throttling_issues[0]
            assert throttling_issue.severity == StorageSeverity.HIGH


@pytest.mark.unit
class TestAzureStorageAnalyzer:
    """Test AzureStorageAnalyzer class functionality."""

    @patch.dict("os.environ", test_env)
    def test_azure_storage_analyzer_initialization(self):
        """Test AzureStorageAnalyzer initialization."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import AzureStorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = AzureStorageAnalyzer(mock_profile_analyzer)

            # Test initialization
            assert analyzer is not None
            assert analyzer.profile_analyzer == mock_profile_analyzer
            assert hasattr(analyzer, "azure_events")
            assert len(analyzer.azure_events) > 10  # Should have Azure-specific events

            # Check for key Azure events
            key_events = [
                "AzureBlobStorageReadMicroseconds",
                "AzureBlobStorageWriteMicroseconds",
                "AzureBlobStorageReadBytes",
                "AzureBlobStorageWriteBytes",
                "DiskAzureReadMicroseconds",
                "DiskAzureWriteMicroseconds",
            ]
            for event in key_events:
                assert event in analyzer.azure_events

    @patch.dict("os.environ", test_env)
    def test_analyze_azure_performance_method(self):
        """Test AzureStorageAnalyzer analyze_azure_performance method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import AzureStorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)

            # Mock get_event_value to return Azure metrics
            def mock_get_event_value(event_name):
                azure_metrics = {
                    "AzureBlobStorageReadMicroseconds": {"current_value": 800000},  # 800ms
                    "AzureBlobStorageWriteMicroseconds": {"current_value": 1200000},  # 1.2s
                    "AzureBlobStorageReadBytes": {"current_value": 150 * 1024 * 1024},  # 150MB
                    "AzureBlobStorageWriteBytes": {"current_value": 75 * 1024 * 1024},  # 75MB
                    "AzureBlobStorageReadRequestsCount": {"current_value": 1500},
                    "AzureBlobStorageWriteRequestsCount": {"current_value": 750},
                }
                return azure_metrics.get(event_name)

            mock_profile_analyzer.get_event_value.side_effect = mock_get_event_value

            analyzer = AzureStorageAnalyzer(mock_profile_analyzer)

            # Test method execution
            result = analyzer.analyze_azure_performance(
                time_range_hours=24, include_historical=True
            )

            # Should return comprehensive Azure analysis
            assert result is not None
            assert hasattr(result, "blob_operation_performance")
            assert hasattr(result, "latency_analysis")
            assert hasattr(result, "error_analysis")
            assert hasattr(result, "throttling_analysis")
            assert hasattr(result, "cost_analysis")
            assert hasattr(result, "tier_optimization")
            assert hasattr(result, "issues")
            assert hasattr(result, "recommendations")

            # Check that operations were analyzed
            assert isinstance(result.blob_operation_performance, dict)
            assert isinstance(result.issues, list)
            assert isinstance(result.recommendations, list)

    @patch.dict("os.environ", test_env)
    def test_azure_operations_analysis_method(self):
        """Test AzureStorageAnalyzer _analyze_azure_operations method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import AzureStorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = AzureStorageAnalyzer(mock_profile_analyzer)

            # Test Azure operations analysis
            mock_metrics = {
                "AzureBlobStorageReadMicroseconds": {"current_value": 1500000},  # 1.5s total
                "AzureBlobStorageReadBytes": {"current_value": 300 * 1024 * 1024},  # 300MB
                "AzureBlobStorageReadRequestsCount": {"current_value": 1500},
                "AzureBlobStorageWriteMicroseconds": {"current_value": 2000000},  # 2s total
                "AzureBlobStorageWriteBytes": {"current_value": 150 * 1024 * 1024},  # 150MB
                "AzureBlobStorageWriteRequestsCount": {"current_value": 1000},
            }

            operations = analyzer._analyze_azure_operations(mock_metrics)

            # Should have read and write operations
            assert "read_operations" in operations
            assert "write_operations" in operations

            # Check read operations analysis
            read_ops = operations["read_operations"]
            assert read_ops["total_requests"] == 1500
            assert read_ops["total_bytes"] == 300 * 1024 * 1024
            assert read_ops["total_time_microseconds"] == 1500000
            assert read_ops["average_latency_ms"] == (1500000 / 1500) / 1000  # 1ms
            assert read_ops["requests_per_second"] == 1500 / 3600  # Assuming 1 hour

            # Check write operations analysis
            write_ops = operations["write_operations"]
            assert write_ops["total_requests"] == 1000
            assert write_ops["total_bytes"] == 150 * 1024 * 1024
            assert write_ops["average_latency_ms"] == (2000000 / 1000) / 1000  # 2ms

    @patch.dict("os.environ", test_env)
    def test_azure_issue_detection_method(self):
        """Test AzureStorageAnalyzer _detect_azure_issues method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import (
                AzureStorageAnalyzer,
                StoragePerformanceIssue,
                StorageSeverity,
            )

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = AzureStorageAnalyzer(mock_profile_analyzer)

            # Create problematic blob performance data
            blob_perf = {
                "read_operations": {"average_latency_ms": 1800.0},  # High latency
                "write_operations": {"average_latency_ms": 2200.0},  # Very high latency
            }

            latency = {}  # Not used in this method
            errors = {}  # Not used in this method
            throttling = {}  # Not used in this method

            # Test issue detection
            issues = analyzer._detect_azure_issues(blob_perf, latency, errors, throttling)

            # Should detect latency issues
            assert len(issues) == 2  # Both read and write latency issues

            # Check both issues are high latency type
            for issue in issues:
                assert issue.type == StoragePerformanceIssue.HIGH_LATENCY
                assert "Azure" in issue.description

            # Check severity levels
            read_issue = next(
                (issue for issue in issues if "read_operations" in issue.description), None
            )
            write_issue = next(
                (issue for issue in issues if "write_operations" in issue.description), None
            )

            assert read_issue is not None
            assert read_issue.severity == StorageSeverity.MEDIUM  # 1800ms < 2000ms
            assert write_issue is not None
            assert write_issue.severity == StorageSeverity.HIGH  # 2200ms > 2000ms


@pytest.mark.unit
class TestCompressionAnalyzer:
    """Test CompressionAnalyzer class functionality."""

    @patch.dict("os.environ", test_env)
    def test_compression_analyzer_initialization(self):
        """Test CompressionAnalyzer initialization."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import CompressionAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Test initialization
            assert analyzer is not None
            assert analyzer.profile_analyzer == mock_profile_analyzer
            assert hasattr(analyzer, "compression_events")
            assert len(analyzer.compression_events) > 8  # Should have compression events

            # Check for key compression events
            key_events = [
                "ReadCompressedBytes",
                "CompressedReadBufferBlocks",
                "CompressedReadBufferBytes",
                "CompressedReadBufferChecksumFailed",
                "UncompressedCacheHits",
                "MarkCacheHits",
            ]
            for event in key_events:
                assert event in analyzer.compression_events

    @patch.dict("os.environ", test_env)
    def test_analyze_compression_performance_method(self):
        """Test CompressionAnalyzer analyze_compression_performance method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import CompressionAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)

            # Mock get_event_value to return compression metrics
            def mock_get_event_value(event_name):
                compression_metrics = {
                    "ReadCompressedBytes": {"current_value": 100 * 1024 * 1024},  # 100MB compressed
                    "CompressedReadBufferBlocks": {"current_value": 1000},
                    "CompressedReadBufferBytes": {
                        "current_value": 300 * 1024 * 1024
                    },  # 300MB uncompressed
                    "CompressedReadBufferChecksumFailed": {"current_value": 5},  # Integrity issues
                    "CompressedReadBufferChecksumDoesntMatch": {"current_value": 3},
                    "UncompressedCacheHits": {"current_value": 8000},
                    "UncompressedCacheMisses": {"current_value": 2000},  # 80% hit rate
                }
                return compression_metrics.get(event_name)

            mock_profile_analyzer.get_event_value.side_effect = mock_get_event_value

            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Test method execution
            result = analyzer.analyze_compression_performance(time_range_hours=24)

            # Should return comprehensive compression analysis
            assert result is not None
            assert hasattr(result, "compression_efficiency")
            assert hasattr(result, "compression_ratios")
            assert hasattr(result, "decompression_performance")
            assert hasattr(result, "integrity_analysis")
            assert hasattr(result, "codec_performance")
            assert hasattr(result, "optimization_opportunities")
            assert hasattr(result, "issues")
            assert hasattr(result, "recommendations")

    @patch.dict("os.environ", test_env)
    def test_compression_efficiency_analysis_method(self):
        """Test CompressionAnalyzer _analyze_compression_efficiency method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import CompressionAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Test good compression efficiency
            good_metrics = {
                "ReadCompressedBytes": {"current_value": 100 * 1024 * 1024},  # 100MB compressed
                "CompressedReadBufferBlocks": {"current_value": 1000},
                "CompressedReadBufferBytes": {
                    "current_value": 300 * 1024 * 1024
                },  # 300MB uncompressed = 3x compression
            }

            efficiency = analyzer._analyze_compression_efficiency(good_metrics)

            assert efficiency["compressed_bytes_read"] == 100 * 1024 * 1024
            assert efficiency["compression_blocks"] == 1000
            assert efficiency["compression_buffer_bytes"] == 300 * 1024 * 1024

            # Should calculate good compression ratio
            performance_indicators = efficiency["performance_indicators"]
            assert performance_indicators["compression_ratio"] == 3.0  # 300MB / 100MB
            assert performance_indicators["is_efficient"] is True  # > 2.0x threshold
            assert efficiency["efficiency_score"] == 30.0  # min(3.0 * 10, 100)

    @patch.dict("os.environ", test_env)
    def test_data_integrity_analysis_method(self):
        """Test CompressionAnalyzer _analyze_data_integrity method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import CompressionAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Test integrity issues scenario
            integrity_issues_metrics = {
                "CompressedReadBufferChecksumFailed": {"current_value": 8},  # Critical failures
                "CompressedReadBufferChecksumDoesntMatch": {"current_value": 5},  # Mismatches
            }

            integrity = analyzer._analyze_data_integrity(integrity_issues_metrics)

            assert integrity["checksum_failures"] == 8
            assert integrity["checksum_mismatches"] == 5
            assert integrity["integrity_score"] == 0.0  # max(0, 100 - (13 * 10)) = 0

            # Check critical issues identification
            critical_issues = integrity["critical_issues"]
            assert len(critical_issues) == 2

            failure_issue = next(
                (issue for issue in critical_issues if issue["type"] == "checksum_failures"), None
            )
            assert failure_issue is not None
            assert failure_issue["count"] == 8
            assert failure_issue["severity"] == "critical"
            assert failure_issue["impact"] == "data_corruption_risk"

            mismatch_issue = next(
                (issue for issue in critical_issues if issue["type"] == "checksum_mismatches"), None
            )
            assert mismatch_issue is not None
            assert mismatch_issue["count"] == 5
            assert mismatch_issue["severity"] == "high"
            assert mismatch_issue["impact"] == "data_integrity_concern"

    @patch.dict("os.environ", test_env)
    def test_compression_issue_detection_method(self):
        """Test CompressionAnalyzer _detect_compression_issues method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import (
                CompressionAnalyzer,
                StoragePerformanceIssue,
                StorageSeverity,
            )

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Create problematic compression analysis results
            efficiency = {
                "performance_indicators": {"compression_ratio": 1.2}  # Poor compression ratio
            }

            ratios = {}  # Not used in this method
            decompression = {}  # Not used in this method

            integrity = {
                "integrity_score": 70.0,  # Low integrity score
                "critical_issues": [
                    {"type": "checksum_failures", "count": 5, "severity": "critical"},
                    {"type": "checksum_mismatches", "count": 3, "severity": "high"},
                ],
            }

            # Test issue detection
            issues = analyzer._detect_compression_issues(
                efficiency, ratios, decompression, integrity
            )

            # Should detect both integrity and efficiency issues
            assert len(issues) == 3  # 2 integrity + 1 efficiency

            # Check integrity issues
            integrity_issues = [
                issue
                for issue in issues
                if issue.type == StoragePerformanceIssue.DATA_INTEGRITY_ISSUES
            ]
            assert len(integrity_issues) == 2

            critical_issue = next(
                (issue for issue in integrity_issues if issue.severity == StorageSeverity.CRITICAL),
                None,
            )
            assert critical_issue is not None
            assert "checksum_failures" in critical_issue.description

            # Check efficiency issue
            efficiency_issues = [
                issue
                for issue in issues
                if issue.type == StoragePerformanceIssue.INEFFICIENT_COMPRESSION
            ]
            assert len(efficiency_issues) == 1
            efficiency_issue = efficiency_issues[0]
            assert efficiency_issue.severity == StorageSeverity.MEDIUM
            assert "1.20x ratio" in efficiency_issue.description


@pytest.mark.unit
class TestStorageOptimizationEngine:
    """Test StorageOptimizationEngine class functionality."""

    @patch.dict("os.environ", test_env)
    def test_storage_optimization_engine_initialization(self):
        """Test StorageOptimizationEngine initialization."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import StorageOptimizationEngine

            engine = StorageOptimizationEngine(mock_client)

            # Test proper initialization
            assert engine is not None
            assert engine.client == mock_client
            assert hasattr(engine, "profile_analyzer")
            assert hasattr(engine, "s3_analyzer")
            assert hasattr(engine, "azure_analyzer")
            assert hasattr(engine, "compression_analyzer")

            # Check that components are properly initialized
            assert engine.profile_analyzer is not None
            assert engine.s3_analyzer is not None
            assert engine.azure_analyzer is not None
            assert engine.compression_analyzer is not None

    @patch.dict("os.environ", test_env)
    def test_generate_comprehensive_storage_report_method(self):
        """Test StorageOptimizationEngine generate_comprehensive_storage_report method."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                StorageIssue,
                StorageOptimizationEngine,
                StoragePerformanceIssue,
                StorageSeverity,
            )

            engine = StorageOptimizationEngine(mock_client)

            # Mock the analyzer methods to return test results
            mock_s3_issue = StorageIssue(
                type=StoragePerformanceIssue.HIGH_LATENCY,
                severity=StorageSeverity.HIGH,
                description="S3 high latency",
                impact_score=80.0,
                recommendations=["Use S3 Transfer Acceleration"],
            )

            mock_s3_analysis = Mock()
            mock_s3_analysis.issues = [mock_s3_issue]
            mock_s3_analysis.recommendations = ["Monitor S3 performance"]
            mock_s3_analysis.cost_analysis = {
                "cost_optimization_opportunities": [
                    {
                        "type": "high_get_request_volume",
                        "description": "CloudFront caching",
                        "potential_savings_percent": 30,
                    }
                ]
            }

            mock_azure_issue = StorageIssue(
                type=StoragePerformanceIssue.COST_INEFFICIENCY,
                severity=StorageSeverity.MEDIUM,
                description="Azure cost inefficiency",
                impact_score=60.0,
            )

            mock_azure_analysis = Mock()
            mock_azure_analysis.issues = [mock_azure_issue]
            mock_azure_analysis.recommendations = ["Optimize Azure tiers"]
            mock_azure_analysis.cost_analysis = {
                "optimization_opportunities": ["Move to cool tier"]
            }

            mock_compression_issue = StorageIssue(
                type=StoragePerformanceIssue.DATA_INTEGRITY_ISSUES,
                severity=StorageSeverity.CRITICAL,
                description="Data integrity issues",
                impact_score=95.0,
            )

            mock_compression_analysis = Mock()
            mock_compression_analysis.issues = [mock_compression_issue]
            mock_compression_analysis.recommendations = ["Check storage hardware"]

            # Mock analyzer methods
            engine.s3_analyzer.analyze_s3_performance = Mock(return_value=mock_s3_analysis)
            engine.azure_analyzer.analyze_azure_performance = Mock(return_value=mock_azure_analysis)
            engine.compression_analyzer.analyze_compression_performance = Mock(
                return_value=mock_compression_analysis
            )

            # Test method execution
            report = engine.generate_comprehensive_storage_report(
                time_range_hours=24, include_s3=True, include_azure=True, include_compression=True
            )

            # Should return comprehensive report
            assert report is not None
            assert report.s3_analysis == mock_s3_analysis
            assert report.azure_analysis == mock_azure_analysis
            assert report.compression_analysis == mock_compression_analysis

            # Check that issues were prioritized
            assert len(report.priority_issues) > 0
            # Critical issue should be first (highest priority)
            assert report.priority_issues[0].severity == StorageSeverity.CRITICAL

            # Check that recommendations were collected
            assert len(report.recommendations) > 0
            assert "Monitor S3 performance" in report.recommendations
            assert "Optimize Azure tiers" in report.recommendations
            assert "Check storage hardware" in report.recommendations

            # Check overall score calculation (should be < 100 due to issues)
            assert 0.0 <= report.overall_score <= 100.0
            assert report.overall_score < 100.0  # Should be reduced due to issues

            # Verify all analyzers were called
            engine.s3_analyzer.analyze_s3_performance.assert_called_once_with(24)
            engine.azure_analyzer.analyze_azure_performance.assert_called_once_with(24)
            engine.compression_analyzer.analyze_compression_performance.assert_called_once_with(24)

    @patch.dict("os.environ", test_env)
    def test_storage_optimization_helper_methods(self):
        """Test StorageOptimizationEngine helper methods."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                StorageIssue,
                StorageOptimizationEngine,
                StoragePerformanceIssue,
                StorageSeverity,
            )

            engine = StorageOptimizationEngine(mock_client)

            # Test _prioritize_issues
            issues = [
                StorageIssue(
                    StoragePerformanceIssue.HIGH_LATENCY,
                    StorageSeverity.MEDIUM,
                    "Medium issue",
                    60.0,
                ),
                StorageIssue(
                    StoragePerformanceIssue.DATA_INTEGRITY_ISSUES,
                    StorageSeverity.CRITICAL,
                    "Critical issue",
                    95.0,
                ),
                StorageIssue(
                    StoragePerformanceIssue.COST_INEFFICIENCY,
                    StorageSeverity.HIGH,
                    "High issue",
                    80.0,
                ),
                StorageIssue(
                    StoragePerformanceIssue.THROTTLING, StorageSeverity.LOW, "Low issue", 30.0
                ),
            ]

            prioritized = engine._prioritize_issues(issues)

            # Should be sorted by severity then impact score
            assert prioritized[0].severity == StorageSeverity.CRITICAL
            assert prioritized[1].severity == StorageSeverity.HIGH
            assert prioritized[2].severity == StorageSeverity.MEDIUM
            assert prioritized[3].severity == StorageSeverity.LOW

            # Test _calculate_overall_score
            mock_report = Mock()
            mock_report.priority_issues = [
                StorageIssue(
                    StoragePerformanceIssue.DATA_INTEGRITY_ISSUES,
                    StorageSeverity.CRITICAL,
                    "Critical",
                    100.0,
                ),
                StorageIssue(
                    StoragePerformanceIssue.HIGH_LATENCY, StorageSeverity.HIGH, "High", 80.0
                ),
                StorageIssue(
                    StoragePerformanceIssue.COST_INEFFICIENCY,
                    StorageSeverity.MEDIUM,
                    "Medium",
                    60.0,
                ),
            ]

            overall_score = engine._calculate_overall_score(mock_report)

            # Should be reduced significantly due to critical issue
            assert overall_score < 80.0  # Significant reduction
            assert overall_score >= 0.0  # But not negative


@pytest.mark.unit
class TestStorageDiagnosticFunctions:
    """Test standalone diagnostic functions."""

    @patch.dict("os.environ", test_env)
    def test_diagnose_high_storage_latency_function(self):
        """Test diagnose_high_storage_latency function."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                StorageIssue,
                StorageOptimizationEngine,
                StoragePerformanceIssue,
                StorageSeverity,
                diagnose_high_storage_latency,
            )

            # Mock the storage optimization engine
            with patch.object(
                StorageOptimizationEngine, "generate_comprehensive_storage_report"
            ) as mock_generate:
                mock_latency_issue = StorageIssue(
                    type=StoragePerformanceIssue.HIGH_LATENCY,
                    severity=StorageSeverity.HIGH,
                    description="High S3 latency detected",
                    impact_score=80.0,
                    recommendations=["Use S3 Transfer Acceleration", "Check network"],
                )

                mock_report = Mock()
                mock_report.priority_issues = [mock_latency_issue]
                mock_report.s3_analysis = Mock()
                mock_report.s3_analysis.latency_analysis = {"read_latency": {"average_ms": 800.0}}
                mock_report.azure_analysis = None

                mock_generate.return_value = mock_report

                # Test function execution
                result = diagnose_high_storage_latency(mock_client, time_range_hours=4)

                # Should return latency diagnosis
                assert "latency_issues" in result
                assert "s3_latency" in result
                assert "azure_latency" in result
                assert "recommendations" in result

                latency_issues = result["latency_issues"]
                assert len(latency_issues) == 1
                assert latency_issues[0].type == StoragePerformanceIssue.HIGH_LATENCY

                recommendations = result["recommendations"]
                assert len(recommendations) == 2
                assert "S3 Transfer Acceleration" in recommendations
                assert "Check network" in recommendations

    @patch.dict("os.environ", test_env)
    def test_diagnose_storage_throttling_function(self):
        """Test diagnose_storage_throttling function."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                StorageIssue,
                StorageOptimizationEngine,
                StoragePerformanceIssue,
                StorageSeverity,
                diagnose_storage_throttling,
            )

            # Mock the storage optimization engine
            with patch.object(
                StorageOptimizationEngine, "generate_comprehensive_storage_report"
            ) as mock_generate:
                mock_throttling_issue = StorageIssue(
                    type=StoragePerformanceIssue.THROTTLING,
                    severity=StorageSeverity.HIGH,
                    description="S3 throttling detected",
                    impact_score=75.0,
                )

                mock_report = Mock()
                mock_report.priority_issues = [mock_throttling_issue]
                mock_report.s3_analysis = Mock()
                mock_report.s3_analysis.throttling_analysis = {"mitigation_needed": True}
                mock_report.azure_analysis = Mock()
                mock_report.azure_analysis.throttling_analysis = {"throttling_detected": False}

                mock_generate.return_value = mock_report

                # Test function execution
                result = diagnose_storage_throttling(mock_client, time_range_hours=4)

                # Should return throttling diagnosis
                assert "throttling_detected" in result
                assert "throttling_issues" in result
                assert "s3_throttling" in result
                assert "azure_throttling" in result
                assert "mitigation_strategies" in result

                assert result["throttling_detected"] is True
                throttling_issues = result["throttling_issues"]
                assert len(throttling_issues) == 1
                assert throttling_issues[0].type == StoragePerformanceIssue.THROTTLING

                mitigation_strategies = result["mitigation_strategies"]
                assert len(mitigation_strategies) == 4
                assert any(
                    "exponential backoff" in strategy.lower() for strategy in mitigation_strategies
                )

    @patch.dict("os.environ", test_env)
    def test_analyze_compression_efficiency_function(self):
        """Test analyze_compression_efficiency function."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                CompressionAnalyzer,
                analyze_compression_efficiency,
            )

            # Mock the compression analyzer
            with patch.object(
                CompressionAnalyzer, "analyze_compression_performance"
            ) as mock_analyze:
                mock_issue = StorageIssue(
                    type=StoragePerformanceIssue.INEFFICIENT_COMPRESSION,
                    severity=StorageSeverity.MEDIUM,
                    description="Low compression ratio",
                    impact_score=50.0,
                    recommendations=["Consider zstd codec"],
                )

                mock_analysis = Mock()
                mock_analysis.compression_efficiency = {"efficiency_score": 65.0}
                mock_analysis.compression_ratios = {"overall_ratio": 2.1}
                mock_analysis.integrity_analysis = {"integrity_score": 98.0}
                mock_analysis.optimization_opportunities = {
                    "cache_optimization": {"needs_optimization": True}
                }
                mock_analysis.issues = [mock_issue]
                mock_analysis.recommendations = ["Consider zstd codec", "Increase cache size"]

                mock_analyze.return_value = mock_analysis

                # Test function execution
                result = analyze_compression_efficiency(mock_client, time_range_hours=24)

                # Should return compression analysis
                assert "compression_efficiency" in result
                assert "compression_ratios" in result
                assert "integrity_analysis" in result
                assert "optimization_opportunities" in result
                assert "issues" in result
                assert "recommendations" in result

                assert result["compression_efficiency"]["efficiency_score"] == 65.0
                assert result["compression_ratios"]["overall_ratio"] == 2.1

                issues = result["issues"]
                assert len(issues) == 1
                assert issues[0]["type"] == "inefficient_compression"
                assert issues[0]["severity"] == "medium"

    @patch.dict("os.environ", test_env)
    def test_identify_storage_cost_optimizations_function(self):
        """Test identify_storage_cost_optimizations function."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import (
                StorageOptimizationEngine,
                identify_storage_cost_optimizations,
            )

            # Mock the storage optimization engine
            with patch.object(
                StorageOptimizationEngine, "generate_comprehensive_storage_report"
            ) as mock_generate:
                mock_report = Mock()
                mock_report.cost_savings_opportunities = [
                    {"type": "high_get_request_volume", "potential_savings_percent": 30},
                    {"type": "tier_optimization", "potential_savings_percent": 15},
                ]
                mock_report.s3_analysis = Mock()
                mock_report.s3_analysis.cost_analysis = {
                    "estimated_monthly_costs": {"total": 1500.0}
                }
                mock_report.azure_analysis = Mock()
                mock_report.azure_analysis.cost_analysis = {"estimated_costs": {"monthly": 800.0}}

                mock_generate.return_value = mock_report

                # Test function execution
                result = identify_storage_cost_optimizations(mock_client, time_range_hours=24)

                # Should return cost optimization analysis
                assert "cost_savings_opportunities" in result
                assert "s3_cost_analysis" in result
                assert "azure_cost_analysis" in result
                assert "estimated_monthly_savings" in result
                assert "priority_optimizations" in result
                assert "recommendations" in result

                savings_opportunities = result["cost_savings_opportunities"]
                assert len(savings_opportunities) == 2

                # Should calculate total potential savings
                assert result["estimated_monthly_savings"] == 45  # 30 + 15

                # Should identify high-impact optimizations (>10% savings)
                priority_optimizations = result["priority_optimizations"]
                assert len(priority_optimizations) == 2  # Both are >10%

                recommendations = result["recommendations"]
                assert len(recommendations) == 4
                assert any("lifecycle policies" in rec.lower() for rec in recommendations)


@pytest.mark.unit
class TestStorageCloudDiagnosticsErrorHandling:
    """Test error handling in storage cloud diagnostics components."""

    @patch.dict("os.environ", test_env)
    def test_storage_engine_with_client_errors(self):
        """Test StorageOptimizationEngine behavior with database client errors."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            from clickhouse_connect.driver.exceptions import ClickHouseError

            mock_client = Mock()
            mock_client.query.side_effect = ClickHouseError("Database connection failed")
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.storage_cloud_diagnostics import StorageOptimizationEngine

            engine = StorageOptimizationEngine(mock_client)

            # Should initialize even with potential client issues
            assert engine is not None
            assert engine.client == mock_client

    @patch.dict("os.environ", test_env)
    def test_analyzer_with_missing_events(self):
        """Test analyzers behavior with missing ProfileEvents."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import S3StorageAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            # Return empty metrics (no S3 data available)
            mock_profile_analyzer.get_events_by_category.return_value = {}

            analyzer = S3StorageAnalyzer(mock_profile_analyzer)

            # Test with missing data
            result = analyzer.analyze_s3_performance(time_range_hours=24, include_historical=True)

            # Should handle missing data gracefully
            assert result is not None
            assert isinstance(result.operation_performance, dict)
            assert isinstance(result.issues, list)
            assert isinstance(result.recommendations, list)

    @patch.dict("os.environ", test_env)
    def test_compression_analyzer_with_no_events(self):
        """Test CompressionAnalyzer behavior with no compression events."""
        with patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
            from agent_zero.monitoring.storage_cloud_diagnostics import CompressionAnalyzer

            mock_profile_analyzer = Mock(spec=ProfileEventsAnalyzer)
            # Return None for all events (no compression data)
            mock_profile_analyzer.get_event_value.return_value = None

            analyzer = CompressionAnalyzer(mock_profile_analyzer)

            # Test with no compression data
            result = analyzer.analyze_compression_performance(time_range_hours=24)

            # Should handle missing data gracefully
            assert result is not None
            assert isinstance(result.compression_efficiency, dict)
            assert isinstance(result.issues, list)
            assert isinstance(result.recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
