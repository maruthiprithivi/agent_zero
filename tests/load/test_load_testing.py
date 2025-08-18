"""Load testing suite for Agent Zero MCP Server.

This module contains comprehensive load tests to validate performance
characteristics under various load conditions and stress scenarios.
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pytest

from agent_zero.server.client import create_clickhouse_client
from agent_zero.server.core import initialize_mcp_server


class LoadTestMetrics:
    """Collect and analyze load test metrics."""

    def __init__(self):
        self.response_times: list[float] = []
        self.error_count: int = 0
        self.success_count: int = 0
        self.start_time: float = 0
        self.end_time: float = 0

    def add_response_time(self, response_time: float):
        """Add a response time measurement."""
        self.response_times.append(response_time)
        self.success_count += 1

    def add_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def start_timer(self):
        """Start the load test timer."""
        self.start_time = time.time()

    def stop_timer(self):
        """Stop the load test timer."""
        self.end_time = time.time()

    @property
    def total_duration(self) -> float:
        """Get total test duration."""
        return self.end_time - self.start_time

    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return self.success_count + self.error_count

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.success_count / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Get error rate percentage."""
        return 100.0 - self.success_rate

    @property
    def throughput(self) -> float:
        """Get requests per second."""
        if self.total_duration == 0:
            return 0.0
        return self.total_requests / self.total_duration

    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def percentile_95(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile

    @property
    def percentile_99(self) -> float:
        """Get 99th percentile response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_duration": self.total_duration,
            "throughput": self.throughput,
            "avg_response_time": self.avg_response_time,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "min_response_time": min(self.response_times) if self.response_times else 0,
            "max_response_time": max(self.response_times) if self.response_times else 0,
        }


@pytest.mark.load
class TestBasicLoadScenarios:
    """Basic load testing scenarios."""

    @pytest.fixture(scope="class")
    def clickhouse_client(self):
        """Create ClickHouse client for load testing."""
        try:
            return create_clickhouse_client()
        except Exception as e:
            pytest.skip(f"ClickHouse not available for load tests: {e}")

    @pytest.mark.slow
    def test_sustained_query_load(self, clickhouse_client):
        """Test sustained query load over extended period."""
        metrics = LoadTestMetrics()
        duration_seconds = 60  # 1 minute test
        target_qps = 10  # 10 queries per second

        metrics.start_timer()

        def execute_query_batch():
            """Execute a batch of queries."""
            batch_metrics = LoadTestMetrics()

            for _ in range(target_qps):
                try:
                    start_time = time.time()
                    result = clickhouse_client.command("SELECT 1")
                    end_time = time.time()

                    assert result == 1
                    batch_metrics.add_response_time(end_time - start_time)

                except Exception:
                    batch_metrics.add_error()

                # Maintain target QPS
                time.sleep(1.0 / target_qps)

            return batch_metrics

        # Run sustained load
        batch_count = 0
        while time.time() - metrics.start_time < duration_seconds:
            batch_metrics = execute_query_batch()

            # Aggregate metrics
            metrics.response_times.extend(batch_metrics.response_times)
            metrics.success_count += batch_metrics.success_count
            metrics.error_count += batch_metrics.error_count

            batch_count += 1

        metrics.stop_timer()

        # Validate performance requirements
        assert metrics.success_rate >= 95.0, f"Success rate too low: {metrics.success_rate:.2f}%"
        assert (
            metrics.avg_response_time <= 1.0
        ), f"Average response time too high: {metrics.avg_response_time:.3f}s"
        assert metrics.throughput >= 8.0, f"Throughput too low: {metrics.throughput:.2f} qps"

        print(f"Load Test Results: {json.dumps(metrics.to_dict(), indent=2)}")

    @pytest.mark.slow
    def test_burst_load_handling(self, clickhouse_client):
        """Test handling of burst load scenarios."""
        metrics = LoadTestMetrics()
        burst_size = 100
        concurrent_workers = 10

        def execute_burst_worker(worker_id: int) -> LoadTestMetrics:
            """Execute burst queries in a worker thread."""
            worker_metrics = LoadTestMetrics()

            for i in range(burst_size // concurrent_workers):
                try:
                    start_time = time.time()
                    result = clickhouse_client.query(
                        f"""
                        SELECT {worker_id} as worker_id, {i} as query_id, now() as timestamp
                    """
                    )
                    end_time = time.time()

                    assert len(result) == 1
                    worker_metrics.add_response_time(end_time - start_time)

                except Exception:
                    worker_metrics.add_error()

            return worker_metrics

        # Execute burst load
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(execute_burst_worker, worker_id)
                for worker_id in range(concurrent_workers)
            ]

            for future in as_completed(futures):
                worker_metrics = future.result()
                metrics.response_times.extend(worker_metrics.response_times)
                metrics.success_count += worker_metrics.success_count
                metrics.error_count += worker_metrics.error_count

        metrics.stop_timer()

        # Validate burst handling
        assert (
            metrics.success_rate >= 90.0
        ), f"Burst success rate too low: {metrics.success_rate:.2f}%"
        assert (
            metrics.percentile_95 <= 5.0
        ), f"95th percentile too high: {metrics.percentile_95:.3f}s"

        print(f"Burst Load Results: {json.dumps(metrics.to_dict(), indent=2)}")

    @pytest.mark.slow
    def test_gradual_load_ramp(self, clickhouse_client):
        """Test gradual load ramp-up and ramp-down."""
        metrics = LoadTestMetrics()
        max_qps = 20
        ramp_duration = 30  # seconds

        metrics.start_timer()

        def calculate_target_qps(elapsed_time: float) -> float:
            """Calculate target QPS based on elapsed time."""
            if elapsed_time < ramp_duration:
                # Ramp up
                return (elapsed_time / ramp_duration) * max_qps
            elif elapsed_time < ramp_duration * 2:
                # Sustain
                return max_qps
            elif elapsed_time < ramp_duration * 3:
                # Ramp down
                remaining = ramp_duration * 3 - elapsed_time
                return (remaining / ramp_duration) * max_qps
            else:
                return 0

        total_duration = ramp_duration * 3

        while time.time() - metrics.start_time < total_duration:
            elapsed = time.time() - metrics.start_time
            target_qps = calculate_target_qps(elapsed)

            if target_qps > 0:
                try:
                    start_time = time.time()
                    result = clickhouse_client.command("SELECT 1")
                    end_time = time.time()

                    assert result == 1
                    metrics.add_response_time(end_time - start_time)

                except Exception:
                    metrics.add_error()

                # Sleep to maintain target QPS
                if target_qps > 0:
                    time.sleep(max(0, 1.0 / target_qps))

        metrics.stop_timer()

        # Validate ramp behavior
        assert (
            metrics.success_rate >= 95.0
        ), f"Ramp test success rate too low: {metrics.success_rate:.2f}%"
        assert (
            metrics.avg_response_time <= 2.0
        ), f"Average response time too high: {metrics.avg_response_time:.3f}s"

        print(f"Ramp Test Results: {json.dumps(metrics.to_dict(), indent=2)}")


@pytest.mark.load
class TestMCPToolLoadTesting:
    """Load testing for MCP tool execution."""

    @pytest.fixture(scope="class")
    def mcp_server(self):
        """Initialize MCP server for load testing."""
        try:
            return initialize_mcp_server()
        except Exception as e:
            pytest.skip(f"MCP server not available for load tests: {e}")

    @pytest.mark.slow
    def test_mcp_tool_concurrent_execution(self, mcp_server):
        """Test concurrent execution of MCP tools."""
        metrics = LoadTestMetrics()
        concurrent_workers = 5
        tools_per_worker = 20

        mcp_tools = [
            ("list_databases", {}),
            ("monitor_cpu_usage", {}),
            ("monitor_memory_usage", {}),
            ("get_cluster_sizing", {}),
        ]

        def execute_mcp_tools_worker(worker_id: int) -> LoadTestMetrics:
            """Execute MCP tools in a worker thread."""
            worker_metrics = LoadTestMetrics()

            for i in range(tools_per_worker):
                tool_name, args = mcp_tools[i % len(mcp_tools)]

                try:
                    start_time = time.time()
                    result = mcp_server.call_tool(tool_name, args)
                    end_time = time.time()

                    assert result is not None
                    worker_metrics.add_response_time(end_time - start_time)

                except Exception:
                    worker_metrics.add_error()

            return worker_metrics

        # Execute concurrent MCP tool calls
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(execute_mcp_tools_worker, worker_id)
                for worker_id in range(concurrent_workers)
            ]

            for future in as_completed(futures):
                worker_metrics = future.result()
                metrics.response_times.extend(worker_metrics.response_times)
                metrics.success_count += worker_metrics.success_count
                metrics.error_count += worker_metrics.error_count

        metrics.stop_timer()

        # Validate MCP tool performance
        assert (
            metrics.success_rate >= 90.0
        ), f"MCP tool success rate too low: {metrics.success_rate:.2f}%"
        assert (
            metrics.avg_response_time <= 5.0
        ), f"MCP tool response time too high: {metrics.avg_response_time:.3f}s"

        print(f"MCP Tool Load Results: {json.dumps(metrics.to_dict(), indent=2)}")

    @pytest.mark.slow
    def test_profile_events_analysis_load(self, mcp_server):
        """Test load on ProfileEvents analysis tools."""
        metrics = LoadTestMetrics()
        concurrent_requests = 10

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        def execute_profile_analysis() -> tuple[float, bool]:
            """Execute ProfileEvents analysis."""
            try:
                start = time.time()
                result = mcp_server.call_tool(
                    "analyze_profile_events_comprehensive",
                    {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "limit": 20,
                    },
                )
                execution_time = time.time() - start

                return execution_time, result is not None
            except Exception:
                return 0.0, False

        # Execute concurrent ProfileEvents analysis
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(execute_profile_analysis) for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                execution_time, success = future.result()

                if success:
                    metrics.add_response_time(execution_time)
                else:
                    metrics.add_error()

        metrics.stop_timer()

        # Validate ProfileEvents analysis performance
        assert (
            metrics.success_rate >= 80.0
        ), f"ProfileEvents analysis success rate too low: {metrics.success_rate:.2f}%"
        assert (
            metrics.avg_response_time <= 10.0
        ), f"ProfileEvents analysis too slow: {metrics.avg_response_time:.3f}s"

        print(f"ProfileEvents Analysis Load Results: {json.dumps(metrics.to_dict(), indent=2)}")


@pytest.mark.load
class TestStressScenarios:
    """Stress testing scenarios to find breaking points."""

    @pytest.fixture(scope="class")
    def stress_client(self):
        """Create client for stress testing."""
        try:
            return create_clickhouse_client()
        except Exception as e:
            pytest.skip(f"ClickHouse not available for stress tests: {e}")

    @pytest.mark.slow
    def test_connection_exhaustion(self, stress_client):
        """Test behavior under connection exhaustion."""
        metrics = LoadTestMetrics()
        max_connections = 50

        def create_connection_worker(worker_id: int) -> LoadTestMetrics:
            """Create and hold connections to exhaust pool."""
            worker_metrics = LoadTestMetrics()

            try:
                start_time = time.time()
                # Hold connection by executing long-running query
                result = stress_client.query(
                    """
                    SELECT sleep(1), number FROM numbers(10)
                """
                )
                end_time = time.time()

                assert len(result) == 10
                worker_metrics.add_response_time(end_time - start_time)

            except Exception:
                worker_metrics.add_error()

            return worker_metrics

        # Attempt to exhaust connections
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=max_connections) as executor:
            futures = [
                executor.submit(create_connection_worker, worker_id)
                for worker_id in range(max_connections)
            ]

            for future in as_completed(futures):
                worker_metrics = future.result()
                metrics.response_times.extend(worker_metrics.response_times)
                metrics.success_count += worker_metrics.success_count
                metrics.error_count += worker_metrics.error_count

        metrics.stop_timer()

        # System should handle connection pressure gracefully
        assert (
            metrics.success_rate >= 70.0
        ), f"Connection exhaustion handling too poor: {metrics.success_rate:.2f}%"

        print(f"Connection Exhaustion Results: {json.dumps(metrics.to_dict(), indent=2)}")

    @pytest.mark.slow
    def test_memory_pressure(self, stress_client):
        """Test behavior under memory pressure."""
        metrics = LoadTestMetrics()
        large_query_count = 20

        def execute_memory_intensive_query(query_id: int) -> LoadTestMetrics:
            """Execute memory-intensive query."""
            query_metrics = LoadTestMetrics()

            try:
                start_time = time.time()
                # Generate large result set
                result = stress_client.query(
                    f"""
                    SELECT
                        {query_id} as query_id,
                        number,
                        toString(number) as str_number,
                        number * {query_id} as multiplied
                    FROM numbers(10000)
                    ORDER BY number
                """
                )
                end_time = time.time()

                assert len(result) == 10000
                query_metrics.add_response_time(end_time - start_time)

            except Exception:
                query_metrics.add_error()

            return query_metrics

        # Execute memory-intensive queries
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(execute_memory_intensive_query, query_id)
                for query_id in range(large_query_count)
            ]

            for future in as_completed(futures):
                query_metrics = future.result()
                metrics.response_times.extend(query_metrics.response_times)
                metrics.success_count += query_metrics.success_count
                metrics.error_count += query_metrics.error_count

        metrics.stop_timer()

        # Should handle memory pressure without failures
        assert (
            metrics.success_rate >= 85.0
        ), f"Memory pressure handling too poor: {metrics.success_rate:.2f}%"
        assert (
            metrics.avg_response_time <= 30.0
        ), f"Memory pressure causing excessive delays: {metrics.avg_response_time:.3f}s"

        print(f"Memory Pressure Results: {json.dumps(metrics.to_dict(), indent=2)}")


@pytest.mark.load
class TestChaosEngineering:
    """Chaos engineering tests for resilience validation."""

    @pytest.fixture(scope="class")
    def chaos_client(self):
        """Create client for chaos testing."""
        try:
            return create_clickhouse_client()
        except Exception as e:
            pytest.skip(f"ClickHouse not available for chaos tests: {e}")

    @pytest.mark.slow
    def test_intermittent_failures(self, chaos_client):
        """Test resilience to intermittent failures."""
        metrics = LoadTestMetrics()
        total_requests = 100
        failure_rate = 0.1  # 10% artificial failure rate

        metrics.start_timer()

        for i in range(total_requests):
            # Introduce artificial failures
            if i % int(1 / failure_rate) == 0:
                # Simulate failure by executing invalid query
                try:
                    chaos_client.query("SELECT * FROM non_existent_table")
                except Exception:
                    metrics.add_error()
            else:
                # Execute normal query
                try:
                    start_time = time.time()
                    result = chaos_client.command("SELECT 1")
                    end_time = time.time()

                    assert result == 1
                    metrics.add_response_time(end_time - start_time)

                except Exception:
                    metrics.add_error()

        metrics.stop_timer()

        # Should handle intermittent failures gracefully
        expected_success_rate = (1 - failure_rate) * 100
        assert (
            metrics.success_rate >= expected_success_rate - 5
        ), f"Intermittent failure handling poor: {metrics.success_rate:.2f}% vs expected {expected_success_rate:.2f}%"

        print(f"Intermittent Failures Results: {json.dumps(metrics.to_dict(), indent=2)}")

    @pytest.mark.slow
    def test_recovery_after_disruption(self, chaos_client):
        """Test recovery capabilities after service disruption."""
        metrics = LoadTestMetrics()

        # Phase 1: Normal operation
        metrics.start_timer()

        for _i in range(20):
            try:
                start_time = time.time()
                result = chaos_client.command("SELECT 1")
                end_time = time.time()

                assert result == 1
                metrics.add_response_time(end_time - start_time)

            except Exception:
                metrics.add_error()

        # Phase 2: Simulate disruption (connection issues)
        disruption_start = time.time()

        for _i in range(10):
            try:
                # Simulate network timeout by using very short timeout
                start_time = time.time()
                result = chaos_client.query("SELECT sleep(10)")  # This should timeout
                end_time = time.time()

                metrics.add_response_time(end_time - start_time)

            except Exception:
                metrics.add_error()

        disruption_duration = time.time() - disruption_start

        # Phase 3: Recovery period
        recovery_start = time.time()

        for _i in range(20):
            try:
                start_time = time.time()
                result = chaos_client.command("SELECT 1")
                end_time = time.time()

                assert result == 1
                metrics.add_response_time(end_time - start_time)

            except Exception:
                metrics.add_error()

        metrics.stop_timer()

        # Should demonstrate recovery after disruption
        recovery_success_rate = (metrics.success_count - 10) / 30 * 100  # Exclude disruption phase
        assert recovery_success_rate >= 90.0, f"Recovery too poor: {recovery_success_rate:.2f}%"

        print(f"Recovery Test Results: {json.dumps(metrics.to_dict(), indent=2)}")
        print(f"Disruption Duration: {disruption_duration:.2f}s")


def generate_load_test_report(test_results: dict[str, LoadTestMetrics]) -> str:
    """Generate comprehensive load test report."""
    report = []
    report.append("# Agent Zero Load Test Report")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")

    for test_name, metrics in test_results.items():
        report.append(f"## {test_name}")
        report.append(f"- **Total Requests**: {metrics.total_requests}")
        report.append(f"- **Success Rate**: {metrics.success_rate:.2f}%")
        report.append(f"- **Throughput**: {metrics.throughput:.2f} req/s")
        report.append(f"- **Average Response Time**: {metrics.avg_response_time:.3f}s")
        report.append(f"- **95th Percentile**: {metrics.percentile_95:.3f}s")
        report.append(f"- **99th Percentile**: {metrics.percentile_99:.3f}s")
        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # Run load tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "load", "--durations=10"])
