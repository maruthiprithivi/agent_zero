"""Simplified unit tests for monitoring functions to achieve 90%+ coverage.

This module tests monitoring functions with direct mocking for better control
over test data and easier debugging.
"""

from unittest.mock import Mock, patch

import pytest

# Set up environment variables before imports
test_env = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
    "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
}

# Import monitoring functions
from agent_zero.monitoring.parts_merges import (
    get_current_merges,
    get_merge_stats,
    get_part_log_events,
    get_partition_stats,
    get_parts_analysis,
)
from agent_zero.monitoring.query_performance import (
    get_current_processes,
    get_normalized_query_stats,
    get_query_duration_stats,
    get_query_kind_breakdown,
)
from agent_zero.monitoring.resource_usage import (
    get_cpu_usage,
    get_memory_usage,
    get_server_sizing,
    get_uptime,
)


@pytest.mark.unit
class TestQueryPerformanceSimple:
    """Simple direct tests for query performance functions."""

    @patch.dict("os.environ", test_env)
    def test_get_current_processes(self):
        """Test get_current_processes function."""
        # Setup mock client with proper query result structure
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "host1",
                "default",
                "Select",
                1.5,
                "1.05 million",
                "1000.00 MiB",
                "1.00 million",
                "0.00",
                "0.00",
                "1000.00 MiB",
                "1000.00 MiB",
                "query-1",
                "hash1",
                "SELECT count() FROM events",
            ],
            [
                "host2",
                "admin",
                "Insert",
                0.8,
                "524.29 thousand",
                "500.00 MiB",
                "0.00",
                "1000.00",
                "10.00 MiB",
                "500.00 MiB",
                "500.00 MiB",
                "query-2",
                "hash2",
                "INSERT INTO logs VALUES",
            ],
        ]
        mock_result.column_names = [
            "hostname",
            "user",
            "query_kind",
            "elapsed",
            "read_rows_",
            "read_bytes_",
            "total_rows_approx_",
            "written_rows_",
            "written_bytes_",
            "memory_usage_",
            "peak_memory_usage_",
            "query_id",
            "query_hash",
            "query",
        ]
        mock_client.query.return_value = mock_result

        result = get_current_processes(mock_client)

        assert isinstance(result, list)
        assert len(result) == 2
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_query_duration_stats(self):
        """Test get_query_duration_stats function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "2024-03-10 12:00:00",
                5,
                150,
                2.5,
                500,
                1200,
                3000,
                "500.00 thousand",
                "1.20 million",
                "5.00 million",
                "50.00 MiB",
                "120.00 MiB",
                "500.00 MiB",
                "2.50 GiB",
                250,
                600,
                1500,
                15000,
                400,
                900,
                2500,
            ]
        ]
        mock_result.column_names = [
            "ts",
            "query_uniq",
            "query_count",
            "qps",
            "time_p50",
            "time_p90",
            "time_max",
            "read_rows_p50",
            "read_rows_p90",
            "read_rows_max",
            "written_rows_p50",
            "written_rows_p90",
            "written_rows_max",
            "mem_p50",
            "mem_p90",
            "mem_max",
            "mem_sum",
            "cpu_p50",
            "cpu_p90",
            "cpu_max",
            "cpu_sum",
        ]
        mock_client.query.return_value = mock_result

        result = get_query_duration_stats(mock_client, query_kind="Select", days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_normalized_query_stats(self):
        """Test get_normalized_query_stats function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "hash123",
                "Select",
                50,
                5,
                "500.00 thousand",
                "1.20 million",
                "5.00 million",
                1500,
                3000,
                8000,
                "10.00 MiB",
                "50.00 MiB",
                "120.00 MiB",
                "500.00 MiB",
                "25.00 GiB",
                750,
                1800,
                5000,
                50000,
                "query-max-1",
                "SELECT * FROM large_table WHERE date > '2024-01-01'",
            ]
        ]
        mock_result.column_names = [
            "normalized_query_hash",
            "query_kind",
            "q_count",
            "q_distinct",
            "read_rows_p50",
            "read_rows_p90",
            "read_rows_max",
            "time_p50",
            "time_p90",
            "time_max",
            "mem_min",
            "mem_p50",
            "mem_p90",
            "mem_max",
            "mem_sum",
            "cpu_p50",
            "cpu_p90",
            "cpu_max",
            "cpu_sum",
            "query_id_of_time_max",
            "query_example",
        ]
        mock_client.query.return_value = mock_result

        result = get_normalized_query_stats(mock_client, days=2, limit=50)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_query_kind_breakdown(self):
        """Test get_query_kind_breakdown function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            ["2024-03-10 12:00:00", 50, 5, 100, 0, 2, 1, 10, 3, 1, 0, 5, 8, 2, 1]
        ]
        mock_result.column_names = [
            "ts",
            "Insert",
            "AsyncInsertFlush",
            "Select",
            "KillQuery",
            "Create",
            "Drop",
            "Show",
            "Describe",
            "Explain",
            "Backup",
            "System",
            "Alter",
            "Delete",
            "Optimize",
        ]
        mock_client.query.return_value = mock_result

        result = get_query_kind_breakdown(mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()


@pytest.mark.unit
class TestResourceUsageSimple:
    """Simple direct tests for resource usage functions."""

    @patch.dict("os.environ", test_env)
    def test_get_memory_usage(self):
        """Test get_memory_usage function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "2024-03-10 12:00:00",
                "host1",
                "32.00 GiB",
                "30.00 GiB",
                "34.00 GiB",
                "35.00 GiB",
                "36.00 GiB",
            ]
        ]
        mock_result.column_names = [
            "ts",
            "hostname_",
            "MemoryTracking_avg",
            "MemoryTracking_p50",
            "MemoryTracking_p90",
            "MemoryTracking_p99",
            "MemoryTracking_max",
        ]
        mock_client.query.return_value = mock_result

        result = get_memory_usage(mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_cpu_usage(self):
        """Test get_cpu_usage function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [["2024-03-10 12:00:00", 2.5, 16, 0.16]]
        mock_result.column_names = [
            "dt",
            "cpu_usage_cluster",
            "cpu_cores_cluster",
            "cpu_usage_cluster_normalized",
        ]
        mock_client.query.return_value = mock_result

        result = get_cpu_usage(mock_client, hours=3)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_server_sizing(self):
        """Test get_server_sizing function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [["server1.example.com", 16, "64.00 GiB"]]
        mock_result.column_names = ["server", "cpu_cores", "memory"]
        mock_client.query.return_value = mock_result

        result = get_server_sizing(mock_client)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_uptime(self):
        """Test get_uptime function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            ["2024-03-10 12:00:00", "2024-03-10 12:00:00", "2024-03-10 12:59:59", 3600]
        ]
        mock_result.column_names = ["ts", "min_event_time", "max_event_time", "uptime"]
        mock_client.query.return_value = mock_result

        result = get_uptime(mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()


@pytest.mark.unit
class TestPartsMergesSimple:
    """Simple direct tests for parts and merges functions."""

    @patch.dict("os.environ", test_env)
    def test_get_current_merges(self):
        """Test get_current_merges function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "host1",
                "default",
                "events_local",
                30,
                0.75,
                "20240310_1_2_1",
                2,
                "1.00 million",
                "950.00 thousand",
                "REGULAR",
                "Horizontal",
            ]
        ]
        mock_result.column_names = [
            "hostname",
            "database",
            "table",
            "elapsed_sec",
            "progress_",
            "result_part_name",
            "source_part_count",
            "rows_read_",
            "rows_written_",
            "merge_type",
            "merge_algorithm",
        ]
        mock_client.query.return_value = mock_result

        result = get_current_merges(mock_client)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_merge_stats(self):
        """Test get_merge_stats function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "2024-03-10 12:00:00",
                50,
                30,
                20,
                35,
                15,
                3,
                241,
                1200,
                2500,
                5000,
                "500.00 thousand",
                "1.20 million",
                "5.00 million",
                "50.00 MiB",
                "120.00 MiB",
                "500.00 MiB",
                "100.00 MiB",
                "250.00 MiB",
                "1.00 GiB",
                "50.00 GiB",
            ]
        ]
        mock_result.column_names = [
            "ts",
            "total_merges",
            "num_Wide",
            "num_Compact",
            "num_Horizontal",
            "num_Vertical",
            "num_Error",
            "errorCode",
            "duration_ms_p50",
            "duration_ms_p90",
            "duration_ms_max",
            "rows_p50",
            "rows_p90",
            "rows_max",
            "size_in_bytes_p50",
            "size_in_bytes_p90",
            "size_in_bytes_max",
            "peak_memory_usage_p50",
            "peak_memory_usage_p90",
            "peak_memory_usage_max",
            "peak_memory_usage_sum",
        ]
        mock_client.query.return_value = mock_result

        result = get_merge_stats(mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_part_log_events(self):
        """Test get_part_log_events function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                "2024-03-10 12:00:00",
                10,
                5,
                2,
                3,
                1,
                4,
                1,
                2,
                0,
                800,
                1500,
                3000,
                "100.00 thousand",
                "500.00 thousand",
                "2.00 million",
                "10.00 MiB",
                "50.00 MiB",
                "200.00 MiB",
                "20.00 MiB",
                "100.00 MiB",
                "500.00 MiB",
                "10.00 GiB",
            ]
        ]
        mock_result.column_names = [
            "ts",
            "NewPart",
            "MergeParts",
            "DownloadPart",
            "RemovePart",
            "MutatePart",
            "MergePartsStart",
            "MutatePartStart",
            "num_Error",
            "errorCode",
            "duration_ms_p50",
            "duration_ms_p90",
            "duration_ms_max",
            "rows_p50",
            "rows_p90",
            "rows_max",
            "size_in_bytes_p50",
            "size_in_bytes_p90",
            "size_in_bytes_max",
            "peak_memory_usage_p50",
            "peak_memory_usage_p90",
            "peak_memory_usage_max",
            "peak_memory_usage_sum",
        ]
        mock_client.query.return_value = mock_result

        result = get_part_log_events(mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_partition_stats(self):
        """Test get_partition_stats function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [["202403", 25, "1.00 million", "500.00 MiB"]]
        mock_result.column_names = ["partition", "part_count", "rows_total", "size_total"]
        mock_client.query.return_value = mock_result

        result = get_partition_stats(mock_client, "default", "events_local")

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()

    @patch.dict("os.environ", test_env)
    def test_get_parts_analysis(self):
        """Test get_parts_analysis function."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.result_rows = [
            [
                0,
                "100.00 thousand",
                "10.00 MiB",
                "500.00 thousand",
                "50.00 MiB",
                "2.00 million",
                "200.00 MiB",
                "Wide",
                "Local",
                25,
            ]
        ]
        mock_result.column_names = [
            "level",
            "rows_min",
            "bytes_min",
            "rows_median",
            "bytes_median",
            "rows_max",
            "bytes_max",
            "part_type",
            "part_storage_type",
            "c",
        ]
        mock_client.query.return_value = mock_result

        result = get_parts_analysis(mock_client, "default", "events_local")

        assert isinstance(result, list)
        assert len(result) == 1
        mock_client.query.assert_called_once()


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in monitoring functions."""

    @patch.dict("os.environ", test_env)
    def test_get_current_processes_error(self):
        """Test error handling in get_current_processes."""
        from clickhouse_connect.driver.exceptions import ClickHouseError

        mock_client = Mock()

        # First call fails, second call succeeds (fallback)
        mock_result_success = Mock()
        mock_result_success.result_rows = [
            [
                "host1",
                "default",
                "Select",
                1.5,
                "1.05 million",
                "1000.00 MiB",
                "1.00 million",
                "0.00",
                "0.00",
                "1000.00 MiB",
                "1000.00 MiB",
                "query-1",
                "hash1",
                "SELECT count() FROM events",
            ]
        ]
        mock_result_success.column_names = [
            "hostname",
            "user",
            "query_kind",
            "elapsed",
            "read_rows_",
            "read_bytes_",
            "total_rows_approx_",
            "written_rows_",
            "written_bytes_",
            "memory_usage_",
            "peak_memory_usage_",
            "query_id",
            "query_hash",
            "query",
        ]

        mock_client.query.side_effect = [
            ClickHouseError("Cluster query failed"),
            mock_result_success,
        ]

        result = get_current_processes(mock_client)

        assert isinstance(result, list)
        assert len(result) == 1
        assert mock_client.query.call_count == 2  # First call failed, second succeeded

    @patch.dict("os.environ", test_env)
    def test_get_cpu_usage_error(self):
        """Test error handling in get_cpu_usage."""
        from clickhouse_connect.driver.exceptions import ClickHouseError

        mock_client = Mock()
        mock_client.query.side_effect = ClickHouseError("Complex query failed")

        result = get_cpu_usage(mock_client, hours=3)

        # Should return empty list on error
        assert isinstance(result, list)
        assert len(result) == 0
        # CPU usage function has retry logic so it may be called multiple times
        assert mock_client.query.call_count >= 1
