"""Direct unit tests for monitoring modules to achieve 90%+ coverage.

This module tests the actual monitoring functions directly rather than just
the MCP tool registration, targeting specific code paths for maximum coverage.
"""

from unittest.mock import patch

import pytest

from agent_zero.monitoring.error_analysis import (
    get_error_stack_traces,
    get_recent_errors,
    get_text_log,
)
from agent_zero.monitoring.insert_operations import (
    get_async_insert_stats,
    get_async_vs_sync_insert_counts,
)
from agent_zero.monitoring.parts_merges import (
    get_current_merges,
    get_merge_stats,
    get_part_log_events,
    get_partition_stats,
    get_parts_analysis,
)

# Import monitoring modules for direct testing
from agent_zero.monitoring.query_performance import (
    get_current_processes,
    get_normalized_query_stats,
    get_query_duration_stats,
)
from agent_zero.monitoring.resource_usage import (
    get_cpu_usage,
    get_memory_usage,
    get_server_sizing,
    get_uptime,
)
from agent_zero.monitoring.system_components import (
    get_blob_storage_stats,
)
from agent_zero.monitoring.table_statistics import (
    get_largest_tables,
    get_table_inactive_parts,
    get_table_stats,
)
from agent_zero.monitoring.utility import (
    create_monitoring_views,
    generate_drop_tables_script,
)


@pytest.mark.unit
class TestQueryPerformanceMonitoring:
    """Test query performance monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_query_performance_testing(self, comprehensive_mock_client):
        """Set up query performance testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_current_processes_success(self):
        """Test successful current processes retrieval."""
        # Setup realistic process data
        self.mock_client.query.return_value.result_rows = [
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
                "host1",
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
        self.mock_client.query.return_value.column_names = [
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

        result = get_current_processes(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 2
        # Verify client was called with correct query
        self.mock_client.query.assert_called_once()

    def test_get_current_processes_empty_result(self):
        """Test current processes with empty result."""
        self.mock_client.query.return_value.result_rows = []
        self.mock_client.query.return_value.column_names = [
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

        result = get_current_processes(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_current_processes_error_handling(self):
        """Test error handling in get_current_processes."""
        self.mock_client.query.side_effect = Exception("Connection failed")

        with pytest.raises((RuntimeError, ValueError, Exception)):
            get_current_processes(self.mock_client)

    def test_get_query_duration_stats_with_filter(self):
        """Test query duration statistics with query kind filter."""
        # Setup query duration data
        self.mock_client.query.return_value.result_rows = [
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
            ],
            [
                "2024-03-10 13:00:00",
                8,
                200,
                3.3,
                600,
                1500,
                4000,
                "600.00 thousand",
                "1.50 million",
                "6.00 million",
                "60.00 MiB",
                "150.00 MiB",
                "600.00 MiB",
                "3.00 GiB",
                300,
                750,
                2000,
                20000,
            ],
        ]
        self.mock_client.query.return_value.column_names = [
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

        result = get_query_duration_stats(self.mock_client, query_kind="Select", days=7)

        assert isinstance(result, list)
        assert len(result) == 2

        # Verify parameters were used
        call_args = self.mock_client.query.call_args[0][0]
        assert "Select" in call_args and "7" in call_args

    def test_get_normalized_query_stats_with_limit(self):
        """Test normalized query statistics with limit parameter."""
        # Setup normalized query data
        self.mock_client.query.return_value.result_rows = [
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
            ],
            [
                "hash456",
                "Insert",
                25,
                3,
                "250.00 thousand",
                "600.00 thousand",
                "2.50 million",
                800,
                1500,
                4000,
                "5.00 MiB",
                "25.00 MiB",
                "60.00 MiB",
                "250.00 MiB",
                "12.50 GiB",
                400,
                900,
                2500,
                25000,
                "query-max-2",
                "INSERT INTO events SELECT * FROM source_table",
            ],
        ]
        self.mock_client.query.return_value.column_names = [
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

        result = get_normalized_query_stats(self.mock_client, days=2, limit=50)

        assert isinstance(result, list)
        assert len(result) == 2

        # Verify parameters were used
        call_args = self.mock_client.query.call_args[0][0]
        assert "2" in call_args and "50" in call_args


@pytest.mark.unit
class TestResourceUsageMonitoring:
    """Test resource usage monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_resource_testing(self, comprehensive_mock_client):
        """Set up resource usage testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_cpu_usage_comprehensive(self):
        """Test comprehensive CPU usage retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", 2.5, 16, 0.16],
            ["2024-03-10 12:01:00", 3.2, 16, 0.20],
            ["2024-03-10 12:02:00", 1.8, 16, 0.11],
        ]
        self.mock_client.query.return_value.column_names = [
            "dt",
            "cpu_usage_cluster",
            "cpu_cores_cluster",
            "cpu_usage_cluster_normalized",
        ]

        result = get_cpu_usage(self.mock_client, hours=3)

        assert isinstance(result, list)
        assert len(result) == 3

        self.mock_client.query.assert_called_once()

    def test_get_memory_usage_detailed(self):
        """Test detailed memory usage retrieval."""
        self.mock_client.query.return_value.result_rows = [
            [
                "2024-03-10 12:00:00",
                "host1",
                "32.00 GiB",
                "30.00 GiB",
                "34.00 GiB",
                "35.00 GiB",
                "36.00 GiB",
            ],
            [
                "2024-03-10 13:00:00",
                "host1",
                "33.00 GiB",
                "31.00 GiB",
                "35.00 GiB",
                "36.00 GiB",
                "37.00 GiB",
            ],
        ]
        self.mock_client.query.return_value.column_names = [
            "ts",
            "hostname_",
            "MemoryTracking_avg",
            "MemoryTracking_p50",
            "MemoryTracking_p90",
            "MemoryTracking_p99",
            "MemoryTracking_max",
        ]

        result = get_memory_usage(self.mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 2

        self.mock_client.query.assert_called_once()

    def test_get_server_sizing_multiple_servers(self):
        """Test server sizing for multiple servers."""
        self.mock_client.query.return_value.result_rows = [
            ["server1.example.com", 16, "64.00 GiB"],
            ["server2.example.com", 32, "128.00 GiB"],
            ["server3.example.com", 8, "32.00 GiB"],
        ]
        self.mock_client.query.return_value.column_names = ["server", "cpu_cores", "memory"]

        result = get_server_sizing(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 3

        self.mock_client.query.assert_called_once()

    def test_get_uptime_stats(self):
        """Test uptime statistics retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", "2024-03-10 12:00:00", "2024-03-10 12:59:59", 3600],
            ["2024-03-10 13:00:00", "2024-03-10 13:00:00", "2024-03-10 13:59:59", 7200],
        ]
        self.mock_client.query.return_value.column_names = [
            "ts",
            "min_event_time",
            "max_event_time",
            "uptime",
        ]

        result = get_uptime(self.mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 2

        self.mock_client.query.assert_called_once()


@pytest.mark.unit
class TestPartsMergesMonitoring:
    """Test parts and merges monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_parts_merges_testing(self, comprehensive_mock_client):
        """Set up parts and merges testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_current_merges_active(self):
        """Test active merges retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["default", "events_local", 30.5, 0.75, "REGULAR"],
            ["logs", "access_log", 15.2, 0.45, "MUTATE"],
        ]
        self.mock_client.query.return_value.column_names = [
            "database",
            "table",
            "elapsed",
            "progress",
            "merge_type",
        ]

        result = get_current_merges(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["database"] == "default"
        assert result[0]["table"] == "events_local"
        assert result[0]["elapsed"] == 30.5
        assert result[0]["progress"] == 0.75

        self.mock_client.query.assert_called_once()

    def test_get_current_merges_no_active_merges(self):
        """Test when no merges are active."""
        self.mock_client.query.return_value.result_rows = []
        self.mock_client.query.return_value.column_names = [
            "database",
            "table",
            "elapsed",
            "progress",
            "merge_type",
        ]

        result = get_current_merges(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_merge_stats_with_days_filter(self):
        """Test merge statistics with days filter."""
        self.mock_client.query.return_value.result_rows = [
            ["default", "events_local", 50, 25.5, 200],
            ["logs", "access_log", 10, 5.2, 30],
        ]
        self.mock_client.query.return_value.column_names = [
            "database",
            "table",
            "merges_count",
            "avg_duration",
            "total_merged_parts",
        ]

        result = get_merge_stats(self.mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["merges_count"] == 50
        assert result[0]["avg_duration"] == 25.5

        # Verify days parameter was used
        call_args = self.mock_client.query.call_args[0][0]
        assert "7" in call_args

    def test_get_part_log_events_with_filters(self):
        """Test part log events with days and limit filters."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", "NEW_PART", "default", "events_local", "20240310_1_1_0"],
            ["2024-03-10 12:01:00", "MERGE_PARTS", "default", "events_local", "20240310_1_2_1"],
        ]
        self.mock_client.query.return_value.column_names = [
            "event_time",
            "event_type",
            "database",
            "table",
            "part_name",
        ]

        result = get_part_log_events(self.mock_client, days=1, limit=100)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["event_type"] == "NEW_PART"
        assert result[1]["event_type"] == "MERGE_PARTS"

        # Verify parameters were used
        call_args = self.mock_client.query.call_args[0][0]
        assert "1" in call_args and "100" in call_args

    def test_get_partition_stats_for_table(self):
        """Test partition statistics for specific table."""
        self.mock_client.query.return_value.result_rows = [
            ["202403", 25, 1000000, 104857600, 10.5],
            ["202404", 30, 1200000, 125829120, 12.0],
            ["202405", 45, 1800000, 188743680, 15.2],
        ]
        self.mock_client.query.return_value.column_names = [
            "partition",
            "parts_count",
            "rows",
            "bytes",
            "compression_ratio",
        ]

        result = get_partition_stats(self.mock_client, "default", "events_local")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["partition"] == "202403"
        assert result[0]["parts_count"] == 25
        assert result[1]["compression_ratio"] == 12.0

        # Verify table parameters were used
        call_args = self.mock_client.query.call_args[0][0]
        assert "default" in call_args and "events_local" in call_args

    def test_get_parts_analysis_comprehensive(self):
        """Test comprehensive parts analysis."""
        self.mock_client.query.return_value.result_rows = [
            ["202403", "events_local", 25, 23, 2, 262144000, 13.4],
            ["202404", "events_local", 30, 28, 2, 314572800, 10.5],
        ]
        self.mock_client.query.return_value.column_names = [
            "partition",
            "table",
            "total_parts",
            "active_parts",
            "inactive_parts",
            "bytes",
            "compression_ratio",
        ]

        result = get_parts_analysis(self.mock_client, "default", "events_local")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["partition"] == "202403"
        assert result[0]["active_parts"] == 23
        assert result[0]["inactive_parts"] == 2


@pytest.mark.unit
class TestInsertOperationsMonitoring:
    """Test insert operations monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_insert_operations_testing(self, comprehensive_mock_client):
        """Set up insert operations testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_async_insert_stats(self):
        """Test async insert statistics retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["async_inserts_total", 15420],
            ["async_inserts_successful", 15250],
            ["async_inserts_failed", 170],
            ["async_insert_bytes", 1073741824],  # 1GB
            ["async_insert_duration_avg", 0.85],
        ]
        self.mock_client.query.return_value.column_names = ["metric", "value"]

        result = get_async_insert_stats(self.mock_client)

        assert isinstance(result, dict)
        assert result["async_inserts_total"] == 15420
        assert result["async_inserts_successful"] == 15250
        assert result["async_inserts_failed"] == 170
        assert result["success_rate"] == pytest.approx(98.9, rel=1e-1)  # 15250/15420

    def test_get_async_vs_sync_insert_counts(self):
        """Test async vs sync insert comparison."""
        self.mock_client.query.return_value.result_rows = [["async", 15420], ["sync", 2680]]
        self.mock_client.query.return_value.column_names = ["insert_type", "count"]

        result = get_async_vs_sync_insert_counts(self.mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.unit
class TestSystemComponentsMonitoring:
    """Test system components monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_system_components_testing(self, comprehensive_mock_client):
        """Set up system components testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_blob_storage_stats(self):
        """Test blob storage statistics retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["s3", "primary-bucket", 1099511627776, 549755813888, "active"],  # 1TB, 50% used
            [
                "azure_blob",
                "backup-container",
                549755813888,
                164926744166,
                "active",
            ],  # 512GB, 30% used
            ["gcs", "analytics-bucket", 274877906944, 82463372083, "active"],  # 256GB, 30% used
        ]
        self.mock_client.query.return_value.column_names = [
            "provider",
            "container",
            "total_bytes",
            "used_bytes",
            "status",
        ]

        result = get_blob_storage_stats(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["provider"] == "s3"
        assert result[0]["container"] == "primary-bucket"
        assert result[0]["usage_percentage"] == 50.0


@pytest.mark.unit
class TestTableStatisticsMonitoring:
    """Test table statistics monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_table_stats_testing(self, comprehensive_mock_client):
        """Set up table statistics testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_table_stats(self):
        """Test table statistics retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["events_local", "default", 10000000, 1073741824, 100, 95],
            ["query_log", "system", 500000, 52428800, 20, 18],
            ["access_log", "logs", 250000, 26214400, 1, 1],
        ]
        self.mock_client.query.return_value.column_names = [
            "table",
            "database",
            "rows",
            "bytes",
            "parts",
            "active_parts",
        ]

        result = get_table_stats(self.mock_client)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_get_largest_tables(self):
        """Test largest tables retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["events_local", "default", "5.00 GiB", "10.00 million"],
            ["query_log", "system", "150.00 MiB", "500.00 thousand"],
            ["access_log", "logs", "25.00 MiB", "250.00 thousand"],
        ]
        self.mock_client.query.return_value.column_names = ["table", "database", "size", "rows"]

        result = get_largest_tables(self.mock_client, limit=10)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_get_table_inactive_parts(self):
        """Test table inactive parts retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["events_local", "default", 5, "2024-03-01"],
            ["query_log", "system", 2, "2024-02-15"],
            ["access_log", "logs", 0, "2024-03-10"],
        ]
        self.mock_client.query.return_value.column_names = [
            "table",
            "database",
            "inactive_parts",
            "oldest_inactive_date",
        ]

        result = get_table_inactive_parts(self.mock_client, database="default")

        assert isinstance(result, list)
        assert len(result) == 3


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions directly."""

    @pytest.fixture(autouse=True)
    def setup_utility_testing(self, comprehensive_mock_client):
        """Set up utility testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_generate_drop_tables_script(self):
        """Test drop tables script generation."""
        self.mock_client.query.return_value.result_rows = [
            ["temp_table_1", "default"],
            ["temp_table_2", "default"],
            ["old_log_table", "logs"],
        ]
        self.mock_client.query.return_value.column_names = ["table", "database"]

        result = generate_drop_tables_script(self.mock_client, database="default")

        assert isinstance(result, list)
        assert len(result) >= 2

    def test_create_monitoring_views(self):
        """Test monitoring views creation."""
        self.mock_client.command.return_value = "OK"

        result = create_monitoring_views(self.mock_client)

        assert isinstance(result, bool)
        assert result is True


@pytest.mark.unit
class TestErrorAnalysisMonitoring:
    """Test error analysis monitoring functions directly."""

    @pytest.fixture(autouse=True)
    def setup_error_analysis_testing(self, comprehensive_mock_client):
        """Set up error analysis testing environment."""
        self.mock_client = comprehensive_mock_client

        # Mock environment variables to prevent config errors
        with patch.dict(
            "os.environ",
            {
                "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
                "AGENT_ZERO_CLICKHOUSE_USER": "default",
                "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
                "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
            },
        ):
            yield

    def test_get_recent_errors(self):
        """Test recent errors retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", 62, "Syntax error", "SELET * FROM table", "default"],
            [
                "2024-03-10 12:01:00",
                241,
                "Memory limit exceeded",
                "SELECT * FROM huge_table",
                "admin",
            ],
            [
                "2024-03-10 12:02:00",
                1001,
                "Connection timeout",
                "INSERT INTO remote_table",
                "user1",
            ],
        ]
        self.mock_client.query.return_value.column_names = [
            "event_time",
            "code",
            "error",
            "query",
            "user",
        ]

        result = get_recent_errors(self.mock_client, hours=24)

        assert isinstance(result, list)
        assert len(result) == 3

        # Verify hours parameter was used
        call_args = self.mock_client.query.call_args[0][0]
        assert "24" in call_args

    def test_get_error_stack_traces(self):
        """Test error stack traces retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", 62, "Syntax error", "Stack trace line 1\nStack trace line 2"],
            [
                "2024-03-10 12:01:00",
                241,
                "Memory limit exceeded",
                "Stack trace line 1\nStack trace line 2",
            ],
        ]
        self.mock_client.query.return_value.column_names = [
            "event_time",
            "code",
            "error",
            "stack_trace",
        ]

        result = get_error_stack_traces(self.mock_client, days=7)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_text_log(self):
        """Test text log retrieval."""
        self.mock_client.query.return_value.result_rows = [
            ["2024-03-10 12:00:00", "Information", "Server started", "main.cpp:123"],
            ["2024-03-10 12:01:00", "Warning", "High memory usage", "memory.cpp:456"],
        ]
        self.mock_client.query.return_value.column_names = [
            "event_time",
            "level",
            "message",
            "source_location",
        ]

        result = get_text_log(self.mock_client, hours=24)

        assert isinstance(result, list)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
