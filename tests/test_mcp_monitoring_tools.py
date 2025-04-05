"""Tests for all monitoring tools in mcp_server.py."""

from unittest.mock import MagicMock, patch

import pytest
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.mcp_server import (
    get_cluster_sizing,
    # Resource Usage
    monitor_cpu_usage,
    # Query Performance
    monitor_current_processes,
    monitor_error_stack_traces,
    monitor_memory_usage,
    monitor_query_duration,
    monitor_query_patterns,
    monitor_query_types,
    # Error Analysis
    monitor_recent_errors,
    monitor_uptime,
    view_text_log,
)


class TestMCPMonitoringTools:
    """Tests for all monitoring tools in mcp_server.py."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        """Set up and tear down test fixtures."""
        self.mock_client = MagicMock(spec=Client)

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

        yield  # This is where the test runs

        # Cleanup
        self.client_patcher.stop()

    # Query Performance Tests

    def test_monitor_current_processes(self):
        """Test monitoring current processes."""
        with patch("agent_zero.mcp_server.get_current_processes") as mock_function:
            # Mock successful execution
            mock_data = [
                {"hostname": "host1", "query_id": "1", "query_kind": "SELECT", "elapsed": 10},
                {"hostname": "host2", "query_id": "2", "query_kind": "INSERT", "elapsed": 20},
            ]
            mock_function.return_value = mock_data
            result = monitor_current_processes()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_current_processes()
            assert isinstance(result, str)
            assert "Error monitoring current processes" in result

    def test_monitor_query_duration(self):
        """Test monitoring query duration."""
        with patch("agent_zero.mcp_server.get_query_duration_stats") as mock_function:
            # Mock successful execution
            mock_data = [
                {
                    "hour": "2024-03-15 10:00:00",
                    "query_kind": "SELECT",
                    "count": 100,
                    "avg_duration": 0.5,
                },
                {
                    "hour": "2024-03-15 11:00:00",
                    "query_kind": "SELECT",
                    "count": 200,
                    "avg_duration": 0.6,
                },
            ]
            mock_function.return_value = mock_data
            result = monitor_query_duration()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, None, 7)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_query_duration(query_kind="INSERT", days=3)
            mock_function.assert_called_once_with(self.mock_client, "INSERT", 3)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_query_duration()
            assert isinstance(result, str)
            assert "Error monitoring query duration" in result

    def test_monitor_query_patterns(self):
        """Test monitoring query patterns."""
        with patch("agent_zero.mcp_server.get_normalized_query_stats") as mock_function:
            # Mock successful execution
            mock_data = [
                {
                    "normalized_query_hash": 123,
                    "query_pattern": "SELECT * FROM table",
                    "count": 100,
                },
                {"normalized_query_hash": 456, "query_pattern": "INSERT INTO table", "count": 50},
            ]
            mock_function.return_value = mock_data
            result = monitor_query_patterns()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 2, 50)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_query_patterns(days=5, limit=100)
            mock_function.assert_called_once_with(self.mock_client, 5, 100)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_query_patterns()
            assert isinstance(result, str)
            assert "Error monitoring query patterns" in result

    def test_monitor_query_types(self):
        """Test monitoring query types."""
        with patch("agent_zero.mcp_server.get_query_kind_breakdown") as mock_function:
            # Mock successful execution
            mock_data = [
                {"hour": "2024-03-15 10:00:00", "query_kind": "SELECT", "count": 100},
                {"hour": "2024-03-15 10:00:00", "query_kind": "INSERT", "count": 50},
            ]
            mock_function.return_value = mock_data
            result = monitor_query_types()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_query_types(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_query_types()
            assert isinstance(result, str)
            assert "Error monitoring query types" in result

    # Resource Usage Tests

    def test_monitor_memory_usage(self):
        """Test monitoring memory usage."""
        with patch("agent_zero.mcp_server.get_memory_usage") as mock_function:
            # Mock successful execution
            mock_data = [
                {"ts": "2024-03-15 10:00:00", "hostname": "host1", "memory_usage": 1024000000},
                {"ts": "2024-03-15 10:00:00", "hostname": "host2", "memory_usage": 2048000000},
            ]
            mock_function.return_value = mock_data
            result = monitor_memory_usage()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_memory_usage(days=30)
            mock_function.assert_called_once_with(self.mock_client, 30)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_memory_usage()
            assert isinstance(result, str)
            assert "Error monitoring memory usage" in result

    def test_monitor_cpu_usage(self):
        """Test monitoring CPU usage."""
        with patch("agent_zero.mcp_server.get_cpu_usage") as mock_function:
            # Mock successful execution
            mock_data = [
                {"dt": "2024-03-15 10:00:00", "cpu_usage_cluster": 50, "cpu_cores_cluster": 100},
                {"dt": "2024-03-15 11:00:00", "cpu_usage_cluster": 60, "cpu_cores_cluster": 100},
            ]
            mock_function.return_value = mock_data
            result = monitor_cpu_usage()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 3)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_cpu_usage(hours=6)
            mock_function.assert_called_once_with(self.mock_client, 6)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_cpu_usage()
            assert isinstance(result, str)
            assert "Error monitoring CPU usage" in result

    def test_get_cluster_sizing(self):
        """Test getting cluster sizing information."""
        with patch("agent_zero.mcp_server.get_server_sizing") as mock_function:
            # Mock successful execution
            mock_data = [
                {"hostname": "host1", "cpu_cores": 32, "total_memory": 128000000000},
                {"hostname": "host2", "cpu_cores": 64, "total_memory": 256000000000},
            ]
            mock_function.return_value = mock_data
            result = get_cluster_sizing()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = get_cluster_sizing()
            assert isinstance(result, str)
            assert "Error getting cluster sizing" in result

    def test_monitor_uptime(self):
        """Test monitoring uptime."""
        with patch("agent_zero.mcp_server.get_uptime") as mock_function:
            # Mock successful execution
            mock_data = [
                {"ts": "2024-03-15 00:00:00", "uptime": 86400},
                {"ts": "2024-03-16 00:00:00", "uptime": 172800},
            ]
            mock_function.return_value = mock_data
            result = monitor_uptime()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_uptime(days=30)
            mock_function.assert_called_once_with(self.mock_client, 30)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_uptime()
            assert isinstance(result, str)
            assert "Error monitoring uptime" in result

    # Error Analysis Tests

    def test_monitor_recent_errors(self):
        """Test monitoring recent errors."""
        with patch("agent_zero.mcp_server.get_recent_errors") as mock_function:
            # Mock successful execution
            mock_data = [
                {"ts": "2024-03-15 10:00:00", "error_code": 123, "error_count": 5},
                {"ts": "2024-03-15 11:00:00", "error_code": 456, "error_count": 3},
            ]
            mock_function.return_value = mock_data
            result = monitor_recent_errors()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 1)

            # Test with parameters
            mock_function.reset_mock()
            result = monitor_recent_errors(days=3)
            mock_function.assert_called_once_with(self.mock_client, 3)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_recent_errors()
            assert isinstance(result, str)
            assert "Error monitoring recent errors" in result

    def test_monitor_error_stack_traces(self):
        """Test monitoring error stack traces."""
        with patch("agent_zero.mcp_server.get_error_stack_traces") as mock_function:
            # Mock successful execution
            mock_data = [
                {
                    "ts": "2024-03-15 10:00:00",
                    "error_code": 123,
                    "stack_trace": "Error at line 1",
                },
                {
                    "ts": "2024-03-15 11:00:00",
                    "error_code": 456,
                    "stack_trace": "Error at line 2",
                },
            ]
            mock_function.return_value = mock_data
            result = monitor_error_stack_traces()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = monitor_error_stack_traces()
            assert isinstance(result, str)
            assert "Error monitoring error stack traces" in result

    def test_view_text_log(self):
        """Test viewing text log."""
        with patch("agent_zero.mcp_server.get_text_log") as mock_function:
            # Mock successful execution
            mock_data = [
                {"ts": "2024-03-15 10:00:00", "level": "ERROR", "message": "Test error"},
                {"ts": "2024-03-15 11:00:00", "level": "INFO", "message": "Test info"},
            ]
            mock_function.return_value = mock_data
            result = view_text_log()
            assert result == mock_data
            mock_function.assert_called_once_with(self.mock_client, 100)

            # Test with parameters
            mock_function.reset_mock()
            result = view_text_log(limit=50)
            mock_function.assert_called_once_with(self.mock_client, 50)

            # Test error handling
            mock_function.side_effect = ClickHouseError("Test exception")
            result = view_text_log()
            assert isinstance(result, str)
            assert "Error viewing text log" in result
