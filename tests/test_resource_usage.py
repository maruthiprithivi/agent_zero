"""Tests for the resource_usage monitoring module."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.monitoring.resource_usage import (
    get_cpu_usage,
    get_memory_usage,
    get_server_sizing,
    get_uptime,
)


class TestResourceUsage(unittest.TestCase):
    """Test cases for resource usage monitoring functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["hostname", "metric_value"]
        self.mock_result.result_rows = [
            ["server1", 1024],
            ["server2", 2048],
        ]
        self.mock_client.query.return_value = self.mock_result

    def test_get_memory_usage(self):
        """Test retrieving memory usage statistics."""
        # Test with default parameters
        result = get_memory_usage(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()
        self.assertIn("clusterAllReplicas", self.mock_client.query.call_args[0][0])

        # Test with custom days parameter
        self.mock_client.query.reset_mock()
        result = get_memory_usage(self.mock_client, days=14)
        self.assertEqual(len(result), 2)
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("toIntervalDay(14)", query)

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Cluster not found"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_memory_usage(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)
        # Second call should use system.metric_log instead of clusterAllReplicas
        self.assertIn("system.metric_log", self.mock_client.query.call_args[0][0])

    def test_get_cpu_usage(self):
        """Test retrieving CPU usage statistics."""
        # Test with default parameters
        result = get_cpu_usage(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test with custom hours parameter
        self.mock_client.query.reset_mock()
        result = get_cpu_usage(self.mock_client, hours=6)
        self.assertEqual(len(result), 2)
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("(now() - 21600)", query)  # 6 * 3600 = 21600

        # Test error handling by patching execute_query_with_retry
        with patch("agent_zero.monitoring.resource_usage.execute_query_with_retry") as mock_execute:
            mock_execute.side_effect = ClickHouseError("Complex query failed")
            result = get_cpu_usage(self.mock_client)
            self.assertEqual(len(result), 0)  # Should return empty list
            self.assertEqual(mock_execute.call_count, 1)

    def test_get_server_sizing(self):
        """Test retrieving server sizing information."""
        # Test successful execution
        result = get_server_sizing(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Error"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_server_sizing(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)

    def test_get_uptime(self):
        """Test retrieving uptime statistics."""
        # Test with default parameters
        result = get_uptime(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test with custom days parameter
        self.mock_client.query.reset_mock()
        result = get_uptime(self.mock_client, days=30)
        self.assertEqual(len(result), 2)
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("toIntervalDay(30)", query)

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Error"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_uptime(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
