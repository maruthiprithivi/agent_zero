"""Tests for the insert_operations monitoring tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client

from agent_zero.mcp_server import (
    monitor_async_insert_stats,
    monitor_insert_bytes_distribution,
    monitor_recent_insert_queries,
)


class TestInsertOperationsTools(unittest.TestCase):
    """Test cases for insert operations monitoring tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["ts", "query", "query_duration"]
        self.mock_result.result_rows = [
            ["2024-03-10 00:00:00", "INSERT INTO table VALUES", 0.5],
            ["2024-03-11 00:00:00", "INSERT INTO another_table VALUES", 1.2],
        ]

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_monitor_recent_insert_queries(self):
        """Test monitoring recent insert queries."""
        # Mock the get_recent_insert_queries function
        with patch("agent_zero.mcp_server.get_recent_insert_queries") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"ts": "2024-03-10 00:00:00", "query": "INSERT INTO table VALUES", "duration": 0.5},
                {
                    "ts": "2024-03-11 00:00:00",
                    "query": "INSERT INTO another_table VALUES",
                    "duration": 1.2,
                },
            ]
            result = monitor_recent_insert_queries()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["query"], "INSERT INTO table VALUES")
            mock_function.assert_called_once_with(self.mock_client, 1, 100)

            # Test with custom parameters
            mock_function.reset_mock()
            result = monitor_recent_insert_queries(days=3, limit=50)
            mock_function.assert_called_once_with(self.mock_client, 3, 50)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_recent_insert_queries()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring recent insert queries", result)

    def test_monitor_async_insert_stats(self):
        """Test monitoring async insert statistics."""
        # Mock the get_async_insert_stats function
        with patch("agent_zero.mcp_server.get_async_insert_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"ts": "2024-03-10", "table": "table1", "async_inserts": 100},
                {"ts": "2024-03-11", "table": "table2", "async_inserts": 200},
            ]
            result = monitor_async_insert_stats()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom parameters
            mock_function.reset_mock()
            result = monitor_async_insert_stats(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_async_insert_stats()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring async insert stats", result)

    def test_monitor_insert_bytes_distribution(self):
        """Test monitoring insert bytes distribution."""
        # Mock the get_insert_written_bytes_distribution function
        with patch("agent_zero.mcp_server.get_insert_written_bytes_distribution") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"table": "table1", "written_bytes": 1024, "count": 10},
                {"table": "table2", "written_bytes": 2048, "count": 20},
            ]
            result = monitor_insert_bytes_distribution()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom parameters
            mock_function.reset_mock()
            result = monitor_insert_bytes_distribution(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_insert_bytes_distribution()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring insert bytes distribution", result)


if __name__ == "__main__":
    unittest.main()
