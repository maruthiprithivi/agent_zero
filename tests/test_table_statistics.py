"""Tests for the table statistics monitoring tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client

from agent_zero.server import (
    monitor_table_inactive_parts,
    monitor_table_stats,
)


class TestTableStatisticsTools(unittest.TestCase):
    """Test cases for table statistics monitoring tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["database", "table", "total_rows", "total_bytes"]
        self.mock_result.result_rows = [
            ["testdb", "table1", 1000000, 1024000000],
            ["testdb", "table2", 2000000, 2048000000],
        ]

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_monitor_table_stats(self):
        """Test monitoring table statistics."""
        # Mock the get_table_stats function
        with patch("agent_zero.server.get_table_stats") as mock_function:
            # Mock successful execution - all tables in database
            mock_function.return_value = [
                {
                    "database": "testdb",
                    "table": "table1",
                    "engine": "MergeTree",
                    "total_rows": 1000000,
                    "total_bytes": 1024000000,
                    "parts_count": 10,
                },
                {
                    "database": "testdb",
                    "table": "table2",
                    "engine": "MergeTree",
                    "total_rows": 2000000,
                    "total_bytes": 2048000000,
                    "parts_count": 20,
                },
            ]
            result = monitor_table_stats("testdb")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["database"], "testdb")
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client, "testdb", None)

            # Test with specific table
            mock_function.reset_mock()
            mock_function.return_value = [
                {
                    "database": "testdb",
                    "table": "table1",
                    "engine": "MergeTree",
                    "total_rows": 1000000,
                    "total_bytes": 1024000000,
                    "parts_count": 10,
                }
            ]
            result = monitor_table_stats("testdb", "table1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client, "testdb", "table1")

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_table_stats("testdb")
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring table stats", result)

    def test_monitor_table_inactive_parts(self):
        """Test monitoring table inactive parts."""
        # Mock the get_table_inactive_parts function
        with patch("agent_zero.server.get_table_inactive_parts") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"partition": "202403", "inactive_parts": 5, "inactive_bytes": 512000},
                {"partition": "202404", "inactive_parts": 3, "inactive_bytes": 256000},
            ]
            result = monitor_table_inactive_parts("testdb", "table1")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["partition"], "202403")
            mock_function.assert_called_once_with(self.mock_client, "testdb", "table1")

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_table_inactive_parts("testdb", "table1")
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring table inactive parts", result)


if __name__ == "__main__":
    unittest.main()
