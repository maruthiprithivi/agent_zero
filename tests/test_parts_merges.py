"""Tests for the parts merges monitoring tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client


class TestPartsMergesTools(unittest.TestCase):
    """Test cases for parts merges monitoring tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["database", "table", "parts_count", "active_parts"]
        self.mock_result.result_rows = [
            ["testdb", "table1", 100, 90],
            ["testdb", "table2", 200, 180],
        ]

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_monitor_current_merges(self):
        """Test monitoring current merges."""
        # Mock the get_current_merges function
        with patch("agent_zero.mcp_server.get_current_merges") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"database": "testdb", "table": "table1", "elapsed": 120, "progress": 0.5},
                {"database": "testdb", "table": "table2", "elapsed": 300, "progress": 0.7},
            ]
            result = monitor_current_merges()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["database"], "testdb")
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_current_merges()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring current merges", result)

    def test_monitor_merge_stats(self):
        """Test monitoring merge statistics."""
        # Mock the get_merge_stats function
        with patch("agent_zero.mcp_server.get_merge_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"database": "testdb", "table": "table1", "merges_count": 10, "avg_duration": 150},
                {"database": "testdb", "table": "table2", "merges_count": 20, "avg_duration": 300},
            ]
            result = monitor_merge_stats()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["database"], "testdb")
            self.assertEqual(result[0]["table"], "table1")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom days parameter
            mock_function.reset_mock()
            result = monitor_merge_stats(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_merge_stats()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring merge stats", result)

    def test_monitor_part_log_events(self):
        """Test monitoring part log events."""
        # Mock the get_part_log_events function
        with patch("agent_zero.mcp_server.get_part_log_events") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "event_time": "2024-03-10 00:00:00",
                    "event_type": "NEW_PART",
                    "database": "testdb",
                    "table": "table1",
                },
                {
                    "event_time": "2024-03-11 00:00:00",
                    "event_type": "MERGE_PARTS",
                    "database": "testdb",
                    "table": "table2",
                },
            ]
            result = monitor_part_log_events()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["event_type"], "NEW_PART")
            mock_function.assert_called_once_with(self.mock_client, 1, 100)

            # Test with custom parameters
            mock_function.reset_mock()
            result = monitor_part_log_events(days=3, limit=50)
            mock_function.assert_called_once_with(self.mock_client, 3, 50)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_part_log_events()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring part log events", result)

    def test_monitor_partition_stats(self):
        """Test monitoring partition statistics."""
        # Mock the get_partition_stats function
        with patch("agent_zero.mcp_server.get_partition_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {"partition": "202403", "parts_count": 10, "rows": 1000, "bytes": 1024000},
                {"partition": "202404", "parts_count": 5, "rows": 500, "bytes": 512000},
            ]
            result = monitor_partition_stats("testdb", "table1")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["partition"], "202403")
            mock_function.assert_called_once_with(self.mock_client, "testdb", "table1")

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_partition_stats("testdb", "table1")
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring partition stats", result)

    def test_monitor_parts_analysis(self):
        """Test monitoring parts analysis."""
        # Mock the get_parts_analysis function
        with patch("agent_zero.mcp_server.get_parts_analysis") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "partition": "202403",
                    "active_parts": 8,
                    "inactive_parts": 2,
                    "compression_ratio": 10.5,
                },
                {
                    "partition": "202404",
                    "active_parts": 4,
                    "inactive_parts": 1,
                    "compression_ratio": 11.2,
                },
            ]
            result = monitor_parts_analysis("testdb", "table1")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["partition"], "202403")
            mock_function.assert_called_once_with(self.mock_client, "testdb", "table1")

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_parts_analysis("testdb", "table1")
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring parts analysis", result)


if __name__ == "__main__":
    unittest.main()
