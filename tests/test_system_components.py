"""Tests for the system components monitoring tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client

from agent_zero.mcp_server import (
    monitor_blob_storage_stats,
    monitor_materialized_view_stats,
    monitor_s3queue_stats,
)


class TestSystemComponentsTools(unittest.TestCase):
    """Test cases for system components monitoring tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["ts", "metric", "value"]
        self.mock_result.result_rows = [
            ["2024-03-10 00:00:00", "blob_read", 1024],
            ["2024-03-11 00:00:00", "blob_write", 2048],
        ]

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_monitor_blob_storage_stats(self):
        """Test monitoring blob storage statistics."""
        # Mock the get_blob_storage_stats function
        with patch("agent_zero.mcp_server.get_blob_storage_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "date": "2024-03-10",
                    "reads": 1000,
                    "writes": 500,
                    "read_bytes": 1024000,
                    "write_bytes": 512000,
                },
                {
                    "date": "2024-03-11",
                    "reads": 2000,
                    "writes": 1000,
                    "read_bytes": 2048000,
                    "write_bytes": 1024000,
                },
            ]
            result = monitor_blob_storage_stats()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["date"], "2024-03-10")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom days parameter
            mock_function.reset_mock()
            result = monitor_blob_storage_stats(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_blob_storage_stats()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring blob storage stats", result)

    def test_monitor_materialized_view_stats(self):
        """Test monitoring materialized view statistics."""
        # Mock the get_mv_query_stats function
        with patch("agent_zero.mcp_server.get_mv_query_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "database": "testdb",
                    "table": "mv1",
                    "source_table": "source1",
                    "query_count": 100,
                    "avg_duration": 0.5,
                },
                {
                    "database": "testdb",
                    "table": "mv2",
                    "source_table": "source2",
                    "query_count": 200,
                    "avg_duration": 0.8,
                },
            ]
            result = monitor_materialized_view_stats()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["table"], "mv1")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom days parameter
            mock_function.reset_mock()
            result = monitor_materialized_view_stats(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_materialized_view_stats()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring materialized view stats", result)

    def test_monitor_s3queue_stats(self):
        """Test monitoring S3 queue statistics."""
        # Mock the get_s3queue_stats function
        with patch("agent_zero.mcp_server.get_s3queue_stats") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "date": "2024-03-10",
                    "queue": "queue1",
                    "files_processed": 100,
                    "bytes_processed": 1024000,
                },
                {
                    "date": "2024-03-11",
                    "queue": "queue2",
                    "files_processed": 200,
                    "bytes_processed": 2048000,
                },
            ]
            result = monitor_s3queue_stats()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["queue"], "queue1")
            mock_function.assert_called_once_with(self.mock_client, 7)

            # Test with custom days parameter
            mock_function.reset_mock()
            result = monitor_s3queue_stats(days=14)
            mock_function.assert_called_once_with(self.mock_client, 14)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = monitor_s3queue_stats()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring S3 Queue stats", result)


if __name__ == "__main__":
    unittest.main()
