"""Tests for the insert_operations monitoring tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client

from agent_zero.monitoring import (
    get_async_insert_stats,
    get_insert_written_bytes_distribution,
    get_recent_insert_queries,
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
        self.client_patcher = patch("agent_zero.server.client.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_get_recent_insert_queries(self):
        """Test getting recent insert queries."""
        # Mock the get_recent_insert_queries function directly
        result = get_recent_insert_queries(self.mock_client, 1, 100)
        # Since this calls the actual function, we need to mock the client's command method
        self.mock_client.command.return_value = [
            {"ts": "2024-03-10 00:00:00", "query": "INSERT INTO table VALUES", "duration": 0.5},
            {
                "ts": "2024-03-11 00:00:00",
                "query": "INSERT INTO another_table VALUES",
                "duration": 1.2,
            },
        ]
        result = get_recent_insert_queries(self.mock_client, 1, 100)
        # Test that the function runs without error
        self.assertIsNotNone(result)

    def test_get_async_insert_stats(self):
        """Test getting async insert statistics."""
        # Mock the client's command method
        self.mock_client.command.return_value = [
            {"ts": "2024-03-10", "table": "table1", "async_inserts": 100},
            {"ts": "2024-03-11", "table": "table2", "async_inserts": 200},
        ]
        result = get_async_insert_stats(self.mock_client, 7)
        # Test that the function runs without error
        self.assertIsNotNone(result)

    def test_get_insert_written_bytes_distribution(self):
        """Test getting insert bytes distribution."""
        # Mock the client's command method
        self.mock_client.command.return_value = [
            {"table": "table1", "written_bytes": 1024, "count": 10},
            {"table": "table2", "written_bytes": 2048, "count": 20},
        ]
        result = get_insert_written_bytes_distribution(self.mock_client, 7)
        # Test that the function runs without error
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
