"""Tests for the query_performance monitoring module."""

import unittest
from unittest.mock import MagicMock

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.monitoring.query_performance import (
    get_current_processes,
    get_normalized_query_stats,
    get_query_duration_stats,
    get_query_kind_breakdown,
)


class TestQueryPerformance(unittest.TestCase):
    """Test cases for query performance monitoring functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = MagicMock()
        self.mock_result.column_names = ["column1", "column2"]
        self.mock_result.result_rows = [["value1", "value2"], ["value3", "value4"]]
        self.mock_client.query.return_value = self.mock_result

    def test_get_current_processes(self):
        """Test retrieving current processes."""
        # Test successful execution
        result = get_current_processes(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()
        self.assertIn("clusterAllReplicas", self.mock_client.query.call_args[0][0])

        # Test error handling and fallback
        self.mock_client.query.side_effect = [
            ClickHouseError("Cluster not found"),  # First call fails
            self.mock_result,  # Second call (fallback) succeeds
        ]
        self.mock_client.query.reset_mock()
        result = get_current_processes(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)
        # Second call should have system.processes instead of clusterAllReplicas
        second_call_query = self.mock_client.query.call_args[0][0]
        self.assertIn("system.processes", second_call_query)

    def test_get_query_duration_stats(self):
        """Test retrieving query duration statistics."""
        # Test with default parameters
        result = get_query_duration_stats(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test with specific query_kind and days
        self.mock_client.query.reset_mock()
        result = get_query_duration_stats(self.mock_client, query_kind="Select", days=3)
        self.assertEqual(len(result), 2)
        # Check that query_kind and days are used in the query
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("AND (query_kind = 'Select')", query)
        self.assertIn("toIntervalDay(3)", query)

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Error"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_query_duration_stats(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)

    def test_get_normalized_query_stats(self):
        """Test retrieving normalized query statistics."""
        # Test with default parameters
        result = get_normalized_query_stats(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test with custom parameters
        self.mock_client.query.reset_mock()
        result = get_normalized_query_stats(self.mock_client, days=5, limit=100)
        self.assertEqual(len(result), 2)
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("toIntervalDay(5)", query)
        self.assertIn("LIMIT 100", query)

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Error"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_normalized_query_stats(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)

    def test_get_query_kind_breakdown(self):
        """Test retrieving query kind breakdown."""
        # Test successful execution
        result = get_query_kind_breakdown(self.mock_client)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

        # Test with custom days parameter
        self.mock_client.query.reset_mock()
        result = get_query_kind_breakdown(self.mock_client, days=14)
        self.assertEqual(len(result), 2)
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("toIntervalDay(14)", query)

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Error"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_query_kind_breakdown(self.mock_client)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
