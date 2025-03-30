"""Tests for the error_analysis monitoring module."""

import unittest
from unittest.mock import MagicMock

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_one.monitoring.error_analysis import (
    get_error_stack_traces,
    get_recent_errors,
    get_text_log,
)

from .utils import assert_query_contains, create_mock_result


class TestErrorAnalysis(unittest.TestCase):
    """Test cases for error analysis monitoring functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = create_mock_result(
            column_names=["ts", "error_code", "count"],
            result_rows=[
                ["2024-03-10 00:00:00", 123, 5],
                ["2024-03-11 00:00:00", 456, 3],
            ],
        )
        self.mock_client.query.return_value = self.mock_result
        self.no_retry_settings = {"disable_retries": True}

    def test_get_recent_errors(self):
        """Test retrieving recent errors."""
        # Test with default parameters
        result = get_recent_errors(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()
        assert_query_contains(self.mock_client.query.call_args[0][0], "clusterAllReplicas")

        # Test with normalized_hash=True
        self.mock_client.query.reset_mock()
        result = get_recent_errors(
            self.mock_client, normalized_hash=True, settings=self.no_retry_settings
        )
        self.assertEqual(len(result), 2)
        assert_query_contains(self.mock_client.query.call_args[0][0], "normalized_query_hash")

        # Test with custom days parameter
        self.mock_client.query.reset_mock()
        result = get_recent_errors(self.mock_client, days=14, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        assert_query_contains(self.mock_client.query.call_args[0][0], "toIntervalDay(14)")

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Cluster not found"),
            self.mock_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_recent_errors(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)

    def test_get_error_stack_traces(self):
        """Test retrieving error stack traces."""
        # Set up mock for stack traces result
        stack_trace_result = create_mock_result(
            column_names=["last_error_time", "last_error_message", "stack_trace"],
            result_rows=[
                ["2024-03-10 00:00:00", "Error message", ["frame1", "frame2"]],
            ],
        )
        self.mock_client.query.return_value = stack_trace_result

        # Test with default parameters
        result = get_error_stack_traces(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 1)
        self.mock_client.query.assert_called_once()
        assert_query_contains(self.mock_client.query.call_args[0][0], "LOGICAL_ERROR")

        # Test with custom error_name
        self.mock_client.query.reset_mock()
        result = get_error_stack_traces(
            self.mock_client, error_name="MEMORY_LIMIT_EXCEEDED", settings=self.no_retry_settings
        )
        self.assertEqual(len(result), 1)
        assert_query_contains(self.mock_client.query.call_args[0][0], "MEMORY_LIMIT_EXCEEDED")

        # Test error handling
        self.mock_client.query.side_effect = ClickHouseError("Introspection not allowed")
        self.mock_client.query.reset_mock()
        result = get_error_stack_traces(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 0)  # Should return empty list
        self.assertEqual(self.mock_client.query.call_count, 1)

    def test_get_text_log(self):
        """Test retrieving text log entries."""
        # Set up mock for text log result
        text_log_result = create_mock_result(
            column_names=[
                "event_time_microseconds",
                "thread_id",
                "level",
                "logger_name",
                "message",
            ],
            result_rows=[
                ["2024-03-10 00:00:00.123456", 12345, "Error", "Logger1", "Error message 1"],
                ["2024-03-10 00:00:01.234567", 12346, "Warning", "Logger2", "Warning message"],
            ],
        )
        self.mock_client.query.return_value = text_log_result

        # Test with default parameters
        result = get_text_log(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()
        assert_query_contains(self.mock_client.query.call_args[0][0], "clusterAllReplicas")
        assert_query_contains(self.mock_client.query.call_args[0][0], "LIMIT 100")

        # Test with query_id and event_date
        self.mock_client.query.reset_mock()
        result = get_text_log(
            self.mock_client,
            query_id="123abc",
            event_date="2024-03-10",
            settings=self.no_retry_settings,
        )
        self.assertEqual(len(result), 2)
        assert_query_contains(self.mock_client.query.call_args[0][0], "query_id = '123abc'")
        assert_query_contains(self.mock_client.query.call_args[0][0], "event_date = '2024-03-10'")

        # Test with custom limit
        self.mock_client.query.reset_mock()
        result = get_text_log(self.mock_client, limit=50, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        assert_query_contains(self.mock_client.query.call_args[0][0], "LIMIT 50")

        # Test error handling
        self.mock_client.query.side_effect = [
            ClickHouseError("Cluster not found"),
            text_log_result,
        ]
        self.mock_client.query.reset_mock()
        result = get_text_log(self.mock_client, settings=self.no_retry_settings)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
