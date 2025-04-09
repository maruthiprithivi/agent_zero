"""Tests for mock_query_metrics.py."""

import pytest
from unittest.mock import MagicMock

from tests.mock_query_metrics import (
    extract_query_type,
    track_query_metrics,
    QUERY_COUNT,
    QUERY_ERRORS,
    QUERY_DURATION,
)


class TestMockQueryMetrics:
    """Tests for the mock query metrics functionality."""

    def test_extract_query_type_select(self):
        """Test that SELECT query type is correctly extracted."""
        query = "SELECT * FROM test_table"
        assert extract_query_type(query) == "SELECT"

        query = "  SELECT * FROM test_table"
        assert extract_query_type(query) == "SELECT"

        query = """
        SELECT *
        FROM test_table
        WHERE id = 1
        """
        assert extract_query_type(query) == "SELECT"

    def test_extract_query_type_insert(self):
        """Test that INSERT query type is correctly extracted."""
        query = "INSERT INTO test_table VALUES (1, 'test')"
        assert extract_query_type(query) == "INSERT"

        query = "  INSERT INTO test_table VALUES (1, 'test')"
        assert extract_query_type(query) == "INSERT"

    def test_extract_query_type_show(self):
        """Test that SHOW query type is correctly extracted."""
        query = "SHOW TABLES"
        assert extract_query_type(query) == "SHOW"

        query = "  SHOW DATABASES"
        assert extract_query_type(query) == "SHOW"

    def test_extract_query_type_with_comments(self):
        """Test that query type is correctly extracted when comments are present."""
        query = "-- This is a comment\nSELECT * FROM test_table"
        assert extract_query_type(query) == "SELECT"

        query = "/* This is a\nmultiline comment */\nSELECT * FROM test_table"
        assert extract_query_type(query) == "SELECT"

        query = "/* This is a comment */ SELECT * FROM test_table -- Another comment"
        assert extract_query_type(query) == "SELECT"

    def test_extract_query_type_empty(self):
        """Test that UNKNOWN is returned for empty queries."""
        assert extract_query_type("") == "UNKNOWN"
        assert extract_query_type(None) == "UNKNOWN"
        assert extract_query_type("  ") == "UNKNOWN"

    def test_track_query_metrics_success(self):
        """Test that query metrics are tracked for successful queries."""
        # Create a mock function to decorate
        mock_function = MagicMock()
        mock_function.return_value = "query_result"

        # Reset metrics
        QUERY_COUNT.values = {}
        QUERY_DURATION.values = {}
        QUERY_ERRORS.values = {}

        # Decorate the function and call it
        decorated_function = track_query_metrics(mock_function)
        result = decorated_function("SELECT * FROM test_table")

        # Verify the function was called and the result was returned
        mock_function.assert_called_with("SELECT * FROM test_table")
        assert result == "query_result"

        # Verify metrics were tracked
        assert QUERY_COUNT.values
        assert QUERY_DURATION.values
        assert not QUERY_ERRORS.values  # No errors

    def test_track_query_metrics_error(self):
        """Test that query metrics are tracked for failed queries."""
        # Create a mock function to decorate that raises an exception
        mock_function = MagicMock()
        mock_function.side_effect = Exception("Query failed")

        # Reset metrics
        QUERY_COUNT.values = {}
        QUERY_DURATION.values = {}
        QUERY_ERRORS.values = {}

        # Decorate the function and call it
        decorated_function = track_query_metrics(mock_function)

        # Call the function and expect an exception
        with pytest.raises(Exception):
            decorated_function("SELECT * FROM test_table")

        # Verify the function was called
        mock_function.assert_called_with("SELECT * FROM test_table")

        # Verify metrics were tracked
        assert QUERY_COUNT.values
        assert QUERY_DURATION.values
        assert QUERY_ERRORS.values  # Error was tracked

    def test_track_query_metrics_with_kwargs(self):
        """Test that query metrics work when the query is passed as a keyword argument."""
        # Create a mock function to decorate
        mock_function = MagicMock()
        mock_function.return_value = "query_result"

        # Reset metrics
        QUERY_COUNT.values = {}
        QUERY_DURATION.values = {}
        QUERY_ERRORS.values = {}

        # Decorate the function and call it with a keyword argument
        decorated_function = track_query_metrics(mock_function)
        result = decorated_function(query="SELECT * FROM test_table")

        # Verify the function was called and the result was returned
        mock_function.assert_called_with(query="SELECT * FROM test_table")
        assert result == "query_result"

        # Verify metrics were tracked
        assert QUERY_COUNT.values
        assert QUERY_DURATION.values
        assert not QUERY_ERRORS.values  # No errors

    def test_track_query_metrics_unknown_query(self):
        """Test that query metrics work when the query can't be identified."""
        # Create a mock function to decorate
        mock_function = MagicMock()
        mock_function.return_value = "query_result"

        # Reset metrics
        QUERY_COUNT.values = {}
        QUERY_DURATION.values = {}
        QUERY_ERRORS.values = {}

        # Decorate the function and call it with a non-query argument
        decorated_function = track_query_metrics(mock_function)
        result = decorated_function(123, "other_arg")

        # Verify the function was called and the result was returned
        mock_function.assert_called_with(123, "other_arg")
        assert result == "query_result"

        # Verify metrics were tracked with unknown type
        assert QUERY_COUNT.values
        assert QUERY_DURATION.values
        assert not QUERY_ERRORS.values  # No errors
