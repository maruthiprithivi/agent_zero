"""Test utilities for agent_one tests."""

from unittest.mock import MagicMock


def assert_query_contains(query: str, expected: str) -> None:
    """Compare queries ignoring whitespace and case.

    Args:
        query: The actual query string
        expected: The expected substring to find in the query
    """
    normalized_query = " ".join(query.lower().split())
    normalized_expected = " ".join(expected.lower().split())
    assert (
        normalized_expected in normalized_query
    ), f"Expected '{expected}' not found in query: '{query}'"


def create_mock_result(column_names: list, result_rows: list):
    """Create a mock query result with the given columns and rows.

    Args:
        column_names: List of column names
        result_rows: List of result rows

    Returns:
        A mock result object with column_names and result_rows attributes
    """
    mock_result = MagicMock()
    mock_result.column_names = column_names
    mock_result.result_rows = result_rows
    return mock_result
