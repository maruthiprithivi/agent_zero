"""Utility functions for the ClickHouse MCP server.

This module provides common utility functions used across the MCP server components.
"""

import logging
import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

# Import the database logger
from agent_zero.database_logger import log_query_execution

# Import query metrics tracking
from agent_zero.query_metrics import track_query_metrics

logger = logging.getLogger("mcp-clickhouse")


@log_query_execution
@track_query_metrics
def execute_query_with_retry(
    client: Client,
    query: str,
    params: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    readonly: bool = True,
) -> list[dict[str, Any]]:
    """Execute a ClickHouse query with retry logic.

    Args:
        client: The ClickHouse client instance
        query: The SQL query to execute
        params: Query parameters (optional)
        settings: Query settings (optional)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        readonly: Whether to enforce readonly mode

    Returns:
        List of dictionaries representing the query results

    Raises:
        ClickHouseError: If the query fails after all retries
    """
    if settings is None:
        settings = {}

    # Enforce readonly mode by default for safety
    if readonly and "readonly" not in settings:
        settings["readonly"] = 1

    # Disable retries if specified in settings
    if settings.get("disable_retries"):
        max_retries = 1

    attempt = 0
    last_error = None

    while attempt < max_retries:
        try:
            res = client.query(query, parameters=params, settings=settings)
            column_names = res.column_names
            rows = []
            for row in res.result_rows:
                row_dict = {}
                for i, col_name in enumerate(column_names):
                    row_dict[col_name] = row[i]
                rows.append(row_dict)
            return rows
        except ClickHouseError as e:
            last_error = e
            attempt += 1
            if attempt < max_retries:
                logger.warning(
                    f"Query failed (attempt {attempt}/{max_retries}): {e!s}. Retrying in"
                    f" {retry_delay}s"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Query failed after {max_retries} attempts: {e!s}")
                raise

    # This should never happen, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Query failed for unknown reason")


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Started executing {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Finished executing {func.__name__} in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error executing {func.__name__} after {elapsed_time:.2f}s: {e!s}")
            raise

    return wrapper


def format_exception(e: Exception) -> str:
    """Format an exception for user-friendly error messages.

    Args:
        e: The exception to format

    Returns:
        A formatted error message
    """
    if isinstance(e, ClickHouseError):
        return f"ClickHouse error: {e!s}"
    return f"Error: {e!s}"


def extract_clickhouse_error_info(error: Exception) -> dict:
    """Extract structured information from ClickHouse exceptions.

    This function parses the string representation of a ClickHouse error
    to extract error codes and other useful information.

    Args:
        error: The ClickHouse exception

    Returns:
        Dictionary with extracted error information
    """
    error_info = {"message": str(error)}

    # Try to extract error code from the error message
    # Common format: "Code: XXX. <error message>"
    error_str = str(error)
    code_match = re.search(r"Code:\s*(\d+)", error_str)
    if code_match:
        error_info["code"] = int(code_match.group(1))

    # Try to extract other common patterns in ClickHouse errors
    if "DB::Exception" in error_str:
        error_info["type"] = "DB::Exception"

    # Extract query ID if present
    query_id_match = re.search(r"QueryID:\s*([a-f0-9-]+)", error_str)
    if query_id_match:
        error_info["query_id"] = query_id_match.group(1)

    return error_info
