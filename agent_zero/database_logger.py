"""Database query logging and monitoring utilities.

This module provides functionality for logging database queries, their execution times,
errors, and warnings.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.mcp_env import config

logger = logging.getLogger("mcp-db-queries")

T = TypeVar("T")


class QueryLogger:
    """Logger for database queries.

    This class provides methods for logging database queries, their execution times,
    errors, and warnings.
    """

    def __init__(self):
        """Initialize the query logger."""
        # Configure detailed formatting for query logs
        self._configure_logger()

    def _configure_logger(self):
        """Configure the logger with appropriate formatting."""
        # Create a handler for the logger if it doesn't already have one
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def log_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """Log a database query if query logging is enabled.

        Args:
            query: The SQL query text
            params: Optional query parameters
            settings: Optional query settings
        """
        if not config.enable_query_logging:
            return

        # Log the query
        logger.info(f"Query: {query}")

        # Log parameters and settings if provided
        if params:
            logger.info(f"Parameters: {params}")
        if settings:
            logger.info(f"Settings: {settings}")

    def log_query_result(self, rows_affected: int) -> None:
        """Log the result of a query.

        Args:
            rows_affected: Number of rows affected or returned
        """
        if not config.enable_query_logging:
            return

        logger.info(f"Query result: {rows_affected} rows affected/returned")

    def log_query_error(self, error: Exception, query: str) -> None:
        """Log a query error.

        Args:
            error: The exception that occurred
            query: The SQL query that caused the error
        """
        if not config.log_query_errors:
            return

        logger.error(f"Query error: {error!s}")
        logger.error(f"Failed query: {query}")

        # For ClickHouse errors, log additional details
        if isinstance(error, ClickHouseError):
            logger.error(f"ClickHouse error code: {error.code}")

    def log_query_warning(self, warning: str, query: str) -> None:
        """Log a query warning.

        Args:
            warning: The warning message
            query: The SQL query that caused the warning
        """
        if not config.log_query_warnings:
            return

        logger.warning(f"Query warning: {warning}")
        logger.warning(f"Query: {query}")


# Create a global instance
query_logger = QueryLogger()


def log_query_execution(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log query execution details.

    This decorator logs the query, parameters, execution time, and any errors.

    Args:
        func: The function to decorate (typically a query execution function)

    Returns:
        The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract query and other details from args/kwargs based on function signature
        query = None
        parameters = None
        settings = None

        # Check each argument to determine which is the query
        for arg in args:
            if isinstance(arg, str) and (
                "SELECT" in arg.upper()
                or "INSERT" in arg.upper()
                or "UPDATE" in arg.upper()
                or "DELETE" in arg.upper()
                or "CREATE" in arg.upper()
                or "ALTER" in arg.upper()
                or "DROP" in arg.upper()
                or "SHOW" in arg.upper()
            ):
                query = arg
                break

        # Extract parameters and settings from kwargs
        parameters = kwargs.get("parameters") or kwargs.get("params")
        settings = kwargs.get("settings")

        # If we couldn't identify the query from args, check kwargs
        if query is None and "query" in kwargs:
            query = kwargs["query"]

        # Log the query details
        if query and config.enable_query_logging:
            query_logger.log_query(query, parameters, settings)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)

            # Log execution time if enabled
            elapsed_time = time.time() - start_time
            if config.log_query_latency:
                logger.info(f"Query executed in {elapsed_time:.4f}s")

            # Log the result summary
            if hasattr(result, "result_rows") and config.enable_query_logging:
                query_logger.log_query_result(len(result.result_rows))

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            if config.log_query_latency:
                logger.info(f"Query failed after {elapsed_time:.4f}s")

            if query and config.log_query_errors:
                query_logger.log_query_error(e, query)
            raise

    return wrapper
