"""Query metrics for the MCP ClickHouse server.

This module provides Prometheus metrics for ClickHouse queries.
"""

import logging
import re
from collections.abc import Callable
from functools import wraps, lru_cache
from typing import TypeVar

from prometheus_client import Counter, Summary

logger = logging.getLogger("mcp-query-metrics")


# Define a function to get the metrics only once
@lru_cache(maxsize=1)
def get_metrics():
    """Get or create Prometheus metrics.

    This function uses lru_cache to ensure metrics are only created once,
    preventing duplicate registration errors during testing.

    Returns:
        Dict containing all Prometheus metrics
    """
    try:
        return {
            "QUERY_COUNT": Counter(
                "clickhouse_queries_total", "Total count of ClickHouse queries", ["type"]
            ),
            "QUERY_ERRORS": Counter(
                "clickhouse_query_errors_total", "Total count of ClickHouse query errors", ["type"]
            ),
            "QUERY_DURATION": Summary(
                "clickhouse_query_duration_seconds",
                "Duration of ClickHouse queries in seconds",
                ["type"],
            ),
        }
    except ValueError as e:
        # If metrics already exist (e.g., in tests), log the error and return empty metrics
        # This prevents tests from failing due to duplicate metric registration
        logger.warning(f"Metrics already registered: {str(e)}")
        # Return a dict of dummy metrics that won't throw errors when used
        # This is primarily for tests that import this module multiple times
        return {}


# Initialize metrics lazily
metrics = None

T = TypeVar("T")


def extract_query_type(query: str) -> str:
    """Extract the query type from the query string.

    Args:
        query: The SQL query

    Returns:
        The query type (e.g., SELECT, INSERT, SHOW)
    """
    if not query:
        return "UNKNOWN"

    # Remove comments
    query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

    # Get the first word (command)
    match = re.match(r"^\s*(\w+)", query)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def track_query_metrics(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to track query metrics with Prometheus.

    This decorator tracks:
    - Query count by type
    - Query duration by type
    - Query errors by type

    Args:
        func: The function to decorate (typically a query execution function)

    Returns:
        The decorated function
    """
    # Initialize metrics on first use
    global metrics
    if not metrics:
        metrics = get_metrics()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract query and other details from args/kwargs based on function signature
        query = None

        # Check each argument to determine which is the query
        for arg in args:
            if isinstance(arg, str) and len(arg) > 5:  # Minimum length to be a query
                query = arg
                break

        # If we couldn't identify the query from args, check kwargs
        if query is None and "query" in kwargs:
            query = kwargs["query"]

        # Extract the query type
        query_type = extract_query_type(query) if query else "UNKNOWN"

        # Increment query count if metrics are available
        if metrics and "QUERY_COUNT" in metrics:
            try:
                metrics["QUERY_COUNT"].labels(type=query_type).inc()
            except (AttributeError, KeyError):
                pass

        # Track query duration
        try:
            # Use metrics if available, otherwise just call the function directly
            if metrics and "QUERY_DURATION" in metrics:
                try:
                    with metrics["QUERY_DURATION"].labels(type=query_type).time():
                        result = func(*args, **kwargs)
                    return result
                except (AttributeError, KeyError):
                    result = func(*args, **kwargs)
                    return result
            else:
                result = func(*args, **kwargs)
                return result
        except Exception:
            # Track query errors
            if metrics and "QUERY_ERRORS" in metrics:
                try:
                    metrics["QUERY_ERRORS"].labels(type=query_type).inc()
                except (AttributeError, KeyError):
                    pass
            raise

    return wrapper
