"""Mock query metrics for the MCP ClickHouse server.

This is a simplified copy of query_metrics.py for isolated testing without dependencies.
"""

import re
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar


# Mock Prometheus metrics for testing
class MockCounter:
    """Mock Prometheus Counter for testing."""

    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.labels_list = labels or []
        self.values = {}

    def labels(self, **kwargs):
        label_key = tuple(sorted([(k, v) for k, v in kwargs.items()]))
        if label_key not in self.values:
            self.values[label_key] = 0
        return self

    def inc(self, value=1):
        """Increment the counter."""
        self.values[tuple()] = self.values.get(tuple(), 0) + value


class MockSummary:
    """Mock Prometheus Summary for testing."""

    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.labels_list = labels or []
        self.values = {}

    def labels(self, **kwargs):
        label_key = tuple(sorted([(k, v) for k, v in kwargs.items()]))
        if label_key not in self.values:
            self.values[label_key] = []
        return self

    def time(self):
        """Create a context manager for timing."""

        class TimerContextManager:
            def __enter__(self_cm):
                self_cm.start_time = time.time()
                return self_cm

            def __exit__(self_cm, exc_type, exc_val, exc_tb):
                duration = time.time() - self_cm.start_time
                self.values[tuple()] = self.values.get(tuple(), []) + [duration]

        return TimerContextManager()


# Create mock metrics
QUERY_COUNT = MockCounter("clickhouse_queries_total", "Total count of ClickHouse queries", ["type"])
QUERY_ERRORS = MockCounter(
    "clickhouse_query_errors_total", "Total count of ClickHouse query errors", ["type"]
)
QUERY_DURATION = MockSummary(
    "clickhouse_query_duration_seconds", "Duration of ClickHouse queries in seconds", ["type"]
)

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

        # Increment query count
        QUERY_COUNT.labels(type=query_type).inc()

        # Track query duration using the context manager's built-in timing
        try:
            with QUERY_DURATION.labels(type=query_type).time():
                result = func(*args, **kwargs)
            return result
        except Exception:
            # Track query errors
            QUERY_ERRORS.labels(type=query_type).inc()
            raise

    return wrapper
