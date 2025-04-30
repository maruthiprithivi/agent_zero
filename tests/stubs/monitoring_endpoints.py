"""Stub implementation of monitoring_endpoints for testing.

This module contains stub implementations of functions from the removed
monitoring_endpoints.py file for test compatibility only.
"""

import logging
from typing import Callable, Any

logger = logging.getLogger("test-stubs")


def setup_monitoring_endpoints(app: Any, create_clickhouse_client: Callable) -> None:
    """Stub for setup_monitoring_endpoints.

    This function is a stub implementation for tests that depend on the removed
    monitoring_endpoints.py functionality.

    Args:
        app: Mock FastAPI application
        create_clickhouse_client: Function to create a ClickHouse client
    """
    logger.info("Using stub implementation of setup_monitoring_endpoints")

    # For testing purposes, add the '/health' endpoint directly to the mock app
    if hasattr(app, "get") and callable(app.get):

        @app.get("/health")
        def health_check():
            """Stub health check endpoint."""
            try:
                client = create_clickhouse_client()
                return {
                    "status": "healthy",
                    "server": "agent-zero-stub",
                    "clickhouse_connected": True,
                    "clickhouse_version": getattr(client, "server_version", "unknown"),
                }
            except Exception as e:
                return {
                    "status": "degraded",
                    "server": "agent-zero-stub",
                    "clickhouse_connected": False,
                    "clickhouse_error": str(e),
                }

        @app.get("/metrics")
        def metrics_endpoint():
            """Stub metrics endpoint."""
            return b"# Test metrics stub\ntest_metric 1.0\n"
