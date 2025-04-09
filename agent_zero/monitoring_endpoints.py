"""Monitoring endpoints for the MCP ClickHouse server.

This module provides monitoring endpoints like /health and /metrics for the MCP server.
"""

import logging
import time
from typing import Callable, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from agent_zero.server_config import server_config

logger = logging.getLogger("mcp-monitoring")

# HTTP Basic Auth security
security = HTTPBasic(auto_error=False)


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
            "REQUEST_COUNT": Counter(
                "http_requests_total",
                "Total count of HTTP requests",
                ["method", "endpoint", "status"],
            ),
            "REQUEST_LATENCY": Histogram(
                "http_request_duration_seconds",
                "HTTP request latency in seconds",
                ["method", "endpoint"],
            ),
            "CLICKHOUSE_CONNECTION_STATUS": Gauge(
                "clickhouse_connection_status", "Status of the ClickHouse connection (1=up, 0=down)"
            ),
            "MCP_TOOL_CALLS": Counter(
                "mcp_tool_calls_total", "Total count of MCP tool calls", ["tool"]
            ),
            "CLICKHOUSE_QUERY_COUNT": Counter(
                "clickhouse_queries_total", "Total count of ClickHouse queries", ["type"]
            ),
            "CLICKHOUSE_QUERY_ERRORS": Counter(
                "clickhouse_query_errors_total", "Total count of ClickHouse query errors", ["type"]
            ),
            "CLICKHOUSE_QUERY_DURATION": Summary(
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
metrics = {}


def setup_monitoring_endpoints(app: FastAPI, create_clickhouse_client: Callable) -> None:
    """Set up monitoring endpoints for the MCP server.

    Args:
        app: FastAPI application
        create_clickhouse_client: Function to create a ClickHouse client
    """
    # Initialize metrics on first use
    global metrics
    if not metrics:
        metrics = get_metrics()

    # Add request middleware for tracking metrics
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        path = request.url.path
        method = request.method

        # Record the start time (used for detailed logging if needed)
        # Using time.time() directly in the context manager below

        # Skip metrics tracking for the metrics endpoint itself to avoid recursion
        if path != "/metrics" and metrics:
            try:
                with metrics["REQUEST_LATENCY"].labels(method=method, endpoint=path).time():
                    response = await call_next(request)
            except (AttributeError, KeyError):
                # Handle case where metrics might not be available (e.g., in tests)
                response = await call_next(request)
        else:
            response = await call_next(request)

        status_code = response.status_code

        # Skip metrics endpoint to avoid recursion
        if path != "/metrics" and metrics:
            try:
                metrics["REQUEST_COUNT"].labels(
                    method=method, endpoint=path, status=status_code
                ).inc()

                # Track MCP tool calls
                if path.startswith("/mcp/") and path != "/mcp/info":
                    tool_name = path.split("/")[-1] if len(path.split("/")) > 2 else "unknown"
                    metrics["MCP_TOOL_CALLS"].labels(tool=tool_name).inc()
            except (AttributeError, KeyError):
                # Handle case where metrics might not be available
                pass

        return response

    # Define authentication dependency
    async def get_authenticated(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
        """Check if the request is authenticated.

        Args:
            credentials: HTTP Basic Auth credentials

        Returns:
            True if authenticated, False otherwise
        """
        auth_config = server_config.get_auth_config()

        # If no auth is configured, allow all requests
        if not auth_config:
            return True

        # If auth is configured but no credentials provided, deny access
        if not credentials:
            return False

        # Check if credentials match
        return (
            credentials.username == auth_config["username"]
            and credentials.password == auth_config["password"]
        )

    async def check_auth(authenticated: bool = Depends(get_authenticated)) -> None:
        """Check if the request is authenticated and raise an exception if not.

        Args:
            authenticated: Whether the request is authenticated

        Raises:
            HTTPException: If the request is not authenticated
        """
        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

    @app.get("/health")
    async def health_check(authenticated: bool = Depends(get_authenticated)) -> Dict[str, Any]:
        """Health check endpoint.

        Args:
            authenticated: Whether the request is authenticated

        Returns:
            Health check information
        """
        # If authentication is required but failed, return 401
        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

        health_info = {
            "status": "healthy",
            "server": "agent-zero",
            "clickhouse_connected": False,
            "timestamp": time.time(),
        }

        # Check ClickHouse connection
        try:
            client = create_clickhouse_client()
            version = client.server_version

            # Update connection status metric
            if metrics:
                try:
                    metrics["CLICKHOUSE_CONNECTION_STATUS"].set(1)
                except (AttributeError, KeyError):
                    pass

            health_info["clickhouse_connected"] = True
            health_info["clickhouse_version"] = version
        except Exception as e:
            # Update connection status metric
            if metrics:
                try:
                    metrics["CLICKHOUSE_CONNECTION_STATUS"].set(0)
                except (AttributeError, KeyError):
                    pass

            health_info["status"] = "degraded"
            health_info["clickhouse_error"] = str(e)

        return health_info

    @app.get("/metrics")
    async def metrics_endpoint(authenticated: bool = Depends(get_authenticated)) -> Response:
        """Prometheus metrics endpoint.

        Args:
            authenticated: Whether the request is authenticated

        Returns:
            Prometheus metrics in text format
        """
        # If authentication is required but failed, return 401
        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

        # Update ClickHouse connection status before generating metrics
        try:
            # Just check if we can create a client, don't need to store it
            create_clickhouse_client()
            if metrics:
                try:
                    metrics["CLICKHOUSE_CONNECTION_STATUS"].set(1)
                except (AttributeError, KeyError):
                    pass
        except Exception:
            if metrics:
                try:
                    metrics["CLICKHOUSE_CONNECTION_STATUS"].set(0)
                except (AttributeError, KeyError):
                    pass

        # Generate latest metrics in Prometheus format
        prometheus_metrics = generate_latest()

        return Response(
            content=prometheus_metrics,
            media_type=CONTENT_TYPE_LATEST,
        )
