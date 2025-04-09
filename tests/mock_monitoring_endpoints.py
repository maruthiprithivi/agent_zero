"""Mock monitoring endpoints for the MCP ClickHouse server.

This is a simplified version of monitoring_endpoints.py for isolated testing.
"""

import time
from typing import Callable, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials


# Mock for ServerConfig
class MockServerConfig:
    """Mock ServerConfig for testing."""

    def __init__(self, auth_config=None, **kwargs):
        self._auth_config = auth_config
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 8505)

    def get_auth_config(self):
        """Get authentication configuration."""
        return self._auth_config


# HTTP Basic Auth security
security = HTTPBasic(auto_error=False)


def setup_monitoring_endpoints(app: FastAPI, create_clickhouse_client: Callable) -> Dict[str, Any]:
    """Set up monitoring endpoints for the MCP server.

    Args:
        app: FastAPI application
        create_clickhouse_client: Function to create a ClickHouse client

    Returns:
        Dict with testing helpers including config_setter function
    """
    # Use a reference for server_config to allow updating it in tests
    config_ref = {"server_config": MockServerConfig()}

    # Add request middleware for tracking metrics
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Middleware to track metrics for HTTP requests."""
        # These variables would be used in a real implementation to track metrics
        # path = request.url.path
        # method = request.method

        # No need to track timing in this mock implementation
        response = await call_next(request)

        # In a real implementation, we would track metrics here

        return response

    async def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
        """Verify authentication credentials.

        Args:
            credentials: HTTP Basic Auth credentials

        Raises:
            HTTPException: If authentication fails
        """
        auth_config = config_ref["server_config"].get_auth_config()

        if auth_config:
            if not credentials:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: Credentials required",
                    headers={"WWW-Authenticate": "Basic"},
                )

            if (
                credentials.username != auth_config["username"]
                or credentials.password != auth_config["password"]
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: Invalid credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )

    @app.get("/health")
    async def health_check(credentials: HTTPBasicCredentials = Depends(security)):
        """Health check endpoint.

        Args:
            credentials: HTTP Basic Auth credentials

        Returns:
            Health check information
        """
        # Check authentication first
        await verify_auth(credentials)

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

            health_info["clickhouse_connected"] = True
            health_info["clickhouse_version"] = version
        except Exception as e:
            health_info["status"] = "degraded"
            health_info["clickhouse_error"] = str(e)

        return health_info

    @app.get("/metrics")
    async def metrics_endpoint(credentials: HTTPBasicCredentials = Depends(security)):
        """Prometheus metrics endpoint.

        Args:
            credentials: HTTP Basic Auth credentials

        Returns:
            Prometheus metrics in text format
        """
        # Check authentication first
        await verify_auth(credentials)

        # In a real implementation, we would generate Prometheus metrics here
        prometheus_metrics = (
            "# HELP test_metric Test metric\n# TYPE test_metric gauge\ntest_metric 1.0\n"
        )

        return Response(
            content=prometheus_metrics,
            media_type="text/plain; version=0.0.4",
            headers={"content-type": "text/plain; version=0.0.4"},
        )

    # Helper function to update the config for tests
    def set_config(new_config):
        config_ref["server_config"] = new_config

    # Return test helpers
    return {"set_config": set_config, "get_config": lambda: config_ref["server_config"]}
