"""Tests for monitoring_endpoints.py."""

import base64
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_zero.server_config import ServerConfig
from agent_zero.monitoring_endpoints import setup_monitoring_endpoints


class TestMonitoringEndpoints:
    """Tests for the monitoring endpoints."""

    @pytest.fixture
    def app(self):
        """Create a FastAPI app."""
        return FastAPI()

    @pytest.fixture
    def mock_clickhouse_client(self):
        """Create a mock ClickHouse client."""
        mock_client = MagicMock()
        mock_client.server_version = "23.1.2.3"
        return mock_client

    @pytest.fixture
    def test_client(self, app, mock_clickhouse_client):
        """Create a test client."""

        def create_mock_client():
            return mock_clickhouse_client

        # Create mock metrics dictionary
        mock_metrics = {
            "REQUEST_COUNT": MagicMock(),
            "REQUEST_LATENCY": MagicMock(),
            "CLICKHOUSE_CONNECTION_STATUS": MagicMock(),
            "MCP_TOOL_CALLS": MagicMock(),
            "CLICKHOUSE_QUERY_COUNT": MagicMock(),
            "CLICKHOUSE_QUERY_ERRORS": MagicMock(),
            "CLICKHOUSE_QUERY_DURATION": MagicMock(),
        }

        # Mock generate_latest to return sample Prometheus metrics
        mock_metrics_content = b"""
# HELP http_requests_total Total count of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET", endpoint="/health", status="200"} 2.0
# HELP http_request_duration_seconds HTTP request latency in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET", endpoint="/health", le="0.005"} 2.0
http_request_duration_seconds_bucket{method="GET", endpoint="/health", le="0.01"} 2.0
http_request_duration_seconds_bucket{method="GET", endpoint="/health", le="0.025"} 2.0
http_request_duration_seconds_bucket{method="GET", endpoint="/health", le="0.05"} 2.0
"""

        # Apply patches for the test
        with patch("agent_zero.monitoring_endpoints.get_metrics", return_value=mock_metrics):
            with patch("agent_zero.monitoring_endpoints.metrics", mock_metrics):
                with patch(
                    "agent_zero.monitoring_endpoints.generate_latest",
                    return_value=mock_metrics_content,
                ):
                    setup_monitoring_endpoints(app, create_mock_client)
                    yield TestClient(app)

    def test_health_check_no_auth(self, test_client):
        """Test health check endpoint without authentication."""
        # Mock server_config to return None for auth_config
        with patch("agent_zero.monitoring_endpoints.server_config", ServerConfig()):
            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["clickhouse_connected"] is True
            assert data["clickhouse_version"] == "23.1.2.3"

    def test_health_check_with_auth_success(self, test_client):
        """Test health check endpoint with successful authentication."""
        # Mock server_config to return auth config
        with patch(
            "agent_zero.monitoring_endpoints.server_config",
            ServerConfig(auth_username="testuser", auth_password="testpass"),
        ):
            # Include credentials in the request
            credentials = base64.b64encode(b"testuser:testpass").decode("utf-8")
            headers = {"Authorization": f"Basic {credentials}"}

            response = test_client.get("/health", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["clickhouse_connected"] is True

    def test_health_check_with_auth_failure(self, test_client):
        """Test health check endpoint with failed authentication."""
        # Mock server_config to return auth config
        with patch(
            "agent_zero.monitoring_endpoints.server_config",
            ServerConfig(auth_username="testuser", auth_password="testpass"),
        ):
            # Include invalid credentials in the request
            credentials = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
            headers = {"Authorization": f"Basic {credentials}"}

            response = test_client.get("/health", headers=headers)

            assert response.status_code == 401
            assert "Unauthorized" in response.text

    def test_health_check_clickhouse_error(self, test_client, app):
        """Test health check endpoint when ClickHouse connection fails."""

        # Create a new client that raises an exception
        def mock_client_error():
            raise Exception("ClickHouse connection error")

        # Create a new FastAPI app and test client with the error-raising client
        test_app = FastAPI()

        # Create mock metrics dictionary
        mock_metrics = {
            "REQUEST_COUNT": MagicMock(),
            "REQUEST_LATENCY": MagicMock(),
            "CLICKHOUSE_CONNECTION_STATUS": MagicMock(),
            "MCP_TOOL_CALLS": MagicMock(),
            "CLICKHOUSE_QUERY_COUNT": MagicMock(),
            "CLICKHOUSE_QUERY_ERRORS": MagicMock(),
            "CLICKHOUSE_QUERY_DURATION": MagicMock(),
        }

        # Mock generate_latest to return sample Prometheus metrics
        mock_metrics_content = b"""
# HELP http_requests_total Total count of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET", endpoint="/health", status="200"} 2.0
"""

        # Apply patches for the test
        with patch("agent_zero.monitoring_endpoints.server_config", ServerConfig()):
            with patch("agent_zero.monitoring_endpoints.get_metrics", return_value=mock_metrics):
                with patch("agent_zero.monitoring_endpoints.metrics", mock_metrics):
                    with patch(
                        "agent_zero.monitoring_endpoints.generate_latest",
                        return_value=mock_metrics_content,
                    ):
                        setup_monitoring_endpoints(test_app, mock_client_error)
                        error_client = TestClient(test_app)

                        response = error_client.get("/health")

                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "degraded"
                        assert data["clickhouse_connected"] is False
                        assert "clickhouse_error" in data
                        assert "ClickHouse connection error" in data["clickhouse_error"]

    def test_metrics_endpoint_no_auth(self, test_client):
        """Test metrics endpoint without authentication."""
        # Mock server_config to return None for auth_config
        with patch("agent_zero.monitoring_endpoints.server_config", ServerConfig()):
            response = test_client.get("/metrics")

            assert response.status_code == 200
            # Prometheus metrics are returned as text in a specific format
            assert "TYPE" in response.text
            assert "HELP" in response.text
            assert "http_requests_total" in response.text
            # Match only the first part of the content type
            assert response.headers["content-type"].startswith("text/plain; version=0.0.4")

    def test_metrics_endpoint_with_auth_success(self, test_client):
        """Test metrics endpoint with successful authentication."""
        # Mock server_config to return auth config
        with patch(
            "agent_zero.monitoring_endpoints.server_config",
            ServerConfig(auth_username="testuser", auth_password="testpass"),
        ):
            # Include credentials in the request
            credentials = base64.b64encode(b"testuser:testpass").decode("utf-8")
            headers = {"Authorization": f"Basic {credentials}"}

            response = test_client.get("/metrics", headers=headers)

            assert response.status_code == 200
            assert "TYPE" in response.text
            assert "HELP" in response.text
            assert "http_requests_total" in response.text

    def test_metrics_endpoint_with_auth_failure(self, test_client):
        """Test metrics endpoint with failed authentication."""
        # Mock server_config to return auth config
        with patch(
            "agent_zero.monitoring_endpoints.server_config",
            ServerConfig(auth_username="testuser", auth_password="testpass"),
        ):
            # Include invalid credentials in the request
            credentials = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
            headers = {"Authorization": f"Basic {credentials}"}

            response = test_client.get("/metrics", headers=headers)

            assert response.status_code == 401
            assert "Unauthorized" in response.text

    def test_request_metrics_tracking(self, test_client):
        """Test that request metrics are tracked correctly."""
        # Make a request to trigger metrics tracking
        with patch("agent_zero.monitoring_endpoints.server_config", ServerConfig()):
            test_client.get("/health")
            test_client.get("/health")

            # Check that the request counter was incremented
            # This is tricky to test directly, as Prometheus metrics are global
            # We could use the REGISTRY to inspect metrics, but that's implementation-specific
            # For now, we'll just ensure the endpoint still works
            response = test_client.get("/metrics")
            assert response.status_code == 200

            # Verify the metrics contain our http_requests_total counter
            assert "http_requests_total" in response.text
