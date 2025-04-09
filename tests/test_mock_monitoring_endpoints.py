"""Tests for mock_monitoring_endpoints.py."""

import base64
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.mock_monitoring_endpoints import setup_monitoring_endpoints, MockServerConfig


class TestMockMonitoringEndpoints:
    """Tests for the mock monitoring endpoints."""

    @pytest.fixture
    def app(self):
        """Create a FastAPI app for testing."""
        app = FastAPI()
        return app

    @pytest.fixture
    def mock_clickhouse_client(self):
        """Create a mock ClickHouse client."""
        mock_client = MagicMock()
        mock_client.server_version = "23.8.1"
        return mock_client

    @pytest.fixture
    def test_helpers(self, app, mock_clickhouse_client):
        """Set up the app and return test helpers."""

        # Create a function that returns the mock client
        def create_mock_client():
            return mock_clickhouse_client

        # Set up monitoring endpoints and get test helpers
        return setup_monitoring_endpoints(app, create_mock_client)

    @pytest.fixture
    def test_client(self, app):
        """Create a test client for the app with monitoring endpoints."""
        return TestClient(app)

    def test_health_check_no_auth(self, test_client, test_helpers):
        """Test health check endpoint without authentication."""
        # Default config has no auth
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["server"] == "agent-zero"
        assert data["clickhouse_connected"] is True
        assert data["clickhouse_version"] == "23.8.1"
        assert "timestamp" in data

    def test_health_check_with_auth_success(self, test_client, test_helpers):
        """Test health check endpoint with successful authentication."""
        # Set auth config
        test_helpers["set_config"](
            MockServerConfig(auth_config={"username": "testuser", "password": "testpass"})
        )

        # Include credentials in the request
        credentials = base64.b64encode(b"testuser:testpass").decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}

        response = test_client.get("/health", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["clickhouse_connected"] is True

    def test_health_check_with_auth_failure(self, test_client, test_helpers):
        """Test health check endpoint with failed authentication."""
        # Set auth config
        test_helpers["set_config"](
            MockServerConfig(auth_config={"username": "testuser", "password": "testpass"})
        )

        # Include invalid credentials in the request
        credentials = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}

        response = test_client.get("/health", headers=headers)

        assert response.status_code == 401
        assert "Unauthorized" in response.text

    def test_health_check_clickhouse_error(self, app):
        """Test health check endpoint when ClickHouse connection fails."""

        # Create a function that raises an exception
        def mock_client_error():
            raise Exception("ClickHouse connection error")

        # Create a new app and set up monitoring endpoints with the error-raising function
        test_app = FastAPI()
        setup_monitoring_endpoints(test_app, mock_client_error)
        test_client = TestClient(test_app)

        # Call the health endpoint
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["clickhouse_connected"] is False
        assert "clickhouse_error" in data
        assert "ClickHouse connection error" in data["clickhouse_error"]

    def test_metrics_endpoint_no_auth(self, test_client, test_helpers):
        """Test metrics endpoint without authentication."""
        # Default config has no auth
        response = test_client.get("/metrics")

        assert response.status_code == 200
        # Check for the mock metrics content
        assert "# HELP test_metric Test metric" in response.text
        assert "# TYPE test_metric gauge" in response.text
        assert "test_metric 1.0" in response.text
        assert response.headers["content-type"] == "text/plain; version=0.0.4"

    def test_metrics_endpoint_with_auth_success(self, test_client, test_helpers):
        """Test metrics endpoint with successful authentication."""
        # Set auth config
        test_helpers["set_config"](
            MockServerConfig(auth_config={"username": "testuser", "password": "testpass"})
        )

        # Include credentials in the request
        credentials = base64.b64encode(b"testuser:testpass").decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}

        response = test_client.get("/metrics", headers=headers)

        assert response.status_code == 200
        assert "# HELP test_metric Test metric" in response.text

    def test_metrics_endpoint_with_auth_failure(self, test_client, test_helpers):
        """Test metrics endpoint with failed authentication."""
        # Set auth config
        test_helpers["set_config"](
            MockServerConfig(auth_config={"username": "testuser", "password": "testpass"})
        )

        # Include invalid credentials in the request
        credentials = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}

        response = test_client.get("/metrics", headers=headers)

        assert response.status_code == 401
        assert "Unauthorized" in response.text
