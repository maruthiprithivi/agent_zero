"""Tests for stub monitoring_endpoints functionality.

This test suite uses the stub implementation provided in tests/stubs/monitoring_endpoints.py
to ensure that tests that previously relied on monitoring_endpoints.py still work after
removing FastAPI and uvicorn dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.stubs.monitoring_endpoints import setup_monitoring_endpoints


class TestMonitoringEndpointsStubs:
    """Tests for the stub monitoring endpoints implementation."""

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

        # Set up monitoring endpoint stubs on the app
        setup_monitoring_endpoints(app, create_mock_client)
        return TestClient(app)

    def test_health_check_endpoint(self, test_client):
        """Test that the stub health check endpoint works."""
        # Mock server_config to return None for auth_config
        with patch("tests.stubs.monitoring_endpoints.logger"):
            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["clickhouse_connected"] is True
            assert data["clickhouse_version"] == "23.1.2.3"

    def test_metrics_endpoint(self, test_client):
        """Test that the stub metrics endpoint works."""
        # Make a request to the stub metrics endpoint
        with patch("tests.stubs.monitoring_endpoints.logger"):
            response = test_client.get("/metrics")

            assert response.status_code == 200
            assert "test_metric" in response.text
