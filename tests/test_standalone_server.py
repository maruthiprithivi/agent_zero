"""
Tests for the standalone server features.

This module tests the integrated functionality of the standalone server features,
including server configuration, SSL/TLS support, basic authentication, and monitoring endpoints.
"""

import base64
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_zero.server_config import ServerConfig
from agent_zero.monitoring_endpoints import setup_monitoring_endpoints


class TestStandaloneServer:
    """Test the standalone server features."""

    def test_server_config_integration(self):
        """Test that the server config can be created and used for configuration."""
        # Create a server config with custom values
        server_config = ServerConfig(
            host="0.0.0.0",
            port=9000,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
            auth_username="admin",
            auth_password="secure_password",
        )

        # Verify the config values
        assert server_config.host == "0.0.0.0"
        assert server_config.port == 9000
        assert server_config.ssl_certfile == "cert.pem"
        assert server_config.ssl_keyfile == "key.pem"
        assert server_config.auth_username == "admin"
        assert server_config.auth_password == "secure_password"

        # Verify SSL config
        ssl_config = server_config.get_ssl_config()
        assert ssl_config["certfile"] == "cert.pem"
        assert ssl_config["keyfile"] == "key.pem"

        # Verify auth config
        auth_config = server_config.get_auth_config()
        assert auth_config["username"] == "admin"
        assert auth_config["password"] == "secure_password"

    def test_setup_monitoring_endpoints(self):
        """Test that monitoring endpoints can be set up."""
        # Create a FastAPI app
        app = FastAPI()

        # Mock ClickHouse client creator
        def mock_clickhouse_client():
            mock_client = MagicMock()
            mock_client.server_version = "23.1.2.3"
            return mock_client

        # Mock the metrics retrieval function
        mock_metrics = {
            "REQUEST_COUNT": MagicMock(),
            "REQUEST_LATENCY": MagicMock(),
            "CLICKHOUSE_CONNECTION_STATUS": MagicMock(),
            "CLICKHOUSE_QUERY_COUNT": MagicMock(),
        }

        # Set up monitoring endpoints
        with patch("agent_zero.monitoring_endpoints.get_metrics", return_value=mock_metrics):
            with patch("agent_zero.monitoring_endpoints.metrics", mock_metrics):
                setup_monitoring_endpoints(app, create_clickhouse_client=mock_clickhouse_client)

                # Create a test client
                client = TestClient(app)

                # Test health endpoint
                response = client.get("/health")
                assert response.status_code == 200
                assert "healthy" in response.text
                assert "clickhouse_connected" in response.text

    def test_auth_middleware(self):
        """Test that authentication middleware works correctly."""
        # Create a FastAPI app
        app = FastAPI()

        # Create a server config with auth
        server_config = ServerConfig(auth_username="admin", auth_password="secure_password")

        # Set up a test endpoint with authentication
        @app.get("/protected")
        def protected_endpoint():
            return {"message": "authenticated"}

        # Mock the metrics retrieval function
        mock_metrics = {
            "REQUEST_COUNT": MagicMock(),
            "REQUEST_LATENCY": MagicMock(),
            "CLICKHOUSE_CONNECTION_STATUS": MagicMock(),
            "CLICKHOUSE_QUERY_COUNT": MagicMock(),
        }

        # Apply auth middleware to the app
        with patch("agent_zero.monitoring_endpoints.server_config", server_config):
            with patch("agent_zero.monitoring_endpoints.get_metrics", return_value=mock_metrics):
                with patch("agent_zero.monitoring_endpoints.metrics", mock_metrics):
                    setup_monitoring_endpoints(app, create_clickhouse_client=lambda: MagicMock())

                    # Create a test client
                    client = TestClient(app)

                    # Test without auth
                    response = client.get("/health")
                    assert response.status_code == 401

                    # Test with correct auth
                    credentials = base64.b64encode(b"admin:secure_password").decode("utf-8")
                    headers = {"Authorization": f"Basic {credentials}"}
                    response = client.get("/health", headers=headers)
                    assert response.status_code == 200

    @patch("uvicorn.run")  # Patch uvicorn.run to prevent actual server startup
    def test_mcp_server_run_integration(self, mock_uvicorn_run):
        """Test that the run function in mcp_server.py correctly passes parameters to uvicorn and setup_monitoring_endpoints."""
        # Create a module-level version of setup_monitoring_endpoints to test
        original_setup_monitoring = setup_monitoring_endpoints

        # Track if setup_monitoring_endpoints is called
        setup_called = False

        def mock_setup_monitoring(app, create_clickhouse_client):
            nonlocal setup_called
            setup_called = True
            return None  # Mock implementation

        # Create a server config to pass to the run function
        server_config = ServerConfig(host="localhost", port=8088)

        try:
            # Replace the setup_monitoring_endpoints function with our mock
            import agent_zero.monitoring_endpoints

            agent_zero.monitoring_endpoints.setup_monitoring_endpoints = mock_setup_monitoring

            # Now import mcp and create a fake app
            import agent_zero.mcp_server
            from unittest.mock import patch

            # Use patching to make mcp.app exist so that setup_monitoring_endpoints is called
            with patch("agent_zero.mcp_server.mcp") as mock_mcp:
                mock_mcp.app = MagicMock(spec=FastAPI)

                # Now import and run the function
                from agent_zero.mcp_server import run

                run(host="localhost", port=8088, server_config=server_config)

                # Check if our mock was called
                assert setup_called, "setup_monitoring_endpoints was not called"

                # Verify uvicorn.run was called with the correct parameters
                mock_uvicorn_run.assert_called_once()
                args, kwargs = mock_uvicorn_run.call_args
                assert kwargs["host"] == "localhost"
                assert kwargs["port"] == 8088
        finally:
            # Restore the original function
            agent_zero.monitoring_endpoints.setup_monitoring_endpoints = original_setup_monitoring
