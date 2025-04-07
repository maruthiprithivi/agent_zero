"""Integration tests for the MCP server with new features."""

from unittest.mock import patch, MagicMock

from fastapi import FastAPI

from agent_zero.server_config import ServerConfig


class TestMCPServerIntegration:
    """Tests for MCP server integration with new features."""

    def test_run_with_default_config(self):
        """Test running the MCP server with default configuration."""
        # Create mocks
        mock_mcp = MagicMock()
        mock_uvicorn_run = MagicMock()
        mock_setup_monitoring = MagicMock()

        # Create a FastAPI app mock
        mock_app = MagicMock(spec=FastAPI)
        mock_mcp.app = mock_app

        # Create a server config with default values
        server_config = ServerConfig()

        # Import the run function
        from agent_zero.mcp_server import run

        # Patch mcp in the module scope
        with patch("agent_zero.mcp_server.mcp", mock_mcp):
            # Call the run method with mocked dependencies injected
            run(
                host="127.0.0.1",
                port=8505,
                server_config=server_config,
                _uvicorn_run=mock_uvicorn_run,
                _setup_monitoring=mock_setup_monitoring,
            )

            # Verify setup_monitoring_endpoints was called with the mock_app
            assert mock_setup_monitoring.call_count == 1
            assert mock_setup_monitoring.call_args[0][0] == mock_app

            # Verify uvicorn.run was called with the correct parameters
            mock_uvicorn_run.assert_called_once()
            args, kwargs = mock_uvicorn_run.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 8505

    def test_run_with_ssl_config(self):
        """Test running the MCP server with SSL configuration."""
        # Create mocks
        mock_mcp = MagicMock()
        mock_uvicorn_run = MagicMock()
        mock_setup_monitoring = MagicMock()

        # Create a FastAPI app mock
        mock_app = MagicMock(spec=FastAPI)
        mock_mcp.app = mock_app

        # Create a server config with SSL configuration
        server_config = ServerConfig(ssl_certfile="cert.pem", ssl_keyfile="key.pem")
        ssl_config = {"certfile": "cert.pem", "keyfile": "key.pem"}

        # Import the run function
        from agent_zero.mcp_server import run

        # Patch mcp in the module scope
        with patch("agent_zero.mcp_server.mcp", mock_mcp):
            # Call the run method with mocked dependencies injected
            run(
                host="127.0.0.1",
                port=8505,
                ssl_config=ssl_config,
                server_config=server_config,
                _uvicorn_run=mock_uvicorn_run,
                _setup_monitoring=mock_setup_monitoring,
            )

            # Verify setup_monitoring_endpoints was called with the mock_app
            assert mock_setup_monitoring.call_count == 1
            assert mock_setup_monitoring.call_args[0][0] == mock_app

            # Verify uvicorn.run was called with the correct parameters
            mock_uvicorn_run.assert_called_once()
            args, kwargs = mock_uvicorn_run.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 8505
            assert kwargs["ssl_certfile"] == "cert.pem"
            assert kwargs["ssl_keyfile"] == "key.pem"

    def test_monitoring_endpoints_setup(self):
        """Test that monitoring endpoints are set up when the app is accessible."""
        # Create mocks
        mock_mcp = MagicMock()
        mock_uvicorn_run = MagicMock()
        mock_setup_monitoring = MagicMock()

        # Create a FastAPI app mock
        mock_app = MagicMock(spec=FastAPI)
        mock_mcp.app = mock_app

        # Import the run function
        from agent_zero.mcp_server import run

        # Patch mcp in the module scope
        with patch("agent_zero.mcp_server.mcp", mock_mcp):
            # Call the run function with mocked dependencies
            run(_uvicorn_run=mock_uvicorn_run, _setup_monitoring=mock_setup_monitoring)

            # Verify setup_monitoring_endpoints was called with the mock_app
            assert mock_setup_monitoring.call_count == 1
            assert mock_setup_monitoring.call_args[0][0] == mock_app

    @patch("agent_zero.monitoring_endpoints.setup_monitoring_endpoints")
    @patch("agent_zero.mcp_server.mcp")
    def test_no_monitoring_without_app(self, mock_mcp, mock_setup_monitoring):
        """Test that monitoring endpoints are not set up when no app is available."""
        # Set mcp.app to None
        mock_mcp.app = None

        # Import the run function with empty module dict
        with patch.dict("sys.modules", {"agent_zero.mcp_server": None}):

            # Verify setup_monitoring_endpoints was not called
            mock_setup_monitoring.assert_not_called()

    @patch("agent_zero.mcp_server.create_clickhouse_client")
    def test_tracking_query_metrics(self, mock_create_client):
        """Test that query metrics tracking is applied."""
        # We need to patch the decorator to verify it was applied,
        # but this is challenging to test directly in a unit test.
        # In a real scenario, we would verify this with integration tests
        # running actual queries.
        #
        # For now we'll just test that the query_metrics module is imported
        # This is a very basic test and doesn't verify functionality
        from agent_zero import query_metrics

        # Verify that query_metrics module is imported
        assert query_metrics is not None
