"""Tests for MCP server integration with ClickHouse.

This module tests the integration between the MCP server and ClickHouse,
focusing on the connection handling and configuration.
"""

import unittest
from unittest.mock import MagicMock, patch

from agent_zero.config.unified import UnifiedConfig


class TestMCPServerIntegration(unittest.TestCase):
    """Tests for MCP server integration with ClickHouse."""

    def test_run_with_default_config(self):
        """Test running the MCP server with default configuration."""
        # Create mocks
        mock_mcp = MagicMock()

        # Create a server config with default values
        server_config = UnifiedConfig.from_env()

        # Import the run function
        from agent_zero.server import run

        # Patch mcp in the module scope
        with patch("agent_zero.server.mcp", mock_mcp):
            # Call the run method with mocked dependencies injected
            run(
                host="127.0.0.1",
                port=8505,
                server_config=server_config,
            )

            # Verify mcp.run was called with the correct parameters
            mock_mcp.run.assert_called_once_with(
                host="127.0.0.1",
                port=8505,
            )

    def test_run_with_ssl_config(self):
        """Test running the MCP server with SSL configuration."""
        # Create mocks
        mock_mcp = MagicMock()

        # Create a server config with SSL configuration
        server_config = UnifiedConfig(ssl_certfile="cert.pem", ssl_keyfile="key.pem")
        ssl_config = {"certfile": "cert.pem", "keyfile": "key.pem"}

        # Import the run function
        from agent_zero.server import run

        # Patch mcp in the module scope
        with patch("agent_zero.server.mcp", mock_mcp):
            # Call the run method with mocked dependencies injected
            run(
                host="127.0.0.1",
                port=8505,
                ssl_config=ssl_config,
                server_config=server_config,
            )

            # Verify mcp.run was called with the correct parameters
            mock_mcp.run.assert_called_once_with(
                host="127.0.0.1",
                port=8505,
                ssl_certfile="cert.pem",
                ssl_keyfile="key.pem",
            )

    @patch("agent_zero.server.create_clickhouse_client")
    def test_client_creation(self, mock_create_client):
        """Test that the ClickHouse client is created correctly."""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.server_version = "23.8.1"
        mock_create_client.return_value = mock_client

        # Import the necessary function
        from agent_zero.server import list_databases

        # Call the function
        list_databases()

        # Verify the client was created
        mock_create_client.assert_called_once()
