"""
Tests for the standalone server features.

This module tests the integrated functionality of the standalone server features,
including server configuration.
"""

from unittest.mock import MagicMock, patch

from agent_zero.config.unified import UnifiedConfig


class TestStandaloneServer:
    """Test the standalone server features."""

    def test_server_config_integration(self):
        """Test that the server config can be created and used for configuration."""
        # Create a server config with custom values
        server_config = UnifiedConfig(
            clickhouse_host="localhost",
            clickhouse_user="default",
            clickhouse_password="",
            server_host="0.0.0.0",
            server_port=9000,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
            ssl_enable=True,
            auth_username="admin",
            auth_password="secure_password",
        )

        # Verify the config values
        assert server_config.server_host == "0.0.0.0"
        assert server_config.server_port == 9000
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

    def test_mcp_server_run_integration(self):
        """Test that the run function in mcp_server.py correctly starts the MCP server."""
        # Create a server config to pass to the run function
        server_config = UnifiedConfig(
            clickhouse_host="localhost",
            clickhouse_user="default",
            clickhouse_password="",
            server_host="localhost",
            server_port=8088,
        )

        try:
            # Import the function to test
            from agent_zero.server import run

            # Use patching to make mcp.run return without actually starting the server
            with patch("agent_zero.server.mcp") as mock_mcp:
                # Make a mock run method that returns immediately
                mock_run = MagicMock(return_value=None)
                mock_mcp.run = mock_run

                # Call the run function
                run(host="localhost", port=8088, server_config=server_config)

                # Verify mcp.run was called with the correct parameters
                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert kwargs["host"] == "localhost"
                assert kwargs["port"] == 8088
        finally:
            # Restore the original function if needed
            pass
