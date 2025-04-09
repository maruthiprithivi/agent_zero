"""Tests for main.py."""

from unittest.mock import patch, MagicMock

import pytest

from agent_zero.main import main


class TestMain:
    """Tests for the main entry point functionality."""

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero"])
    def test_main_default_args(self, mock_server_config, mock_mcp):
        """Test main function with default arguments."""
        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "127.0.0.1"
        mock_config_instance.port = 8505
        mock_config_instance.get_ssl_config.return_value = None
        mock_config_instance.get_auth_config.return_value = None

        # Call the main function
        main()

        # Verify ServerConfig was called with no arguments
        mock_server_config.assert_called_once_with()

        # Verify the mcp.run was called with the correct arguments
        mock_mcp.run.assert_called_once_with(
            host="127.0.0.1", port=8505, ssl_config=None, server_config=mock_config_instance
        )

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero", "--host", "0.0.0.0", "--port", "9000"])
    def test_main_custom_host_port(self, mock_server_config, mock_mcp):
        """Test main function with custom host and port."""
        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "0.0.0.0"
        mock_config_instance.port = 9000
        mock_config_instance.get_ssl_config.return_value = None
        mock_config_instance.get_auth_config.return_value = None

        # Call the main function
        main()

        # Verify ServerConfig was called with the correct arguments
        mock_server_config.assert_called_once_with(host="0.0.0.0", port=9000)

        # Verify the mcp.run was called with the correct arguments
        mock_mcp.run.assert_called_once_with(
            host="0.0.0.0", port=9000, ssl_config=None, server_config=mock_config_instance
        )

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero", "--ssl-certfile", "cert.pem", "--ssl-keyfile", "key.pem"])
    def test_main_ssl_config(self, mock_server_config, mock_mcp):
        """Test main function with SSL configuration."""
        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "127.0.0.1"
        mock_config_instance.port = 8505

        # Configure SSL
        ssl_config = {"certfile": "cert.pem", "keyfile": "key.pem"}
        mock_config_instance.get_ssl_config.return_value = ssl_config
        mock_config_instance.get_auth_config.return_value = None

        # Call the main function
        main()

        # Verify ServerConfig was called with the correct arguments
        mock_server_config.assert_called_once_with(ssl_certfile="cert.pem", ssl_keyfile="key.pem")

        # Verify the mcp.run was called with the correct arguments
        mock_mcp.run.assert_called_once_with(
            host="127.0.0.1", port=8505, ssl_config=ssl_config, server_config=mock_config_instance
        )

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch(
        "sys.argv", ["ch-agent-zero", "--auth-username", "testuser", "--auth-password", "testpass"]
    )
    def test_main_auth_config(self, mock_server_config, mock_mcp):
        """Test main function with authentication configuration."""
        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "127.0.0.1"
        mock_config_instance.port = 8505
        mock_config_instance.get_ssl_config.return_value = None

        # Configure authentication
        auth_config = {"username": "testuser", "password": "testpass"}
        mock_config_instance.get_auth_config.return_value = auth_config

        # Call the main function
        main()

        # Verify ServerConfig was called with the correct arguments
        mock_server_config.assert_called_once_with(
            auth_username="testuser", auth_password="testpass"
        )

        # Verify the mcp.run was called with the correct arguments
        mock_mcp.run.assert_called_once_with(
            host="127.0.0.1", port=8505, ssl_config=None, server_config=mock_config_instance
        )

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch(
        "sys.argv",
        ["ch-agent-zero", "--auth-username", "testuser", "--auth-password-file", "password.txt"],
    )
    def test_main_auth_password_file(self, mock_server_config, mock_mcp):
        """Test main function with authentication password file configuration."""
        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "127.0.0.1"
        mock_config_instance.port = 8505
        mock_config_instance.get_ssl_config.return_value = None

        # Configure authentication
        auth_config = {"username": "testuser", "password": "password_from_file"}
        mock_config_instance.get_auth_config.return_value = auth_config

        # Call the main function
        main()

        # Verify ServerConfig was called with the correct arguments
        mock_server_config.assert_called_once_with(
            auth_username="testuser", auth_password_file="password.txt"
        )

        # Verify the mcp.run was called with the correct arguments
        mock_mcp.run.assert_called_once_with(
            host="127.0.0.1", port=8505, ssl_config=None, server_config=mock_config_instance
        )

    @patch("agent_zero.main.mcp")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero"])
    def test_main_exception_handling(self, mock_server_config, mock_mcp):
        """Test main function handles exceptions correctly."""
        # Make mcp.run raise an exception
        mock_mcp.run.side_effect = Exception("Test exception")

        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.get_ssl_config.return_value = None
        mock_config_instance.get_auth_config.return_value = None

        # Call the main function and expect it to raise the exception
        with pytest.raises(Exception):
            main()
