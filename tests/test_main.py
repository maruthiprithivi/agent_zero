"""Tests for main.py module."""

import sys
from unittest.mock import patch, MagicMock
import pytest

from agent_zero.main import main


class TestMain:
    """Tests for main entry point."""

    @patch("agent_zero.mcp_server.run")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero"])
    def test_main_default_args(self, mock_server_config, mock_run):
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

        # Verify run was called with correct arguments
        mock_run.assert_called_once_with(
            host="127.0.0.1",
            port=8505,
            server_config=mock_config_instance,
        )

    @patch("agent_zero.mcp_server.run")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero", "--host", "0.0.0.0", "--port", "9000"])
    def test_main_custom_host_port(self, mock_server_config, mock_run):
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

        # Verify run was called with the correct arguments
        mock_run.assert_called_once_with(
            host="0.0.0.0",
            port=9000,
            server_config=mock_config_instance,
        )

    @patch("agent_zero.mcp_server.run")
    @patch("agent_zero.main.ServerConfig")
    @patch(
        "sys.argv", ["ch-agent-zero", "--auth-username", "testuser", "--auth-password", "testpass"]
    )
    def test_main_auth_config(self, mock_server_config, mock_run):
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

        # Verify run was called with the correct arguments
        mock_run.assert_called_once_with(
            host="127.0.0.1",
            port=8505,
            server_config=mock_config_instance,
        )

    @patch("agent_zero.mcp_server.run")
    @patch("agent_zero.main.ServerConfig")
    @patch(
        "sys.argv",
        ["ch-agent-zero", "--auth-username", "testuser", "--auth-password-file", "password.txt"],
    )
    def test_main_auth_password_file(self, mock_server_config, mock_run):
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

        # Verify run was called with the correct arguments
        mock_run.assert_called_once_with(
            host="127.0.0.1",
            port=8505,
            server_config=mock_config_instance,
        )

    @patch("agent_zero.mcp_server.run")
    @patch("agent_zero.main.ServerConfig")
    @patch("sys.argv", ["ch-agent-zero"])
    def test_main_exception_handling(self, mock_server_config, mock_run):
        """Test main function handles exceptions correctly."""
        # Make run raise an exception
        mock_run.side_effect = Exception("Test exception")

        # Setup mock objects
        mock_config_instance = MagicMock()
        mock_server_config.return_value = mock_config_instance
        mock_config_instance.host = "127.0.0.1"
        mock_config_instance.port = 8505
        mock_config_instance.get_ssl_config.return_value = None
        mock_config_instance.get_auth_config.return_value = None

        # Call the main function and expect it to raise the exception
        with pytest.raises(Exception, match="Test exception"):
            main()
