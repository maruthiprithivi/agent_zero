"""Tests for main.py module."""

import os
import sys
import unittest
from unittest import mock

from agent_zero.main import main


class TestMain(unittest.TestCase):
    """Test the main entry point."""

    def setUp(self):
        """Set up the test environment."""
        # Save sys.argv
        self.original_argv = sys.argv
        # Clear environment variables that might affect the tests
        self.clear_env()
        # Patch sys.exit to prevent the tests from actually exiting
        self.exit_patch = mock.patch("sys.exit")
        self.mock_exit = self.exit_patch.start()
        # Patch the run function to prevent it from actually running
        self.run_patch = mock.patch("agent_zero.main.run")
        self.mock_run = self.run_patch.start()
        # Patch logger to prevent it from actually logging
        self.logger_patch = mock.patch("agent_zero.main.logger")
        self.mock_logger = self.logger_patch.start()

    def tearDown(self):
        """Tear down the test environment."""
        # Restore sys.argv
        sys.argv = self.original_argv
        # Stop all patches
        self.exit_patch.stop()
        self.run_patch.stop()
        self.logger_patch.stop()

    def clear_env(self):
        """Clear environment variables that might affect the tests."""
        env_vars = [
            "MCP_SERVER_HOST",
            "MCP_SERVER_PORT",
            "MCP_SSL_CERTFILE",
            "MCP_SSL_KEYFILE",
            "MCP_AUTH_USERNAME",
            "MCP_AUTH_PASSWORD",
            "MCP_AUTH_PASSWORD_FILE",
            "MCP_CURSOR_MODE",
            "MCP_CURSOR_TRANSPORT",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_main_default_args(self):
        """Test main with default arguments."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 8505)
        self.assertIsNone(kwargs.get("cursor_mode"))

    def test_main_custom_host_port(self):
        """Test main with custom host and port."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--host", "0.0.0.0", "--port", "8080"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        self.assertEqual(kwargs["host"], "0.0.0.0")
        self.assertEqual(kwargs["port"], 8080)

    def test_main_auth_args(self):
        """Test main with authentication arguments."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--auth-username", "admin", "--auth-password", "pass"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.auth_username, "admin")
        self.assertEqual(server_config.auth_password, "pass")

    def test_main_auth_password_file(self):
        """Test main with auth password file."""
        # Set up sys.argv
        sys.argv = [
            "ch-agent-zero",
            "--auth-username",
            "admin",
            "--auth-password-file",
            "/path/to/password.txt",
        ]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.auth_username, "admin")
        self.assertEqual(server_config.auth_password_file, "/path/to/password.txt")

    def test_main_cursor_mode_agent(self):
        """Test main with Cursor IDE in agent mode."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--cursor-mode", "agent"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.cursor_mode, "agent")

    def test_main_cursor_mode_ask(self):
        """Test main with Cursor IDE in ask mode."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--cursor-mode", "ask"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.cursor_mode, "ask")

    def test_main_cursor_mode_edit(self):
        """Test main with Cursor IDE in edit mode."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--cursor-mode", "edit"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.cursor_mode, "edit")

    def test_main_cursor_transport(self):
        """Test main with custom Cursor IDE transport."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--cursor-mode", "agent", "--cursor-transport", "websocket"]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.cursor_mode, "agent")
        from agent_zero.server_config import TransportType

        self.assertEqual(server_config.cursor_transport, TransportType.WEBSOCKET)

    def test_main_cursor_mode_invalid(self):
        """Test main with invalid Cursor IDE mode."""
        # Set up sys.argv
        sys.argv = ["ch-agent-zero", "--cursor-mode", "invalid"]

        # We need to patch argparse.ArgumentParser.error to properly handle this case
        # since it calls sys.exit(2) internally
        with mock.patch("argparse.ArgumentParser.error", side_effect=SystemExit(2)):
            # Call main
            with self.assertRaises(SystemExit):
                main()

    def test_main_cursor_mode_with_auth(self):
        """Test main with Cursor IDE mode and authentication."""
        # Set up sys.argv
        sys.argv = [
            "ch-agent-zero",
            "--cursor-mode",
            "agent",
            "--auth-username",
            "admin",
            "--auth-password",
            "pass",
        ]
        # Call main
        main()
        # Check that run was called with the correct arguments
        self.mock_run.assert_called_once()
        args, kwargs = self.mock_run.call_args
        server_config = kwargs["server_config"]
        self.assertEqual(server_config.cursor_mode, "agent")
        self.assertEqual(server_config.auth_username, "admin")
        self.assertEqual(server_config.auth_password, "pass")
