"""Tests for Cursor IDE support in the MCP server."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest

from agent_zero.server_config import ServerConfig

# Skip tests if mcp is not available
pytest.importorskip("mcp")


class TestCursorIDESupport(unittest.TestCase):
    """Test Cursor IDE support in the MCP server."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = os.environ.copy()

        # We need to patch mcp first, then import the run function after
        # This prevents import-time issues with circular references
        self.mock_mcp = MagicMock()
        self.mock_run = MagicMock()
        self.mock_mcp.run = self.mock_run

        # Apply multiple patches to ensure we don't get recursion
        self.patches = [
            patch("agent_zero.mcp_server.mcp", self.mock_mcp),
            patch("agent_zero.mcp_server._original_mcp", self.mock_mcp),
            patch("agent_zero.mcp_server._original_run", self.mock_run),
        ]

        # Start all patches
        for p in self.patches:
            p.start()

        # Import run function only after patching to avoid circular imports
        from agent_zero.mcp_server import run

        self.run_func = run

    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

        # Stop all patches
        for p in self.patches:
            p.stop()

    def test_cursor_mode_agent(self):
        """Test Cursor IDE in agent mode."""
        # Set up config
        server_config = ServerConfig(cursor_mode="agent", cursor_transport="sse")

        # Run the server
        self.run_func(server_config=server_config)

        # Check if run was called with correct arguments
        self.mock_run.assert_called_once()
        call_args = self.mock_run.call_args[1]
        self.assertEqual(call_args["transport"], "sse")
        self.assertEqual(call_args["host"], "127.0.0.1")
        self.assertEqual(call_args["port"], 8505)

    def test_cursor_mode_ask(self):
        """Test Cursor IDE in ask mode."""
        # Set up config
        server_config = ServerConfig(cursor_mode="ask", cursor_transport="websocket")

        # Run the server
        self.run_func(host="localhost", port=9000, server_config=server_config)

        # Check if run was called with correct arguments
        self.mock_run.assert_called_once()
        call_args = self.mock_run.call_args[1]
        self.assertEqual(call_args["transport"], "websocket")
        self.assertEqual(call_args["host"], "localhost")
        self.assertEqual(call_args["port"], 9000)

    def test_cursor_mode_edit(self):
        """Test Cursor IDE in edit mode."""
        # Set up config
        server_config = ServerConfig(cursor_mode="edit")

        # Run the server
        self.run_func(server_config=server_config)

        # Check if run was called with correct arguments
        self.mock_run.assert_called_once()
        call_args = self.mock_run.call_args[1]
        self.assertEqual(call_args["transport"], "sse")  # Default transport

    def test_cursor_transport_override(self):
        """Test overriding Cursor IDE transport via environment variable."""
        # Set environment variables
        os.environ["MCP_CURSOR_MODE"] = "agent"
        os.environ["MCP_CURSOR_TRANSPORT"] = "websocket"

        # Create config from environment
        server_config = ServerConfig()

        # Run the server
        self.run_func(server_config=server_config)

        # Check if run was called with correct arguments
        self.mock_run.assert_called_once()
        call_args = self.mock_run.call_args[1]
        self.assertEqual(call_args["transport"], "websocket")

    def test_no_cursor_mode(self):
        """Test server without Cursor IDE mode."""
        # Run the server without Cursor IDE mode
        self.run_func()

        # Check if run was called with default arguments
        self.mock_run.assert_called_once()
        call_args = self.mock_run.call_args[1]
        self.assertNotIn("transport", call_args)


if __name__ == "__main__":
    unittest.main()
