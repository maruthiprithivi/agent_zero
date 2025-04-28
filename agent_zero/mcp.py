"""Module to expose the MCP app object for ASGI loading."""

from agent_zero.mcp_server import mcp

# Export the app object for ASGI loading
app = mcp.app
