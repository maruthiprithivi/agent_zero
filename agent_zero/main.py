"""Entry point for the ClickHouse Monitoring MCP Server."""

from .mcp_server import mcp


def main():
    """Run the ClickHouse Monitoring MCP Server."""
    mcp.run()


if __name__ == "__main__":
    main()