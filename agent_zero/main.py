"""Entry point for the ClickHouse Monitoring MCP Server."""

import logging
import sys

from .mcp_server import mcp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ch-agent-zero")


def main():
    """Run the ClickHouse Monitoring MCP Server."""
    try:
        logger.debug("Starting ch-agent-zero entry point")
        logger.debug(f"Python path: {sys.path}")
        logger.debug(f"Current working directory: {sys.path[0]}")
        mcp.run()
    except Exception as e:
        logger.error(f"Error in main entry point: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
