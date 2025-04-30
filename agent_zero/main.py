"""Entry point for the ClickHouse Monitoring MCP Server."""

import argparse
import logging
import sys

from .mcp_server import run
from .server_config import ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ch-agent-zero")


def main():
    """Run the ClickHouse Monitoring MCP Server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ClickHouse Agent Zero MCP Server")

    # Server configuration
    parser.add_argument("--host", default=None, help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 8505)")

    # Authentication configuration
    parser.add_argument("--auth-username", help="Username for basic authentication")
    parser.add_argument(
        "--auth-password",
        help="Password for basic authentication (not recommended, use --auth-password-file instead)",
    )
    parser.add_argument(
        "--auth-password-file", help="Path to file containing password for authentication"
    )

    # Cursor IDE integration
    cursor_group = parser.add_argument_group("Cursor IDE Integration")
    cursor_group.add_argument(
        "--cursor-mode",
        choices=["agent", "ask", "edit"],
        help="Cursor IDE mode to support (agent, ask, or edit)",
    )
    cursor_group.add_argument(
        "--cursor-transport",
        choices=["sse", "websocket"],
        help="Transport to use with Cursor IDE (default: sse)",
    )

    args = parser.parse_args()

    # Create ServerConfig with command-line overrides
    server_config_values = {}
    if args.host:
        server_config_values["host"] = args.host
    if args.port:
        server_config_values["port"] = args.port
    if args.auth_username:
        server_config_values["auth_username"] = args.auth_username
    if args.auth_password:
        server_config_values["auth_password"] = args.auth_password
    if args.auth_password_file:
        server_config_values["auth_password_file"] = args.auth_password_file
    if args.cursor_mode:
        server_config_values["cursor_mode"] = args.cursor_mode
    if args.cursor_transport:
        server_config_values["cursor_transport"] = args.cursor_transport

    server_config = ServerConfig(**server_config_values)

    try:
        logger.debug("Starting ch-agent-zero entry point")
        logger.debug(f"Python path: {sys.path}")
        logger.debug(f"Current working directory: {sys.path[0]}")

        # Log server configuration
        logger.info(f"Starting server on {server_config.host}:{server_config.port}")

        auth_config = server_config.get_auth_config()
        if auth_config:
            logger.info(f"Authentication is enabled for user: {auth_config['username']}")

        # Log Cursor IDE mode if specified
        if server_config.cursor_mode:
            logger.info(f"Running in Cursor IDE mode: {server_config.cursor_mode}")

        # Run the MCP server with the configuration
        run(
            host=server_config.host,
            port=server_config.port,
            server_config=server_config,
        )
    except Exception as e:
        logger.error(f"Error in main entry point: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
