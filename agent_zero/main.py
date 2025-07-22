"""Entry point for the ClickHouse Monitoring MCP Server.

Supports multiple deployment modes and IDE integrations (2025 edition):
- Local development and testing
- Standalone server deployment  
- Enterprise deployment with advanced features
- IDE integrations: Claude Desktop, Claude Code, Cursor, Windsurf, VS Code
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .mcp_server import run
from .server_config import ServerConfig, DeploymentMode, IDEType, TransportType

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ch-agent-zero")


def generate_config():
    """Generate configuration files for supported IDEs."""
    parser = argparse.ArgumentParser(description="Generate MCP configuration for IDEs")
    parser.add_argument(
        "--ide",
        choices=["claude-desktop", "claude-code", "cursor", "windsurf", "vscode"],
        required=True,
        help="Target IDE for configuration generation",
    )
    parser.add_argument(
        "--deployment-mode",
        choices=["local", "standalone", "enterprise"],
        default="local",
        help="Deployment mode (default: local)",
    )
    parser.add_argument(
        "--output", 
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--install-path",
        help="Path to ch-agent-zero installation (for local configs)"
    )
    
    args = parser.parse_args()
    
    # Create server config for the specified deployment mode
    server_config = ServerConfig(deployment_mode=args.deployment_mode)
    
    # Map CLI IDE names to enum values
    ide_map = {
        "claude-desktop": IDEType.CLAUDE_DESKTOP,
        "claude-code": IDEType.CLAUDE_CODE,
        "cursor": IDEType.CURSOR,
        "windsurf": IDEType.WINDSURF,
        "vscode": IDEType.VSCODE,
    }
    
    ide_type = ide_map[args.ide]
    config = server_config.generate_ide_config(ide_type, args.install_path)
    
    # Output configuration
    config_json = json.dumps(config, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(config_json)
        print(f"Configuration written to {args.output}")
    else:
        print(config_json)

def main():
    """Run the ClickHouse Monitoring MCP Server."""
    # Check if this is a config generation command
    if len(sys.argv) > 1 and sys.argv[1] == "generate-config":
        # Remove the generate-config argument and run config generation
        sys.argv.pop(1)
        return generate_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="ClickHouse Agent Zero MCP Server - 2025 Multi-IDE Edition",
        epilog="Examples:\n"
               "  ch-agent-zero                                    # Local stdio mode\n"
               "  ch-agent-zero --deployment-mode standalone      # Standalone SSE server\n"
               "  ch-agent-zero --ide-type claude-code           # Optimized for Claude Code\n"
               "  ch-agent-zero generate-config --ide cursor     # Generate Cursor config\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Deployment configuration
    deployment_group = parser.add_argument_group("Deployment Configuration")
    deployment_group.add_argument(
        "--deployment-mode",
        choices=["local", "standalone", "enterprise"],
        help="Deployment mode (local|standalone|enterprise)"
    )
    deployment_group.add_argument(
        "--ide-type",
        choices=["claude-desktop", "claude-code", "cursor", "windsurf", "vscode"],
        help="Target IDE type for optimization"
    )
    deployment_group.add_argument(
        "--transport",
        choices=["stdio", "sse", "websocket", "http"],
        help="Default transport type"
    )

    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--host", help="Host to bind to (default: 127.0.0.1)")
    server_group.add_argument("--port", type=int, help="Port to bind to (default: 8505)")
    server_group.add_argument("--ssl-enable", action="store_true", help="Enable SSL")
    server_group.add_argument("--ssl-certfile", help="SSL certificate file path")
    server_group.add_argument("--ssl-keyfile", help="SSL key file path")

    # Authentication configuration
    auth_group = parser.add_argument_group("Authentication Configuration")
    auth_group.add_argument("--auth-username", help="Username for basic authentication")
    auth_group.add_argument(
        "--auth-password",
        help="Password for basic authentication (not recommended, use --auth-password-file)",
    )
    auth_group.add_argument(
        "--auth-password-file", help="Path to file containing password for authentication"
    )
    auth_group.add_argument("--oauth-enable", action="store_true", help="Enable OAuth 2.0")
    auth_group.add_argument("--oauth-client-id", help="OAuth client ID")
    auth_group.add_argument("--oauth-client-secret", help="OAuth client secret")

    # IDE-specific configuration
    ide_group = parser.add_argument_group("IDE-Specific Configuration")
    ide_group.add_argument(
        "--cursor-mode",
        choices=["agent", "ask", "edit"],
        help="Cursor IDE mode (agent|ask|edit)",
    )
    ide_group.add_argument(
        "--cursor-transport",
        choices=["sse", "websocket"],
        help="Transport for Cursor IDE (default: sse)",
    )
    ide_group.add_argument(
        "--windsurf-plugins", 
        action="store_true", 
        help="Enable Windsurf plugin integration"
    )

    # Feature configuration
    features_group = parser.add_argument_group("Feature Configuration")
    features_group.add_argument("--enable-metrics", action="store_true", help="Enable metrics collection")
    features_group.add_argument("--enable-health-check", action="store_true", help="Enable health check endpoint")
    features_group.add_argument("--rate-limit", action="store_true", help="Enable rate limiting")
    features_group.add_argument("--rate-limit-requests", type=int, help="Max requests per minute (default: 100)")
    features_group.add_argument("--tool-limit", type=int, help="Maximum tools to expose (default: 100)")
    features_group.add_argument("--resource-limit", type=int, help="Maximum resources to expose (default: 50)")

    # Utility commands
    parser.add_argument("--version", action="version", version="%(prog)s 0.0.1")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration and exit")

    args = parser.parse_args()

    # Create ServerConfig with command-line overrides
    server_config_values = {}
    
    # Deployment configuration
    if hasattr(args, 'deployment_mode') and args.deployment_mode:
        server_config_values["deployment_mode"] = args.deployment_mode
    if hasattr(args, 'ide_type') and args.ide_type:
        server_config_values["ide_type"] = args.ide_type.replace("-", "_")
    if hasattr(args, 'transport') and args.transport:
        server_config_values["transport"] = args.transport
    
    # Server configuration
    if hasattr(args, 'host') and args.host:
        server_config_values["host"] = args.host
    if hasattr(args, 'port') and args.port:
        server_config_values["port"] = args.port
    if hasattr(args, 'ssl_enable') and args.ssl_enable:
        server_config_values["ssl_enable"] = True
    if hasattr(args, 'ssl_certfile') and args.ssl_certfile:
        server_config_values["ssl_certfile"] = args.ssl_certfile
    if hasattr(args, 'ssl_keyfile') and args.ssl_keyfile:
        server_config_values["ssl_keyfile"] = args.ssl_keyfile
    
    # Authentication configuration
    if hasattr(args, 'auth_username') and args.auth_username:
        server_config_values["auth_username"] = args.auth_username
    if hasattr(args, 'auth_password') and args.auth_password:
        server_config_values["auth_password"] = args.auth_password
    if hasattr(args, 'auth_password_file') and args.auth_password_file:
        server_config_values["auth_password_file"] = args.auth_password_file
    if hasattr(args, 'oauth_enable') and args.oauth_enable:
        server_config_values["oauth_enable"] = True
    if hasattr(args, 'oauth_client_id') and args.oauth_client_id:
        server_config_values["oauth_client_id"] = args.oauth_client_id
    if hasattr(args, 'oauth_client_secret') and args.oauth_client_secret:
        server_config_values["oauth_client_secret"] = args.oauth_client_secret
    
    # IDE-specific configuration
    if hasattr(args, 'cursor_mode') and args.cursor_mode:
        server_config_values["cursor_mode"] = args.cursor_mode
    if hasattr(args, 'cursor_transport') and args.cursor_transport:
        server_config_values["cursor_transport"] = args.cursor_transport
    if hasattr(args, 'windsurf_plugins') and args.windsurf_plugins:
        server_config_values["windsurf_plugins_enabled"] = True
    
    # Feature configuration
    if hasattr(args, 'enable_metrics') and args.enable_metrics:
        server_config_values["enable_metrics"] = True
    if hasattr(args, 'enable_health_check') and args.enable_health_check:
        server_config_values["enable_health_check"] = True
    if hasattr(args, 'rate_limit') and args.rate_limit:
        server_config_values["rate_limit_enabled"] = True
    if hasattr(args, 'rate_limit_requests') and args.rate_limit_requests:
        server_config_values["rate_limit_requests"] = args.rate_limit_requests
    if hasattr(args, 'tool_limit') and args.tool_limit:
        server_config_values["tool_limit"] = args.tool_limit
    if hasattr(args, 'resource_limit') and args.resource_limit:
        server_config_values["resource_limit"] = args.resource_limit

    server_config = ServerConfig(**server_config_values)
    
    # Show configuration and exit if requested
    if hasattr(args, 'show_config') and args.show_config:
        config_dict = server_config.get_deployment_config()
        print(json.dumps(config_dict, indent=2))
        return

    try:
        logger.debug("Starting ch-agent-zero entry point")
        logger.debug(f"Python path: {sys.path}")
        logger.debug(f"Current working directory: {sys.path[0]}")

        # Log deployment configuration
        logger.info(f"Starting Agent Zero MCP Server (2025 Multi-IDE Edition)")
        logger.info(f"Deployment mode: {server_config.deployment_mode.value}")
        logger.info(f"Server: {server_config.host}:{server_config.port}")
        logger.info(f"Transport: {server_config.transport.value}")
        
        if server_config.ide_type:
            logger.info(f"Optimized for IDE: {server_config.ide_type.value}")
            optimal_transport = server_config.get_transport_for_ide()
            logger.info(f"Optimal transport for {server_config.ide_type.value}: {optimal_transport.value}")

        # Log security configuration
        ssl_config = server_config.get_ssl_config()
        if ssl_config:
            logger.info(f"SSL enabled with cert: {ssl_config['certfile']}")
        
        auth_config = server_config.get_auth_config()
        if auth_config:
            logger.info(f"Basic authentication enabled for user: {auth_config['username']}")
        
        oauth_config = server_config.get_oauth_config()
        if oauth_config:
            logger.info(f"OAuth 2.0 authentication enabled for client: {oauth_config['client_id']}")

        # Log IDE-specific configuration
        if server_config.cursor_mode:
            logger.info(f"Cursor IDE mode: {server_config.cursor_mode}")
            logger.info(f"Cursor transport: {server_config.cursor_transport.value}")
        
        if server_config.ide_type == IDEType.WINDSURF and server_config.windsurf_plugins_enabled:
            logger.info("Windsurf plugin integration enabled")
        
        # Log feature configuration
        features = []
        if server_config.enable_metrics:
            features.append("metrics")
        if server_config.enable_health_check:
            features.append("health-check")
        if server_config.rate_limit_enabled:
            features.append(f"rate-limiting({server_config.rate_limit_requests}/min)")
        if server_config.enable_structured_output:
            features.append("structured-output")
        
        if features:
            logger.info(f"Enabled features: {', '.join(features)}")
        
        logger.info(f"Tool limit: {server_config.tool_limit}, Resource limit: {server_config.resource_limit}")

        # Run the MCP server with the enhanced configuration
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
