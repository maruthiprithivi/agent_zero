#!/usr/bin/env python3
"""
Production server example for Agent Zero ClickHouse MCP Server.

This example demonstrates how to run the Agent Zero MCP server with full
production-grade operations capabilities including health monitoring,
metrics collection, distributed tracing, performance monitoring, and
backup systems.
"""

import asyncio
import logging
import os
from typing import Any

from agent_zero.config import UnifiedConfig
from agent_zero.server import create_clickhouse_client, run_production_server


async def main():
    """Run the production MCP server with comprehensive monitoring."""

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Agent Zero MCP Server in production mode")

    # Load configuration
    try:
        config = UnifiedConfig.from_env(
            # Override for production
            deployment_mode="remote",
            server_host="0.0.0.0",
            server_port=8505,
            enable_health_check=True,
            enable_mcp_tracing=True,
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure all required environment variables are set:")
        logger.error("- AGENT_ZERO_CLICKHOUSE_HOST")
        logger.error("- AGENT_ZERO_CLICKHOUSE_USER")
        logger.error("- AGENT_ZERO_CLICKHOUSE_PASSWORD")
        return 1

    # Create ClickHouse client factory
    def clickhouse_client_factory():
        """Factory function to create ClickHouse clients."""
        return create_clickhouse_client(config)

    # Production server configuration
    server_config: dict[str, Any] = {
        # Basic server settings
        "service_name": "agent-zero-mcp",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "host": config.server_host,
        "port": config.server_port,
        # ClickHouse integration
        "clickhouse_client_factory": clickhouse_client_factory,
        # Monitoring and observability
        "enable_metrics": True,
        "enable_tracing": True,
        "enable_performance_monitoring": True,
        "enable_backups": True,
        # Metrics configuration
        "metrics_prefix": "agent_zero",
        # Tracing configuration
        "tracing_endpoint": os.getenv("TRACING_ENDPOINT"),
        # Backup configuration
        "backup_dir": os.getenv("BACKUP_DIR", "/var/backups/agent_zero"),
        "backup_retention_days": int(os.getenv("BACKUP_RETENTION_DAYS", "30")),
        # SSL configuration (if certificates are provided)
        "ssl_config": (
            {
                "certfile": os.getenv("SSL_CERT_FILE"),
                "keyfile": os.getenv("SSL_KEY_FILE"),
            }
            if os.getenv("SSL_CERT_FILE") and os.getenv("SSL_KEY_FILE")
            else None
        ),
    }

    # Log configuration summary
    logger.info("Production server configuration:")
    logger.info(f"  Service: {server_config['service_name']}")
    logger.info(f"  Environment: {server_config['environment']}")
    logger.info(f"  Listen: {server_config['host']}:{server_config['port']}")
    logger.info(f"  ClickHouse: {config.clickhouse_host}:{config.clickhouse_port}")
    logger.info(f"  SSL: {'Enabled' if server_config['ssl_config'] else 'Disabled'}")
    logger.info(f"  Monitoring: {'Enabled' if server_config['enable_metrics'] else 'Disabled'}")
    logger.info(f"  Tracing: {'Enabled' if server_config['enable_tracing'] else 'Disabled'}")
    logger.info(f"  Backups: {'Enabled' if server_config['enable_backups'] else 'Disabled'}")

    try:
        # Start the production server
        await run_production_server(server_config)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping server...")

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1

    logger.info("Server stopped successfully")
    return 0


if __name__ == "__main__":
    """
    Example usage:

    # Set required environment variables
    export AGENT_ZERO_CLICKHOUSE_HOST="your-clickhouse-host"
    export AGENT_ZERO_CLICKHOUSE_USER="your-username"
    export AGENT_ZERO_CLICKHOUSE_PASSWORD="your-password"

    # Optional environment variables
    export ENVIRONMENT="production"
    export TRACING_ENDPOINT="http://jaeger:14268/api/traces"
    export BACKUP_DIR="/var/backups/agent_zero"
    export SSL_CERT_FILE="/path/to/cert.pem"
    export SSL_KEY_FILE="/path/to/key.pem"

    # Run the server
    python examples/production_server_example.py

    # Test the endpoints
    curl http://localhost:8505/health
    curl http://localhost:8505/metrics
    curl http://localhost:8505/performance
    """

    exit_code = asyncio.run(main())
    exit(exit_code)
