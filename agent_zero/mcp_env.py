"""Environment configuration for the MCP ClickHouse server.

This module handles all environment variable configuration with sensible defaults
and type conversion.
"""

import os
from dataclasses import dataclass


@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection settings.

    This class handles all environment variable configuration with sensible defaults
    and type conversion. It provides typed methods for accessing each configuration value.

    Required environment variables:
        CLICKHOUSE_HOST: The hostname of the ClickHouse server
        CLICKHOUSE_USER: The username for authentication
        CLICKHOUSE_PASSWORD: The password for authentication

    Optional environment variables (with defaults):
        CLICKHOUSE_PORT: The port number (default: 8443 if secure=True, 8123 if secure=False)
        CLICKHOUSE_SECURE: Enable HTTPS (default: true)
        CLICKHOUSE_VERIFY: Verify SSL certificates (default: true)
        CLICKHOUSE_CONNECT_TIMEOUT: Connection timeout in seconds (default: 30)
        CLICKHOUSE_SEND_RECEIVE_TIMEOUT: Send/receive timeout in seconds (default: 300)
        CLICKHOUSE_DATABASE: Default database to use (default: None)
        CLICKHOUSE_ENABLE_QUERY_LOGGING: Enable detailed query logging (default: false)
        CLICKHOUSE_LOG_QUERY_LATENCY: Log query execution times (default: false)
        CLICKHOUSE_LOG_QUERY_ERRORS: Log query errors (default: true)
        CLICKHOUSE_LOG_QUERY_WARNINGS: Log query warnings (default: true)
        MCP_ENABLE_TRACING: Enable tracing for MCP server communications (default: false)
    """

    def __init__(self):
        """Initialize the configuration from environment variables."""
        self._validate_required_vars()

    @property
    def host(self) -> str:
        """Get the ClickHouse host."""
        return os.environ["CLICKHOUSE_HOST"]

    @property
    def port(self) -> int:
        """Get the ClickHouse port.

        Defaults to 8443 if secure=True, 8123 if secure=False.
        Can be overridden by CLICKHOUSE_PORT environment variable.
        """
        if "CLICKHOUSE_PORT" in os.environ:
            return int(os.environ["CLICKHOUSE_PORT"])
        return 8443 if self.secure else 8123

    @property
    def username(self) -> str:
        """Get the ClickHouse username."""
        return os.environ["CLICKHOUSE_USER"]

    @property
    def password(self) -> str:
        """Get the ClickHouse password."""
        return os.environ["CLICKHOUSE_PASSWORD"]

    @property
    def database(self) -> str | None:
        """Get the default database name if set."""
        return os.getenv("CLICKHOUSE_DATABASE")

    @property
    def secure(self) -> bool:
        """Get whether HTTPS is enabled.

        Default: True
        """
        return os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true"

    @property
    def verify(self) -> bool:
        """Get whether SSL certificate verification is enabled.

        Default: True
        """
        return os.getenv("CLICKHOUSE_VERIFY", "true").lower() == "true"

    @property
    def connect_timeout(self) -> int:
        """Get the connection timeout in seconds.

        Default: 30
        """
        return int(os.getenv("CLICKHOUSE_CONNECT_TIMEOUT", "30"))

    @property
    def send_receive_timeout(self) -> int:
        """Get the send/receive timeout in seconds.

        Default: 300 (ClickHouse default)
        """
        return int(os.getenv("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "300"))

    @property
    def enable_query_logging(self) -> bool:
        """Get whether detailed query logging is enabled.

        Default: False
        """
        return os.getenv("CLICKHOUSE_ENABLE_QUERY_LOGGING", "false").lower() == "true"

    @property
    def log_query_latency(self) -> bool:
        """Get whether query latency logging is enabled.

        Default: False
        """
        return os.getenv("CLICKHOUSE_LOG_QUERY_LATENCY", "false").lower() == "true"

    @property
    def log_query_errors(self) -> bool:
        """Get whether query error logging is enabled.

        Default: True
        """
        return os.getenv("CLICKHOUSE_LOG_QUERY_ERRORS", "true").lower() == "true"

    @property
    def log_query_warnings(self) -> bool:
        """Get whether query warning logging is enabled.

        Default: True
        """
        return os.getenv("CLICKHOUSE_LOG_QUERY_WARNINGS", "true").lower() == "true"

    @property
    def enable_mcp_tracing(self) -> bool:
        """Get whether MCP server tracing is enabled.

        Default: False
        """
        return os.getenv("MCP_ENABLE_TRACING", "false").lower() == "true"

    def get_client_config(self) -> dict:
        """Get the configuration dictionary for clickhouse_connect client.

        Returns:
            dict: Configuration ready to be passed to clickhouse_connect.get_client()
        """
        config = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "secure": self.secure,
            "verify": self.verify,
            "connect_timeout": self.connect_timeout,
            "send_receive_timeout": self.send_receive_timeout,
            "client_name": "mcp_clickhouse",
        }

        # Add optional database if set
        if self.database:
            config["database"] = self.database

        return config

    def _validate_required_vars(self) -> None:
        """Validate that all required environment variables are set.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        missing_vars = []
        for var in ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"]:
            if var not in os.environ:
                missing_vars.append(var)

        if missing_vars:
            # For development/debugging: use defaults when variables are missing
            if os.environ.get("CH_AGENT_ZERO_DEBUG") == "1":
                import logging

                logger = logging.getLogger("ch-agent-zero")
                logger.warning(
                    f"Using default values for missing environment variables: {', '.join(missing_vars)}"
                )
                for var in missing_vars:
                    if var == "CLICKHOUSE_HOST":
                        os.environ[var] = "localhost"
                    elif var == "CLICKHOUSE_USER":
                        os.environ[var] = "default"
                    elif var == "CLICKHOUSE_PASSWORD":
                        os.environ[var] = ""
            else:
                raise ValueError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )


# Global instance for easy access
config = ClickHouseConfig()
