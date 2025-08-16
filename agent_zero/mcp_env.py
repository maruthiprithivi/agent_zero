"""Environment configuration for the MCP ClickHouse server.

DEPRECATED: This module is deprecated in favor of agent_zero.config.UnifiedConfig.
It provides backward compatibility but will be removed in a future version.
"""

import warnings
from typing import Any

# Import the new unified configuration
from agent_zero.config import get_config


class ClickHouseConfig:
    """DEPRECATED: ClickHouse configuration wrapper for backward compatibility.

    This class provides backward compatibility with the old ClickHouseConfig interface
    while delegating to the new UnifiedConfig system.
    """

    def __init__(self):
        """Initialize with a deprecation warning."""
        warnings.warn(
            "ClickHouseConfig is deprecated. Use agent_zero.config.UnifiedConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._config = get_config()

    @property
    def host(self) -> str:
        """Get the ClickHouse host."""
        return self._config.clickhouse_host

    @property
    def port(self) -> int:
        """Get the ClickHouse port."""
        return self._config.clickhouse_port

    @property
    def username(self) -> str:
        """Get the ClickHouse username."""
        return self._config.clickhouse_user

    @property
    def password(self) -> str:
        """Get the ClickHouse password."""
        return self._config.clickhouse_password

    @property
    def database(self) -> str | None:
        """Get the default database name if set."""
        return self._config.clickhouse_database

    @property
    def secure(self) -> bool:
        """Get whether HTTPS is enabled."""
        return self._config.clickhouse_secure

    @property
    def verify(self) -> bool:
        """Get whether SSL certificate verification is enabled."""
        return self._config.clickhouse_verify

    @property
    def connect_timeout(self) -> int:
        """Get the connection timeout in seconds."""
        return self._config.clickhouse_connect_timeout

    @property
    def send_receive_timeout(self) -> int:
        """Get the send/receive timeout in seconds."""
        return self._config.clickhouse_send_receive_timeout

    @property
    def enable_query_logging(self) -> bool:
        """Get whether detailed query logging is enabled."""
        return self._config.enable_query_logging

    @property
    def log_query_latency(self) -> bool:
        """Get whether query latency logging is enabled."""
        return self._config.log_query_latency

    @property
    def log_query_errors(self) -> bool:
        """Get whether query error logging is enabled."""
        return self._config.log_query_errors

    @property
    def log_query_warnings(self) -> bool:
        """Get whether query warning logging is enabled."""
        return self._config.log_query_warnings

    @property
    def enable_mcp_tracing(self) -> bool:
        """Get whether MCP server tracing is enabled."""
        return self._config.enable_mcp_tracing

    def get_client_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for clickhouse_connect client."""
        return self._config.get_clickhouse_client_config()


# Global instance for backward compatibility - lazy initialization
_config_instance = None


def get_legacy_config() -> ClickHouseConfig:
    """Get the legacy config instance with lazy initialization."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ClickHouseConfig()
    return _config_instance


# For backward compatibility, expose the lazy-loaded config
class _ConfigProxy:
    """Proxy to provide backward compatibility while using lazy loading."""

    def __getattr__(self, name):
        return getattr(get_legacy_config(), name)


config = _ConfigProxy()
