"""DEPRECATED: Server configuration for the MCP ClickHouse server.

DEPRECATED: This module is deprecated in favor of agent_zero.config.UnifiedConfig.
It will be removed in a future version. Please use the unified configuration system.
"""

import warnings

from agent_zero.config import DeploymentMode, IDEType, TransportType, UnifiedConfig

# Re-export the enums for backward compatibility
__all__ = ["DeploymentMode", "IDEType", "ServerConfig", "TransportType"]

warnings.warn(
    "agent_zero.server_config is deprecated. Use agent_zero.config.UnifiedConfig instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Alias for backward compatibility
class _ServerConfigFactory:
    def __call__(self, **kwargs):
        # Provide a lazy config that tolerates missing ClickHouse env in tests
        try:
            return UnifiedConfig.from_env(**kwargs)
        except Exception:
            base = UnifiedConfig(
                clickhouse_host="localhost",
                clickhouse_user="default",
                clickhouse_password="",
            )
            for k, v in kwargs.items():
                setattr(base, k, v)
            return base


ServerConfig = _ServerConfigFactory()
