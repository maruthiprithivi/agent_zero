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


# Compatibility wrapper for old ServerConfig interface
class _ServerConfigWrapper:
    """Backward compatibility wrapper for ServerConfig."""

    def __init__(self, unified_config: UnifiedConfig):
        self._config = unified_config

    # Map old attribute names to new ones
    @property
    def host(self) -> str:
        return self._config.server_host

    @property
    def port(self) -> int:
        return self._config.server_port

    @property
    def ssl_certfile(self) -> str | None:
        return getattr(self._config, "ssl_certfile", None)

    @property
    def ssl_keyfile(self) -> str | None:
        return getattr(self._config, "ssl_keyfile", None)

    @property
    def auth_username(self) -> str | None:
        return self._config.auth_username

    @property
    def auth_password(self) -> str | None:
        return self._config.auth_password

    @property
    def auth_password_file(self) -> str | None:
        return self._config.auth_password_file


class _ServerConfigFactory:
    def __call__(self, **kwargs):
        # Map old environment variable names to override values for backward compatibility
        import os

        # Read old env vars and map them to new parameter names
        overrides = kwargs.copy()

        old_to_new_mapping = {
            "MCP_SERVER_HOST": "server_host",
            "MCP_SERVER_PORT": "server_port",
            "MCP_SSL_CERTFILE": "ssl_certfile",
            "MCP_SSL_KEYFILE": "ssl_keyfile",
            "MCP_AUTH_USERNAME": "auth_username",
            "MCP_AUTH_PASSWORD": "auth_password",
            "MCP_AUTH_PASSWORD_FILE": "auth_password_file",
        }

        for old_env_key, param_name in old_to_new_mapping.items():
            if old_env_key in os.environ and param_name not in overrides:
                value = os.environ[old_env_key]
                # Convert port to int if needed
                if param_name == "server_port":
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                overrides[param_name] = value

        # Provide a lazy config that tolerates missing ClickHouse env in tests
        try:
            unified = UnifiedConfig.from_env(**overrides)
        except Exception:
            # Fallback for test environments
            default_overrides = {
                "clickhouse_host": "localhost",
                "clickhouse_user": "default",
                "clickhouse_password": "",
            }
            default_overrides.update(overrides)
            unified = UnifiedConfig(**default_overrides)

        return _ServerConfigWrapper(unified)


ServerConfig = _ServerConfigFactory()
