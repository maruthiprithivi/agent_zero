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

    # Map old attribute names to new ones with fallback support
    @property
    def host(self) -> str:
        return getattr(self._config, "server_host", getattr(self._config, "host", "127.0.0.1"))

    @property
    def port(self) -> int:
        port_val = getattr(self._config, "server_port", getattr(self._config, "port", 8505))
        return int(port_val) if isinstance(port_val, str) else port_val

    @property
    def ssl_certfile(self) -> str | None:
        return getattr(self._config, "ssl_certfile", None)

    @property
    def ssl_keyfile(self) -> str | None:
        return getattr(self._config, "ssl_keyfile", None)

    @property
    def auth_username(self) -> str | None:
        return getattr(self._config, "auth_username", None)

    @property
    def auth_password(self) -> str | None:
        return getattr(self._config, "auth_password", None)

    @property
    def auth_password_file(self) -> str | None:
        return getattr(self._config, "auth_password_file", None)

    @property
    def ssl_enable(self) -> bool:
        return getattr(self._config, "ssl_enable", False)

    def get_ssl_config(self) -> dict[str, str] | None:
        """Get SSL configuration if both cert and key files are provided."""
        if self.ssl_certfile and self.ssl_keyfile:
            return {
                "certfile": self.ssl_certfile,
                "keyfile": self.ssl_keyfile,
            }
        return None

    def get_auth_config(self) -> dict[str, str] | None:
        """Get authentication configuration if username is provided."""
        if not self.auth_username:
            return None

        password = None
        if self.auth_password:
            password = self.auth_password
        elif self.auth_password_file:
            try:
                with open(self.auth_password_file, "r") as f:
                    password = f.read().strip()
            except (OSError, IOError):
                return None

        if password:
            return {
                "username": self.auth_username,
                "password": password,
            }
        return None


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
            # Fallback for test environments - map old parameter names to new ones
            default_overrides = {
                "clickhouse_host": "localhost",
                "clickhouse_user": "default",
                "clickhouse_password": "",
            }
            # Map old parameter names to new ones for the fallback
            param_mapping = {
                "host": "server_host",
                "port": "server_port",
                "ssl_enable": "ssl_enable",
                "ssl_certfile": "ssl_certfile",
                "ssl_keyfile": "ssl_keyfile",
                "auth_username": "auth_username",
                "auth_password": "auth_password",
                "auth_password_file": "auth_password_file",
            }

            mapped_overrides = {}
            for old_key, value in overrides.items():
                new_key = param_mapping.get(old_key, old_key)
                mapped_overrides[new_key] = value

            default_overrides.update(mapped_overrides)

            # Create a minimal config without validation for tests
            unified = type("MinimalConfig", (), default_overrides)()

        return _ServerConfigWrapper(unified)


ServerConfig = _ServerConfigFactory()
