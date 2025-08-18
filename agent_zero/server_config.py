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
        import os

        # CRITICAL: Process direct parameters FIRST, then environment variables as fallback
        # This ensures direct parameters always take precedence

        # Start with default values
        config_values = {
            "clickhouse_host": "localhost",
            "clickhouse_user": "default",
            "clickhouse_password": "",
            "server_host": "127.0.0.1",
            "server_port": 8505,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "ssl_enable": False,
            "auth_username": None,
            "auth_password": None,
            "auth_password_file": None,
        }

        # Map old MCP environment variables to new parameter names (as fallback only)
        old_to_new_env_mapping = {
            "MCP_SERVER_HOST": "server_host",
            "MCP_SERVER_PORT": "server_port",
            "MCP_SSL_CERTFILE": "ssl_certfile",
            "MCP_SSL_KEYFILE": "ssl_keyfile",
            "MCP_AUTH_USERNAME": "auth_username",
            "MCP_AUTH_PASSWORD": "auth_password",
            "MCP_AUTH_PASSWORD_FILE": "auth_password_file",
        }

        # Apply environment variables as fallback (lower priority)
        for old_env_key, param_name in old_to_new_env_mapping.items():
            if old_env_key in os.environ:
                value = os.environ[old_env_key]
                # Convert port to int if needed
                if param_name == "server_port":
                    try:
                        value = int(value)
                    except ValueError:
                        value = 8505  # fallback
                config_values[param_name] = value

        # Map old-style direct parameters to new parameter names
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

        # Apply direct parameters (HIGHEST priority - they override everything)
        for old_param, new_param in param_mapping.items():
            if old_param in kwargs:
                value = kwargs[old_param]
                # Convert port to int if needed
                if old_param == "port" and isinstance(value, str):
                    value = int(value)
                config_values[new_param] = value

        # Also apply any new-style direct parameters
        for key, value in kwargs.items():
            if key in config_values:
                config_values[key] = value

        # Try to create UnifiedConfig first (production path)
        try:
            # Use specific environment variables to avoid conflicts
            unified = UnifiedConfig(
                clickhouse_host=config_values["clickhouse_host"],
                clickhouse_user=config_values["clickhouse_user"],
                clickhouse_password=config_values["clickhouse_password"],
                server_host=config_values["server_host"],
                server_port=config_values["server_port"],
                **{
                    k: v
                    for k, v in config_values.items()
                    if k
                    not in [
                        "clickhouse_host",
                        "clickhouse_user",
                        "clickhouse_password",
                        "server_host",
                        "server_port",
                    ]
                },
            )
        except Exception:
            # Fallback for test environments - create simple object
            unified = type("MinimalConfig", (), config_values)()

        return _ServerConfigWrapper(unified)


ServerConfig = _ServerConfigFactory()
