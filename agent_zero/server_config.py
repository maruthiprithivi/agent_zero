"""Server configuration for the MCP ClickHouse server.

This module handles server-specific configuration with sensible defaults
and type conversion.
"""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for MCP server settings.

    This class handles all server-specific configuration with sensible defaults
    and type conversion. It provides typed methods for accessing each configuration value.

    Optional environment variables (with defaults):
        MCP_SERVER_HOST: Host to bind to (default: 127.0.0.1)
        MCP_SERVER_PORT: Port to bind to (default: 8505)
        MCP_SSL_CERTFILE: SSL certificate file path (default: None)
        MCP_SSL_KEYFILE: SSL key file path (default: None)
        MCP_AUTH_USERNAME: Basic auth username (default: None)
        MCP_AUTH_PASSWORD: Basic auth password (default: None)
        MCP_AUTH_PASSWORD_FILE: Path to file containing basic auth password (default: None)
        MCP_CURSOR_MODE: Cursor IDE mode to support (default: None)
        MCP_CURSOR_TRANSPORT: Transport to use with Cursor IDE (default: sse)
    """

    def __init__(self, **override_values):
        """Initialize the configuration from environment variables.

        Args:
            **override_values: Values to override environment variables
        """
        self._override_values = override_values

    @property
    def host(self) -> str:
        """Get the MCP server host."""
        if "host" in self._override_values:
            return self._override_values["host"]
        return os.getenv("MCP_SERVER_HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        """Get the MCP server port."""
        if "port" in self._override_values:
            return int(self._override_values["port"])
        return int(os.getenv("MCP_SERVER_PORT", "8505"))

    @property
    def ssl_certfile(self) -> str | None:
        """Get the SSL certificate file path."""
        if "ssl_certfile" in self._override_values:
            return self._override_values["ssl_certfile"]
        return os.getenv("MCP_SSL_CERTFILE")

    @property
    def ssl_keyfile(self) -> str | None:
        """Get the SSL key file path."""
        if "ssl_keyfile" in self._override_values:
            return self._override_values["ssl_keyfile"]
        return os.getenv("MCP_SSL_KEYFILE")

    @property
    def auth_username(self) -> str | None:
        """Get the basic auth username."""
        if "auth_username" in self._override_values:
            return self._override_values["auth_username"]
        return os.getenv("MCP_AUTH_USERNAME")

    @property
    def auth_password(self) -> str | None:
        """Get the basic auth password."""
        if "auth_password" in self._override_values:
            return self._override_values["auth_password"]
        return os.getenv("MCP_AUTH_PASSWORD")

    @property
    def auth_password_file(self) -> str | None:
        """Get the basic auth password file path."""
        if "auth_password_file" in self._override_values:
            return self._override_values["auth_password_file"]
        return os.getenv("MCP_AUTH_PASSWORD_FILE")

    @property
    def cursor_mode(self) -> str | None:
        """Get the Cursor IDE mode.

        Valid values: "agent", "ask", "edit", or None
        """
        if "cursor_mode" in self._override_values:
            return self._override_values["cursor_mode"]
        return os.getenv("MCP_CURSOR_MODE")

    @property
    def cursor_transport(self) -> str:
        """Get the transport type to use with Cursor IDE.

        Valid values: "sse", "websocket"
        """
        if "cursor_transport" in self._override_values:
            return self._override_values["cursor_transport"]
        return os.getenv("MCP_CURSOR_TRANSPORT", "sse")

    def get_ssl_config(self) -> dict | None:
        """Get the SSL configuration dictionary.

        Returns:
            dict|None: SSL configuration for MCP server, or None if SSL is not configured
        """
        if self.ssl_certfile and self.ssl_keyfile:
            return {
                "certfile": self.ssl_certfile,
                "keyfile": self.ssl_keyfile,
            }
        return None

    def get_auth_config(self) -> dict | None:
        """Get the authentication configuration dictionary.

        If auth_password_file is set, the password is read from the file.
        If both auth_password and auth_password_file are set, auth_password takes precedence.

        Returns:
            dict|None: Authentication configuration, or None if authentication is not configured
        """
        if self.auth_username:
            password = self.auth_password

            # If no password is set but a password file is, read from the file
            if not password and self.auth_password_file:
                try:
                    with open(self.auth_password_file, "r") as f:
                        password = f.read().strip()
                except Exception:
                    # If we can't read the password file, authentication is not configured
                    return None

            if password:
                return {"username": self.auth_username, "password": password}

        return None


# Global instance for easy access
server_config = ServerConfig()
