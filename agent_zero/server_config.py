"""Server configuration for the MCP ClickHouse server.

This module handles server-specific configuration with sensible defaults
and type conversion. Supports multiple deployment modes and IDE integrations.
"""

import os
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class DeploymentMode(Enum):
    """Supported deployment modes for the MCP server."""
    LOCAL = "local"  # Local development/testing
    STANDALONE = "standalone"  # Standalone server mode
    ENTERPRISE = "enterprise"  # Enterprise deployment with advanced features

class IDEType(Enum):
    """Supported IDE types."""
    CLAUDE_DESKTOP = "claude_desktop"
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    WINDSURF = "windsurf"
    VSCODE = "vscode"

class TransportType(Enum):
    """Supported transport types."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"
    HTTP = "http"

@dataclass
class ServerConfig:
    """Configuration for MCP server settings.

    This class handles all server-specific configuration with sensible defaults
    and type conversion. It provides typed methods for accessing each configuration value.
    Supports multiple deployment modes and IDE integrations (2025 edition).

    Environment variables:
        # Server Configuration
        MCP_SERVER_HOST: Host to bind to (default: 127.0.0.1)
        MCP_SERVER_PORT: Port to bind to (default: 8505)
        MCP_DEPLOYMENT_MODE: Deployment mode (local|standalone|enterprise)
        MCP_TRANSPORT: Default transport type (stdio|sse|websocket|http)
        
        # SSL Configuration
        MCP_SSL_CERTFILE: SSL certificate file path (default: None)
        MCP_SSL_KEYFILE: SSL key file path (default: None)
        MCP_SSL_ENABLE: Enable SSL (default: false)
        
        # Authentication
        MCP_AUTH_USERNAME: Basic auth username (default: None)
        MCP_AUTH_PASSWORD: Basic auth password (default: None)
        MCP_AUTH_PASSWORD_FILE: Path to file containing basic auth password (default: None)
        MCP_OAUTH_ENABLE: Enable OAuth 2.0 authentication (default: false)
        MCP_OAUTH_CLIENT_ID: OAuth client ID
        MCP_OAUTH_CLIENT_SECRET: OAuth client secret
        
        # IDE-Specific Configuration
        MCP_IDE_TYPE: Target IDE type (claude_desktop|claude_code|cursor|windsurf|vscode)
        MCP_CURSOR_MODE: Cursor IDE mode (agent|ask|edit)
        MCP_CURSOR_TRANSPORT: Transport for Cursor IDE (sse|websocket)
        MCP_WINDSURF_PLUGINS_ENABLED: Enable Windsurf plugin integration
        
        # Server Features
        MCP_ENABLE_METRICS: Enable metrics collection (default: false)
        MCP_ENABLE_HEALTH_CHECK: Enable health check endpoint (default: true)
        MCP_RATE_LIMIT_ENABLED: Enable rate limiting (default: false)
        MCP_RATE_LIMIT_REQUESTS: Max requests per minute (default: 100)
        
        # Advanced Features
        MCP_TOOL_LIMIT: Maximum number of tools to expose (default: 100)
        MCP_RESOURCE_LIMIT: Maximum number of resources to expose (default: 50)
        MCP_ENABLE_STRUCTURED_OUTPUT: Enable structured output (default: true)
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
    def deployment_mode(self) -> DeploymentMode:
        """Get the deployment mode."""
        if "deployment_mode" in self._override_values:
            mode = self._override_values["deployment_mode"]
        else:
            mode = os.getenv("MCP_DEPLOYMENT_MODE", "local")
        return DeploymentMode(mode)
    
    @property
    def ide_type(self) -> Optional[IDEType]:
        """Get the target IDE type."""
        if "ide_type" in self._override_values:
            ide = self._override_values["ide_type"]
        else:
            ide = os.getenv("MCP_IDE_TYPE")
        return IDEType(ide) if ide else None
    
    @property
    def transport(self) -> TransportType:
        """Get the default transport type."""
        if "transport" in self._override_values:
            transport = self._override_values["transport"]
        else:
            transport = os.getenv("MCP_TRANSPORT", "stdio")
        return TransportType(transport)

    @property
    def cursor_transport(self) -> TransportType:
        """Get the transport type to use with Cursor IDE."""
        if "cursor_transport" in self._override_values:
            transport = self._override_values["cursor_transport"]
        else:
            transport = os.getenv("MCP_CURSOR_TRANSPORT", "sse")
        return TransportType(transport)
    
    @property
    def ssl_enable(self) -> bool:
        """Get whether SSL is enabled."""
        if "ssl_enable" in self._override_values:
            return bool(self._override_values["ssl_enable"])
        return os.getenv("MCP_SSL_ENABLE", "false").lower() == "true"
    
    @property
    def oauth_enable(self) -> bool:
        """Get whether OAuth 2.0 authentication is enabled."""
        if "oauth_enable" in self._override_values:
            return bool(self._override_values["oauth_enable"])
        return os.getenv("MCP_OAUTH_ENABLE", "false").lower() == "true"
    
    @property
    def oauth_client_id(self) -> str | None:
        """Get the OAuth client ID."""
        if "oauth_client_id" in self._override_values:
            return self._override_values["oauth_client_id"]
        return os.getenv("MCP_OAUTH_CLIENT_ID")
    
    @property
    def oauth_client_secret(self) -> str | None:
        """Get the OAuth client secret."""
        if "oauth_client_secret" in self._override_values:
            return self._override_values["oauth_client_secret"]
        return os.getenv("MCP_OAUTH_CLIENT_SECRET")
    
    @property
    def windsurf_plugins_enabled(self) -> bool:
        """Get whether Windsurf plugin integration is enabled."""
        if "windsurf_plugins_enabled" in self._override_values:
            return bool(self._override_values["windsurf_plugins_enabled"])
        return os.getenv("MCP_WINDSURF_PLUGINS_ENABLED", "true").lower() == "true"
    
    @property
    def enable_metrics(self) -> bool:
        """Get whether metrics collection is enabled."""
        if "enable_metrics" in self._override_values:
            return bool(self._override_values["enable_metrics"])
        return os.getenv("MCP_ENABLE_METRICS", "false").lower() == "true"
    
    @property
    def enable_health_check(self) -> bool:
        """Get whether health check endpoint is enabled."""
        if "enable_health_check" in self._override_values:
            return bool(self._override_values["enable_health_check"])
        return os.getenv("MCP_ENABLE_HEALTH_CHECK", "true").lower() == "true"
    
    @property
    def rate_limit_enabled(self) -> bool:
        """Get whether rate limiting is enabled."""
        if "rate_limit_enabled" in self._override_values:
            return bool(self._override_values["rate_limit_enabled"])
        return os.getenv("MCP_RATE_LIMIT_ENABLED", "false").lower() == "true"
    
    @property
    def rate_limit_requests(self) -> int:
        """Get the maximum requests per minute for rate limiting."""
        if "rate_limit_requests" in self._override_values:
            return int(self._override_values["rate_limit_requests"])
        return int(os.getenv("MCP_RATE_LIMIT_REQUESTS", "100"))
    
    @property
    def tool_limit(self) -> int:
        """Get the maximum number of tools to expose."""
        if "tool_limit" in self._override_values:
            return int(self._override_values["tool_limit"])
        return int(os.getenv("MCP_TOOL_LIMIT", "100"))
    
    @property
    def resource_limit(self) -> int:
        """Get the maximum number of resources to expose."""
        if "resource_limit" in self._override_values:
            return int(self._override_values["resource_limit"])
        return int(os.getenv("MCP_RESOURCE_LIMIT", "50"))
    
    @property
    def enable_structured_output(self) -> bool:
        """Get whether structured output is enabled (2025-06-18 spec)."""
        if "enable_structured_output" in self._override_values:
            return bool(self._override_values["enable_structured_output"])
        return os.getenv("MCP_ENABLE_STRUCTURED_OUTPUT", "true").lower() == "true"

    def get_ssl_config(self) -> dict | None:
        """Get the SSL configuration dictionary.

        Returns:
            dict|None: SSL configuration for MCP server, or None if SSL is not configured
        """
        if self.ssl_certfile and self.ssl_keyfile:
            return {
                "certfile": self.ssl_certfile,
                "keyfile": self.ssl_keyfile,
                "enabled": self.ssl_enable,
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
    
    def get_oauth_config(self) -> dict | None:
        """Get the OAuth configuration dictionary.
        
        Returns:
            dict|None: OAuth configuration, or None if OAuth is not configured
        """
        if self.oauth_enable and self.oauth_client_id and self.oauth_client_secret:
            return {
                "client_id": self.oauth_client_id,
                "client_secret": self.oauth_client_secret,
                "enabled": True,
            }
        return None
    
    def get_transport_for_ide(self, ide_type: Optional[IDEType] = None) -> TransportType:
        """Get the appropriate transport type for a specific IDE.
        
        Args:
            ide_type: The IDE type, defaults to self.ide_type
            
        Returns:
            The appropriate transport type for the IDE
        """
        ide = ide_type or self.ide_type
        
        if ide == IDEType.CLAUDE_CODE:
            # Claude Code supports stdio, sse, and http transports
            return self.transport
        elif ide == IDEType.CLAUDE_DESKTOP:
            # Claude Desktop primarily uses stdio transport
            return TransportType.STDIO
        elif ide == IDEType.CURSOR:
            # Cursor supports sse and websocket
            return self.cursor_transport
        elif ide == IDEType.WINDSURF:
            # Windsurf supports stdio and sse
            return TransportType.SSE if self.deployment_mode == DeploymentMode.STANDALONE else TransportType.STDIO
        elif ide == IDEType.VSCODE:
            # VS Code with MCP extension supports stdio and sse
            return TransportType.SSE if self.deployment_mode == DeploymentMode.STANDALONE else TransportType.STDIO
        else:
            # Default transport
            return self.transport
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get a complete deployment configuration dictionary.
        
        Returns:
            A dictionary with all deployment-relevant configuration
        """
        config = {
            "deployment_mode": self.deployment_mode.value,
            "host": self.host,
            "port": self.port,
            "transport": self.transport.value,
            "ide_type": self.ide_type.value if self.ide_type else None,
            "ssl": self.get_ssl_config(),
            "auth": self.get_auth_config(),
            "oauth": self.get_oauth_config(),
            "features": {
                "metrics": self.enable_metrics,
                "health_check": self.enable_health_check,
                "rate_limiting": self.rate_limit_enabled,
                "structured_output": self.enable_structured_output,
            },
            "limits": {
                "tools": self.tool_limit,
                "resources": self.resource_limit,
                "rate_limit_requests": self.rate_limit_requests,
            },
        }
        
        # Add IDE-specific configuration
        if self.ide_type == IDEType.CURSOR:
            config["cursor"] = {
                "mode": self.cursor_mode,
                "transport": self.cursor_transport.value,
            }
        elif self.ide_type == IDEType.WINDSURF:
            config["windsurf"] = {
                "plugins_enabled": self.windsurf_plugins_enabled,
            }
        
        return config
    
    def generate_ide_config(self, ide_type: IDEType, install_path: str = None) -> Dict[str, Any]:
        """Generate IDE-specific configuration.
        
        Args:
            ide_type: The target IDE type
            install_path: Path to the installation (for local configs)
            
        Returns:
            IDE-specific configuration dictionary
        """
        transport = self.get_transport_for_ide(ide_type)
        
        base_config = {
            "name": "agent-zero",
            "description": "ClickHouse monitoring and analysis MCP server",
        }
        
        if ide_type == IDEType.CLAUDE_DESKTOP:
            return {
                "mcpServers": {
                    "agent-zero": {
                        **base_config,
                        "command": install_path or "ch-agent-zero",
                        "env": self._get_env_config(),
                    }
                }
            }
        
        elif ide_type == IDEType.CLAUDE_CODE:
            if transport == TransportType.SSE:
                return {
                    "mcpServers": {
                        "agent-zero": {
                            **base_config,
                            "transport": "sse",
                            "url": f"http{'s' if self.ssl_enable else ''}://{self.host}:{self.port}/sse",
                            "headers": self._get_auth_headers(),
                        }
                    }
                }
            else:
                return {
                    "mcpServers": {
                        "agent-zero": {
                            **base_config,
                            "command": install_path or "ch-agent-zero",
                            "env": self._get_env_config(),
                        }
                    }
                }
        
        elif ide_type == IDEType.CURSOR:
            config = {
                **base_config,
                "transport": transport.value,
            }
            
            if transport == TransportType.SSE:
                config["url"] = f"http{'s' if self.ssl_enable else ''}://{self.host}:{self.port}/sse"
                config["headers"] = self._get_auth_headers()
            else:
                config["command"] = install_path or "ch-agent-zero"
                config["args"] = ["--cursor-mode", self.cursor_mode or "agent"]
                config["env"] = self._get_env_config()
            
            return config
        
        elif ide_type == IDEType.WINDSURF:
            if transport == TransportType.SSE:
                return {
                    "servers": {
                        "agent-zero": {
                            **base_config,
                            "type": "sse",
                            "url": f"http{'s' if self.ssl_enable else ''}://{self.host}:{self.port}/sse",
                            "headers": self._get_auth_headers(),
                        }
                    }
                }
            else:
                return {
                    "servers": {
                        "agent-zero": {
                            **base_config,
                            "command": install_path or "ch-agent-zero",
                            "env": self._get_env_config(),
                        }
                    }
                }
        
        return base_config
    
    def _get_env_config(self) -> Dict[str, str]:
        """Get environment configuration for IDE configs."""
        env_config = {}
        
        # ClickHouse configuration
        clickhouse_vars = [
            "CLICKHOUSE_HOST", "CLICKHOUSE_PORT", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD",
            "CLICKHOUSE_SECURE", "CLICKHOUSE_VERIFY", "CLICKHOUSE_CONNECT_TIMEOUT",
            "CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "CLICKHOUSE_DATABASE"
        ]
        
        for var in clickhouse_vars:
            value = os.getenv(var)
            if value:
                env_config[var] = value
        
        # MCP configuration
        if self.enable_metrics:
            env_config["MCP_ENABLE_TRACING"] = "true"
        
        return env_config
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HTTP-based transports."""
        headers = {}
        
        auth_config = self.get_auth_config()
        if auth_config:
            import base64
            credentials = f"{auth_config['username']}:{auth_config['password']}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"
        
        oauth_config = self.get_oauth_config()
        if oauth_config:
            # OAuth headers would be set during the OAuth flow
            headers["X-OAuth-Client-ID"] = oauth_config["client_id"]
        
        return headers


# Global instance for easy access
server_config = ServerConfig()
