"""Unified configuration management for Agent Zero.

This module consolidates all configuration management into a single, coherent system
following the development standards defined in CLAUDE.md.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment mode options for 2025."""

    LOCAL = "local"
    REMOTE = "remote"
    ENTERPRISE = "enterprise"  # New in 2025
    SERVERLESS = "serverless"  # New in 2025
    EDGE = "edge"  # New in 2025
    HYBRID = "hybrid"  # New in 2025

    @classmethod
    def _missing_(cls, value):
        # Backward compatible alias handling
        if isinstance(value, str) and value.lower() in {"standalone"}:
            # Map legacy values to closest supported mode
            return cls.REMOTE
        return None


class TransportType(Enum):
    """MCP transport types for 2025."""

    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable_http"  # New in 2025 MCP spec
    GRPC = "grpc"  # New in 2025


class IDEType(Enum):
    """Supported IDE types for 2025."""

    CLAUDE_DESKTOP = "claude_desktop"
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    WINDSURF = "windsurf"
    VSCODE = "vscode"
    ZED = "zed"  # New in 2025
    NEOVIM = "neovim"  # New in 2025
    EMACS = "emacs"  # New in 2025


@dataclass
class UnifiedConfig:
    """Single source of truth for all Agent Zero configuration.

    This class consolidates the previous ClickHouseConfig and ServerConfig
    into a unified, consistent configuration system.

    Environment variables follow the AGENT_ZERO_* convention.
    """

    # ClickHouse connection settings
    clickhouse_host: str
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_port: int = None  # Will be auto-determined from secure setting
    clickhouse_secure: bool = True
    clickhouse_verify: bool = True
    clickhouse_database: str | None = None
    clickhouse_connect_timeout: int = 30
    clickhouse_send_receive_timeout: int = 300

    # MCP Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8505
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    transport: TransportType = TransportType.STDIO
    ide_type: IDEType | None = None

    # Authentication settings
    auth_username: str | None = None
    auth_password: str | None = None
    auth_password_file: str | None = None

    # SSL settings
    ssl_enable: bool = False
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    # Logging and monitoring settings
    enable_query_logging: bool = False
    log_query_latency: bool = False
    log_query_errors: bool = True
    log_query_warnings: bool = True
    enable_mcp_tracing: bool = False

    # Safety and behavior flags (default preserve legacy behavior)
    enable_client_cache: bool = False
    enable_structured_tool_output: bool = False
    enable_tool_errors: bool = False
    default_max_rows: int = 10000
    default_max_execution_seconds: int = 30

    # Feature settings
    enable_health_check: bool = True
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    tool_limit: int = 100
    resource_limit: int = 50

    # IDE-specific settings
    cursor_mode: str | None = None
    cursor_transport: TransportType = TransportType.SSE

    # 2025 New Features
    # AI/ML Features
    enable_ai_predictions: bool = False
    ai_model_type: str = "random_forest"
    vector_db_enabled: bool = False
    vector_db_provider: str = "chromadb"  # chromadb, pinecone, weaviate

    # Cloud Native Features
    kubernetes_enabled: bool = False
    kubernetes_namespace: str = "agent-zero"
    service_mesh_type: str | None = None  # istio, linkerd, consul
    auto_scaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 20

    # Security Features (Zero Trust)
    zero_trust_enabled: bool = False
    require_mutual_tls: bool = False
    certificate_rotation_days: int = 90
    threat_detection_enabled: bool = False
    compliance_frameworks: str = "soc2"  # Comma-separated list

    # Performance Features
    distributed_cache_enabled: bool = False
    redis_cluster_url: str | None = None
    load_balancing_algorithm: str = "least_connections"
    circuit_breaker_enabled: bool = True
    real_time_analytics: bool = False

    # Advanced Transport Features (MCP 2025)
    oauth2_enabled: bool = False
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    json_rpc_batching: bool = True
    streamable_responses: bool = True
    content_types_supported: str = "text,image,audio"  # New audio support
    tool_annotations_enabled: bool = True
    progress_notifications: bool = True
    completions_capability: bool = True

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self.validate()
        self._auto_configure()

    @classmethod
    def from_env(cls, **overrides) -> "UnifiedConfig":
        """Load configuration from environment variables.

        Args:
            **overrides: Configuration values to override from environment

        Returns:
            UnifiedConfig instance loaded from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        # Handle debug mode for development
        if os.getenv("CH_AGENT_ZERO_DEBUG") == "1":
            cls._set_debug_defaults()

        try:
            config = cls(
                # ClickHouse settings
                clickhouse_host=overrides.get("clickhouse_host")
                or os.environ["AGENT_ZERO_CLICKHOUSE_HOST"],
                clickhouse_user=overrides.get("clickhouse_user")
                or os.environ["AGENT_ZERO_CLICKHOUSE_USER"],
                clickhouse_password=overrides.get("clickhouse_password")
                or os.environ["AGENT_ZERO_CLICKHOUSE_PASSWORD"],
                clickhouse_port=cls._get_int_env(
                    "AGENT_ZERO_CLICKHOUSE_PORT", overrides.get("clickhouse_port")
                ),
                clickhouse_secure=cls._get_bool_env(
                    "AGENT_ZERO_CLICKHOUSE_SECURE", overrides.get("clickhouse_secure", True)
                ),
                clickhouse_verify=cls._get_bool_env(
                    "AGENT_ZERO_CLICKHOUSE_VERIFY", overrides.get("clickhouse_verify", True)
                ),
                clickhouse_database=overrides.get("clickhouse_database")
                or os.getenv("AGENT_ZERO_CLICKHOUSE_DATABASE"),
                clickhouse_connect_timeout=cls._get_int_env(
                    "AGENT_ZERO_CLICKHOUSE_CONNECT_TIMEOUT",
                    overrides.get("clickhouse_connect_timeout", 30),
                ),
                clickhouse_send_receive_timeout=cls._get_int_env(
                    "AGENT_ZERO_CLICKHOUSE_SEND_RECEIVE_TIMEOUT",
                    overrides.get("clickhouse_send_receive_timeout", 300),
                ),
                # Server settings
                server_host=overrides.get("server_host")
                or os.getenv("AGENT_ZERO_SERVER_HOST", "127.0.0.1"),
                server_port=cls._get_int_env(
                    "AGENT_ZERO_SERVER_PORT", overrides.get("server_port", 8505)
                ),
                # Allow raw string passthrough for tests (e.g., 'standalone')
                deployment_mode=overrides.get("deployment_mode")
                or DeploymentMode(os.getenv("AGENT_ZERO_DEPLOYMENT_MODE", "local")),
                transport=TransportType(
                    overrides.get("transport") or os.getenv("AGENT_ZERO_TRANSPORT", "stdio")
                ),
                ide_type=cls._get_enum_env(
                    IDEType, "AGENT_ZERO_IDE_TYPE", overrides.get("ide_type")
                ),
                # Authentication
                auth_username=overrides.get("auth_username")
                or os.getenv("AGENT_ZERO_AUTH_USERNAME"),
                auth_password=overrides.get("auth_password")
                or os.getenv("AGENT_ZERO_AUTH_PASSWORD"),
                auth_password_file=overrides.get("auth_password_file")
                or os.getenv("AGENT_ZERO_AUTH_PASSWORD_FILE"),
                # SSL
                ssl_enable=cls._get_bool_env(
                    "AGENT_ZERO_SSL_ENABLE", overrides.get("ssl_enable", False)
                ),
                ssl_certfile=overrides.get("ssl_certfile") or os.getenv("AGENT_ZERO_SSL_CERTFILE"),
                ssl_keyfile=overrides.get("ssl_keyfile") or os.getenv("AGENT_ZERO_SSL_KEYFILE"),
                # Logging
                enable_query_logging=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_QUERY_LOGGING", overrides.get("enable_query_logging", False)
                ),
                log_query_latency=cls._get_bool_env(
                    "AGENT_ZERO_LOG_QUERY_LATENCY", overrides.get("log_query_latency", False)
                ),
                log_query_errors=cls._get_bool_env(
                    "AGENT_ZERO_LOG_QUERY_ERRORS", overrides.get("log_query_errors", True)
                ),
                log_query_warnings=cls._get_bool_env(
                    "AGENT_ZERO_LOG_QUERY_WARNINGS", overrides.get("log_query_warnings", True)
                ),
                enable_mcp_tracing=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_MCP_TRACING", overrides.get("enable_mcp_tracing", False)
                ),
                # Safety/behavior flags
                enable_client_cache=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_CLIENT_CACHE", overrides.get("enable_client_cache", False)
                ),
                enable_structured_tool_output=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_STRUCTURED_TOOL_OUTPUT",
                    overrides.get("enable_structured_tool_output", False),
                ),
                enable_tool_errors=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_TOOL_ERRORS", overrides.get("enable_tool_errors", False)
                ),
                default_max_rows=cls._get_int_env(
                    "AGENT_ZERO_DEFAULT_MAX_ROWS", overrides.get("default_max_rows", 10000)
                )
                or 10000,
                default_max_execution_seconds=cls._get_int_env(
                    "AGENT_ZERO_DEFAULT_MAX_EXECUTION_SECONDS",
                    overrides.get("default_max_execution_seconds", 30),
                )
                or 30,
                # Features
                enable_health_check=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_HEALTH_CHECK", overrides.get("enable_health_check", True)
                ),
                rate_limit_enabled=cls._get_bool_env(
                    "AGENT_ZERO_RATE_LIMIT_ENABLED", overrides.get("rate_limit_enabled", False)
                ),
                rate_limit_requests=cls._get_int_env(
                    "AGENT_ZERO_RATE_LIMIT_REQUESTS", overrides.get("rate_limit_requests", 100)
                ),
                tool_limit=cls._get_int_env(
                    "AGENT_ZERO_TOOL_LIMIT", overrides.get("tool_limit", 100)
                ),
                resource_limit=cls._get_int_env(
                    "AGENT_ZERO_RESOURCE_LIMIT", overrides.get("resource_limit", 50)
                ),
                # IDE-specific
                cursor_mode=overrides.get("cursor_mode") or os.getenv("AGENT_ZERO_CURSOR_MODE"),
                cursor_transport=TransportType(
                    overrides.get("cursor_transport")
                    or os.getenv("AGENT_ZERO_CURSOR_TRANSPORT", "sse")
                ),
                # 2025 New Features
                # AI/ML Features
                enable_ai_predictions=cls._get_bool_env(
                    "AGENT_ZERO_ENABLE_AI_PREDICTIONS",
                    overrides.get("enable_ai_predictions", False),
                ),
                ai_model_type=overrides.get("ai_model_type")
                or os.getenv("AGENT_ZERO_AI_MODEL_TYPE", "random_forest"),
                vector_db_enabled=cls._get_bool_env(
                    "AGENT_ZERO_VECTOR_DB_ENABLED", overrides.get("vector_db_enabled", False)
                ),
                vector_db_provider=overrides.get("vector_db_provider")
                or os.getenv("AGENT_ZERO_VECTOR_DB_PROVIDER", "chromadb"),
                # Cloud Native Features
                kubernetes_enabled=cls._get_bool_env(
                    "AGENT_ZERO_KUBERNETES_ENABLED", overrides.get("kubernetes_enabled", False)
                ),
                kubernetes_namespace=overrides.get("kubernetes_namespace")
                or os.getenv("AGENT_ZERO_KUBERNETES_NAMESPACE", "agent-zero"),
                service_mesh_type=overrides.get("service_mesh_type")
                or os.getenv("AGENT_ZERO_SERVICE_MESH_TYPE"),
                auto_scaling_enabled=cls._get_bool_env(
                    "AGENT_ZERO_AUTO_SCALING_ENABLED", overrides.get("auto_scaling_enabled", True)
                ),
                min_replicas=cls._get_int_env(
                    "AGENT_ZERO_MIN_REPLICAS", overrides.get("min_replicas", 2)
                ),
                max_replicas=cls._get_int_env(
                    "AGENT_ZERO_MAX_REPLICAS", overrides.get("max_replicas", 20)
                ),
                # Security Features
                zero_trust_enabled=cls._get_bool_env(
                    "AGENT_ZERO_ZERO_TRUST_ENABLED", overrides.get("zero_trust_enabled", False)
                ),
                require_mutual_tls=cls._get_bool_env(
                    "AGENT_ZERO_REQUIRE_MUTUAL_TLS", overrides.get("require_mutual_tls", False)
                ),
                certificate_rotation_days=cls._get_int_env(
                    "AGENT_ZERO_CERT_ROTATION_DAYS", overrides.get("certificate_rotation_days", 90)
                ),
                threat_detection_enabled=cls._get_bool_env(
                    "AGENT_ZERO_THREAT_DETECTION_ENABLED",
                    overrides.get("threat_detection_enabled", False),
                ),
                compliance_frameworks=overrides.get("compliance_frameworks")
                or os.getenv("AGENT_ZERO_COMPLIANCE_FRAMEWORKS", "soc2"),
                # Performance Features
                distributed_cache_enabled=cls._get_bool_env(
                    "AGENT_ZERO_DISTRIBUTED_CACHE_ENABLED",
                    overrides.get("distributed_cache_enabled", False),
                ),
                redis_cluster_url=overrides.get("redis_cluster_url")
                or os.getenv("AGENT_ZERO_REDIS_CLUSTER_URL"),
                load_balancing_algorithm=overrides.get("load_balancing_algorithm")
                or os.getenv("AGENT_ZERO_LOAD_BALANCING_ALGORITHM", "least_connections"),
                circuit_breaker_enabled=cls._get_bool_env(
                    "AGENT_ZERO_CIRCUIT_BREAKER_ENABLED",
                    overrides.get("circuit_breaker_enabled", True),
                ),
                real_time_analytics=cls._get_bool_env(
                    "AGENT_ZERO_REAL_TIME_ANALYTICS", overrides.get("real_time_analytics", False)
                ),
                # Advanced Transport Features
                oauth2_enabled=cls._get_bool_env(
                    "AGENT_ZERO_OAUTH2_ENABLED", overrides.get("oauth2_enabled", False)
                ),
                oauth2_client_id=overrides.get("oauth2_client_id")
                or os.getenv("AGENT_ZERO_OAUTH2_CLIENT_ID"),
                oauth2_client_secret=overrides.get("oauth2_client_secret")
                or os.getenv("AGENT_ZERO_OAUTH2_CLIENT_SECRET"),
                json_rpc_batching=cls._get_bool_env(
                    "AGENT_ZERO_JSON_RPC_BATCHING", overrides.get("json_rpc_batching", True)
                ),
                streamable_responses=cls._get_bool_env(
                    "AGENT_ZERO_STREAMABLE_RESPONSES", overrides.get("streamable_responses", True)
                ),
                content_types_supported=overrides.get("content_types_supported")
                or os.getenv("AGENT_ZERO_CONTENT_TYPES_SUPPORTED", "text,image,audio"),
                tool_annotations_enabled=cls._get_bool_env(
                    "AGENT_ZERO_TOOL_ANNOTATIONS_ENABLED",
                    overrides.get("tool_annotations_enabled", True),
                ),
                progress_notifications=cls._get_bool_env(
                    "AGENT_ZERO_PROGRESS_NOTIFICATIONS",
                    overrides.get("progress_notifications", True),
                ),
                completions_capability=cls._get_bool_env(
                    "AGENT_ZERO_COMPLETIONS_CAPABILITY",
                    overrides.get("completions_capability", True),
                ),
            )

            return config

        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required environment variable: {missing_var}")

    def validate(self) -> None:
        """Validate configuration values and raise descriptive errors."""
        # Validate required ClickHouse settings
        if not self.clickhouse_host:
            raise ValueError("ClickHouse host is required")

        if not self.clickhouse_user:
            raise ValueError("ClickHouse user is required")

        if self.clickhouse_password is None:
            raise ValueError("ClickHouse password is required (can be empty string)")

        # Validate port ranges
        if not (1 <= self.server_port <= 65535):
            raise ValueError(f"Server port must be between 1-65535, got: {self.server_port}")

        if self.clickhouse_port and not (1 <= self.clickhouse_port <= 65535):
            raise ValueError(
                f"ClickHouse port must be between 1-65535, got: {self.clickhouse_port}"
            )

        # Validate timeouts
        if self.clickhouse_connect_timeout <= 0:
            raise ValueError("ClickHouse connect timeout must be positive")

        if self.clickhouse_send_receive_timeout <= 0:
            raise ValueError("ClickHouse send/receive timeout must be positive")

        # Validate SSL configuration
        if self.ssl_enable:
            if not self.ssl_certfile or not self.ssl_keyfile:
                raise ValueError("SSL certificate and key files are required when SSL is enabled")

        # Validate authentication
        if self.auth_username and not (self.auth_password or self.auth_password_file):
            raise ValueError(
                "Authentication password or password file is required when username is set"
            )

        # Validate rate limiting
        if self.rate_limit_enabled and self.rate_limit_requests <= 0:
            raise ValueError("Rate limit requests must be positive when rate limiting is enabled")

        # Validate limits
        if self.tool_limit <= 0:
            raise ValueError("Tool limit must be positive")

        if self.resource_limit <= 0:
            raise ValueError("Resource limit must be positive")

    def _auto_configure(self) -> None:
        """Auto-configure derived settings based on primary settings."""
        # Auto-determine ClickHouse port if not explicitly set
        if self.clickhouse_port is None:
            self.clickhouse_port = 8443 if self.clickhouse_secure else 8123

        # Auto-determine transport based on deployment mode and IDE type
        if self.deployment_mode == DeploymentMode.REMOTE:
            if self.transport == TransportType.STDIO:
                self.transport = TransportType.SSE

        # IDE-specific optimizations
        if self.ide_type == IDEType.CURSOR and self.cursor_mode:
            self.transport = self.cursor_transport

    def get_clickhouse_client_config(self) -> dict[str, Any]:
        """Get configuration dictionary for ClickHouse client.

        Returns:
            Dictionary ready for clickhouse_connect.get_client()
        """
        config = {
            "host": self.clickhouse_host,
            "port": self.clickhouse_port,
            "username": self.clickhouse_user,
            "password": self.clickhouse_password,
            "secure": self.clickhouse_secure,
            "verify": self.clickhouse_verify,
            "connect_timeout": self.clickhouse_connect_timeout,
            "send_receive_timeout": self.clickhouse_send_receive_timeout,
            "client_name": "agent-zero-mcp",
        }

        if self.clickhouse_database:
            config["database"] = self.clickhouse_database

        return config

    def get_auth_config(self) -> dict[str, str] | None:
        """Get authentication configuration.

        Returns:
            Dictionary with auth config or None if auth is not configured
        """
        if not self.auth_username:
            return None

        password = self.auth_password

        # Read password from file if specified
        if not password and self.auth_password_file:
            try:
                with open(self.auth_password_file) as f:
                    password = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read password file {self.auth_password_file}: {e}")
                return None

        if password is not None:
            return {"username": self.auth_username, "password": password}

        return None

    def get_ssl_config(self) -> dict[str, str] | None:
        """Get SSL configuration.

        Returns:
            Dictionary with SSL config or None if SSL is not configured
        """
        if not self.ssl_enable or not self.ssl_certfile or not self.ssl_keyfile:
            return None

        return {
            "certfile": self.ssl_certfile,
            "keyfile": self.ssl_keyfile,
        }

    def determine_optimal_transport(self) -> TransportType:
        """Determine the optimal transport type based on configuration.

        Returns:
            The optimal transport type for the current configuration
        """
        # IDE-specific transport preferences
        if self.ide_type == IDEType.CLAUDE_DESKTOP:
            return TransportType.STDIO

        if self.ide_type == IDEType.CURSOR:
            return self.cursor_transport

        # General rules
        if self.deployment_mode == DeploymentMode.REMOTE:
            return TransportType.SSE

        if self.server_host != "127.0.0.1" or self.server_port != 8505:
            return TransportType.SSE

        return self.transport

    @staticmethod
    def _get_bool_env(env_var: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _get_int_env(env_var: str, default: int | None) -> int | None:
        """Get integer value from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        return int(value)

    @staticmethod
    def _get_enum_env(enum_class, env_var: str, default: Any | None) -> Any | None:
        """Get enum value from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        try:
            return enum_class(value)
        except ValueError:
            return default

    @staticmethod
    def _set_debug_defaults() -> None:
        """Set default environment variables for debug mode."""
        debug_defaults = {
            "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
            "AGENT_ZERO_CLICKHOUSE_USER": "default",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
        }

        for var, default_value in debug_defaults.items():
            if var not in os.environ:
                os.environ[var] = default_value
                logger.debug(f"Set debug default: {var}={default_value}")


# Global configuration instance
config = None


def get_config(**overrides) -> UnifiedConfig:
    """Get the global configuration instance.

    Args:
        **overrides: Configuration values to override

    Returns:
        The global UnifiedConfig instance
    """
    global config
    if config is None:
        config = UnifiedConfig.from_env(**overrides)
    return config


def reset_config() -> None:
    """Reset the global configuration instance (useful for testing)."""
    global config
    config = None
