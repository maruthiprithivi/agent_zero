"""ClickHouse client management for Agent Zero MCP server.

This module handles ClickHouse client creation and connection management
with proper error handling and logging. It supports an optional cached
client to avoid connection churn under load.
"""

import atexit
import logging
import threading

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from agent_zero.config import get_config

logger = logging.getLogger(__name__)

# Optional cached client and lock for thread-safety
_cached_client: Client | None = None
_client_lock = threading.Lock()


def _close_cached_client() -> None:
    global _cached_client
    with _client_lock:
        if _cached_client is not None:
            try:
                # clickhouse-connect Client exposes close() to release resources
                _cached_client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            finally:
                _cached_client = None


atexit.register(_close_cached_client)


def create_clickhouse_client() -> Client:
    """Create and return a ClickHouse client connection.

    When caching is enabled (default), reuse a process-wide client if healthy.

    Returns:
        A configured ClickHouse client instance.

    Raises:
        Exception: If connection fails.
    """
    global _cached_client
    # Read config once to avoid multiple get_config() calls (tests expect this)
    config = get_config()
    should_cache = bool(getattr(config, "enable_client_cache", False))

    if should_cache:
        with _client_lock:
            if _cached_client is not None:
                try:
                    # Lightweight health check; server_version touches connection
                    _ = _cached_client.server_version
                    return _cached_client
                except Exception:
                    # Drop broken client and recreate
                    _close_cached_client()

    client_config = config.get_clickhouse_client_config()

    logger.info(
        f"Creating ClickHouse client connection to {client_config['host']}:{client_config['port']} "
        f"as {client_config['username']} "
        f"(secure={client_config['secure']}, verify={client_config['verify']}, "
        f"connect_timeout={client_config['connect_timeout']}s, "
        f"send_receive_timeout={client_config['send_receive_timeout']}s)"
    )

    try:
        client = clickhouse_connect.get_client(**client_config)
        # Test the connection
        version = client.server_version
        logger.info(f"Successfully connected to ClickHouse server version {version}")

        # Enable instrumentation for logging if query logging is enabled
        if config.enable_query_logging or config.log_query_latency or config.log_query_errors:
            logger.info("Database query logging is enabled")

        if should_cache:
            with _client_lock:
                _cached_client = client

        return client
    except Exception as e:
        logger.error(f"Failed to connect to ClickHouse: {e!s}")
        raise
