"""Production-grade Prometheus metrics for Agent Zero MCP Server.

This module implements comprehensive application metrics following 2025 best practices
for observability, including custom ClickHouse metrics, request tracking, and
performance monitoring.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# Optional imports for metrics functionality
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    prometheus_available = True
except ImportError:
    logger.warning("prometheus_client not available. Metrics will be disabled.")
    prometheus_available = False
    Counter = Gauge = Histogram = Info = CollectorRegistry = None
    CONTENT_TYPE_LATEST = "text/plain"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enabled: bool = True
    include_high_cardinality: bool = False
    prefix: str = "agent_zero"
    deployment_mode: str = "production"
    registry: Any | None = None


class MetricsManager:
    """Manages Prometheus metrics for the MCP server."""

    def __init__(self, config: MetricsConfig | None = None):
        """Initialize metrics manager.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self.enabled = self.config.enabled and prometheus_available

        if not self.enabled:
            logger.warning("Metrics collection disabled")
            return

        # Use custom registry or default
        self.registry = self.config.registry or CollectorRegistry()

        # Initialize core metrics
        self._init_core_metrics()
        self._init_mcp_metrics()
        self._init_clickhouse_metrics()
        self._init_system_metrics()

    def _init_core_metrics(self) -> None:
        """Initialize core application metrics."""
        if not self.enabled:
            return

        prefix = self.config.prefix

        # Application info
        self.app_info = Info(
            f"{prefix}_app_info", "Application information", registry=self.registry
        )
        self.app_info.info(
            {
                "version": __version__,
                "deployment_mode": self.config.deployment_mode,
            }
        )

        # Request metrics
        self.http_requests_total = Counter(
            f"{prefix}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            f"{prefix}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Active connections
        self.active_connections = Gauge(
            f"{prefix}_active_connections",
            "Number of active connections",
            ["connection_type"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors by type",
            ["error_type", "component"],
            registry=self.registry,
        )

    def _init_mcp_metrics(self) -> None:
        """Initialize MCP-specific metrics."""
        if not self.enabled:
            return

        prefix = self.config.prefix

        # MCP tool calls
        self.mcp_tool_calls_total = Counter(
            f"{prefix}_mcp_tool_calls_total",
            "Total MCP tool calls",
            ["tool_name", "status"],
            registry=self.registry,
        )

        self.mcp_tool_duration_seconds = Histogram(
            f"{prefix}_mcp_tool_duration_seconds",
            "MCP tool execution duration in seconds",
            ["tool_name"],
            buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # MCP transport metrics
        self.mcp_messages_total = Counter(
            f"{prefix}_mcp_messages_total",
            "Total MCP messages",
            ["transport", "direction", "message_type"],
            registry=self.registry,
        )

        # Resource usage by tools
        self.mcp_tool_memory_bytes = Gauge(
            f"{prefix}_mcp_tool_memory_bytes",
            "Memory usage by MCP tools",
            ["tool_name"],
            registry=self.registry,
        )

    def _init_clickhouse_metrics(self) -> None:
        """Initialize ClickHouse-specific metrics."""
        if not self.enabled:
            return

        prefix = self.config.prefix

        # Database connections
        self.clickhouse_connections_active = Gauge(
            f"{prefix}_clickhouse_connections_active",
            "Active ClickHouse connections",
            registry=self.registry,
        )

        self.clickhouse_connections_total = Counter(
            f"{prefix}_clickhouse_connections_total",
            "Total ClickHouse connections",
            ["status"],
            registry=self.registry,
        )

        # Query metrics
        self.clickhouse_queries_total = Counter(
            f"{prefix}_clickhouse_queries_total",
            "Total ClickHouse queries",
            ["query_type", "status"],
            registry=self.registry,
        )

        self.clickhouse_query_duration_seconds = Histogram(
            f"{prefix}_clickhouse_query_duration_seconds",
            "ClickHouse query duration in seconds",
            ["query_type"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry,
        )

        # Data metrics
        self.clickhouse_rows_processed = Counter(
            f"{prefix}_clickhouse_rows_processed",
            "Total rows processed by ClickHouse",
            ["operation"],
            registry=self.registry,
        )

        self.clickhouse_bytes_transferred = Counter(
            f"{prefix}_clickhouse_bytes_transferred",
            "Total bytes transferred to/from ClickHouse",
            ["direction"],
            registry=self.registry,
        )

        # Performance metrics
        self.clickhouse_server_memory_usage = Gauge(
            f"{prefix}_clickhouse_server_memory_usage_bytes",
            "ClickHouse server memory usage in bytes",
            registry=self.registry,
        )

        self.clickhouse_server_cpu_usage = Gauge(
            f"{prefix}_clickhouse_server_cpu_usage_percent",
            "ClickHouse server CPU usage percentage",
            registry=self.registry,
        )

    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        if not self.enabled:
            return

        prefix = self.config.prefix

        # Process metrics
        self.process_memory_bytes = Gauge(
            f"{prefix}_process_memory_bytes",
            "Process memory usage in bytes",
            ["memory_type"],
            registry=self.registry,
        )

        self.process_cpu_usage_percent = Gauge(
            f"{prefix}_process_cpu_usage_percent",
            "Process CPU usage percentage",
            registry=self.registry,
        )

        self.process_open_fds = Gauge(
            f"{prefix}_process_open_file_descriptors",
            "Number of open file descriptors",
            registry=self.registry,
        )

        # Garbage collection metrics
        self.gc_collections_total = Counter(
            f"{prefix}_gc_collections_total",
            "Total garbage collections",
            ["generation"],
            registry=self.registry,
        )

        self.gc_objects_collected_total = Counter(
            f"{prefix}_gc_objects_collected_total",
            "Total objects collected by garbage collector",
            ["generation"],
            registry=self.registry,
        )

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ) -> None:
        """Record HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        if not self.enabled:
            return

        try:
            self.http_requests_total.labels(
                method=method, endpoint=endpoint, status=str(status_code)
            ).inc()

            self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                duration
            )
        except Exception as e:
            logger.error(f"Failed to record HTTP request metrics: {e}")

    def record_mcp_tool_call(self, tool_name: str, status: str, duration: float) -> None:
        """Record MCP tool call metrics.

        Args:
            tool_name: Name of the MCP tool
            status: Call status (success, error, timeout)
            duration: Call duration in seconds
        """
        if not self.enabled:
            return

        try:
            self.mcp_tool_calls_total.labels(tool_name=tool_name, status=status).inc()

            self.mcp_tool_duration_seconds.labels(tool_name=tool_name).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record MCP tool call metrics: {e}")

    def record_clickhouse_query(
        self, query_type: str, status: str, duration: float, rows: int = 0
    ) -> None:
        """Record ClickHouse query metrics.

        Args:
            query_type: Type of query (SELECT, INSERT, etc.)
            status: Query status (success, error, timeout)
            duration: Query duration in seconds
            rows: Number of rows processed
        """
        if not self.enabled:
            return

        try:
            self.clickhouse_queries_total.labels(query_type=query_type, status=status).inc()

            self.clickhouse_query_duration_seconds.labels(query_type=query_type).observe(duration)

            if rows > 0:
                self.clickhouse_rows_processed.labels(operation=query_type.lower()).inc(rows)
        except Exception as e:
            logger.error(f"Failed to record ClickHouse query metrics: {e}")

    def record_error(self, error_type: str, component: str) -> None:
        """Record error metrics.

        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        if not self.enabled:
            return

        try:
            self.errors_total.labels(error_type=error_type, component=component).inc()
        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}")

    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        if not self.enabled:
            return

        try:
            import gc

            import psutil

            # Process metrics
            process = psutil.Process()
            memory_info = process.memory_info()

            self.process_memory_bytes.labels(memory_type="rss").set(memory_info.rss)
            self.process_memory_bytes.labels(memory_type="vms").set(memory_info.vms)
            self.process_cpu_usage_percent.set(process.cpu_percent())

            # File descriptors (Unix only)
            try:
                self.process_open_fds.set(process.num_fds())
            except (AttributeError, psutil.AccessDenied):
                pass  # Not available on Windows or access denied

            # Garbage collection metrics
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                self.gc_collections_total.labels(generation=str(i)).inc(
                    stats["collections"] - getattr(self, f"_last_gc_collections_{i}", 0)
                )
                self.gc_objects_collected_total.labels(generation=str(i)).inc(
                    stats["collected"] - getattr(self, f"_last_gc_collected_{i}", 0)
                )
                setattr(self, f"_last_gc_collections_{i}", stats["collections"])
                setattr(self, f"_last_gc_collected_{i}", stats["collected"])

        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def set_active_connections(self, connection_type: str, count: int) -> None:
        """Set active connection count.

        Args:
            connection_type: Type of connection (websocket, http, sse)
            count: Current connection count
        """
        if not self.enabled:
            return

        try:
            self.active_connections.labels(connection_type=connection_type).set(count)
        except Exception as e:
            logger.error(f"Failed to set active connections: {e}")

    @contextmanager
    def time_operation(
        self, operation_name: str, labels: dict[str, str] | None = None
    ) -> Generator[None, None, None]:
        """Context manager to time operations.

        Args:
            operation_name: Name of the operation to time
            labels: Additional labels for the metric

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            # This is a generic timing context manager
            # Specific metrics should use dedicated methods

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Metrics data in Prometheus text format
        """
        if not self.enabled:
            return "# Metrics collection disabled\n"

        try:
            # Update system metrics before export
            self.update_system_metrics()
            return generate_latest(self.registry).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return f"# Error exporting metrics: {e}\n"

    def get_content_type(self) -> str:
        """Get the content type for metrics export.

        Returns:
            Content type string for HTTP response
        """
        return CONTENT_TYPE_LATEST


def metrics_decorator(metrics_manager: "MetricsManager", tool_name: str):
    """Decorator for automatic MCP tool metrics collection.

    Args:
        metrics_manager: MetricsManager instance
        tool_name: Name of the MCP tool

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics_manager.record_error(type(e).__name__, "mcp_tool")
                raise
            finally:
                duration = time.time() - start_time
                metrics_manager.record_mcp_tool_call(tool_name, status, duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics_manager.record_error(type(e).__name__, "mcp_tool")
                raise
            finally:
                duration = time.time() - start_time
                metrics_manager.record_mcp_tool_call(tool_name, status, duration)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global metrics manager instance
metrics_manager: MetricsManager | None = None


def get_metrics_manager(config: MetricsConfig | None = None) -> MetricsManager:
    """Get or create the global metrics manager.

    Args:
        config: Metrics configuration

    Returns:
        Global metrics manager instance
    """
    global metrics_manager
    if metrics_manager is None:
        metrics_manager = MetricsManager(config)
    return metrics_manager


def reset_metrics_manager() -> None:
    """Reset the global metrics manager (useful for testing)."""
    global metrics_manager
    metrics_manager = None
