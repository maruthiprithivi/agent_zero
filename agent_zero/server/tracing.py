"""Production-grade OpenTelemetry distributed tracing for Agent Zero MCP Server.

This module implements comprehensive distributed tracing following 2025 best practices
with OpenTelemetry integration, custom instrumentation, and trace correlation.
"""

import asyncio
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# Optional imports for OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.http import get_excluded_urls

    opentelemetry_available = True
except ImportError:
    logger.warning("OpenTelemetry not available. Tracing will be disabled.")
    opentelemetry_available = False
    trace = None


T = TypeVar("T")


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    enabled: bool = True
    service_name: str = "agent-zero-mcp"
    service_version: str = __version__
    environment: str = "production"
    sample_rate: float = 1.0
    export_endpoint: str | None = None
    export_headers: dict[str, str] | None = None
    enable_prometheus_metrics: bool = True
    enable_logging_instrumentation: bool = True
    enable_aiohttp_instrumentation: bool = True


class TracingManager:
    """Manages OpenTelemetry tracing configuration and instrumentation."""

    def __init__(self, config: TracingConfig | None = None):
        """Initialize tracing manager.

        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfig()
        self.enabled = self.config.enabled and opentelemetry_available
        self.tracer: Any | None = None
        self._initialized = False

        if not self.enabled:
            logger.warning("Distributed tracing disabled")
            return

        self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not self.enabled:
            return

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: self.config.service_name,
                    ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
                }
            )

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Configure sampling
            if self.config.sample_rate < 1.0:
                from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

                provider._sampler = TraceIdRatioBased(self.config.sample_rate)

            # Setup exporters
            self._setup_exporters(provider)

            # Set as global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self.tracer = trace.get_tracer(self.config.service_name, self.config.service_version)

            # Setup automatic instrumentation
            self._setup_auto_instrumentation()

            self._initialized = True
            logger.info(f"OpenTelemetry tracing initialized for {self.config.service_name}")

        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
            self.enabled = False

    def _setup_exporters(self, provider: Any) -> None:
        """Setup trace exporters.

        Args:
            provider: Tracer provider
        """
        # Console exporter for development
        if self.config.environment == "development":
            from opentelemetry.exporter.console import ConsoleSpanExporter

            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # OTLP exporter for production
        if self.config.export_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.export_endpoint, headers=self.config.export_headers or {}
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OTLP trace exporter configured for {self.config.export_endpoint}")

            except ImportError:
                logger.warning("OTLP exporter not available")
            except Exception as e:
                logger.error(f"Failed to setup OTLP exporter: {e}")

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation."""
        try:
            # Logging instrumentation
            if self.config.enable_logging_instrumentation:
                LoggingInstrumentor().instrument(set_logging_format=True)
                logger.debug("Logging instrumentation enabled")

            # HTTP client instrumentation
            if self.config.enable_aiohttp_instrumentation:
                AioHttpClientInstrumentor().instrument()
                logger.debug("aiohttp client instrumentation enabled")

        except Exception as e:
            logger.error(f"Failed to setup auto instrumentation: {e}")

    def create_span(
        self,
        name: str,
        kind: Any | None = None,
        attributes: dict[str, Any] | None = None,
        parent: Any | None = None,
    ) -> Any:
        """Create a new span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            parent: Parent span or context

        Returns:
            New span or dummy span if tracing disabled
        """
        if not self.enabled or not self.tracer:
            return DummySpan()

        try:
            span = self.tracer.start_span(
                name=name, kind=kind, attributes=attributes, context=parent
            )
            return span
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return DummySpan()

    @contextmanager
    def trace_operation(
        self, name: str, attributes: dict[str, Any] | None = None, kind: Any | None = None
    ) -> Generator[Any, None, None]:
        """Context manager for tracing operations.

        Args:
            name: Operation name
            attributes: Span attributes
            kind: Span kind

        Yields:
            Span object
        """
        if not self.enabled:
            yield DummySpan()
            return

        span = self.create_span(name, kind=kind, attributes=attributes)

        try:
            with trace.use_span(span, end_on_exit=True):
                yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

    def trace_mcp_tool(self, tool_name: str):
        """Decorator for tracing MCP tool calls.

        Args:
            tool_name: Name of the MCP tool

        Returns:
            Decorator function
        """

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attributes = {
                    "mcp.tool.name": tool_name,
                    "mcp.tool.type": "async",
                }

                with self.trace_operation(f"mcp.tool.{tool_name}", attributes):
                    try:
                        result = await func(*args, **kwargs)
                        # Add result metadata to span if available
                        current_span = trace.get_current_span()
                        if current_span and hasattr(result, "__len__"):
                            current_span.set_attribute("mcp.tool.result_size", len(result))
                        return result
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("mcp.tool.error", str(e))
                            current_span.set_attribute("mcp.tool.error_type", type(e).__name__)
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attributes = {
                    "mcp.tool.name": tool_name,
                    "mcp.tool.type": "sync",
                }

                with self.trace_operation(f"mcp.tool.{tool_name}", attributes):
                    try:
                        result = func(*args, **kwargs)
                        # Add result metadata to span if available
                        current_span = trace.get_current_span()
                        if current_span and hasattr(result, "__len__"):
                            current_span.set_attribute("mcp.tool.result_size", len(result))
                        return result
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("mcp.tool.error", str(e))
                            current_span.set_attribute("mcp.tool.error_type", type(e).__name__)
                        raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def trace_clickhouse_query(self, query_type: str = "unknown"):
        """Decorator for tracing ClickHouse queries.

        Args:
            query_type: Type of query (SELECT, INSERT, etc.)

        Returns:
            Decorator function
        """

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attributes = {
                    "db.system": "clickhouse",
                    "db.operation": query_type,
                    "db.name": "default",  # TODO: Get from config
                }

                # Try to extract query from arguments
                query = None
                if args and isinstance(args[0], str):
                    query = args[0][:100]  # Truncate for safety
                elif "query" in kwargs:
                    query = kwargs["query"][:100]

                if query:
                    attributes["db.statement"] = query

                with self.trace_operation(f"db.clickhouse.{query_type.lower()}", attributes):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)

                        # Add result metadata to span
                        current_span = trace.get_current_span()
                        if current_span:
                            duration = time.time() - start_time
                            current_span.set_attribute("db.duration_ms", duration * 1000)

                            if hasattr(result, "__len__"):
                                current_span.set_attribute("db.rows_affected", len(result))

                        return result
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("db.error", str(e))
                            current_span.set_attribute("db.error_type", type(e).__name__)
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attributes = {
                    "db.system": "clickhouse",
                    "db.operation": query_type,
                    "db.name": "default",  # TODO: Get from config
                }

                # Try to extract query from arguments
                query = None
                if args and isinstance(args[0], str):
                    query = args[0][:100]  # Truncate for safety
                elif "query" in kwargs:
                    query = kwargs["query"][:100]

                if query:
                    attributes["db.statement"] = query

                with self.trace_operation(f"db.clickhouse.{query_type.lower()}", attributes):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)

                        # Add result metadata to span
                        current_span = trace.get_current_span()
                        if current_span:
                            duration = time.time() - start_time
                            current_span.set_attribute("db.duration_ms", duration * 1000)

                            if hasattr(result, "__len__"):
                                current_span.set_attribute("db.rows_affected", len(result))

                        return result
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("db.error", str(e))
                            current_span.set_attribute("db.error_type", type(e).__name__)
                        raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def trace_http_request(self, method: str, endpoint: str):
        """Decorator for tracing HTTP requests.

        Args:
            method: HTTP method
            endpoint: Request endpoint

        Returns:
            Decorator function
        """

        def decorator(func):
            @wraps(func)
            async def async_wrapper(request, *args, **kwargs):
                attributes = {
                    SpanAttributes.HTTP_METHOD: method,
                    SpanAttributes.HTTP_URL: f"{request.scheme}://{request.host}{request.path}",
                    SpanAttributes.HTTP_ROUTE: endpoint,
                    SpanAttributes.HTTP_USER_AGENT: request.headers.get("User-Agent", ""),
                    SpanAttributes.HTTP_CLIENT_IP: getattr(request, "remote", ""),
                }

                with self.trace_operation(f"http.{method.lower()}.{endpoint}", attributes):
                    start_time = time.time()
                    try:
                        response = await func(request, *args, **kwargs)

                        # Add response metadata to span
                        current_span = trace.get_current_span()
                        if current_span:
                            duration = time.time() - start_time
                            current_span.set_attribute("http.duration_ms", duration * 1000)

                            if hasattr(response, "status"):
                                current_span.set_attribute(
                                    SpanAttributes.HTTP_STATUS_CODE, response.status
                                )

                                # Set span status based on HTTP status
                                if response.status >= 400:
                                    current_span.set_status(Status(StatusCode.ERROR))

                        return response
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("http.error", str(e))
                            current_span.set_attribute("http.error_type", type(e).__name__)
                            current_span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            @wraps(func)
            def sync_wrapper(request, *args, **kwargs):
                attributes = {
                    SpanAttributes.HTTP_METHOD: method,
                    SpanAttributes.HTTP_URL: f"{request.scheme}://{request.host}{request.path}",
                    SpanAttributes.HTTP_ROUTE: endpoint,
                    SpanAttributes.HTTP_USER_AGENT: request.headers.get("User-Agent", ""),
                    SpanAttributes.HTTP_CLIENT_IP: getattr(request, "remote", ""),
                }

                with self.trace_operation(f"http.{method.lower()}.{endpoint}", attributes):
                    start_time = time.time()
                    try:
                        response = func(request, *args, **kwargs)

                        # Add response metadata to span
                        current_span = trace.get_current_span()
                        if current_span:
                            duration = time.time() - start_time
                            current_span.set_attribute("http.duration_ms", duration * 1000)

                            if hasattr(response, "status"):
                                current_span.set_attribute(
                                    SpanAttributes.HTTP_STATUS_CODE, response.status
                                )

                                # Set span status based on HTTP status
                                if response.status >= 400:
                                    current_span.set_status(Status(StatusCode.ERROR))

                        return response
                    except Exception as e:
                        current_span = trace.get_current_span()
                        if current_span:
                            current_span.set_attribute("http.error", str(e))
                            current_span.set_attribute("http.error_type", type(e).__name__)
                            current_span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_trace_context(self) -> dict[str, str]:
        """Get current trace context for propagation.

        Returns:
            Trace context headers
        """
        if not self.enabled:
            return {}

        try:
            headers = {}
            inject(headers)
            return headers
        except Exception as e:
            logger.error(f"Failed to get trace context: {e}")
            return {}

    def set_trace_context(self, headers: dict[str, str]) -> None:
        """Set trace context from headers.

        Args:
            headers: Headers containing trace context
        """
        if not self.enabled:
            return

        try:
            context = extract(headers)
            trace.set_span_in_context(trace.get_current_span(), context)
        except Exception as e:
            logger.error(f"Failed to set trace context: {e}")

    def add_span_attribute(self, key: str, value: Any) -> None:
        """Add attribute to current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(key, value)
        except Exception as e:
            logger.error(f"Failed to add span attribute: {e}")

    def add_span_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event(name, attributes or {})
        except Exception as e:
            logger.error(f"Failed to add span event: {e}")

    def record_exception(self, exception: Exception) -> None:
        """Record exception in current span.

        Args:
            exception: Exception to record
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.record_exception(exception)
                current_span.set_status(Status(StatusCode.ERROR, str(exception)))
        except Exception as e:
            logger.error(f"Failed to record exception: {e}")


class DummySpan:
    """Dummy span for when tracing is disabled."""

    def __init__(self):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Global tracing manager
tracing_manager: TracingManager | None = None


def get_tracing_manager(config: TracingConfig | None = None) -> TracingManager:
    """Get or create the global tracing manager.

    Args:
        config: Tracing configuration

    Returns:
        Global tracing manager instance
    """
    global tracing_manager
    if tracing_manager is None:
        tracing_manager = TracingManager(config)
    return tracing_manager


def reset_tracing_manager() -> None:
    """Reset the global tracing manager (useful for testing)."""
    global tracing_manager
    tracing_manager = None


# Convenience decorators using global tracing manager
def trace_mcp_tool(tool_name: str):
    """Convenience decorator for MCP tool tracing."""
    return get_tracing_manager().trace_mcp_tool(tool_name)


def trace_clickhouse_query(query_type: str = "unknown"):
    """Convenience decorator for ClickHouse query tracing."""
    return get_tracing_manager().trace_clickhouse_query(query_type)


def trace_http_request(method: str, endpoint: str):
    """Convenience decorator for HTTP request tracing."""
    return get_tracing_manager().trace_http_request(method, endpoint)
