"""MCP server tracing utilities.

This module provides functionality for tracing and logging communications in the MCP server.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from agent_zero.mcp_env import config

logger = logging.getLogger("mcp-tracing")

T = TypeVar("T")


class MCPTracer:
    """Tracer for MCP server communications.

    This class provides methods for tracing and logging communications in the MCP server.
    """

    def __init__(self):
        """Initialize the MCP tracer."""
        # Configure detailed formatting for trace logs
        self._configure_logger()
        self.trace_id_counter = 0

    def _configure_logger(self):
        """Configure the logger with appropriate formatting."""
        # Create a handler for the logger if it doesn't already have one
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def generate_trace_id(self) -> str:
        """Generate a unique trace ID.

        Returns:
            A unique trace ID string
        """
        self.trace_id_counter += 1
        return f"{uuid.uuid4().hex[:8]}-{self.trace_id_counter}"

    def log_request(self, endpoint: str, method: str, payload: Any, trace_id: str) -> None:
        """Log an incoming request.

        Args:
            endpoint: The API endpoint
            method: The HTTP method
            payload: The request payload
            trace_id: The trace ID for correlation
        """
        if not config.enable_mcp_tracing:
            return

        try:
            payload_str = json.dumps(payload) if payload else "None"
            logger.info(f"TRACE-IN [{trace_id}] {method} {endpoint} - Payload: {payload_str}")
        except Exception as e:
            logger.error(f"Failed to log request: {e}")

    def log_response(
        self,
        endpoint: str,
        status_code: int,
        response_data: Any,
        trace_id: str,
        elapsed_time: float,
    ) -> None:
        """Log an outgoing response.

        Args:
            endpoint: The API endpoint
            status_code: The HTTP status code
            response_data: The response data
            trace_id: The trace ID for correlation
            elapsed_time: The time taken to process the request in seconds
        """
        if not config.enable_mcp_tracing:
            return

        try:
            # Truncate large responses to avoid log flooding
            response_str = str(response_data)
            if len(response_str) > 1000:
                response_str = response_str[:1000] + "... [truncated]"

            logger.info(
                f"TRACE-OUT [{trace_id}] {endpoint} - Status: {status_code}, "
                f"Time: {elapsed_time:.4f}s, Response: {response_str}"
            )
        except Exception as e:
            logger.error(f"Failed to log response: {e}")

    def log_error(self, endpoint: str, error: Exception, trace_id: str) -> None:
        """Log an error that occurred during request processing.

        Args:
            endpoint: The API endpoint
            error: The exception that occurred
            trace_id: The trace ID for correlation
        """
        if not config.enable_mcp_tracing:
            return

        logger.error(f"TRACE-ERROR [{trace_id}] {endpoint} - Error: {error!s}")


# Create a global instance
mcp_tracer = MCPTracer()


def trace_mcp_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to trace MCP tool calls.

    This decorator logs the incoming request, the outgoing response, and any errors
    that occur during processing.

    Args:
        func: The function to decorate (typically an MCP tool function)

    Returns:
        The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not config.enable_mcp_tracing:
            return func(*args, **kwargs)

        # Generate a trace ID for this request
        trace_id = mcp_tracer.generate_trace_id()

        # Log the request
        endpoint = func.__name__
        mcp_tracer.log_request(endpoint, "CALL", kwargs, trace_id)

        # Process the request
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            # Log the response
            mcp_tracer.log_response(endpoint, 200, result, trace_id, elapsed_time)

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            mcp_tracer.log_error(endpoint, e, trace_id)
            raise

    return wrapper
