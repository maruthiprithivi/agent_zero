"""MCP server tracing utilities.

This module provides functionality for tracing and logging communications in the MCP server.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from agent_zero.config import get_config

logger = logging.getLogger("mcp-tracing")

T = TypeVar("T")


class MCPTracer:
    """Tracer for MCP server communications and lightweight in-memory traces."""

    def __init__(self, enabled: bool | None = None):
        """Initialize the MCP tracer.

        Args:
            enabled: Optional explicit toggle; if None, reads from config
        """
        self._configure_logger()
        self.trace_id_counter = 0
        # Simple in-memory trace store for tests and debugging
        self.traces: list[dict[str, Any]] = []
        # If not explicitly provided, derive from config lazily when used
        self.enabled = bool(enabled) if enabled is not None else False

    def _configure_logger(self) -> None:
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _is_enabled(self) -> bool:
        # Prefer explicit flag on instance; otherwise consult global config
        if self.enabled:
            return True
        try:
            config = get_config()
            return bool(getattr(config, "enable_mcp_tracing", False))
        except Exception:
            return False

    def generate_trace_id(self) -> str:
        self.trace_id_counter += 1
        return f"{uuid.uuid4().hex[:8]}-{self.trace_id_counter}"

    def add_trace(self, trace: dict[str, Any]) -> None:
        self.traces.append(trace)

    def get_traces(self) -> list[dict[str, Any]]:
        return list(self.traces)

    def clear_traces(self) -> None:
        self.traces.clear()

    def log_request(self, endpoint: str, method: str, payload: Any, trace_id: str) -> None:
        if not self._is_enabled():
            return
        try:
            payload_str = json.dumps(payload) if payload else "None"
            logger.info(f"TRACE-IN [{trace_id}] {method} {endpoint} - Payload: {payload_str}")
            self.add_trace({"direction": "in", "id": trace_id, "endpoint": endpoint})
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
        if not self._is_enabled():
            return
        try:
            response_str = str(response_data)
            if len(response_str) > 1000:
                response_str = response_str[:1000] + "... [truncated]"
            logger.info(
                f"TRACE-OUT [{trace_id}] {endpoint} - Status: {status_code}, Time: {elapsed_time:.4f}s, Response: {response_str}"
            )
            self.add_trace(
                {
                    "direction": "out",
                    "id": trace_id,
                    "endpoint": endpoint,
                    "ms": int(elapsed_time * 1000),
                }
            )
        except Exception as e:
            logger.error(f"Failed to log response: {e}")

    def log_error(self, endpoint: str, error: Exception, trace_id: str) -> None:
        if not self._is_enabled():
            return
        logger.error(f"TRACE-ERROR [{trace_id}] {endpoint} - Error: {error!s}")
        self.add_trace(
            {"direction": "error", "id": trace_id, "endpoint": endpoint, "error": str(error)}
        )


# Global tracer instance used by decorator
mcp_tracer = MCPTracer()


def trace_mcp_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to trace MCP tool calls (sync and async)."""

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):  # type: ignore[no-redef]
            if not mcp_tracer._is_enabled():
                return await cast(Callable[..., Any], func)(*args, **kwargs)

            trace_id = mcp_tracer.generate_trace_id()
            endpoint = func.__name__
            mcp_tracer.log_request(endpoint, "CALL", kwargs, trace_id)

            start_time = time.time()
            try:
                result = await cast(Callable[..., Any], func)(*args, **kwargs)
                elapsed_time = time.time() - start_time
                mcp_tracer.log_response(endpoint, 200, result, trace_id, elapsed_time)
                return result
            except Exception as e:
                mcp_tracer.log_error(endpoint, e, trace_id)
                raise

        return async_wrapper  # type: ignore[return-value]

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not mcp_tracer._is_enabled():
            return cast(Callable[..., Any], func)(*args, **kwargs)

        trace_id = mcp_tracer.generate_trace_id()
        endpoint = func.__name__
        mcp_tracer.log_request(endpoint, "CALL", kwargs, trace_id)

        start_time = time.time()
        try:
            result = cast(Callable[..., Any], func)(*args, **kwargs)
            elapsed_time = time.time() - start_time
            mcp_tracer.log_response(endpoint, 200, result, trace_id, elapsed_time)
            return result
        except Exception as e:
            mcp_tracer.log_error(endpoint, e, trace_id)
            raise

    return wrapper  # type: ignore[return-value]
