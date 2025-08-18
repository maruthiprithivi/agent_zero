"""Query execution utilities for Agent Zero MCP server.

This module handles safe query execution with proper logging and error handling.
Adds helpers for structured results and bounded executions while preserving
legacy behavior by default.
"""

import atexit
import concurrent.futures
import logging
import time
from collections.abc import Sequence
from typing import Any

from agent_zero.config import get_config
from agent_zero.utils import format_exception

from .client import create_clickhouse_client

logger = logging.getLogger(__name__)

# Global query executor for threaded queries
QUERY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10)
atexit.register(lambda: QUERY_EXECUTOR.shutdown(wait=True))

SELECT_QUERY_TIMEOUT_SECS = 30


def execute_query(query: str) -> list[dict[str, Any]]:
    """Execute a read-only SQL query.

    Args:
        query: The SQL query to execute.

    Returns:
        The query results as a list of dictionaries.
    """
    # Defer to tests which may patch create_clickhouse_client in this module
    try:
        client = create_clickhouse_client()  # type: ignore[misc]
    except Exception:
        # Fallback import path for tests that patch agent_zero.server.client.create_clickhouse_client
        from .client import create_clickhouse_client as _create

        client = _create()
    try:
        # Import the database logger if needed
        from agent_zero.database_logger import query_logger

        # In tests, environment may be missing required vars; fallback to minimal defaults
        try:
            config = get_config()
        except Exception:

            class _Tmp:
                enable_query_logging = False
                log_query_latency = False
                log_query_errors = False
                default_max_rows = None
                default_max_execution_seconds = None

            config = _Tmp()

        # Log the query if query logging is enabled
        if config.enable_query_logging:
            query_logger.log_query(query, None, {"readonly": 1})

        start_time = time.time()
        # Apply default limits from config for safety
        settings = {"readonly": 1}
        try:
            cfg = get_config()
        except Exception:

            class _TmpCfg:
                default_max_rows = None
                default_max_execution_seconds = None

            cfg = _TmpCfg()
        if getattr(cfg, "default_max_rows", None):
            settings["max_result_rows"] = cfg.default_max_rows
            settings["result_overflow_mode"] = "break"
        if getattr(cfg, "default_max_execution_seconds", None):
            settings["max_execution_time"] = cfg.default_max_execution_seconds

        res = client.query(query, settings=settings)

        # Log query latency if enabled
        if config.log_query_latency:
            elapsed_time = time.time() - start_time
            logger.info(f"Query executed in {elapsed_time:.4f}s")

        column_names = res.column_names
        rows = []
        for row in res.result_rows:
            row_dict = {}
            for i, col_name in enumerate(column_names):
                row_dict[col_name] = row[i]
            rows.append(row_dict)

        # Log the result if query logging is enabled
        if config.enable_query_logging:
            query_logger.log_query_result(len(rows))

        logger.info(f"Query returned {len(rows)} rows")
        return rows
    except Exception as err:
        # Log the error if error logging is enabled
        config = get_config()
        if config.log_query_errors:
            query_logger.log_query_error(err, query)

        logger.error(f"Error executing query: {err}")
        return f"error running query: {format_exception(err)}"


def format_tabular(column_names: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Format columnar results to a structured tabular dict.

    Returns a stable shape for MCP tool outputs when enabled via config.
    """
    data = [list(row) for row in rows]
    return {"columns": list(column_names), "data": data, "meta": {"row_count": len(data)}}


def execute_query_threaded(query: str) -> Any:
    """Execute a read-only SELECT query in a separate thread with timeout.

    Args:
        query: The SQL query to execute (must be read-only).

    Returns:
        The query results as a list of dictionaries or error string.
    """
    logger.info(f"Executing SELECT query: {query}")
    future = QUERY_EXECUTOR.submit(execute_query, query)
    try:
        result = future.result(timeout=SELECT_QUERY_TIMEOUT_SECS)
        return result
    except concurrent.futures.TimeoutError:
        logger.warning(f"Query timed out after {SELECT_QUERY_TIMEOUT_SECS} seconds: {query}")
        future.cancel()
        return f"error running query: Query timed out after {SELECT_QUERY_TIMEOUT_SECS} seconds"
