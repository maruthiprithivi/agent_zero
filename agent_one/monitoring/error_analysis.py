"""Error analysis tools for ClickHouse.

This module provides tools for monitoring and analyzing errors and exceptions
in a ClickHouse database.
"""

import logging
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_one.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_recent_errors(
    client: Client,
    days: int = 7,
    normalized_hash: bool = False,
    settings: dict[str, Any] | None = None,
) -> list[dict[str, str | int]]:
    """Get a summary of recent errors from the query log.

    This function retrieves information about recent query errors, grouped by day
    and exception code.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)
        normalized_hash: Group errors by normalized query hash instead of day (default: False)

    Returns:
        List of dictionaries with error information
    """
    if normalized_hash:
        # Use the view that groups errors by normalized query hash
        query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            normalized_query_hash,
            any(query_id) AS query_id,
            query_kind,
            type,
            exception_code,
            errorCodeToName(exception_code) AS exception_name,
            count() AS c
        FROM clusterAllReplicas(default, merge(system, '^query_log*'))
        WHERE (event_date >= (today() - toIntervalDay({days})))
          AND (user NOT ILIKE '%internal%')
          AND (type IN (3, 4))
        GROUP BY ALL WITH TOTALS
        ORDER BY ts DESC, normalized_query_hash ASC
        """
    else:
        # Use the view that groups errors by day
        query = f"""
        SELECT
            toStartOfDay(event_time) AS ts,
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            any(user) AS user,
            query_kind,
            exception_code,
            errorCodeToName(exception_code) AS exception_name,
            count() AS c,
            any(query_id) AS query_id
        FROM clusterAllReplicas(default, merge(system, '^query_log*'))
        WHERE (event_date >= (today() - toIntervalDay({days})))
          AND (user NOT ILIKE '%internal%')
          AND (type IN (3, 4))
        GROUP BY ALL
        ORDER BY ts ASC, max_event_time ASC, c ASC
        SETTINGS skip_unavailable_shards = 1
        """

    grouping = "normalized query hash" if normalized_hash else "day"
    logger.info(f"Retrieving recent errors grouped by {grouping} for the past {days} days")

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error retrieving recent errors: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            toStartOfDay(event_time) AS ts,
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            any(user) AS user,
            query_kind,
            exception_code,
            errorCodeToName(exception_code) AS exception_name,
            count() AS c,
            any(query_id) AS query_id
        FROM system.query_log
        WHERE (event_date >= (today() - toIntervalDay({days})))
          AND (user NOT ILIKE '%internal%')
          AND (type IN (3, 4))
        GROUP BY ts, query_kind, exception_code
        ORDER BY ts ASC, max_event_time ASC, c ASC
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query, settings=settings)


@log_execution_time
def get_error_stack_traces(
    client: Client, error_name: str | None = "LOGICAL_ERROR", settings: dict[str, Any] | None = None
) -> list[dict[str, str | list[str]]]:
    """Get stack traces for specific error types.

    This function retrieves stack traces for errors of a specific type (default: LOGICAL_ERROR).

    Args:
        client: The ClickHouse client instance
        error_name: The name of the error type to retrieve stack traces for (default: LOGICAL_ERROR)
        settings: Optional query settings

    Returns:
        List of dictionaries with error stack trace information
    """
    query = f"""
    SELECT
        last_error_time,
        last_error_message,
        arrayMap(x -> demangle(addressToSymbol(x)), last_error_trace) AS stack_trace
    FROM system.errors
    WHERE name = '{error_name}'
    SETTINGS allow_introspection_functions = 1
    """

    logger.info(f"Retrieving stack traces for error type '{error_name}'")

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error retrieving stack traces: {e!s}")
        # For stack traces, there's no good fallback that doesn't need introspection functions
        logger.warning("No fallback available for stack traces query")
        return []


@log_execution_time
def get_text_log(
    client: Client,
    query_id: str | None = None,
    event_date: str | None = None,
    limit: int = 100,
    settings: dict[str, Any] | None = None,
) -> list[dict[str, str | int]]:
    """Get entries from the text log.

    This function retrieves entries from the ClickHouse text log, optionally filtered
    by query_id and event_date.

    Args:
        client: The ClickHouse client instance
        query_id: Filter by specific query ID (default: None)
        event_date: Filter by event date in 'YYYY-MM-DD' format (default: None)
        limit: Maximum number of log entries to return (default: 100)

    Returns:
        List of dictionaries with text log entries
    """
    where_clauses = []
    if query_id:
        where_clauses.append(f"query_id = '{query_id}'")
    if event_date:
        where_clauses.append(f"event_date = '{event_date}'")

    where_clause = " AND ".join(where_clauses)
    if where_clause:
        where_clause = f"WHERE {where_clause}"

    query = f"""
    SELECT
        event_time_microseconds,
        thread_id,
        level,
        logger_name,
        message
    FROM clusterAllReplicas(default, system.text_log)
    {where_clause}
    ORDER BY event_time_microseconds ASC
    LIMIT {limit}
    """

    filters = []
    if query_id:
        filters.append(f"query_id '{query_id}'")
    if event_date:
        filters.append(f"date '{event_date}'")

    filter_text = " and ".join(filters) if filters else "all entries"
    logger.info(f"Retrieving text log entries for {filter_text} (limit: {limit})")

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error retrieving text log entries: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            event_time_microseconds,
            thread_id,
            level,
            logger_name,
            message
        FROM system.text_log
        {where_clause}
        ORDER BY event_time_microseconds ASC
        LIMIT {limit}
        """
        logger.info("Falling back to local text_log query")
        return execute_query_with_retry(client, fallback_query, settings=settings)
