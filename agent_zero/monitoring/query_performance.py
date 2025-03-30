"""Query performance monitoring tools for ClickHouse.

This module provides tools for monitoring query performance metrics in a ClickHouse cluster.
"""

import logging
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_current_processes(
    client: Client, settings: dict[str, Any] | None = None
) -> list[dict[str, str | int | float]]:
    """Get information about currently running processes on the ClickHouse cluster.

    This function retrieves information about all currently running queries on the cluster,
    including resource usage, query type, and elapsed time.

    Args:
        client: The ClickHouse client instance
        settings: Optional query settings

    Returns:
        List of dictionaries with information about each running process
    """
    base_query = """
    SELECT
        hostname(),
        user,
        query_kind,
        elapsed,
        formatReadableQuantity(read_rows) AS read_rows_,
        formatReadableSize(read_bytes) AS read_bytes_,
        formatReadableQuantity(total_rows_approx) AS total_rows_approx_,
        formatReadableQuantity(written_rows) AS written_rows_,
        formatReadableSize(written_bytes) AS written_bytes_,
        formatReadableSize(memory_usage) AS memory_usage_,
        formatReadableSize(peak_memory_usage) AS peak_memory_usage_,
        query_id,
        normalizedQueryHash(query) AS query_hash,
        query
    FROM {table}
    ORDER BY elapsed DESC
    """

    logger.info("Retrieving information about currently running processes")
    try:
        return execute_query_with_retry(
            client,
            base_query.format(table="clusterAllReplicas(default, system.processes)"),
            settings=settings,
        )
    except ClickHouseError as e:
        logger.error(f"Error retrieving current processes: {e!s}")
        # Fall back to local processes if cluster query fails
        logger.info("Falling back to local processes query")
        return execute_query_with_retry(
            client, base_query.format(table="system.processes"), settings=settings
        )


@log_execution_time
def get_query_duration_stats(
    client: Client, query_kind: str | None = None, days: int = 7
) -> list[dict[str, str | int | float]]:
    """Get query duration statistics grouped by hour.

    This function retrieves aggregated query statistics (duration, memory usage, CPU usage)
    grouped by hour for the specified time period.

    Args:
        client: The ClickHouse client instance
        query_kind: Filter by specific query kind (e.g., 'Select', 'Insert'), or None for all queries
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with hourly query statistics
    """
    # Build the WHERE clause based on query_kind
    where_kind = f"AND (query_kind = '{query_kind}')" if query_kind else ""

    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        countDistinct(normalized_query_hash) AS query_uniq,
        count() AS query_count,
        round(query_count / 3600) AS qps,
        round(quantile(0.5)(query_duration_ms)) AS time_p50,
        round(quantile(0.9)(query_duration_ms)) AS time_p90,
        max(query_duration_ms) AS time_max,
        formatReadableQuantity(round(quantile(0.5)(read_rows))) AS read_rows_p50,
        formatReadableQuantity(round(quantile(0.9)(read_rows))) AS read_rows_p90,
        formatReadableQuantity(max(read_rows)) AS read_rows_max,
        formatReadableQuantity(round(quantile(0.5)(written_rows))) AS written_rows_p50,
        formatReadableQuantity(round(quantile(0.9)(written_rows))) AS written_rows_p90,
        formatReadableQuantity(max(written_rows)) AS written_rows_max,
        formatReadableSize(quantile(0.5)(memory_usage)) AS mem_p50,
        formatReadableSize(quantile(0.9)(memory_usage)) AS mem_p90,
        formatReadableSize(max(memory_usage)) AS mem_max,
        formatReadableSize(sum(memory_usage)) AS mem_sum,
        round(quantile(0.5)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p50,
        round(quantile(0.9)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p90,
        max(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_max,
        sum(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_sum
    FROM clusterAllReplicas(default, merge(system, '^query_log*'))
    WHERE (type != 'QueryStart')
      AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
      AND (user NOT ILIKE '%internal%')
      {where_kind}
    GROUP BY ts
    ORDER BY ts ASC
    SETTINGS skip_unavailable_shards = 1
    """

    kind_desc = f"'{query_kind}'" if query_kind else "all"
    logger.info(
        f"Retrieving query duration statistics for {kind_desc} queries over the past {days} days"
    )

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving query duration stats: {e!s}")
        # Fallback to non-clustered query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            countDistinct(normalized_query_hash) AS query_uniq,
            count() AS query_count,
            round(query_count / 3600) AS qps,
            round(quantile(0.5)(query_duration_ms)) AS time_p50,
            round(quantile(0.9)(query_duration_ms)) AS time_p90,
            max(query_duration_ms) AS time_max,
            formatReadableQuantity(round(quantile(0.5)(read_rows))) AS read_rows_p50,
            formatReadableQuantity(round(quantile(0.9)(read_rows))) AS read_rows_p90,
            formatReadableQuantity(max(read_rows)) AS read_rows_max,
            formatReadableQuantity(round(quantile(0.5)(written_rows))) AS written_rows_p50,
            formatReadableQuantity(round(quantile(0.9)(written_rows))) AS written_rows_p90,
            formatReadableQuantity(max(written_rows)) AS written_rows_max,
            formatReadableSize(quantile(0.5)(memory_usage)) AS mem_p50,
            formatReadableSize(quantile(0.9)(memory_usage)) AS mem_p90,
            formatReadableSize(max(memory_usage)) AS mem_max,
            formatReadableSize(sum(memory_usage)) AS mem_sum,
            round(quantile(0.5)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p50,
            round(quantile(0.9)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p90,
            max(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_max,
            sum(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_sum
        FROM system.query_log
        WHERE (type != 'QueryStart')
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
          AND (user NOT ILIKE '%internal%')
          {where_kind}
        GROUP BY ts
        ORDER BY ts ASC
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_normalized_query_stats(
    client: Client, days: int = 2, limit: int = 50
) -> list[dict[str, str | int | float]]:
    """Get performance statistics grouped by normalized query hash.

    This function identifies the most resource-intensive query patterns by
    grouping queries with the same normalized hash and calculating performance metrics.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 2)
        limit: Maximum number of query patterns to return (default: 50)

    Returns:
        List of dictionaries with statistics for each query pattern
    """
    query = f"""
    SELECT
        normalized_query_hash,
        query_kind,
        count() AS q_count,
        countDistinct(query) AS q_distinct,
        formatReadableQuantity(round(quantile(0.5)(read_rows))) AS read_rows_p50,
        formatReadableQuantity(round(quantile(0.9)(read_rows))) AS read_rows_p90,
        formatReadableQuantity(max(read_rows)) AS read_rows_max,
        round(quantile(0.5)(query_duration_ms)) AS time_p50,
        round(quantile(0.9)(query_duration_ms)) AS time_p90,
        max(query_duration_ms) AS time_max,
        formatReadableSize(min(memory_usage)) AS mem_min,
        formatReadableSize(quantile(0.5)(memory_usage)) AS mem_p50,
        formatReadableSize(quantile(0.9)(memory_usage)) AS mem_p90,
        formatReadableSize(max(memory_usage)) AS mem_max,
        formatReadableSize(sum(memory_usage)) AS mem_sum,
        round(quantile(0.5)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p50,
        round(quantile(0.9)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p90,
        max(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_max,
        sum(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_sum,
        argMax(query_id, query_duration_ms) AS query_id_of_time_max,
        argMax(query, query_duration_ms) AS query_example
    FROM clusterAllReplicas(default, system.query_log)
    WHERE (type != 'QueryStart')
      AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
      AND (user NOT ILIKE '%internal%')
    GROUP BY normalized_query_hash, query_kind
    ORDER BY time_p90 DESC
    LIMIT {limit}
    """

    logger.info(f"Retrieving normalized query statistics for the past {days} days (limit: {limit})")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving normalized query stats: {e!s}")
        # Fallback to non-clustered query
        fallback_query = f"""
        SELECT
            normalized_query_hash,
            query_kind,
            count() AS q_count,
            countDistinct(query) AS q_distinct,
            formatReadableQuantity(round(quantile(0.5)(read_rows))) AS read_rows_p50,
            formatReadableQuantity(round(quantile(0.9)(read_rows))) AS read_rows_p90,
            formatReadableQuantity(max(read_rows)) AS read_rows_max,
            round(quantile(0.5)(query_duration_ms)) AS time_p50,
            round(quantile(0.9)(query_duration_ms)) AS time_p90,
            max(query_duration_ms) AS time_max,
            formatReadableSize(min(memory_usage)) AS mem_min,
            formatReadableSize(quantile(0.5)(memory_usage)) AS mem_p50,
            formatReadableSize(quantile(0.9)(memory_usage)) AS mem_p90,
            formatReadableSize(max(memory_usage)) AS mem_max,
            formatReadableSize(sum(memory_usage)) AS mem_sum,
            round(quantile(0.5)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p50,
            round(quantile(0.9)(ProfileEvents['OSCPUVirtualTimeMicroseconds'])) AS cpu_p90,
            max(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_max,
            sum(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS cpu_sum,
            argMax(query_id, query_duration_ms) AS query_id_of_time_max,
            argMax(query, query_duration_ms) AS query_example
        FROM system.query_log
        WHERE (type != 'QueryStart')
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
          AND (user NOT ILIKE '%internal%')
        GROUP BY normalized_query_hash, query_kind
        ORDER BY time_p90 DESC
        LIMIT {limit}
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_query_kind_breakdown(client: Client, days: int = 7) -> list[dict[str, str | int]]:
    """Get a breakdown of query types by hour.

    This function retrieves the count of different query types (SELECT, INSERT, etc.)
    grouped by hour over the specified time period.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with hourly query type breakdowns
    """
    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        countIf(query_kind = 'Insert') AS Insert,
        countIf(query_kind = 'AsyncInsertFlush') AS AsyncInsertFlush,
        countIf(query_kind = 'Select') AS Select,
        countIf(query_kind = 'KillQuery') AS KillQuery,
        countIf(query_kind = 'Create') AS Create,
        countIf(query_kind = 'Drop') AS Drop,
        countIf(query_kind = 'Show') AS Show,
        countIf(query_kind = 'Describe') AS Describe,
        countIf(query_kind = 'Explain') AS Explain,
        countIf(query_kind = 'Backup') AS Backup,
        countIf(query_kind = 'System') AS System,
        countIf(query_kind = 'Alter') AS Alter,
        countIf(query_kind = 'Delete') AS Delete,
        countIf(query_kind = 'Optimize') AS Optimize
    FROM clusterAllReplicas(default, system.query_log)
    WHERE ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
      AND (type = 'QueryStart')
      AND (user NOT ILIKE '%internal%')
    GROUP BY ts
    ORDER BY ts ASC
    """

    logger.info(f"Retrieving query kind breakdown for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving query kind breakdown: {e!s}")
        # Fallback to non-clustered query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            countIf(query_kind = 'Insert') AS Insert,
            countIf(query_kind = 'AsyncInsertFlush') AS AsyncInsertFlush,
            countIf(query_kind = 'Select') AS Select,
            countIf(query_kind = 'KillQuery') AS KillQuery,
            countIf(query_kind = 'Create') AS Create,
            countIf(query_kind = 'Drop') AS Drop,
            countIf(query_kind = 'Show') AS Show,
            countIf(query_kind = 'Describe') AS Describe,
            countIf(query_kind = 'Explain') AS Explain,
            countIf(query_kind = 'Backup') AS Backup,
            countIf(query_kind = 'System') AS System,
            countIf(query_kind = 'Alter') AS Alter,
            countIf(query_kind = 'Delete') AS Delete,
            countIf(query_kind = 'Optimize') AS Optimize
        FROM system.query_log
        WHERE ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
          AND (type = 'QueryStart')
          AND (user NOT ILIKE '%internal%')
        GROUP BY ts
        ORDER BY ts ASC
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)
