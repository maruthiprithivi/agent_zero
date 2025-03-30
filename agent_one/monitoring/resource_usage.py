"""Resource usage monitoring tools for ClickHouse.

This module provides tools for monitoring resource usage, including memory, CPU, and server sizing.
"""

import logging
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_one.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_memory_usage(
    client: Client, days: int = 7, settings: dict[str, Any] | None = None
) -> list[dict[str, str | int | float]]:
    """Get memory usage statistics over time by host.

    This function retrieves memory usage metrics (avg, p50, p90, p99, max) over time
    for each host in the cluster.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)
        settings: Optional query settings

    Returns:
        List of dictionaries with memory usage statistics
    """
    base_query = f"""
    SELECT
        event_time AS ts,
        splitByChar('.', hostname)[1] AS hostname_,
        formatReadableSize(avg(CurrentMetric_MemoryTracking)) AS MemoryTracking_avg,
        formatReadableSize(quantile(0.5)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p50,
        formatReadableSize(quantile(0.9)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p90,
        formatReadableSize(quantile(0.99)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p99,
        formatReadableSize(max(CurrentMetric_MemoryTracking)) AS MemoryTracking_max
    FROM {{table}}
    WHERE (event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now())
    GROUP BY ALL
    ORDER BY ts ASC, hostname_ ASC
    """

    logger.info(f"Retrieving memory usage statistics for the past {days} days")

    try:
        return execute_query_with_retry(
            client,
            base_query.format(table="clusterAllReplicas(default, system.metric_log)"),
            settings=settings,
        )
    except ClickHouseError as e:
        logger.error(f"Error retrieving memory usage statistics: {e!s}")
        # Fallback to local query
        logger.info("Falling back to local metric_log query")
        return execute_query_with_retry(
            client, base_query.format(table="system.metric_log"), settings=settings
        )


@log_execution_time
def get_cpu_usage(
    client: Client, hours: int = 3, settings: dict[str, Any] | None = None
) -> list[dict[str, str | float]]:
    """Get CPU usage statistics over time.

    This function retrieves CPU usage metrics, including total cluster CPU usage,
    available CPU cores, and normalized usage (usage/available).

    Args:
        client: The ClickHouse client instance
        hours: Number of hours to look back in history (default: 3)
        settings: Optional query settings

    Returns:
        List of dictionaries with CPU usage statistics
    """
    interval = f"(now() - {hours * 3600})"
    query = f"""
    WITH cpu_cores_in_cluster_table AS (
        SELECT t, sum(value) AS cpu_cores_in_cluster
        FROM (
            SELECT hostname, t, argMax(value, t) AS value
            FROM (
                SELECT
                    hostname() AS hostname,
                    CAST(toStartOfInterval(event_time, toIntervalSecond(60)), 'INT') AS t,
                    value
                FROM clusterAllReplicas(default, system.asynchronous_metric_log)
                WHERE (metric = 'CGroupMaxCPU')
                  AND (event_date >= toDate({interval}))
                  AND (event_time >= {interval})
            )
            GROUP BY t, hostname
        )
        GROUP BY t
        ORDER BY t ASC
    ),
    cpu_cores_usage_in_cluster_table AS (
        SELECT
            CAST(toStartOfInterval(event_time, toIntervalSecond(60)), 'INT') AS t,
            avg(metric) / 1000000 AS cpu_cores_usage_in_cluster
        FROM (
            SELECT
                event_time,
                sum(ProfileEvent_OSCPUVirtualTimeMicroseconds) AS metric
            FROM clusterAllReplicas(default, system.metric_log)
            WHERE (event_date >= toDate({interval}))
              AND (event_time >= {interval})
            GROUP BY event_time
        )
        GROUP BY t
        ORDER BY t ASC
        WITH FILL STEP 60
    )
    SELECT
        toDateTime(t) AS dt,
        round(cpu_cores_usage_in_cluster_table.cpu_cores_usage_in_cluster, 2) AS cpu_usage_cluster,
        cpu_cores_in_cluster_table.cpu_cores_in_cluster AS cpu_cores_cluster,
        round(cpu_usage_cluster / cpu_cores_cluster, 2) AS cpu_usage_cluster_normalized
    FROM cpu_cores_usage_in_cluster_table
    INNER JOIN cpu_cores_in_cluster_table ON cpu_cores_usage_in_cluster_table.t = cpu_cores_in_cluster_table.t
    ORDER BY t ASC
    """

    logger.info(f"Retrieving CPU usage statistics for the past {hours} hours")

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error retrieving CPU usage statistics: {e!s}")
        # This query is too complex for a simple fallback, return an empty result
        logger.warning("No fallback available for CPU usage query")
        return []


@log_execution_time
def get_server_sizing(client: Client) -> list[dict[str, str | int]]:
    """Get server sizing information for all nodes in the cluster.

    This function retrieves the current configuration for each server in the cluster,
    including CPU cores and allocated memory.

    Args:
        client: The ClickHouse client instance

    Returns:
        List of dictionaries with server sizing information
    """
    query = """
    SELECT *
    FROM clusterAllReplicas('default', view(
        SELECT
            hostname() AS server,
            getSetting('max_threads') AS cpu_cores,
            formatReadableSize(getSetting('max_memory_usage')) AS memory
        FROM system.one
    ))
    ORDER BY server ASC
    SETTINGS skip_unavailable_shards = 1
    """

    logger.info("Retrieving server sizing information")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving server sizing information: {e!s}")
        # Fallback to local query
        fallback_query = """
        SELECT
            hostname() AS server,
            getSetting('max_threads') AS cpu_cores,
            formatReadableSize(getSetting('max_memory_usage')) AS memory
        FROM system.one
        """
        logger.info("Falling back to local server sizing query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_uptime(client: Client, days: int = 7) -> list[dict[str, str | float]]:
    """Get server uptime statistics.

    This function retrieves uptime metrics for all servers in the cluster over time.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with uptime statistics
    """
    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        min(event_time) AS min_event_time,
        max(event_time) AS max_event_time,
        round(avg(value)) AS uptime
    FROM clusterAllReplicas(default, merge(system, '^asynchronous_metric_log*'))
    WHERE (metric = 'Uptime') AND (event_date >= (today() - toIntervalDay({days})))
    GROUP BY ALL
    ORDER BY ts ASC
    """

    logger.info(f"Retrieving uptime statistics for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving uptime statistics: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            round(avg(value)) AS uptime
        FROM system.asynchronous_metric_log
        WHERE (metric = 'Uptime') AND (event_date >= (today() - toIntervalDay({days})))
        GROUP BY ALL
        ORDER BY ts ASC
        """
        logger.info("Falling back to local asynchronous_metric_log query")
        return execute_query_with_retry(client, fallback_query)
