"""Insert operations monitoring tools for ClickHouse.

This module provides tools for monitoring insert operations in a ClickHouse cluster.
"""

import logging

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_one.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_async_insert_stats(client: Client, days: int = 5) -> list[dict[str, str | int | float]]:
    """Get statistics on asynchronous insert batches.

    This function retrieves statistics about asynchronous insert batches,
    including batch sizes and average rows per batch.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 5)

    Returns:
        List of dictionaries with asynchronous insert statistics
    """
    query = f"""
    SELECT
        c,
        count(),
        avg(d) AS avg_rows
    FROM (
        SELECT
            flush_query_id,
            count() AS c,
            min(rows),
            max(rows),
            sum(rows) AS d
        FROM clusterAllReplicas(default, system.asynchronous_insert_log)
        WHERE event_time >= (now() - toIntervalDay({days}))
        GROUP BY flush_query_id
    )
    GROUP BY c
    """

    logger.info(f"Retrieving asynchronous insert statistics for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving asynchronous insert statistics: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            c,
            count(),
            avg(d) AS avg_rows
        FROM (
            SELECT
                flush_query_id,
                count() AS c,
                min(rows),
                max(rows),
                sum(rows) AS d
            FROM system.asynchronous_insert_log
            WHERE event_time >= (now() - toIntervalDay({days}))
            GROUP BY flush_query_id
        )
        GROUP BY c
        """
        logger.info("Falling back to local asynchronous_insert_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_insert_written_bytes_distribution(
    client: Client, days: int = 7
) -> list[dict[str, str | int | float]]:
    """Get distribution of written bytes for insert operations.

    This function retrieves statistics about the distribution of bytes written by
    insert operations, bucketed by power-of-2 sizes.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with insert bytes distribution
    """
    query = f"""
    SELECT
        bucket,
        bucket_readable,
        c,
        sum(c) OVER (ORDER BY bucket ASC) AS running_c,
        round(running_c / sum(c) OVER (), 4) AS running_pct_of_tot
    FROM (
        SELECT
            exp2(round(log2(written_bytes))) AS bucket,
            formatReadableSize(bucket) AS bucket_readable,
            uniq(query_id) AS c
        FROM clusterAllReplicas(default, merge('system', '^query_log'))
        WHERE (type = 2)
          AND (query_kind = 'Insert')
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
        GROUP BY ALL
    )
    ORDER BY bucket ASC
    """

    logger.info(f"Retrieving insert written bytes distribution for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving insert written bytes distribution: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            bucket,
            bucket_readable,
            c,
            sum(c) OVER (ORDER BY bucket ASC) AS running_c,
            round(running_c / sum(c) OVER (), 4) AS running_pct_of_tot
        FROM (
            SELECT
                exp2(round(log2(written_bytes))) AS bucket,
                formatReadableSize(bucket) AS bucket_readable,
                uniq(query_id) AS c
            FROM system.query_log
            WHERE (type = 2)
              AND (query_kind = 'Insert')
              AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
            GROUP BY bucket, bucket_readable
        )
        ORDER BY bucket ASC
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_async_vs_sync_insert_counts(client: Client, days: int = 7) -> list[dict[str, str | int]]:
    """Get hourly counts of async vs. sync inserts.

    This function retrieves hourly counts of asynchronous vs. synchronous
    insert operations.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with hourly insert counts
    """
    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        count() AS num_insert,
        countIf(if(has(Settings, 'async_insert'), CAST(Settings['async_insert'], 'Int'), 0) = 1) AS num_async_insert,
        num_insert - num_async_insert AS num_sync_insert
    FROM clusterAllReplicas(default, merge(system, '^query_log*'))
    WHERE (type = 'QueryFinish')
      AND (query_kind = 'Insert')
      AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
      AND (user NOT ILIKE '%internal%')
      AND (match(query, 'insert.*select.*') = 0)
    GROUP BY ts
    ORDER BY ts ASC
    SETTINGS skip_unavailable_shards = 1
    """

    logger.info(f"Retrieving async vs. sync insert counts for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving async vs. sync insert counts: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            count() AS num_insert,
            countIf(if(has(Settings, 'async_insert'), CAST(Settings['async_insert'], 'Int'), 0) = 1) AS num_async_insert,
            num_insert - num_async_insert AS num_sync_insert
        FROM system.query_log
        WHERE (type = 'QueryFinish')
          AND (query_kind = 'Insert')
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
          AND (user NOT ILIKE '%internal%')
          AND (match(query, 'insert.*select.*') = 0)
        GROUP BY ts
        ORDER BY ts ASC
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_recent_insert_queries(
    client: Client, days: int = 1, limit: int = 10, min_count: int = 1000
) -> list[dict[str, str | int]]:
    """Get most frequent recent insert queries.

    This function retrieves the most frequently executed insert queries,
    optionally filtered by a minimum count threshold.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 1)
        limit: Maximum number of queries to return (default: 10)
        min_count: Minimum count threshold for inclusion (default: 1000)

    Returns:
        List of dictionaries with frequent insert queries
    """
    query = f"""
    SELECT
        min(event_time) AS min_event_time,
        max(event_time) AS max_event_time,
        normalized_query_hash,
        substring(replaceAll(query, '\n', ''), 1, 50) AS q,
        Settings['async_insert'] AS is_async_insert,
        count() AS c
    FROM clusterAllReplicas(default, system.query_log)
    WHERE ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
      AND (type = 'QueryStart')
      AND (user NOT ILIKE '%internal%')
      AND (query_kind = 'Insert')
    GROUP BY ALL
    HAVING c > {min_count}
    ORDER BY c DESC
    LIMIT {limit}
    """

    logger.info(
        f"Retrieving recent insert queries for the past {days} days (limit: {limit}, min_count:"
        f" {min_count})"
    )

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving recent insert queries: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            normalized_query_hash,
            substring(replaceAll(query, '\n', ''), 1, 50) AS q,
            Settings['async_insert'] AS is_async_insert,
            count() AS c
        FROM system.query_log
        WHERE ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
          AND (type = 'QueryStart')
          AND (user NOT ILIKE '%internal%')
          AND (query_kind = 'Insert')
        GROUP BY normalized_query_hash, q, Settings['async_insert']
        HAVING c > {min_count}
        ORDER BY c DESC
        LIMIT {limit}
        """
        logger.info("Falling back to local query_log query")
        return execute_query_with_retry(client, fallback_query)
