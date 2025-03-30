"""System components monitoring tools for ClickHouse.

This module provides tools for monitoring specific system components in a ClickHouse database,
such as materialized views, blob storage, and S3 queues.
"""

import logging
from typing import Dict, List, Optional, Union

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_mv_query_stats(
    client: Client,
    days: int = 7
) -> List[Dict[str, Union[str, int, float]]]:
    """Get statistics about materialized view queries.

    This function retrieves statistics about materialized view query executions,
    including performance metrics and exception counts.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with materialized view query statistics
    """
    query = f"""
    SELECT 
        toStartOfDay(event_time) AS ts, 
        min(event_time) AS min_event_time, 
        max(event_time) AS max_event_time, 
        view_name, 
        exception_code, 
        errorCodeToName(exception_code) AS exception_name, 
        count() AS c, 
        round(avg(view_duration_ms / 1000), 2) AS view_duration_ms_avg, 
        round(quantile(0.5)(view_duration_ms)) AS view_duration_ms_p50, 
        round(quantile(0.9)(view_duration_ms)) AS view_duration_ms_p90, 
        formatReadableSize(avg(peak_memory_usage)) AS peak_memory_avg, 
        formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_p50, 
        formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_p90, 
        formatReadableQuantity(avg(read_rows)) AS read_rows_avg, 
        formatReadableQuantity(quantile(0.5)(read_rows)) AS read_rows_p50, 
        formatReadableQuantity(quantile(0.9)(read_rows)) AS read_rows_p90, 
        formatReadableQuantity(avg(written_rows)) AS written_rows_avg, 
        formatReadableQuantity(quantile(0.5)(written_rows)) AS written_rows_p50, 
        formatReadableQuantity(quantile(0.9)(written_rows)) AS written_rows_p90 
    FROM clusterAllReplicas(default, merge(system, '^query_views_log*')) 
    WHERE event_date >= (today() - toIntervalDay({days})) 
    GROUP BY ALL 
    ORDER BY ts ASC
    """
    
    logger.info(f"Retrieving materialized view query statistics for the past {days} days")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving materialized view query statistics: {str(e)}")
        # Fallback to local query
        fallback_query = f"""
        SELECT 
            toStartOfDay(event_time) AS ts, 
            min(event_time) AS min_event_time, 
            max(event_time) AS max_event_time, 
            view_name, 
            exception_code, 
            errorCodeToName(exception_code) AS exception_name, 
            count() AS c, 
            round(avg(view_duration_ms / 1000), 2) AS view_duration_ms_avg, 
            round(quantile(0.5)(view_duration_ms)) AS view_duration_ms_p50, 
            round(quantile(0.9)(view_duration_ms)) AS view_duration_ms_p90, 
            formatReadableSize(avg(peak_memory_usage)) AS peak_memory_avg, 
            formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_p50, 
            formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_p90, 
            formatReadableQuantity(avg(read_rows)) AS read_rows_avg, 
            formatReadableQuantity(quantile(0.5)(read_rows)) AS read_rows_p50, 
            formatReadableQuantity(quantile(0.9)(read_rows)) AS read_rows_p90, 
            formatReadableQuantity(avg(written_rows)) AS written_rows_avg, 
            formatReadableQuantity(quantile(0.5)(written_rows)) AS written_rows_p50, 
            formatReadableQuantity(quantile(0.9)(written_rows)) AS written_rows_p90 
        FROM system.query_views_log
        WHERE event_date >= (today() - toIntervalDay({days})) 
        GROUP BY ts, view_name, exception_code, exception_name
        ORDER BY ts ASC
        """
        logger.info("Falling back to local query_views_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_mv_deduplicated_blocks(
    client: Client,
    view_name: str,
    days: int = 7
) -> List[Dict[str, Union[str, int]]]:
    """Get statistics about deduplicated blocks for a specific materialized view.

    This function retrieves information about deduplicated inserted blocks
    for a specific materialized view.

    Args:
        client: The ClickHouse client instance
        view_name: The name of the materialized view (format: database.view_name)
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with deduplicated blocks statistics
    """
    if '.' not in view_name:
        raise ValueError("view_name must be in format 'database.view_name'")
    
    query = f"""
    WITH query_ids AS (
        SELECT 
            initial_query_id, 
            written_rows 
        FROM clusterAllReplicas(default, system.query_views_log) 
        WHERE (view_name = '{view_name}') 
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now())) 
        ORDER BY event_time DESC 
        LIMIT 1000
    )
    SELECT 
        event_time, 
        query_id, 
        ProfileEvents['DuplicatedInsertedBlocks'] AS duplicated_blocks, 
        query_ids.written_rows 
    FROM clusterAllReplicas(default, system.query_log) AS query_log 
    INNER JOIN query_ids ON query_ids.initial_query_id = query_log.query_id 
    WHERE (type = 'QueryFinish') 
      AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
    """
    
    logger.info(f"Retrieving deduplicated blocks for materialized view '{view_name}' for the past {days} days")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving deduplicated blocks: {str(e)}")
        # Fallback to local query but without the join
        fallback_query = f"""
        SELECT 
            event_time, 
            query_id, 
            ProfileEvents['DuplicatedInsertedBlocks'] AS duplicated_blocks
        FROM system.query_log
        WHERE (view_name LIKE '%{view_name.split(".")[-1]}%')
          AND (type = 'QueryFinish') 
          AND ((event_time >= (now() - toIntervalDay({days}))) AND (event_time <= now()))
        ORDER BY event_time DESC
        LIMIT 1000
        """
        logger.info("Falling back to simplified local query_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_blob_storage_stats(
    client: Client
) -> List[Dict[str, Union[str, int]]]:
    """Get statistics about blob storage operations.

    This function retrieves counts of blob storage operations by event type,
    linked to their respective tables.

    Args:
        client: The ClickHouse client instance

    Returns:
        List of dictionaries with blob storage statistics
    """
    query = """
    SELECT 
        l.event_type, 
        l.c, 
        l.table_uuid, 
        t.name 
    FROM (
        SELECT 
            event_type, 
            count() AS c, 
            substring(any(local_path), 11, 36) AS table_uuid 
        FROM clusterAllReplicas(default, system.blob_storage_log) 
        WHERE event_date = today() 
        GROUP BY event_type
    ) AS l 
    INNER JOIN system.tables AS t ON l.table_uuid = CAST(t.uuid, 'String')
    """
    
    logger.info("Retrieving blob storage statistics")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving blob storage statistics: {str(e)}")
        # Fallback to local query without join
        fallback_query = """
        SELECT 
            event_type, 
            count() AS c, 
            substring(any(local_path), 11, 36) AS table_uuid 
        FROM system.blob_storage_log 
        WHERE event_date = today() 
        GROUP BY event_type
        """
        logger.info("Falling back to simplified local blob_storage_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_s3queue_stats(
    client: Client,
    database: str = "default",
    table: str = None,
    limit: int = 400
) -> List[Dict[str, Union[str, List[str], int]]]:
    """Get statistics about S3 queue processing.

    This function retrieves information about S3 queue processing operations,
    including file names, host counts, and timestamps.

    Args:
        client: The ClickHouse client instance
        database: Filter by database name (default: "default")
        table: Filter by table name (optional)
        limit: Maximum number of records to return (default: 400)

    Returns:
        List of dictionaries with S3 queue statistics
    """
    where_clause = f"(database = '{database}')"
    if table:
        where_clause += f" AND (`table` = '{table}')"
    
    query = f"""
    SELECT 
        file_name, 
        arraySort(groupUniqArray(splitByChar('.', hostname)[1] AS hostname)) AS hostnames, 
        length(hostnames) AS num_hostnames, 
        min(event_time) AS min_event_time, 
        max(event_time) AS max_event_time, 
        count() AS num_processed_by_all_replicas 
    FROM clusterAllReplicas(default, system.s3queue_log) 
    WHERE {where_clause} AND (exception = '') 
    GROUP BY file_name 
    ORDER BY num_processed_by_all_replicas ASC 
    LIMIT {limit}
    """
    
    table_filter = f" and table '{table}'" if table else ""
    logger.info(f"Retrieving S3 queue statistics for database '{database}'{table_filter} (limit: {limit})")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving S3 queue statistics: {str(e)}")
        # Fallback to local query
        fallback_query = f"""
        SELECT 
            file_name, 
            [hostname()] AS hostnames, 
            1 AS num_hostnames, 
            min(event_time) AS min_event_time, 
            max(event_time) AS max_event_time, 
            count() AS num_processed_by_all_replicas 
        FROM system.s3queue_log 
        WHERE {where_clause} AND (exception = '') 
        GROUP BY file_name 
        ORDER BY num_processed_by_all_replicas ASC 
        LIMIT {limit}
        """
        logger.info("Falling back to local s3queue_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_s3queue_with_names(
    client: Client
) -> List[Dict[str, Union[str, int]]]:
    """Get S3 queue entries with database and table names.

    This function retrieves S3 queue entries enriched with database and table names
    instead of just UUIDs.

    Args:
        client: The ClickHouse client instance

    Returns:
        List of dictionaries with S3 queue entries including database and table names
    """
    query = """
    WITH system_s3queue AS (
        SELECT 
            hostname(), 
            toUUID(splitByChar('/', zookeeper_path)[4]) AS database_uuid, 
            toUUID(splitByChar('/', zookeeper_path)[5]) AS table_uuid, 
            * 
        FROM clusterAllReplicas(default, system.s3queue)
    )
    SELECT 
        system_s3queue.*, 
        databases.name AS database_name, 
        tables.name AS table_name 
    FROM system_s3queue 
    INNER JOIN system.databases ON system_s3queue.database_uuid = system.databases.uuid 
    INNER JOIN system.tables ON system_s3queue.table_uuid = system.tables.uuid
    """
    
    logger.info("Retrieving S3 queue entries with database and table names")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving S3 queue entries with names: {str(e)}")
        # Fallback to local query without joins
        fallback_query = """
        SELECT 
            hostname(), 
            toUUID(splitByChar('/', zookeeper_path)[4]) AS database_uuid, 
            toUUID(splitByChar('/', zookeeper_path)[5]) AS table_uuid, 
            * 
        FROM system.s3queue
        """
        logger.info("Falling back to simplified local s3queue query")
        return execute_query_with_retry(client, fallback_query)