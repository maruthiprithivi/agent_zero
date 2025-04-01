"""Parts and merges monitoring tools for ClickHouse.

This module provides tools for monitoring table parts, merges, and related operations
in a ClickHouse database.
"""

import logging

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_parts_analysis(
    client: Client, database: str = "system", table: str = "query_log"
) -> list[dict[str, str | int]]:
    """Get analysis of parts for a specific table.

    This function analyzes the parts of a table, grouped by level, providing statistics
    about rows, sizes, part types, etc.

    Args:
        client: The ClickHouse client instance
        database: The database name (default: "system")
        table: The table name (default: "query_log")

    Returns:
        List of dictionaries with part analysis information
    """
    query = f"""
    SELECT
        level,
        formatReadableQuantity(min(rows)) AS rows_min,
        formatReadableSize(min(data_compressed_bytes)) AS bytes_min,
        formatReadableQuantity(quantile(0.5)(rows)) AS rows_median,
        formatReadableSize(quantile(0.5)(data_compressed_bytes)) AS bytes_median,
        formatReadableQuantity(max(rows)) AS rows_max,
        formatReadableSize(max(data_compressed_bytes)) AS bytes_max,
        part_type,
        part_storage_type,
        count() AS c
    FROM system.parts
    WHERE active AND (`table` = '{table}') AND (database = '{database}')
    GROUP BY ALL WITH TOTALS
    ORDER BY level ASC, c ASC
    """

    logger.info(f"Analyzing parts for table {database}.{table}")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error analyzing parts: {e!s}")
        return []


@log_execution_time
def get_current_merges(client: Client) -> list[dict[str, str | int | float]]:
    """Get information about currently running merges.

    This function retrieves details about all currently running merge operations
    across the cluster.

    Args:
        client: The ClickHouse client instance

    Returns:
        List of dictionaries with information about each merge operation
    """
    query = """
    SELECT
        hostname(),
        database,
        `table`,
        round(elapsed) AS elapsed_sec,
        round(progress, 2) AS progress_,
        result_part_name,
        length(source_part_names) AS source_part_count,
        formatReadableQuantity(rows_read) AS rows_read_,
        formatReadableQuantity(rows_written) AS rows_written_,
        merge_type,
        merge_algorithm
    FROM clusterAllReplicas(default, system.merges)
    ORDER BY elapsed ASC
    SETTINGS skip_unavailable_shards = 1
    """

    logger.info("Retrieving information about currently running merges")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving current merges: {e!s}")
        # Fallback to local query
        fallback_query = """
        SELECT
            hostname(),
            database,
            `table`,
            round(elapsed) AS elapsed_sec,
            round(progress, 2) AS progress_,
            result_part_name,
            length(source_part_names) AS source_part_count,
            formatReadableQuantity(rows_read) AS rows_read_,
            formatReadableQuantity(rows_written) AS rows_written_,
            merge_type,
            merge_algorithm
        FROM system.merges
        ORDER BY elapsed ASC
        """
        logger.info("Falling back to local merges query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_merge_stats(client: Client, days: int = 7) -> list[dict[str, str | int | float]]:
    """Get merge operation statistics over time.

    This function retrieves statistics about merge operations, grouped by hour,
    including counts by type, durations, rows processed, etc.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with merge statistics
    """
    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        count() AS total_merges,
        countIf(part_type = 'Wide') AS num_Wide,
        countIf(part_type = 'Compact') AS num_Compact,
        countIf(merge_algorithm = 'Horizontal') AS num_Horizontal,
        countIf(merge_algorithm = 'Vertical') AS num_Vertical,
        countIf(error > 0) AS num_Error,
        if(error > 0, any(error), 0) AS errorCode,
        round(quantile(0.5)(duration_ms)) AS duration_ms_p50,
        round(quantile(0.9)(duration_ms)) AS duration_ms_p90,
        max(duration_ms) AS duration_ms_max,
        formatReadableQuantity(quantile(0.5)(rows)) AS rows_p50,
        formatReadableQuantity(quantile(0.9)(rows)) AS rows_p90,
        formatReadableQuantity(max(rows)) AS rows_max,
        formatReadableSize(quantile(0.5)(size_in_bytes)) AS size_in_bytes_p50,
        formatReadableSize(quantile(0.9)(size_in_bytes)) AS size_in_bytes_p90,
        formatReadableSize(max(size_in_bytes)) AS size_in_bytes_max,
        formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_usage_p50,
        formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_usage_p90,
        formatReadableSize(max(peak_memory_usage)) AS peak_memory_usage_max,
        formatReadableSize(sum(peak_memory_usage)) AS peak_memory_usage_sum
    FROM clusterAllReplicas(default, system.part_log)
    WHERE (event_type = 'MergeParts') AND (event_time >= (now() - toIntervalDay({days})))
    GROUP BY ALL
    ORDER BY ts ASC
    """

    logger.info(f"Retrieving merge statistics for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving merge statistics: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            count() AS total_merges,
            countIf(part_type = 'Wide') AS num_Wide,
            countIf(part_type = 'Compact') AS num_Compact,
            countIf(merge_algorithm = 'Horizontal') AS num_Horizontal,
            countIf(merge_algorithm = 'Vertical') AS num_Vertical,
            countIf(error > 0) AS num_Error,
            if(error > 0, any(error), 0) AS errorCode,
            round(quantile(0.5)(duration_ms)) AS duration_ms_p50,
            round(quantile(0.9)(duration_ms)) AS duration_ms_p90,
            max(duration_ms) AS duration_ms_max,
            formatReadableQuantity(quantile(0.5)(rows)) AS rows_p50,
            formatReadableQuantity(quantile(0.9)(rows)) AS rows_p90,
            formatReadableQuantity(max(rows)) AS rows_max,
            formatReadableSize(quantile(0.5)(size_in_bytes)) AS size_in_bytes_p50,
            formatReadableSize(quantile(0.9)(size_in_bytes)) AS size_in_bytes_p90,
            formatReadableSize(max(size_in_bytes)) AS size_in_bytes_max,
            formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_usage_p50,
            formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_usage_p90,
            formatReadableSize(max(peak_memory_usage)) AS peak_memory_usage_max,
            formatReadableSize(sum(peak_memory_usage)) AS peak_memory_usage_sum
        FROM system.part_log
        WHERE (event_type = 'MergeParts') AND (event_time >= (now() - toIntervalDay({days})))
        GROUP BY ts
        ORDER BY ts ASC
        """
        logger.info("Falling back to local part_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_part_log_events(client: Client, days: int = 7) -> list[dict[str, str | int | float]]:
    """Get part log events by type over time.

    This function retrieves counts of different part log event types (e.g., NewPart,
    MergeParts, RemovePart) over time.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back in history (default: 7)

    Returns:
        List of dictionaries with part log event statistics
    """
    query = f"""
    SELECT
        toStartOfHour(event_time) AS ts,
        countIf(event_type = 'NewPart') AS NewPart,
        countIf(event_type = 'MergeParts') AS MergeParts,
        countIf(event_type = 'DownloadPart') AS DownloadPart,
        countIf(event_type = 'RemovePart') AS RemovePart,
        countIf(event_type = 'MutatePart') AS MutatePart,
        countIf(event_type = 'MergePartsStart') AS MergePartsStart,
        countIf(event_type = 'MutatePartStart') AS MutatePartStart,
        countIf(1, error > 0) AS num_Error,
        if(error > 0, any(error), 0) AS errorCode,
        round(quantile(0.5)(duration_ms)) AS duration_ms_p50,
        round(quantile(0.9)(duration_ms)) AS duration_ms_p90,
        max(duration_ms) AS duration_ms_max,
        formatReadableQuantity(quantile(0.5)(rows)) AS rows_p50,
        formatReadableQuantity(quantile(0.9)(rows)) AS rows_p90,
        formatReadableQuantity(max(rows)) AS rows_max,
        formatReadableSize(quantile(0.5)(size_in_bytes)) AS size_in_bytes_p50,
        formatReadableSize(quantile(0.9)(size_in_bytes)) AS size_in_bytes_p90,
        formatReadableSize(max(size_in_bytes)) AS size_in_bytes_max,
        formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_usage_p50,
        formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_usage_p90,
        formatReadableSize(max(peak_memory_usage)) AS peak_memory_usage_max,
        formatReadableSize(sum(peak_memory_usage)) AS peak_memory_usage_sum
    FROM clusterAllReplicas(default, system.part_log)
    WHERE event_time >= (now() - toIntervalDay({days}))
    GROUP BY ALL
    ORDER BY ts ASC
    """

    logger.info(f"Retrieving part log events for the past {days} days")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving part log events: {e!s}")
        # Fallback to local query
        fallback_query = f"""
        SELECT
            toStartOfHour(event_time) AS ts,
            countIf(event_type = 'NewPart') AS NewPart,
            countIf(event_type = 'MergeParts') AS MergeParts,
            countIf(event_type = 'DownloadPart') AS DownloadPart,
            countIf(event_type = 'RemovePart') AS RemovePart,
            countIf(event_type = 'MutatePart') AS MutatePart,
            countIf(event_type = 'MergePartsStart') AS MergePartsStart,
            countIf(event_type = 'MutatePartStart') AS MutatePartStart,
            countIf(1, error > 0) AS num_Error,
            if(error > 0, any(error), 0) AS errorCode,
            round(quantile(0.5)(duration_ms)) AS duration_ms_p50,
            round(quantile(0.9)(duration_ms)) AS duration_ms_p90,
            max(duration_ms) AS duration_ms_max,
            formatReadableQuantity(quantile(0.5)(rows)) AS rows_p50,
            formatReadableQuantity(quantile(0.9)(rows)) AS rows_p90,
            formatReadableQuantity(max(rows)) AS rows_max,
            formatReadableSize(quantile(0.5)(size_in_bytes)) AS size_in_bytes_p50,
            formatReadableSize(quantile(0.9)(size_in_bytes)) AS size_in_bytes_p90,
            formatReadableSize(max(size_in_bytes)) AS size_in_bytes_max,
            formatReadableSize(quantile(0.5)(peak_memory_usage)) AS peak_memory_usage_p50,
            formatReadableSize(quantile(0.9)(peak_memory_usage)) AS peak_memory_usage_p90,
            formatReadableSize(max(peak_memory_usage)) AS peak_memory_usage_max,
            formatReadableSize(sum(peak_memory_usage)) AS peak_memory_usage_sum
        FROM system.part_log
        WHERE event_time >= (now() - toIntervalDay({days}))
        GROUP BY ts
        ORDER BY ts ASC
        """
        logger.info("Falling back to local part_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_partition_stats(
    client: Client, database: str = "system", table: str = "part_log"
) -> list[dict[str, str | int]]:
    """Get statistics by partition for a specific table.

    This function retrieves statistics about partitions for a specific table,
    including part count, total rows, and total size.

    Args:
        client: The ClickHouse client instance
        database: The database name (default: "system")
        table: The table name (default: "part_log")

    Returns:
        List of dictionaries with partition statistics
    """
    query = f"""
    SELECT
        partition,
        count() AS part_count,
        formatReadableQuantity(sum(rows)) AS rows_total,
        formatReadableSize(sum(bytes_on_disk)) AS size_total
    FROM system.parts
    WHERE (`table` = '{table}') AND (database = '{database}') AND active
    GROUP BY partition
    ORDER BY partition ASC
    """

    logger.info(f"Retrieving partition statistics for table {database}.{table}")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving partition statistics: {e!s}")
        return []
