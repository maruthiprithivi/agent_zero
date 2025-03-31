"""Table statistics monitoring tools for ClickHouse.

This module provides tools for monitoring table statistics in a ClickHouse cluster.
"""

import logging

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_one.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


@log_execution_time
def get_table_stats(
    client: Client, detailed: bool = False, database_like: str = "%", exclude_system: bool = True
) -> list[dict[str, str | int | list[str]]]:
    """Get statistics for tables in the database.

    This function retrieves statistics about tables, including row counts,
    sizes, column counts, and more.

    Args:
        client: The ClickHouse client instance
        detailed: Whether to include detailed information like create statements and column types (default: False)
        database_like: LIKE pattern to filter databases (default: "%")
        exclude_system: Whether to exclude system databases (default: True)

    Returns:
        List of dictionaries with table statistics
    """
    exclude_clause = ""
    if exclude_system:
        exclude_clause = (
            "WHERE t.database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')"
        )

    if detailed:
        query = f"""
        WITH columns AS (
            SELECT
                database,
                `table`,
                arrayDistinct(groupArray(type)) AS unique_col_type,
                count() AS c
            FROM system.columns
            WHERE database LIKE '{database_like}'
            {" AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
            GROUP BY ALL
        ),
        partitions AS (
            SELECT
                database,
                `table`,
                arraySort(groupArrayDistinct(partition)) AS partition_name,
                length(partition_name) AS partition_count
            FROM system.parts
            WHERE active AND database LIKE '{database_like}'
            {" AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
            GROUP BY ALL
        )
        SELECT
            t.database AS database,
            t.name AS name,
            t.create_table_query AS create_table_query,
            t.engine AS engine,
            t.metadata_modification_time AS modification_time,
            t.partition_key AS partition_key,
            t.sorting_key AS sorting_key,
            t.primary_key AS primary_key,
            columns.unique_col_type AS unique_col_type,
            formatReadableQuantity(t.total_rows) AS total_rows_,
            formatReadableSize(t.total_bytes) AS total_bytes_,
            partitions.partition_count AS partitions,
            t.active_parts AS active_parts,
            t.total_marks AS marks,
            columns.c AS columns
        FROM system.tables AS t
        LEFT JOIN columns ON (t.database = columns.database) AND (t.name = columns.`table`)
        LEFT JOIN partitions ON (t.database = partitions.database) AND (t.name = partitions.`table`)
        WHERE t.database LIKE '{database_like}'
        {" AND t.database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
        """
    else:
        query = f"""
        WITH columns AS (
            SELECT
                database,
                `table`,
                arrayDistinct(groupArray(type)) AS unique_col_type,
                count() AS c
            FROM system.columns
            WHERE database LIKE '{database_like}'
            {" AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
            GROUP BY ALL
        ),
        partitions AS (
            SELECT
                database,
                `table`,
                arraySort(groupArrayDistinct(partition)) AS partition_name,
                length(partition_name) AS partition_count
            FROM system.parts
            WHERE active AND database LIKE '{database_like}'
            {" AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
            GROUP BY ALL
        )
        SELECT
            t.database AS database,
            t.name AS name,
            t.engine AS engine,
            t.metadata_modification_time AS modification_time,
            t.partition_key AS partition_key,
            t.sorting_key AS sorting_key,
            t.primary_key AS primary_key,
            formatReadableQuantity(t.total_rows) AS total_rows_,
            formatReadableSize(t.total_bytes) AS total_bytes_,
            partitions.partition_count AS partitions,
            t.active_parts AS active_parts,
            t.total_marks AS marks,
            columns.c AS columns
        FROM system.tables AS t
        LEFT JOIN columns ON (t.database = columns.database) AND (t.name = columns.`table`)
        LEFT JOIN partitions ON (t.database = partitions.database) AND (t.name = partitions.`table`)
        WHERE t.database LIKE '{database_like}'
        {" AND t.database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
        """

    db_filter = f"matching '{database_like}'" if database_like != "%" else "all"
    detailed_str = "detailed " if detailed else ""
    logger.info(f"Retrieving {detailed_str}table statistics for {db_filter} databases")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving table statistics: {e!s}")
        # Simplified fallback query
        fallback_query = f"""
        SELECT
            database,
            name,
            engine,
            metadata_modification_time AS modification_time,
            partition_key,
            sorting_key,
            primary_key,
            formatReadableQuantity(total_rows) AS total_rows_,
            formatReadableSize(total_bytes) AS total_bytes_,
            active_parts
        FROM system.tables
        WHERE database LIKE '{database_like}'
        {" AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')" if exclude_system else ""}
        """
        logger.info("Falling back to simplified table statistics query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def get_table_inactive_parts(
    client: Client, database_like: str = "%", table_like: str = "%", limit: int = 10
) -> list[dict[str, str | int]]:
    """Get statistics about inactive parts by table.

    This function retrieves information about inactive parts,
    such as those marked for deletion or being merged.

    Args:
        client: The ClickHouse client instance
        database_like: LIKE pattern to filter databases (default: "%")
        table_like: LIKE pattern to filter tables (default: "%")
        limit: Maximum number of tables to return (default: 10)

    Returns:
        List of dictionaries with inactive parts statistics
    """
    query = f"""
    SELECT
        database,
        `table`,
        formatReadableSize(sum(data_compressed_bytes) AS size) AS compressed,
        sum(rows) AS rows,
        count() AS part_count
    FROM system.parts
    WHERE (active = 0)
      AND (database LIKE '{database_like}')
      AND (`table` LIKE '{table_like}')
    GROUP BY database, `table`
    ORDER BY size DESC
    LIMIT {limit}
    """

    db_filter = f"matching '{database_like}'" if database_like != "%" else "all"
    table_filter = f"matching '{table_like}'" if table_like != "%" else "all"
    logger.info(
        f"Retrieving inactive parts for {db_filter} databases and {table_filter} tables (limit:"
        f" {limit})"
    )

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving inactive parts: {e!s}")
        return []


@log_execution_time
def get_recent_table_modifications(
    client: Client, days: int = 7, exclude_system: bool = True, limit: int = 50
) -> list[dict[str, str | int]]:
    """Get recently modified tables.

    This function identifies tables that have been modified recently,
    sorted by modification time.

    Args:
        client: The ClickHouse client instance
        days: Number of days to look back (default: 7)
        exclude_system: Whether to exclude system databases (default: True)
        limit: Maximum number of tables to return (default: 50)

    Returns:
        List of dictionaries with recently modified tables
    """
    exclude_clause = ""
    if exclude_system:
        exclude_clause = (
            "AND database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')"
        )

    query = f"""
    SELECT
        database,
        name,
        engine,
        metadata_modification_time,
        formatReadableQuantity(total_rows) AS total_rows_,
        formatReadableSize(total_bytes) AS total_bytes_,
        toUnixTimestamp(now()) - toUnixTimestamp(metadata_modification_time) AS seconds_since_modification
    FROM system.tables
    WHERE metadata_modification_time >= (now() - toIntervalDay({days}))
      {exclude_clause}
    ORDER BY metadata_modification_time DESC
    LIMIT {limit}
    """

    logger.info(f"Retrieving tables modified in the last {days} days (limit: {limit})")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving recently modified tables: {e!s}")
        return []


@log_execution_time
def get_largest_tables(
    client: Client, exclude_system: bool = True, limit: int = 20
) -> list[dict[str, str | int]]:
    """Get the largest tables by size.

    This function identifies the largest tables in the database by total bytes.

    Args:
        client: The ClickHouse client instance
        exclude_system: Whether to exclude system databases (default: True)
        limit: Maximum number of tables to return (default: 20)

    Returns:
        List of dictionaries with the largest tables
    """
    exclude_clause = ""
    if exclude_system:
        exclude_clause = (
            "WHERE database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')"
        )

    query = f"""
    SELECT
        database,
        name,
        engine,
        formatReadableQuantity(total_rows) AS total_rows_,
        formatReadableSize(total_bytes) AS total_bytes_,
        total_bytes
    FROM system.tables
    {exclude_clause}
    ORDER BY total_bytes DESC
    LIMIT {limit}
    """

    logger.info(f"Retrieving the {limit} largest tables by size")

    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving largest tables: {e!s}")
        return []
