"""Utility tools for ClickHouse database administration.

This module provides utility functions for database administration tasks.
"""

import logging
from typing import Dict, List, Optional, Union

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("agent-zero")


@log_execution_time
def generate_drop_tables_script(
    client: Client,
    database: str = "default"
) -> List[Dict[str, str]]:
    """Generate a script to drop all tables in a database.

    This function creates DROP TABLE statements for all tables in the specified database.

    Args:
        client: The ClickHouse client instance
        database: The database name

    Returns:
        List of dictionaries with drop table statements
    """
    query = f"""
    SELECT concat('DROP TABLE {database}.`', name, '`;') AS drop_table_query 
    FROM system.tables 
    WHERE database = '{database}'
    """
    
    logger.info(f"Generating script to drop all tables in database '{database}'")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error generating drop tables script: {str(e)}")
        return []


@log_execution_time
def get_user_defined_functions(
    client: Client
) -> List[Dict[str, str]]:
    """Get a list of user-defined SQL functions.

    This function retrieves all SQL user-defined functions in the database.

    Args:
        client: The ClickHouse client instance

    Returns:
        List of dictionaries with function information
    """
    query = """
    SELECT name, create_query 
    FROM system.functions 
    WHERE origin = 'SQLUserDefined'
    """
    
    logger.info("Retrieving user-defined SQL functions")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving user-defined functions: {str(e)}")
        return []


@log_execution_time
def prewarm_cache_on_all_replicas(
    client: Client,
    database: str,
    table: str
) -> List[Dict[str, int]]:
    """Prewarm the cache on all replicas.

    This function executes a query that forces all replicas to load data into cache.

    Args:
        client: The ClickHouse client instance
        database: The database name
        table: The table name

    Returns:
        List of dictionaries with a sum of ignore results
    """
    query = f"""
    SELECT sum(ignore(*)) 
    FROM clusterAllReplicas(default, {database}.{table})
    """
    
    logger.info(f"Prewarming cache for table {database}.{table} on all replicas")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error prewarming cache: {str(e)}")
        return []


@log_execution_time
def get_thread_name_distributions(
    client: Client,
    start_time: str,
    end_time: str
) -> List[Dict[str, Union[str, int, float]]]:
    """Get thread name distribution by host.

    This function analyzes thread name distribution across different hosts.

    Args:
        client: The ClickHouse client instance
        start_time: Start time in 'YYYY-MM-DD HH:MM:SS' format
        end_time: End time in 'YYYY-MM-DD HH:MM:SS' format

    Returns:
        List of dictionaries with thread name distribution
    """
    query = f"""
    SELECT 
        thread_name, 
        count() AS c
    FROM clusterAllReplicas(default, system.text_log) 
    WHERE (event_time >= '{start_time}') AND (event_time <= '{end_time}') 
    GROUP BY ALL 
    ORDER BY c ASC
    """
    
    logger.info(f"Retrieving thread name distribution from {start_time} to {end_time}")
    
    try:
        return execute_query_with_retry(client, query)
    except ClickHouseError as e:
        logger.error(f"Error retrieving thread name distribution: {str(e)}")
        # Fallback to local query
        fallback_query = f"""
        SELECT 
            thread_name, 
            count() AS c
        FROM system.text_log 
        WHERE (event_time >= '{start_time}') AND (event_time <= '{end_time}') 
        GROUP BY thread_name
        ORDER BY c ASC
        """
        logger.info("Falling back to local text_log query")
        return execute_query_with_retry(client, fallback_query)


@log_execution_time
def create_monitoring_views(
    client: Client
) -> bool:
    """Create or update the monitoring views.

    This function creates or updates all the monitoring views used by the
    ClickHouse Monitoring MCP Service.

    Args:
        client: The ClickHouse client instance

    Returns:
        True if all views were created/updated successfully, False otherwise
    """
    # Create 'tools' database if it doesn't exist
    try:
        client.command("CREATE DATABASE IF NOT EXISTS tools")
        logger.info("Ensured tools database exists")
    except ClickHouseError as e:
        logger.error(f"Error creating tools database: {str(e)}")
        return False
    
    # List of view definitions
    views = [
        """
        CREATE OR REPLACE VIEW tools.view_analyze_parts (`level` UInt32, `rows_min` String, `bytes_min` String, `rows_median` String, `bytes_median` String, `rows_max` String, `bytes_max` String, `part_type` String, `part_storage_type` String, `c` UInt64) AS 
        SELECT level, formatReadableQuantity(min(rows)) AS rows_min, formatReadableSize(min(data_compressed_bytes)) AS bytes_min, formatReadableQuantity(quantile(0.5)(rows)) AS rows_median, formatReadableSize(quantile(0.5)(data_compressed_bytes)) AS bytes_median, formatReadableQuantity(max(rows)) AS rows_max, formatReadableSize(max(data_compressed_bytes)) AS bytes_max, part_type, part_storage_type, count() AS c 
        FROM system.parts 
        WHERE active AND (`table` = 'query_log') AND (database = 'system') 
        GROUP BY ALL WITH TOTALS 
        ORDER BY level ASC, c ASC
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_async_insert_batches (`c` UInt64, `count()` UInt64, `avg_rows` Float64) AS 
        SELECT c, count(), avg(d) AS avg_rows 
        FROM (
            SELECT flush_query_id, count() AS c, min(rows), max(rows), sum(rows) AS d 
            FROM clusterAllReplicas(default, system.asynchronous_insert_log) 
            WHERE event_time >= (now() - toIntervalDay(5)) 
            GROUP BY 1
        ) 
        GROUP BY 1
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_blob_storage_log (`event_type` Enum8('Upload' = 1, 'Delete' = 2, 'MultiPartUploadCreate' = 3, 'MultiPartUploadWrite' = 4, 'MultiPartUploadComplete' = 5, 'MultiPartUploadAbort' = 6) COMMENT 'https://pastila.nl/?019af080/4bcdfe0d9387d697b56f51e3833c5fb5#1bGHvnAXOb1U+ro/LwmJVQ==', `c` UInt64, `table_uuid` String, `name` String) AS 
        SELECT l.event_type, l.c, l.table_uuid, t.name 
        FROM (
            SELECT event_type, count() AS c, substring(any(local_path), 11, 36) AS table_uuid 
            FROM clusterAllReplicas(default, system.blob_storage_log) 
            WHERE event_date = today() 
            GROUP BY event_type
        ) AS l 
        INNER JOIN system.tables AS t ON l.table_uuid = CAST(t.uuid, 'String')
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_cpu_usage (`dt` DateTime, `cpu_usage_cluster` Float64, `cpu_cores_cluster` Float64, `cpu_usage_cluster_normalized` Float64) AS 
        WITH cpu_cores_in_cluster_table AS (
            SELECT t, sum(value) AS cpu_cores_in_cluster 
            FROM (
                SELECT hostname, t, argMax(value, t) AS value 
                FROM (
                    SELECT hostname() AS hostname, CAST(toStartOfInterval(event_time, toIntervalSecond(60)), 'INT') AS t, value 
                    FROM clusterAllReplicas(default, system.asynchronous_metric_log) 
                    WHERE (metric = 'CGroupMaxCPU') AND (event_date >= toDate(now() - 10800)) AND (event_time >= (now() - 10800))
                ) 
                GROUP BY t, hostname
            ) 
            GROUP BY t 
            ORDER BY t ASC
        ), 
        cpu_cores_usage_in_cluster_table AS (
            SELECT CAST(toStartOfInterval(event_time, toIntervalSecond(60)), 'INT') AS t, avg(metric) / 1000000 AS cpu_cores_usage_in_cluster 
            FROM (
                SELECT event_time, sum(ProfileEvent_OSCPUVirtualTimeMicroseconds) AS metric 
                FROM clusterAllReplicas(default, system.metric_log) 
                WHERE (event_date >= toDate(now() - 10800)) AND (event_time >= (now() - 10800)) 
                GROUP BY event_time
            ) 
            GROUP BY t 
            ORDER BY t ASC 
            WITH FILL STEP 60
        ) 
        SELECT toDateTime(t) AS dt, round(cpu_cores_usage_in_cluster_table.cpu_cores_usage_in_cluster, 2) AS cpu_usage_cluster, cpu_cores_in_cluster_table.cpu_cores_in_cluster AS cpu_cores_cluster, round(cpu_usage_cluster / cpu_cores_cluster, 2) AS cpu_usage_cluster_normalized 
        FROM cpu_cores_usage_in_cluster_table 
        INNER JOIN cpu_cores_in_cluster_table ON cpu_cores_usage_in_cluster_table.t = cpu_cores_in_cluster_table.t 
        ORDER BY t ASC
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_current_processes (`hostname()` String, `user` String, `query_kind` String, `elapsed` Float64, `read_rows_` String, `read_bytes_` String, `total_rows_approx_` String, `written_rows_` String, `written_bytes_` String, `memory_usage_` String, `peak_memory_usage_` String, `query_id` String, `normalizedQueryHash(query)` UInt64) AS 
        SELECT hostname(), user, query_kind, elapsed, formatReadableQuantity(read_rows) AS read_rows_, formatReadableSize(read_bytes) AS read_bytes_, formatReadableQuantity(total_rows_approx) AS total_rows_approx_, formatReadableQuantity(written_rows) AS written_rows_, formatReadableSize(written_bytes) AS written_bytes_, formatReadableSize(memory_usage) AS memory_usage_, formatReadableSize(peak_memory_usage) AS peak_memory_usage_, query_id, normalizedQueryHash(query) 
        FROM clusterAllReplicas(default, system.processes)
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_error_stack_trace (`last_error_time` DateTime, `last_error_message` String, `stack_trace` Array(String)) AS 
        SELECT last_error_time, last_error_message, arrayMap(x -> demangle(addressToSymbol(x)), last_error_trace) AS stack_trace 
        FROM system.errors 
        WHERE name = 'LOGICAL_ERROR' 
        SETTINGS allow_introspection_functions = 1
        """,
        
        """
        CREATE OR REPLACE VIEW tools.view_check_memory_usage (`ts` DateTime, `hostname_` String, `MemoryTracking_avg` String, `MemoryTracking_p50` String, `MemoryTracking_p90` String, `MemoryTracking_p99` String, `MemoryTracking_max` String) AS 
        SELECT event_time AS ts, splitByChar('.', hostname)[1] AS hostname_, formatReadableSize(avg(CurrentMetric_MemoryTracking)) AS MemoryTracking_avg, formatReadableSize(quantile(0.5)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p50, formatReadableSize(quantile(0.9)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p90, formatReadableSize(quantile(0.99)(CurrentMetric_MemoryTracking)) AS MemoryTracking_p99, formatReadableSize(max(CurrentMetric_MemoryTracking)) AS MemoryTracking_max 
        FROM clusterAllReplicas(default, system.metric_log) 
        WHERE (event_time >= (now() - toIntervalDay(7))) AND (event_time <= now()) 
        GROUP BY ALL 
        ORDER BY ts ASC, hostname_ ASC
        """,
    ]
    
    # Create or replace each view
    success = True
    for view_definition in views:
        try:
            client.command(view_definition, settings={"allow_ddl": 1})
            logger.info(f"Created/updated view: {view_definition.split('CREATE OR REPLACE VIEW ')[1].split(' ')[0]}")
        except ClickHouseError as e:
            logger.error(f"Error creating view: {str(e)}")
            success = False
    
    return success