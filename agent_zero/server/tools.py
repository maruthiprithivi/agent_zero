"""MCP tool registration for Agent Zero.

This module contains all MCP tool definitions organized by category,
following the development standards defined in CLAUDE.md.
"""

import logging
from collections.abc import Sequence
from datetime import datetime

from clickhouse_connect.driver.binding import format_query_value, quote_identifier

from agent_zero.ai_diagnostics import (
    create_ai_bottleneck_detector,
    create_anomaly_detection_engine,
    create_pattern_analysis_engine,
    create_performance_advisor,
)
from agent_zero.config import get_config
from agent_zero.mcp_tracer import trace_mcp_call
from agent_zero.monitoring import (
    AzureStorageAnalyzer,
    CacheAnalyzer,
    CompressionAnalyzer,
    # Hardware Diagnostics
    CPUAnalyzer,
    HardwareHealthEngine,
    IOPerformanceAnalyzer,
    MemoryAnalyzer,
    # Performance Diagnostics
    PerformanceDiagnosticEngine,
    # ProfileEvents Core Analysis
    ProfileEventsAnalyzer,
    ProfileEventsCategory,
    ProfileEventsComparator,
    QueryExecutionAnalyzer,
    # Storage & Cloud Diagnostics
    S3StorageAnalyzer,
    StorageOptimizationEngine,
    ThreadPoolAnalyzer,
    create_monitoring_views,
    detect_profile_event_anomalies,
    generate_drop_tables_script,
    # Insert Operations
    get_async_insert_stats,
    get_async_vs_sync_insert_counts,
    # System Components
    get_blob_storage_stats,
    # Resource Usage
    get_cpu_usage,
    # Parts Merges
    get_current_merges,
    # Query Performance
    get_current_processes,
    # Error Analysis
    get_error_stack_traces,
    get_insert_written_bytes_distribution,
    get_largest_tables,
    get_memory_usage,
    get_merge_stats,
    get_mv_deduplicated_blocks,
    get_mv_query_stats,
    get_normalized_query_stats,
    get_part_log_events,
    get_partition_stats,
    get_parts_analysis,
    get_query_duration_stats,
    get_query_kind_breakdown,
    get_recent_errors,
    get_recent_insert_queries,
    get_recent_table_modifications,
    get_s3queue_stats,
    get_s3queue_with_names,
    get_server_sizing,
    # Table Statistics
    get_table_inactive_parts,
    get_table_stats,
    get_text_log,
    get_thread_name_distributions,
    get_uptime,
    get_user_defined_functions,
    prewarm_cache_on_all_replicas,
)
from agent_zero.utils import format_exception

from .client import create_clickhouse_client
from .errors import MCPToolError
from .query import execute_query_threaded

logger = logging.getLogger(__name__)


def register_all_tools(mcp):
    """Register all MCP tools with the FastMCP instance.

    Args:
        mcp: FastMCP instance to register tools with
    """
    register_database_tools(mcp)
    register_query_performance_tools(mcp)
    register_resource_usage_tools(mcp)
    register_error_analysis_tools(mcp)
    register_insert_operations_tools(mcp)
    register_parts_merges_tools(mcp)
    register_system_components_tools(mcp)
    register_table_statistics_tools(mcp)
    register_utility_tools(mcp)
    register_ai_diagnostics_tools(mcp)
    register_profile_events_tools(mcp)
    register_performance_diagnostics_tools(mcp)
    register_storage_cloud_tools(mcp)
    register_distributed_systems_tools(mcp)
    register_hardware_diagnostics_tools(mcp)
    register_ai_powered_analysis_tools(mcp)


def register_database_tools(mcp):
    """Register basic database interaction tools."""

    @mcp.tool()
    @trace_mcp_call
    def list_databases():
        """List all databases in the ClickHouse server.

        Returns:
            A list of database names.
        """
        logger.info("Listing all databases")
        client = create_clickhouse_client()
        try:
            result = client.command("SHOW DATABASES")
            logger.info(f"Found {len(result) if isinstance(result, list) else 1} databases")
            cfg = get_config()
            if getattr(cfg, "enable_structured_tool_output", False):
                data = [[db] for db in (result or [])]
                return {"columns": ["database"], "data": data, "meta": {"row_count": len(data)}}
            return result
        except Exception as e:
            logger.error(f"Error listing databases: {e!s}")
            cfg = get_config()
            if getattr(cfg, "enable_tool_errors", False):
                raise MCPToolError(code="LIST_DATABASES_FAILED", message=format_exception(e))
            return f"Error listing databases: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def list_tables(database: str, like: str = None):
        """List all tables in a specified database.

        Args:
            database: The name of the database.
            like: Optional filter pattern for table names.

        Returns:
            A list of table information including schema details.
        """
        logger.info(f"Listing tables in database '{database}'")
        client = create_clickhouse_client()
        try:
            query = f"SHOW TABLES FROM {quote_identifier(database)}"
            if like:
                query += f" LIKE {format_query_value(like)}"
            result = client.command(query)

            # Get all table comments in one query
            table_comments_query = (
                "SELECT name, comment FROM system.tables WHERE database ="
                f" {format_query_value(database)}"
            )
            table_comments_result = client.query(table_comments_query)
            table_comments = {row[0]: row[1] for row in table_comments_result.result_rows}

            # Get all column comments in one query
            column_comments_query = (
                "SELECT table, name, comment FROM system.columns WHERE database ="
                f" {format_query_value(database)}"
            )
            column_comments_result = client.query(column_comments_query)
            column_comments = {}
            for row in column_comments_result.result_rows:
                table, col_name, comment = row
                if table not in column_comments:
                    column_comments[table] = {}
                column_comments[table][col_name] = comment

            def get_table_info(table):
                logger.info(f"Getting schema info for table {database}.{table}")
                schema_query = (
                    f"DESCRIBE TABLE {quote_identifier(database)}.{quote_identifier(table)}"
                )
                schema_result = client.query(schema_query)

                columns = []
                column_names = schema_result.column_names
                for row in schema_result.result_rows:
                    column_dict = {}
                    for i, col_name in enumerate(column_names):
                        column_dict[col_name] = row[i]
                    # Add comment from our pre-fetched comments
                    if table in column_comments and column_dict["name"] in column_comments[table]:
                        column_dict["comment"] = column_comments[table][column_dict["name"]]
                    else:
                        column_dict["comment"] = None
                    columns.append(column_dict)

                create_table_query = f"SHOW CREATE TABLE {database}.`{table}`"
                create_table_result = client.command(create_table_query)

                return {
                    "database": database,
                    "name": table,
                    "comment": table_comments.get(table),
                    "columns": columns,
                    "create_table_query": create_table_result,
                }

            tables = []
            if isinstance(result, str):
                # Single table result
                for table in (t.strip() for t in result.split()):
                    if table:
                        tables.append(get_table_info(table))
            elif isinstance(result, Sequence):
                # Multiple table results
                for table in result:
                    tables.append(get_table_info(table))

            logger.info(f"Found {len(tables)} tables")
            cfg = get_config()
            if getattr(cfg, "enable_structured_tool_output", False):
                if not tables:
                    return {"columns": [], "data": [], "meta": {"row_count": 0}}
                columns = list(tables[0].keys())
                data = [[t.get(c) for c in columns] for t in tables]
                return {"columns": columns, "data": data, "meta": {"row_count": len(data)}}
            return tables
        except Exception as e:
            logger.error(f"Error listing tables in database '{database}': {e!s}")
            cfg = get_config()
            if getattr(cfg, "enable_tool_errors", False):
                raise MCPToolError(
                    code="LIST_TABLES_FAILED",
                    message=format_exception(e),
                    context={"database": database},
                )
            return f"Error listing tables: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def run_select_query(query: str):
        """Execute a read-only SELECT query against the ClickHouse database.

        Args:
            query: The SQL query to execute (must be read-only).

        Returns:
            The query results as a list of dictionaries.
        """
        return execute_query_threaded(query)


def register_query_performance_tools(mcp):
    """Register query performance monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_current_processes():
        """Get information about currently running processes on the ClickHouse cluster."""
        logger.info("Monitoring current processes")
        client = create_clickhouse_client()
        try:
            return get_current_processes(client)
        except Exception as e:
            logger.error(f"Error monitoring current processes: {e!s}")
            return f"Error monitoring current processes: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_query_duration(query_kind: str | None = None, days: int = 7):
        """Get query duration statistics grouped by hour."""
        kind_desc = f"'{query_kind}'" if query_kind else "all"
        logger.info(f"Monitoring query duration for {kind_desc} queries over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_query_duration_stats(client, query_kind, days)
        except Exception as e:
            logger.error(f"Error monitoring query duration: {e!s}")
            return f"Error monitoring query duration: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_query_patterns(days: int = 2, limit: int = 50):
        """Identify the most resource-intensive query patterns."""
        logger.info(f"Monitoring query patterns over the past {days} days (limit: {limit})")
        client = create_clickhouse_client()
        try:
            return get_normalized_query_stats(client, days, limit)
        except Exception as e:
            logger.error(f"Error monitoring query patterns: {e!s}")
            return f"Error monitoring query patterns: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_query_types(days: int = 7):
        """Get a breakdown of query types by hour."""
        logger.info(f"Monitoring query types over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_query_kind_breakdown(client, days)
        except Exception as e:
            logger.error(f"Error monitoring query types: {e!s}")
            return f"Error monitoring query types: {format_exception(e)}"


def register_resource_usage_tools(mcp):
    """Register resource usage monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_memory_usage(days: int = 7):
        """Get memory usage statistics over time by host."""
        logger.info(f"Monitoring memory usage over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_memory_usage(client, days)
        except Exception as e:
            logger.error(f"Error monitoring memory usage: {e!s}")
            return f"Error monitoring memory usage: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_cpu_usage(hours: int = 3):
        """Get CPU usage statistics over time."""
        logger.info(f"Monitoring CPU usage over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            return get_cpu_usage(client, hours)
        except Exception as e:
            logger.error(f"Error monitoring CPU usage: {e!s}")
            return f"Error monitoring CPU usage: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_cluster_sizing():
        """Get server sizing information for all nodes in the cluster."""
        logger.info("Getting cluster sizing information")
        client = create_clickhouse_client()
        try:
            return get_server_sizing(client)
        except Exception as e:
            logger.error(f"Error getting cluster sizing: {e!s}")
            return f"Error getting cluster sizing: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_uptime(days: int = 7):
        """Get server uptime statistics."""
        logger.info(f"Monitoring uptime over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_uptime(client, days)
        except Exception as e:
            logger.error(f"Error monitoring uptime: {e!s}")
            return f"Error monitoring uptime: {format_exception(e)}"


def register_error_analysis_tools(mcp):
    """Register error analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_recent_errors(days: int = 1):
        """Get recent errors from ClickHouse system.errors table."""
        logger.info(f"Monitoring recent errors over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_recent_errors(client, days)
        except Exception as e:
            logger.error(f"Error monitoring recent errors: {e!s}")
            return f"Error monitoring recent errors: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_error_stack_traces():
        """Get error stack traces for logical errors in the system."""
        logger.info("Monitoring error stack traces")
        client = create_clickhouse_client()
        try:
            return get_error_stack_traces(client)
        except Exception as e:
            logger.error(f"Error monitoring error stack traces: {e!s}")
            return f"Error monitoring error stack traces: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def view_text_log(limit: int = 100):
        """Get recent entries from the text log."""
        logger.info(f"Viewing text log (limit: {limit})")
        client = create_clickhouse_client()
        try:
            return get_text_log(client, limit)
        except Exception as e:
            logger.error(f"Error viewing text log: {e!s}")
            return f"Error viewing text log: {format_exception(e)}"


def register_insert_operations_tools(mcp):
    """Register insert operations monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_recent_insert_queries(days: int = 1, limit: int = 100):
        """Get recent insert queries."""
        logger.info(f"Monitoring recent insert queries over the past {days} days (limit: {limit})")
        client = create_clickhouse_client()
        try:
            return get_recent_insert_queries(client, days, limit)
        except Exception as e:
            logger.error(f"Error monitoring recent insert queries: {e!s}")
            return f"Error monitoring recent insert queries: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_async_insert_stats(days: int = 7):
        """Get asynchronous insert statistics."""
        logger.info(f"Monitoring async insert stats over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_async_insert_stats(client, days)
        except Exception as e:
            logger.error(f"Error monitoring async insert stats: {e!s}")
            return f"Error monitoring async insert stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_insert_bytes_distribution(days: int = 7):
        """Get distribution of written bytes for insert operations."""
        logger.info(f"Monitoring insert bytes distribution over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_insert_written_bytes_distribution(client, days)
        except Exception as e:
            logger.error(f"Error monitoring insert bytes distribution: {e!s}")
            return f"Error monitoring insert bytes distribution: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_async_vs_sync_inserts(days: int = 7):
        """Get hourly counts of asynchronous vs. synchronous insert operations."""
        logger.info(f"Monitoring async vs. sync insert counts for the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_async_vs_sync_insert_counts(client, days)
        except Exception as e:
            logger.error(f"Error monitoring async vs. sync insert counts: {e!s}")
            return f"Error monitoring async vs. sync insert counts: {format_exception(e)}"


def register_parts_merges_tools(mcp):
    """Register parts and merges monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_current_merges():
        """Get information about currently running merge operations."""
        logger.info("Monitoring current merges")
        client = create_clickhouse_client()
        try:
            return get_current_merges(client)
        except Exception as e:
            logger.error(f"Error monitoring current merges: {e!s}")
            return f"Error monitoring current merges: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_merge_stats(days: int = 7):
        """Get merge performance statistics."""
        logger.info(f"Monitoring merge stats over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_merge_stats(client, days)
        except Exception as e:
            logger.error(f"Error monitoring merge stats: {e!s}")
            return f"Error monitoring merge stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_part_log_events(days: int = 1, limit: int = 100):
        """Get recent part log events."""
        logger.info(f"Monitoring part log events over the past {days} days (limit: {limit})")
        client = create_clickhouse_client()
        try:
            return get_part_log_events(client, days, limit)
        except Exception as e:
            logger.error(f"Error monitoring part log events: {e!s}")
            return f"Error monitoring part log events: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_partition_stats(database: str, table: str):
        """Get partition statistics for a specific table."""
        logger.info(f"Monitoring partition stats for {database}.{table}")
        client = create_clickhouse_client()
        try:
            return get_partition_stats(client, database, table)
        except Exception as e:
            logger.error(f"Error monitoring partition stats: {e!s}")
            return f"Error monitoring partition stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_parts_analysis(database: str, table: str):
        """Get parts analysis for a specific table."""
        logger.info(f"Monitoring parts analysis for {database}.{table}")
        client = create_clickhouse_client()
        try:
            return get_parts_analysis(client, database, table)
        except Exception as e:
            logger.error(f"Error monitoring parts analysis: {e!s}")
            return f"Error monitoring parts analysis: {format_exception(e)}"


def register_system_components_tools(mcp):
    """Register system components monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_blob_storage_stats(days: int = 7):
        """Get statistics for blob storage operations."""
        logger.info(f"Monitoring blob storage stats over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_blob_storage_stats(client, days)
        except Exception as e:
            logger.error(f"Error monitoring blob storage stats: {e!s}")
            return f"Error monitoring blob storage stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_materialized_view_stats(days: int = 7):
        """Get statistics for materialized view queries."""
        logger.info(f"Monitoring materialized view stats over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_mv_query_stats(client, days)
        except Exception as e:
            logger.error(f"Error monitoring materialized view stats: {e!s}")
            return f"Error monitoring materialized view stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_s3queue_stats(days: int = 7):
        """Get statistics for S3 Queue operations."""
        logger.info(f"Monitoring S3 Queue stats over the past {days} days")
        client = create_clickhouse_client()
        try:
            return get_s3queue_stats(client, days)
        except Exception as e:
            logger.error(f"Error monitoring S3 Queue stats: {e!s}")
            return f"Error monitoring S3 Queue stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_mv_deduplicated_blocks(view_name: str, days: int = 7):
        """Get statistics about deduplicated blocks for a specific materialized view."""
        logger.info(
            f"Monitoring deduplicated blocks for materialized view '{view_name}' for the past {days} days"
        )
        client = create_clickhouse_client()
        try:
            return get_mv_deduplicated_blocks(client, view_name, days)
        except Exception as e:
            logger.error(f"Error monitoring deduplicated blocks: {e!s}")
            return f"Error monitoring deduplicated blocks: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def list_s3queue_with_names():
        """Get S3 queue entries with database and table names."""
        logger.info("Retrieving S3 queue entries with database and table names")
        client = create_clickhouse_client()
        try:
            return get_s3queue_with_names(client)
        except Exception as e:
            logger.error(f"Error retrieving S3 queue entries with names: {e!s}")
            return f"Error retrieving S3 queue entries with names: {format_exception(e)}"


def register_table_statistics_tools(mcp):
    """Register table statistics monitoring tools."""

    @mcp.tool()
    @trace_mcp_call
    def monitor_table_stats(database: str, table: str = None):
        """Get detailed statistics for tables."""
        table_desc = f"{database}.{table}" if table else f"all tables in {database}"
        logger.info(f"Monitoring stats for {table_desc}")
        client = create_clickhouse_client()
        try:
            return get_table_stats(client, database, table)
        except Exception as e:
            logger.error(f"Error monitoring table stats: {e!s}")
            return f"Error monitoring table stats: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def monitor_table_inactive_parts(database: str, table: str):
        """Get information about inactive parts for a table."""
        logger.info(f"Monitoring inactive parts for {database}.{table}")
        client = create_clickhouse_client()
        try:
            return get_table_inactive_parts(client, database, table)
        except Exception as e:
            logger.error(f"Error monitoring table inactive parts: {e!s}")
            return f"Error monitoring table inactive parts: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def list_recent_table_modifications(
        days: int = 7, exclude_system: bool = True, limit: int = 50
    ):
        """Get recently modified tables."""
        logger.info(f"Retrieving tables modified in the last {days} days (limit: {limit})")
        client = create_clickhouse_client()
        try:
            return get_recent_table_modifications(client, days, exclude_system, limit)
        except Exception as e:
            logger.error(f"Error retrieving recently modified tables: {e!s}")
            return f"Error retrieving recently modified tables: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def list_largest_tables(exclude_system: bool = True, limit: int = 20):
        """Get the largest tables by size."""
        logger.info(f"Retrieving the {limit} largest tables by size")
        client = create_clickhouse_client()
        try:
            return get_largest_tables(client, exclude_system, limit)
        except Exception as e:
            logger.error(f"Error retrieving largest tables: {e!s}")
            return f"Error retrieving largest tables: {format_exception(e)}"


def register_utility_tools(mcp):
    """Register utility tools."""

    @mcp.tool()
    @trace_mcp_call
    def health_check():
        """Health check for MCP server and ClickHouse connection."""
        client = create_clickhouse_client()
        cfg = get_config()
        try:
            version = getattr(client, "server_version", "unknown")
            return {
                "status": "ok",
                "server": "mcp-clickhouse",
                "clickhouse_version": str(version),
                "transport": (
                    cfg.transport.value if hasattr(cfg.transport, "value") else str(cfg.transport)
                ),
                "deployment_mode": (
                    cfg.deployment_mode.value
                    if hasattr(cfg.deployment_mode, "value")
                    else str(cfg.deployment_mode)
                ),
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": format_exception(e),
            }

    @mcp.tool()
    @trace_mcp_call
    def server_info():
        """Return server information and non-sensitive configuration flags."""
        cfg = get_config()
        return {
            "name": "mcp-clickhouse",
            "transport": (
                cfg.transport.value if hasattr(cfg.transport, "value") else str(cfg.transport)
            ),
            "deployment_mode": (
                cfg.deployment_mode.value
                if hasattr(cfg.deployment_mode, "value")
                else str(cfg.deployment_mode)
            ),
            "features": {
                "enable_client_cache": getattr(cfg, "enable_client_cache", True),
                "enable_structured_tool_output": getattr(
                    cfg, "enable_structured_tool_output", False
                ),
                "enable_tool_errors": getattr(cfg, "enable_tool_errors", False),
                "enable_mcp_tracing": getattr(cfg, "enable_mcp_tracing", False),
            },
        }

    @mcp.tool()
    @trace_mcp_call
    def generate_table_drop_script(database: str):
        """Generate a script to drop all tables in a database."""
        logger.info(f"Generating drop tables script for database {database}")
        client = create_clickhouse_client()
        try:
            return generate_drop_tables_script(client, database)
        except Exception as e:
            logger.error(f"Error generating drop tables script: {e!s}")
            return f"Error generating drop tables script: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def list_user_defined_functions():
        """Get information about user-defined functions."""
        logger.info("Listing user-defined functions")
        client = create_clickhouse_client()
        try:
            return get_user_defined_functions(client)
        except Exception as e:
            logger.error(f"Error listing user-defined functions: {e!s}")
            return f"Error listing user-defined functions: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def prewarm_cache(database: str, table: str):
        """Prewarm the cache on all replicas."""
        logger.info(f"Prewarming cache for table {database}.{table} on all replicas")
        client = create_clickhouse_client()
        try:
            return prewarm_cache_on_all_replicas(client, database, table)
        except Exception as e:
            logger.error(f"Error prewarming cache: {e!s}")
            return f"Error prewarming cache: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_thread_distribution(start_time: str, end_time: str):
        """Get thread name distribution by host."""
        logger.info(f"Retrieving thread name distribution from {start_time} to {end_time}")
        client = create_clickhouse_client()
        try:
            return get_thread_name_distributions(client, start_time, end_time)
        except Exception as e:
            logger.error(f"Error retrieving thread name distribution: {e!s}")
            return f"Error retrieving thread name distribution: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def setup_monitoring_views():
        """Create or update the monitoring views."""
        logger.info("Creating or updating monitoring views")
        client = create_clickhouse_client()
        try:
            result = create_monitoring_views(client)
            if result:
                return "Successfully created/updated all monitoring views"
            else:
                return "Failed to create/update some monitoring views"
        except Exception as e:
            logger.error(f"Error creating monitoring views: {e!s}")
            return f"Error creating monitoring views: {format_exception(e)}"


def register_ai_diagnostics_tools(mcp):
    """Register AI diagnostics and performance advisor tools."""

    @mcp.tool()
    @trace_mcp_call
    def generate_performance_recommendations(max_recommendations: int = 20):
        """Generate comprehensive AI-powered performance optimization recommendations.

        Analyzes system performance data and provides actionable recommendations
        with impact predictions, implementation complexity assessments, and priority rankings.

        Args:
            max_recommendations: Maximum number of recommendations to return (default: 20)

        Returns:
            Dictionary containing categorized recommendations with implementation details
        """
        logger.info(f"Generating performance recommendations (max: {max_recommendations})")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector for integration
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Create performance advisor
            advisor = create_performance_advisor(client, bottleneck_detector)

            # Generate comprehensive recommendations
            recommendations = advisor.generate_comprehensive_recommendations(
                max_recommendations=max_recommendations
            )

            logger.info(
                f"Generated {recommendations.get('total_recommendations', 0)} recommendations"
            )
            return recommendations

        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e!s}")
            return f"Error generating performance recommendations: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_performance_recommendation_summary():
        """Get a quick summary of available performance optimization opportunities.

        Provides a high-level overview of system health and top optimization opportunities
        without detailed implementation steps.

        Returns:
            Summary of system health and key recommendation categories
        """
        logger.info("Generating performance recommendation summary")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Create performance advisor
            advisor = create_performance_advisor(client, bottleneck_detector)

            # Generate limited recommendations for summary
            recommendations = advisor.generate_comprehensive_recommendations(max_recommendations=5)

            # Extract summary information
            summary = {
                "system_health": recommendations.get("summary", {}),
                "context": recommendations.get("context", {}),
                "top_opportunities": [],
            }

            # Get top recommendations from each category
            recommendations_by_category = recommendations.get("recommendations_by_category", {})
            for category, recs in recommendations_by_category.items():
                if recs:  # If there are recommendations in this category
                    top_rec = recs[0]  # Get the highest priority recommendation
                    summary["top_opportunities"].append(
                        {
                            "category": category,
                            "title": top_rec["title"],
                            "impact": top_rec["impact_percentage"],
                            "complexity": top_rec["complexity"],
                            "priority_score": top_rec["priority_score"],
                        }
                    )

            return summary

        except Exception as e:
            logger.error(f"Error generating recommendation summary: {e!s}")
            return f"Error generating recommendation summary: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_configuration_optimization_recommendations():
        """Get specific ClickHouse configuration optimization recommendations.

        Analyzes current server configuration and provides specific setting
        adjustments to improve performance based on current workload patterns.

        Returns:
            List of configuration optimization recommendations
        """
        logger.info("Generating configuration optimization recommendations")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector for context
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Create performance advisor
            advisor = create_performance_advisor(client, bottleneck_detector)

            # Get system context
            context = advisor._gather_system_context()

            # Get configuration-specific recommendations
            config_recommendations = advisor.configuration_advisor.analyze_server_configuration(
                context
            )

            # Format recommendations for response
            formatted_recs = []
            for rec in config_recommendations:
                formatted_recs.append(
                    {
                        "recommendation_id": rec.recommendation_id,
                        "title": rec.title,
                        "description": rec.description,
                        "current_settings": rec.current_state,
                        "recommended_settings": rec.recommended_changes,
                        "implementation_steps": rec.implementation_steps,
                        "impact_percentage": rec.impact_percentage,
                        "confidence_score": rec.confidence_score,
                        "complexity": rec.complexity.value,
                        "risk_level": rec.risk_level.value,
                        "estimated_time_hours": rec.estimated_time_hours,
                        "priority_score": rec.priority_score,
                        "business_impact": rec.business_impact,
                    }
                )

            logger.info(f"Generated {len(formatted_recs)} configuration recommendations")
            return {
                "total_recommendations": len(formatted_recs),
                "recommendations": formatted_recs,
            }

        except Exception as e:
            logger.error(f"Error generating configuration recommendations: {e!s}")
            return f"Error generating configuration recommendations: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_query_optimization_recommendations():
        """Get query optimization recommendations based on slow query analysis.

        Analyzes query performance patterns and provides specific optimization
        suggestions for frequently executed slow queries.

        Returns:
            List of query optimization recommendations
        """
        logger.info("Generating query optimization recommendations")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector for context
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Create performance advisor
            advisor = create_performance_advisor(client, bottleneck_detector)

            # Get system context
            context = advisor._gather_system_context()

            # Get query optimization recommendations
            query_recommendations = advisor.query_optimizer.analyze_query_patterns(context)

            # Format recommendations for response
            formatted_recs = []
            for rec in query_recommendations:
                formatted_recs.append(
                    {
                        "recommendation_id": rec.recommendation_id,
                        "title": rec.title,
                        "description": rec.description,
                        "current_performance": rec.current_state,
                        "optimization_approach": rec.recommended_changes,
                        "implementation_steps": rec.implementation_steps,
                        "validation_queries": rec.validation_queries,
                        "evidence": rec.evidence,
                        "impact_percentage": rec.impact_percentage,
                        "confidence_score": rec.confidence_score,
                        "complexity": rec.complexity.value,
                        "estimated_time_hours": rec.estimated_time_hours,
                        "priority_score": rec.priority_score,
                    }
                )

            logger.info(f"Generated {len(formatted_recs)} query optimization recommendations")
            return {
                "total_recommendations": len(formatted_recs),
                "recommendations": formatted_recs,
            }

        except Exception as e:
            logger.error(f"Error generating query optimization recommendations: {e!s}")
            return f"Error generating query optimization recommendations: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_capacity_planning_recommendations():
        """Get hardware and capacity planning recommendations.

        Analyzes resource utilization trends and provides recommendations
        for scaling hardware resources, storage, and infrastructure.

        Returns:
            List of capacity planning recommendations
        """
        logger.info("Generating capacity planning recommendations")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector for context
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Create performance advisor
            advisor = create_performance_advisor(client, bottleneck_detector)

            # Get system context
            context = advisor._gather_system_context()

            # Get capacity planning recommendations
            capacity_recommendations = advisor.capacity_planner.analyze_capacity_requirements(
                context
            )

            # Format recommendations for response
            formatted_recs = []
            for rec in capacity_recommendations:
                formatted_recs.append(
                    {
                        "recommendation_id": rec.recommendation_id,
                        "title": rec.title,
                        "description": rec.description,
                        "current_capacity": rec.current_state,
                        "recommended_scaling": rec.recommended_changes,
                        "implementation_steps": rec.implementation_steps,
                        "impact_percentage": rec.impact_percentage,
                        "confidence_score": rec.confidence_score,
                        "complexity": rec.complexity.value,
                        "risk_level": rec.risk_level.value,
                        "estimated_time_hours": rec.estimated_time_hours,
                        "estimated_cost_savings": rec.estimated_cost_savings,
                        "implementation_cost": rec.implementation_cost,
                        "priority_score": rec.priority_score,
                        "business_impact": rec.business_impact,
                    }
                )

            logger.info(f"Generated {len(formatted_recs)} capacity planning recommendations")
            return {
                "total_recommendations": len(formatted_recs),
                "recommendations": formatted_recs,
            }

        except Exception as e:
            logger.error(f"Error generating capacity planning recommendations: {e!s}")
            return f"Error generating capacity planning recommendations: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def detect_performance_bottlenecks():
        """Detect current performance bottlenecks using AI analysis.

        Uses machine learning to identify active performance bottlenecks
        with confidence scoring and severity assessment.

        Returns:
            List of detected bottlenecks with analysis details
        """
        logger.info("Detecting performance bottlenecks with AI analysis")
        client = create_clickhouse_client()
        try:
            # Create bottleneck detector
            bottleneck_detector = create_ai_bottleneck_detector(client)

            # Detect bottlenecks
            bottlenecks = bottleneck_detector.detect_bottlenecks()

            # Format bottlenecks for response
            formatted_bottlenecks = []
            for bottleneck in bottlenecks:
                formatted_bottlenecks.append(
                    {
                        "category": bottleneck.signature.category.value,
                        "name": bottleneck.signature.name,
                        "description": bottleneck.signature.description,
                        "severity": bottleneck.severity.value,
                        "confidence": bottleneck.confidence,
                        "confidence_level": bottleneck.confidence_level.value,
                        "impact_percentage": bottleneck.estimated_performance_impact,
                        "trend_direction": bottleneck.trend_direction.value,
                        "affected_operations": bottleneck.affected_operations,
                        "immediate_actions": bottleneck.immediate_actions,
                        "optimization_recommendations": bottleneck.optimization_recommendations,
                        "evidence": bottleneck.triggering_events,
                        "detection_timestamp": bottleneck.detection_timestamp.isoformat(),
                    }
                )

            # Calculate system health score
            health_score = bottleneck_detector.calculate_system_health_score()

            logger.info(
                f"Detected {len(bottlenecks)} bottlenecks, system health: {health_score.overall_score:.1f}"
            )
            return {
                "total_bottlenecks": len(bottlenecks),
                "bottlenecks": formatted_bottlenecks,
                "system_health_score": health_score.overall_score,
                "component_health": health_score.component_scores,
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error detecting performance bottlenecks: {e!s}")
            return f"Error detecting performance bottlenecks: {format_exception(e)}"


def register_profile_events_tools(mcp):
    """Register ProfileEvents analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def analyze_profile_events_comprehensive(hours: int = 24):
        """Perform comprehensive ProfileEvents analysis by category.

        Analyzes all ProfileEvents across different categories to provide
        a complete breakdown of system activity and performance metrics.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Comprehensive ProfileEvents analysis by category
        """
        logger.info(f"Performing comprehensive ProfileEvents analysis for the past {hours} hours")
        client = create_clickhouse_client()
        try:
            analyzer = ProfileEventsAnalyzer(client)
            return analyzer.analyze_comprehensive(hours)
        except Exception as e:
            logger.error(f"Error in comprehensive ProfileEvents analysis: {e!s}")
            return f"Error in comprehensive ProfileEvents analysis: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def get_profile_events_by_category(category: str, hours: int = 24, limit: int = 50):
        """Get ProfileEvents filtered by specific category.

        Args:
            category: ProfileEvents category (e.g., 'query_execution', 'memory_allocation')
            hours: Number of hours to analyze (default: 24)
            limit: Maximum number of events to return (default: 50)

        Returns:
            ProfileEvents analysis for the specified category
        """
        logger.info(f"Getting ProfileEvents for category '{category}' over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Validate category
            try:
                category_enum = ProfileEventsCategory(category)
            except ValueError:
                return f"Invalid category '{category}'. Valid categories: {[c.value for c in ProfileEventsCategory]}"

            return get_profile_events_by_category(client, category_enum, hours, limit)
        except Exception as e:
            logger.error(f"Error getting ProfileEvents by category: {e!s}")
            return f"Error getting ProfileEvents by category: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def compare_profile_events_time_periods(
        period1_hours: int = 24, period2_hours: int = 24, period2_offset_hours: int = 24
    ):
        """Compare ProfileEvents between two time periods.

        Args:
            period1_hours: Duration of first period in hours (default: 24)
            period2_hours: Duration of second period in hours (default: 24)
            period2_offset_hours: How many hours back to start period2 (default: 24)

        Returns:
            Side-by-side comparison of ProfileEvents between time periods
        """
        logger.info(
            f"Comparing ProfileEvents: period1({period1_hours}h) vs period2({period2_hours}h, {period2_offset_hours}h offset)"
        )
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            period1_start = end_time - timedelta(hours=period1_hours)
            period1_end = end_time

            period2_end = end_time - timedelta(hours=period2_offset_hours)
            period2_start = period2_end - timedelta(hours=period2_hours)

            # Create analyzer and comparator
            analyzer = ProfileEventsAnalyzer(client)
            comparator = ProfileEventsComparator(analyzer)

            # Get available events to compare
            available_events = analyzer.get_available_profile_events(days=1)
            event_names = available_events[:50]  # Limit to top 50 events

            comparisons = comparator.compare_time_periods(
                event_names, period2_start, period2_end, period1_start, period1_end
            )

            return {
                "period1": {
                    "start": period1_start.isoformat(),
                    "end": period1_end.isoformat(),
                    "hours": period1_hours,
                },
                "period2": {
                    "start": period2_start.isoformat(),
                    "end": period2_end.isoformat(),
                    "hours": period2_hours,
                },
                "comparisons": [
                    {
                        "event_name": comp.event_name,
                        "baseline_value": comp.baseline_value,
                        "comparison_value": comp.comparison_value,
                        "change_percentage": comp.change_percentage,
                        "is_anomaly": comp.is_anomaly,
                        "significance_score": comp.significance_score,
                        "anomaly_reason": comp.anomaly_reason,
                    }
                    for comp in comparisons
                ],
                "total_events_compared": len(comparisons),
                "anomalies_detected": len([c for c in comparisons if c.is_anomaly]),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error comparing ProfileEvents time periods: {e!s}")
            return f"Error comparing ProfileEvents time periods: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def detect_profile_events_anomalies(hours: int = 24, threshold_multiplier: float = 3.0):
        """Detect anomalies in ProfileEvents data.

        Args:
            hours: Number of hours to analyze (default: 24)
            threshold_multiplier: Anomaly detection sensitivity (default: 3.0)

        Returns:
            Detected ProfileEvents anomalies with severity and impact assessment
        """
        logger.info(f"Detecting ProfileEvents anomalies over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            return detect_profile_event_anomalies(client, hours, threshold_multiplier)
        except Exception as e:
            logger.error(f"Error detecting ProfileEvents anomalies: {e!s}")
            return f"Error detecting ProfileEvents anomalies: {format_exception(e)}"


def register_performance_diagnostics_tools(mcp):
    """Register performance diagnostics tools."""

    @mcp.tool()
    @trace_mcp_call
    def analyze_query_execution_performance(hours: int = 24):
        """Deep query performance analysis with execution bottleneck detection.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Comprehensive query execution performance analysis
        """
        logger.info(f"Analyzing query execution performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            analyzer = QueryExecutionAnalyzer(client)
            return analyzer.analyze_query_execution(hours)
        except Exception as e:
            logger.error(f"Error analyzing query execution performance: {e!s}")
            return f"Error analyzing query execution performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_io_performance(hours: int = 24):
        """Analyze I/O operations and detect bottlenecks.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            I/O performance analysis with bottleneck identification
        """
        logger.info(f"Analyzing I/O performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            analyzer = IOPerformanceAnalyzer(client)
            return analyzer.analyze_io_performance(hours)
        except Exception as e:
            logger.error(f"Error analyzing I/O performance: {e!s}")
            return f"Error analyzing I/O performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_cache_efficiency(hours: int = 24):
        """Analyze cache performance and optimization opportunities.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Cache efficiency analysis with optimization recommendations
        """
        logger.info(f"Analyzing cache efficiency over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            analyzer = CacheAnalyzer(client)
            return analyzer.analyze_cache_efficiency(hours)
        except Exception as e:
            logger.error(f"Error analyzing cache efficiency: {e!s}")
            return f"Error analyzing cache efficiency: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def generate_performance_diagnostic_report(hours: int = 24):
        """Generate comprehensive performance diagnostic report.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Complete performance diagnostic report with all analysis categories
        """
        logger.info(f"Generating performance diagnostic report for the past {hours} hours")
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            engine = PerformanceDiagnosticEngine(client)
            report = engine.generate_comprehensive_report(start_time, end_time)

            # Convert to dictionary format for MCP response
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hours": hours,
                },
                "query_execution": {
                    "function_performance": report.query_analysis.function_performance,
                    "null_handling": report.query_analysis.null_handling_efficiency,
                    "memory_allocation": report.query_analysis.memory_allocation_patterns,
                    "primary_key_usage": report.query_analysis.primary_key_usage,
                },
                "io_performance": {
                    "file_operations": report.io_analysis.file_operations,
                    "network_performance": report.io_analysis.network_performance,
                    "disk_performance": report.io_analysis.disk_performance,
                },
                "cache_analysis": {
                    "mark_cache": report.cache_analysis.mark_cache,
                    "uncompressed_cache": report.cache_analysis.uncompressed_cache,
                    "query_cache": report.cache_analysis.query_cache,
                    "overall_score": report.cache_analysis.overall_score,
                },
                "bottlenecks": [
                    {
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "impact": b.impact_percentage,
                        "recommendations": b.recommendations,
                    }
                    for b in report.bottlenecks
                ],
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating performance diagnostic report: {e!s}")
            return f"Error generating performance diagnostic report: {format_exception(e)}"


def register_storage_cloud_tools(mcp):
    """Register storage and cloud analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def analyze_s3_storage_performance(hours: int = 24):
        """Analyze S3 storage operations and cost optimization.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            S3 storage performance analysis with cost optimization recommendations
        """
        logger.info(f"Analyzing S3 storage performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = S3StorageAnalyzer(profile_analyzer)

            s3_analysis = analyzer.analyze_s3_performance(hours)

            # Convert to dictionary format
            return {
                "analysis_period_hours": hours,
                "cost_efficiency_score": s3_analysis.cost_efficiency_score,
                "performance_score": s3_analysis.performance_score,
                "operation_analysis": s3_analysis.operation_analysis,
                "throughput_analysis": s3_analysis.throughput_analysis,
                "latency_analysis": s3_analysis.latency_analysis,
                "error_analysis": s3_analysis.error_analysis,
                "cost_optimization": s3_analysis.cost_optimization,
                "performance_issues": [
                    {
                        "issue": issue.issue.value,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "impact": issue.impact_percentage,
                        "recommendations": issue.recommendations,
                    }
                    for issue in s3_analysis.performance_issues
                ],
                "recommendations": s3_analysis.recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing S3 storage performance: {e!s}")
            return f"Error analyzing S3 storage performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_azure_storage_performance(hours: int = 24):
        """Analyze Azure blob storage operations and performance.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Azure storage performance analysis with optimization recommendations
        """
        logger.info(f"Analyzing Azure storage performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = AzureStorageAnalyzer(profile_analyzer)

            azure_analysis = analyzer.analyze_azure_performance(hours)

            # Convert to dictionary format
            return {
                "analysis_period_hours": hours,
                "cost_efficiency_score": azure_analysis.cost_efficiency_score,
                "performance_score": azure_analysis.performance_score,
                "operation_analysis": azure_analysis.operation_analysis,
                "throughput_analysis": azure_analysis.throughput_analysis,
                "latency_analysis": azure_analysis.latency_analysis,
                "error_analysis": azure_analysis.error_analysis,
                "tier_optimization": azure_analysis.tier_optimization,
                "performance_issues": [
                    {
                        "issue": issue.issue.value,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "impact": issue.impact_percentage,
                        "recommendations": issue.recommendations,
                    }
                    for issue in azure_analysis.performance_issues
                ],
                "recommendations": azure_analysis.recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing Azure storage performance: {e!s}")
            return f"Error analyzing Azure storage performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_compression_efficiency(hours: int = 24):
        """Analyze compression performance and data integrity.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Compression efficiency analysis with performance recommendations
        """
        logger.info(f"Analyzing compression efficiency over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = CompressionAnalyzer(profile_analyzer)

            compression_analysis = analyzer.analyze_compression_performance(hours)

            # Convert to dictionary format
            return {
                "analysis_period_hours": hours,
                "efficiency_score": compression_analysis.efficiency_score,
                "compression_ratio_analysis": compression_analysis.compression_ratio_analysis,
                "algorithm_performance": compression_analysis.algorithm_performance,
                "storage_savings": compression_analysis.storage_savings,
                "cpu_impact": compression_analysis.cpu_impact,
                "integrity_analysis": compression_analysis.integrity_analysis,
                "performance_issues": [
                    {
                        "issue": issue.issue.value,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "impact": issue.impact_percentage,
                        "recommendations": issue.recommendations,
                    }
                    for issue in compression_analysis.performance_issues
                ],
                "recommendations": compression_analysis.recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing compression efficiency: {e!s}")
            return f"Error analyzing compression efficiency: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def identify_storage_optimizations(hours: int = 24):
        """Identify storage optimization opportunities across all storage types.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Storage optimization recommendations with cost and performance impact
        """
        logger.info(f"Identifying storage optimizations over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Create engine with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            engine = StorageOptimizationEngine(profile_analyzer)

            optimization_report = engine.generate_optimization_report(hours)

            # Convert to dictionary format
            return {
                "analysis_period_hours": hours,
                "overall_efficiency_score": optimization_report.overall_efficiency_score,
                "s3_optimization": {
                    "efficiency_score": optimization_report.s3_analysis.cost_efficiency_score,
                    "recommendations": optimization_report.s3_analysis.recommendations,
                    "cost_optimization": optimization_report.s3_analysis.cost_optimization,
                },
                "azure_optimization": {
                    "efficiency_score": optimization_report.azure_analysis.cost_efficiency_score,
                    "recommendations": optimization_report.azure_analysis.recommendations,
                    "tier_optimization": optimization_report.azure_analysis.tier_optimization,
                },
                "compression_optimization": {
                    "efficiency_score": optimization_report.compression_analysis.efficiency_score,
                    "recommendations": optimization_report.compression_analysis.recommendations,
                    "storage_savings": optimization_report.compression_analysis.storage_savings,
                },
                "cost_savings_potential": optimization_report.cost_savings_potential,
                "performance_improvements": optimization_report.performance_improvements,
                "implementation_priorities": optimization_report.implementation_priorities,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error identifying storage optimizations: {e!s}")
            return f"Error identifying storage optimizations: {format_exception(e)}"


def register_distributed_systems_tools(mcp):
    """Register distributed systems analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def analyze_zookeeper_health(hours: int = 24):
        """Analyze ZooKeeper performance and reliability.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            ZooKeeper health analysis with performance recommendations
        """
        logger.info(f"Analyzing ZooKeeper health over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Use ProfileEvents analysis for ZooKeeper operations
            analyzer = ProfileEventsAnalyzer(client)
            return analyzer.analyze_zookeeper_operations(hours)
        except Exception as e:
            logger.error(f"Error analyzing ZooKeeper health: {e!s}")
            return f"Error analyzing ZooKeeper health: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_replication_health(hours: int = 24):
        """Analyze replication health and performance.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Replication health analysis with performance optimization recommendations
        """
        logger.info(f"Analyzing replication health over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Use ProfileEvents analysis for replication operations
            analyzer = ProfileEventsAnalyzer(client)
            return analyzer.analyze_replication_performance(hours)
        except Exception as e:
            logger.error(f"Error analyzing replication health: {e!s}")
            return f"Error analyzing replication health: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_distributed_query_performance(hours: int = 24):
        """Analyze distributed query performance across cluster nodes.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Distributed query performance analysis with cluster optimization recommendations
        """
        logger.info(f"Analyzing distributed query performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            # Use ProfileEvents analysis for distributed query operations
            analyzer = ProfileEventsAnalyzer(client)
            return analyzer.analyze_distributed_queries(hours)
        except Exception as e:
            logger.error(f"Error analyzing distributed query performance: {e!s}")
            return f"Error analyzing distributed query performance: {format_exception(e)}"


def register_hardware_diagnostics_tools(mcp):
    """Register hardware analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def analyze_cpu_performance(hours: int = 24):
        """Analyze CPU efficiency and bottleneck detection.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            CPU performance analysis with bottleneck identification and optimization recommendations
        """
        logger.info(f"Analyzing CPU performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = CPUAnalyzer(profile_analyzer)

            cpu_analysis = analyzer.analyze_cpu_performance(start_time, end_time)

            # Convert to dictionary format
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hours": hours,
                },
                "efficiency_score": cpu_analysis.efficiency_score,
                "utilization_metrics": cpu_analysis.utilization_metrics,
                "instruction_efficiency": cpu_analysis.instruction_efficiency,
                "cache_performance": cpu_analysis.cache_performance,
                "context_switching": cpu_analysis.context_switching,
                "scaling_analysis": cpu_analysis.scaling_analysis,
                "bottlenecks": [
                    {
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "impact": b.impact_percentage,
                        "recommendations": b.recommendations,
                    }
                    for b in cpu_analysis.bottlenecks
                ],
                "recommendations": cpu_analysis.recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing CPU performance: {e!s}")
            return f"Error analyzing CPU performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_memory_performance(hours: int = 24):
        """Analyze memory usage and allocation patterns.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Memory performance analysis with usage optimization recommendations
        """
        logger.info(f"Analyzing memory performance over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = MemoryAnalyzer(profile_analyzer)

            memory_analysis = analyzer.analyze_memory_performance(start_time, end_time)

            # Convert to dictionary format
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hours": hours,
                },
                "efficiency_score": memory_analysis.efficiency_score,
                "allocation_patterns": memory_analysis.allocation_patterns,
                "fragmentation_analysis": memory_analysis.fragmentation_analysis,
                "gc_performance": memory_analysis.gc_performance,
                "memory_leaks": memory_analysis.memory_leaks,
                "cache_behavior": memory_analysis.cache_behavior,
                "swap_analysis": memory_analysis.swap_analysis,
                "bottlenecks": [
                    {
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "impact": b.impact_percentage,
                        "recommendations": b.recommendations,
                    }
                    for b in memory_analysis.bottlenecks
                ],
                "recommendations": memory_analysis.recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing memory performance: {e!s}")
            return f"Error analyzing memory performance: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_thread_pool_efficiency(hours: int = 24):
        """Analyze thread pool performance and efficiency.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Thread pool efficiency analysis with configuration recommendations
        """
        logger.info(f"Analyzing thread pool efficiency over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Create analyzer with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            analyzer = ThreadPoolAnalyzer(profile_analyzer)

            thread_analysis = analyzer.analyze_thread_pool_performance(start_time, end_time)

            # Convert to dictionary format
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hours": hours,
                },
                "efficiency_score": thread_analysis.efficiency_score,
                "thread_utilization": thread_analysis.thread_utilization,
                "contention_analysis": thread_analysis.contention_analysis,
                "queue_efficiency": thread_analysis.queue_efficiency,
                "scaling_analysis": thread_analysis.scaling_analysis,
                "lock_contention": thread_analysis.lock_contention,
                "thread_migration": thread_analysis.thread_migration,
                "bottlenecks": [
                    {
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "impact": b.impact_percentage,
                        "recommendations": b.recommendations,
                    }
                    for b in thread_analysis.bottlenecks
                ],
                "recommendations": thread_analysis.recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error analyzing thread pool efficiency: {e!s}")
            return f"Error analyzing thread pool efficiency: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def generate_hardware_health_report(hours: int = 24):
        """Generate complete hardware analysis report.

        Args:
            hours: Number of hours to analyze (default: 24)

        Returns:
            Comprehensive hardware health report with all component analyses
        """
        logger.info(f"Generating hardware health report for the past {hours} hours")
        client = create_clickhouse_client()
        try:
            from datetime import datetime, timedelta

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Create engine with ProfileEventsAnalyzer
            profile_analyzer = ProfileEventsAnalyzer(client)
            engine = HardwareHealthEngine(profile_analyzer)

            report = engine.generate_comprehensive_report(start_time, end_time)

            # Convert to dictionary format
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hours": hours,
                },
                "overall_health_score": report.overall_health_score,
                "cpu_analysis": {
                    "efficiency_score": report.cpu_analysis.efficiency_score,
                    "utilization_metrics": report.cpu_analysis.utilization_metrics,
                    "instruction_efficiency": report.cpu_analysis.instruction_efficiency,
                    "cache_performance": report.cpu_analysis.cache_performance,
                    "recommendations": report.cpu_analysis.recommendations,
                },
                "memory_analysis": {
                    "efficiency_score": report.memory_analysis.efficiency_score,
                    "allocation_patterns": report.memory_analysis.allocation_patterns,
                    "fragmentation_analysis": report.memory_analysis.fragmentation_analysis,
                    "recommendations": report.memory_analysis.recommendations,
                },
                "thread_pool_analysis": {
                    "efficiency_score": report.thread_pool_analysis.efficiency_score,
                    "thread_utilization": report.thread_pool_analysis.thread_utilization,
                    "contention_analysis": report.thread_pool_analysis.contention_analysis,
                    "recommendations": report.thread_pool_analysis.recommendations,
                },
                "system_efficiency": report.system_efficiency,
                "capacity_planning": report.capacity_planning,
                "critical_bottlenecks": [
                    {
                        "type": b.type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "impact": b.impact_percentage,
                        "recommendations": b.recommendations,
                    }
                    for b in report.critical_bottlenecks
                ],
                "optimization_priorities": report.optimization_priorities,
                "performance_trends": report.performance_trends,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating hardware health report: {e!s}")
            return f"Error generating hardware health report: {format_exception(e)}"


def register_ai_powered_analysis_tools(mcp):
    """Register AI-powered analysis tools."""

    @mcp.tool()
    @trace_mcp_call
    def detect_performance_bottlenecks_advanced():
        """Advanced AI bottleneck detection with ML pattern recognition.

        Uses machine learning algorithms to identify complex performance patterns
        and predict potential bottlenecks before they impact system performance.

        Returns:
            Advanced bottleneck detection results with ML-based predictions
        """
        logger.info("Performing advanced AI bottleneck detection")
        client = create_clickhouse_client()
        try:
            detector = create_ai_bottleneck_detector(client)
            bottlenecks = detector.detect_bottlenecks()

            # Get predictive analysis
            predictive_metrics = detector.get_predictive_analysis()

            return {
                "current_bottlenecks": [
                    {
                        "category": b.signature.category.value,
                        "name": b.signature.name,
                        "severity": b.severity.value,
                        "confidence": b.confidence,
                        "impact_percentage": b.estimated_performance_impact,
                        "recommendations": b.optimization_recommendations,
                    }
                    for b in bottlenecks
                ],
                "predictive_analysis": {
                    "degradation_risk": predictive_metrics.degradation_risk,
                    "projected_bottlenecks": predictive_metrics.projected_bottlenecks,
                    "recommended_actions": predictive_metrics.recommended_preventive_actions,
                    "confidence_score": predictive_metrics.confidence_score,
                },
                "system_health_score": detector.calculate_system_health_score().overall_score,
            }
        except Exception as e:
            logger.error(f"Error in advanced bottleneck detection: {e!s}")
            return f"Error in advanced bottleneck detection: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def generate_performance_recommendations_advanced(focus_area: str = "all"):
        """Generate ML-based optimization recommendations.

        Args:
            focus_area: Specific area to focus on ('query', 'storage', 'hardware', 'all')

        Returns:
            AI-generated performance recommendations with impact predictions
        """
        logger.info(f"Generating advanced performance recommendations (focus: {focus_area})")
        client = create_clickhouse_client()
        try:
            bottleneck_detector = create_ai_bottleneck_detector(client)
            advisor = create_performance_advisor(client, bottleneck_detector)

            recommendations = advisor.generate_comprehensive_recommendations(max_recommendations=25)

            # Filter by focus area if specified
            if focus_area != "all":
                filtered_recs = {}
                for category, recs in recommendations.get(
                    "recommendations_by_category", {}
                ).items():
                    if focus_area.lower() in category.lower():
                        filtered_recs[category] = recs
                recommendations["recommendations_by_category"] = filtered_recs

            return recommendations
        except Exception as e:
            logger.error(f"Error generating advanced recommendations: {e!s}")
            return f"Error generating advanced recommendations: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def analyze_performance_patterns(pattern_type: str = "all", hours: int = 168):
        """Analyze performance patterns using AI pattern recognition.

        Args:
            pattern_type: Type of patterns to analyze ('Query', 'Memory', 'I/O', 'all')
            hours: Number of hours to analyze (default: 168 - one week)

        Returns:
            AI-powered pattern analysis with anomaly detection and trend predictions
        """
        logger.info(f"Analyzing performance patterns ({pattern_type}) over the past {hours} hours")
        client = create_clickhouse_client()
        try:
            pattern_engine = create_pattern_analysis_engine(client)

            if pattern_type.lower() == "all":
                # Analyze all pattern types
                results = {}
                for ptype in ["Query", "Memory", "I/O", "Cache"]:
                    try:
                        results[ptype.lower()] = pattern_engine.analyze_patterns(ptype, hours)
                    except Exception as e:
                        logger.warning(f"Failed to analyze {ptype} patterns: {e!s}")
                        results[ptype.lower()] = f"Analysis failed: {e!s}"
                return results
            else:
                return pattern_engine.analyze_patterns(pattern_type, hours)
        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e!s}")
            return f"Error analyzing performance patterns: {format_exception(e)}"

    @mcp.tool()
    @trace_mcp_call
    def predict_performance_trends(prediction_hours: int = 24, confidence_threshold: float = 0.7):
        """Predict performance trends using machine learning.

        Args:
            prediction_hours: Number of hours to predict ahead (default: 24)
            confidence_threshold: Minimum confidence for predictions (default: 0.7)

        Returns:
            Performance trend predictions with confidence scores and recommended actions
        """
        logger.info(f"Predicting performance trends for the next {prediction_hours} hours")
        client = create_clickhouse_client()
        try:
            detector = create_ai_bottleneck_detector(client)
            predictive_metrics = detector.get_predictive_analysis()

            # Get pattern analysis for trend prediction
            pattern_engine = create_pattern_analysis_engine(client)
            anomaly_engine = create_anomaly_detection_engine(client)

            # Analyze current trends
            trend_data = pattern_engine.get_trend_analysis(lookback_hours=72)
            anomaly_summary = anomaly_engine.get_anomaly_summary(lookback_hours=24)

            return {
                "prediction_horizon_hours": prediction_hours,
                "confidence_threshold": confidence_threshold,
                "predicted_trends": {
                    "performance_degradation_risk": predictive_metrics.degradation_risk,
                    "projected_issues": predictive_metrics.projected_bottlenecks,
                    "confidence_score": predictive_metrics.confidence_score,
                },
                "current_trends": trend_data,
                "anomaly_indicators": anomaly_summary,
                "recommended_preventive_actions": predictive_metrics.recommended_preventive_actions,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error predicting performance trends: {e!s}")
            return f"Error predicting performance trends: {format_exception(e)}"
