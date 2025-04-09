import atexit
import concurrent.futures
import logging
import os
import sys
import time
from collections.abc import Sequence
from typing import Dict, Any, Optional

import clickhouse_connect
from clickhouse_connect.driver.binding import format_query_value, quote_identifier
from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP

    logging.getLogger("agent_zero").debug("Successfully imported FastMCP")
except ImportError as e:
    logging.getLogger("agent_zero").error(f"Failed to import FastMCP: {e}")
    sys.stderr.write(f"ERROR: Failed to import FastMCP: {e}\n")
    sys.stderr.write(f"Python path: {sys.path}\n")

from agent_zero.mcp_env import config
from agent_zero.mcp_tracer import trace_mcp_call
from agent_zero.monitoring import (
    # Utility
    generate_drop_tables_script,
    # Insert Operations
    get_async_insert_stats,
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
    get_memory_usage,
    get_merge_stats,
    get_mv_query_stats,
    get_normalized_query_stats,
    get_part_log_events,
    get_partition_stats,
    get_parts_analysis,
    get_query_duration_stats,
    get_query_kind_breakdown,
    get_recent_errors,
    get_recent_insert_queries,
    get_s3queue_stats,
    get_server_sizing,
    # Table Statistics
    get_table_inactive_parts,
    get_table_stats,
    get_text_log,
    get_uptime,
    get_user_defined_functions,
)
from agent_zero.utils import format_exception

# Import monitoring utilities but don't set them up yet
from agent_zero.monitoring_endpoints import setup_monitoring_endpoints
from agent_zero.server_config import ServerConfig

MCP_SERVER_NAME = "mcp-clickhouse"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(MCP_SERVER_NAME)

QUERY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10)
atexit.register(lambda: QUERY_EXECUTOR.shutdown(wait=True))
SELECT_QUERY_TIMEOUT_SECS = 30

load_dotenv()

# Log environment information
logger.debug(f"Environment variables: {os.environ.keys()}")
try:
    import pkg_resources

    installed_packages = [f"{p.key}=={p.version}" for p in pkg_resources.working_set]
    logger.debug(f"Installed packages: {installed_packages}")
except Exception as e:
    logger.error(f"Error listing packages: {e}")

deps = [
    "clickhouse-connect",
    "python-dotenv",
    "uvicorn",
    "pip-system-certs",
]

try:
    logger.debug(f"Creating FastMCP with server name: {MCP_SERVER_NAME}")
    mcp = FastMCP(MCP_SERVER_NAME, dependencies=deps)
    logger.debug("Successfully created FastMCP instance")
except Exception as e:
    logger.error(f"Error creating FastMCP instance: {e}", exc_info=True)
    sys.stderr.write(f"ERROR: Failed to create FastMCP: {e}\n")
    raise

# Extract the FastAPI app from the FastMCP instance to add custom endpoints
app = getattr(mcp, "app", None)
if app:
    # Set up monitoring endpoints
    setup_monitoring_endpoints(app, create_clickhouse_client=lambda: create_clickhouse_client())
else:
    logger.warning("Could not access FastAPI app from FastMCP, monitoring endpoints not set up")


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
        return result
    except Exception as e:
        logger.error(f"Error listing databases: {e!s}")
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
            schema_query = f"DESCRIBE TABLE {quote_identifier(database)}.{quote_identifier(table)}"
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
        return tables
    except Exception as e:
        logger.error(f"Error listing tables in database '{database}': {e!s}")
        return f"Error listing tables: {format_exception(e)}"


def execute_query(query: str):
    """Execute a read-only SQL query.

    Args:
        query: The SQL query to execute.

    Returns:
        The query results as a list of dictionaries.
    """
    client = create_clickhouse_client()
    try:
        # Import the database logger if needed
        from agent_zero.database_logger import query_logger

        # Log the query if query logging is enabled
        if config.enable_query_logging:
            query_logger.log_query(query, None, {"readonly": 1})

        start_time = time.time()
        res = client.query(query, settings={"readonly": 1})

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
        if config.log_query_errors:
            query_logger.log_query_error(err, query)

        logger.error(f"Error executing query: {err}")
        return f"error running query: {format_exception(err)}"


@mcp.tool()
@trace_mcp_call
def run_select_query(query: str):
    """Execute a read-only SELECT query against the ClickHouse database.

    Args:
        query: The SQL query to execute (must be read-only).

    Returns:
        The query results as a list of dictionaries.
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


def create_clickhouse_client():
    """Create and return a ClickHouse client connection.

    Returns:
        A configured ClickHouse client instance.

    Raises:
        Exception: If connection fails.
    """
    client_config = config.get_client_config()
    logger.info(
        f"Creating ClickHouse client connection to {client_config['host']}:{client_config['port']} "
        f"as {client_config['username']} "
        f"(secure={client_config['secure']}, verify={client_config['verify']}, "
        f"connect_timeout={client_config['connect_timeout']}s, "
        f"send_receive_timeout={client_config['send_receive_timeout']}s)"
    )

    try:
        client = clickhouse_connect.get_client(**client_config)
        # Test the connection
        version = client.server_version
        logger.info(f"Successfully connected to ClickHouse server version {version}")

        # Enable instrumentation for logging if query logging is enabled
        if config.enable_query_logging or config.log_query_latency or config.log_query_errors:
            # Log a message indicating logging is active
            logger.info("Database query logging is enabled")

        return client
    except Exception as e:
        logger.error(f"Failed to connect to ClickHouse: {e!s}")
        raise


# ===== MONITORING TOOLS =====

# Query Performance Tools


@mcp.tool()
@trace_mcp_call
def monitor_current_processes():
    """Get information about currently running processes on the ClickHouse cluster.

    This function retrieves details about all currently running queries including resource usage,
    query type, and elapsed time.

    Returns:
        A list of dictionaries with information about each running process.
    """
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
    """Get query duration statistics grouped by hour.

    Args:
        query_kind: Filter by specific query kind (e.g., 'Select', 'Insert'), or None for all queries.
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with hourly query statistics.
    """
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
    """Identify the most resource-intensive query patterns.

    Args:
        days: Number of days to look back in history (default: 2).
        limit: Maximum number of query patterns to return (default: 50).

    Returns:
        A list of dictionaries with statistics for each query pattern.
    """
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
    """Get a breakdown of query types by hour.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with hourly query type breakdowns.
    """
    logger.info(f"Monitoring query types over the past {days} days")
    client = create_clickhouse_client()
    try:
        return get_query_kind_breakdown(client, days)
    except Exception as e:
        logger.error(f"Error monitoring query types: {e!s}")
        return f"Error monitoring query types: {format_exception(e)}"


# Resource Usage Tools


@mcp.tool()
@trace_mcp_call
def monitor_memory_usage(days: int = 7):
    """Get memory usage statistics over time by host.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with memory usage statistics.
    """
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
    """Get CPU usage statistics over time.

    Args:
        hours: Number of hours to look back in history (default: 3).

    Returns:
        A list of dictionaries with CPU usage statistics.
    """
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
    """Get server sizing information for all nodes in the cluster.

    Returns:
        A list of dictionaries with server sizing information.
    """
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
    """Get server uptime statistics.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with uptime statistics.
    """
    logger.info(f"Monitoring uptime over the past {days} days")
    client = create_clickhouse_client()
    try:
        return get_uptime(client, days)
    except Exception as e:
        logger.error(f"Error monitoring uptime: {e!s}")
        return f"Error monitoring uptime: {format_exception(e)}"


# Error Analysis Tools


@mcp.tool()
@trace_mcp_call
def monitor_recent_errors(days: int = 1):
    """Get recent errors from ClickHouse system.errors table.

    Args:
        days: Number of days to look back in history (default: 1).

    Returns:
        A list of dictionaries with recent error information.
    """
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
    """Get error stack traces for logical errors in the system.

    Returns:
        A list of dictionaries with error stack trace information.
    """
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
    """Get recent entries from the text log.

    Args:
        limit: Maximum number of log entries to return (default: 100).

    Returns:
        A list of dictionaries with text log entries.
    """
    logger.info(f"Viewing text log (limit: {limit})")
    client = create_clickhouse_client()
    try:
        return get_text_log(client, limit)
    except Exception as e:
        logger.error(f"Error viewing text log: {e!s}")
        return f"Error viewing text log: {format_exception(e)}"


# Insert Operations Tools


@mcp.tool()
@trace_mcp_call
def monitor_recent_insert_queries(days: int = 1, limit: int = 100):
    """Get recent insert queries.

    Args:
        days: Number of days to look back in history (default: 1).
        limit: Maximum number of queries to return (default: 100).

    Returns:
        A list of dictionaries with insert query information.
    """
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
    """Get asynchronous insert statistics.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with async insert statistics.
    """
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
    """Get distribution of written bytes for insert operations.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with insert bytes distribution statistics.
    """
    logger.info(f"Monitoring insert bytes distribution over the past {days} days")
    client = create_clickhouse_client()
    try:
        return get_insert_written_bytes_distribution(client, days)
    except Exception as e:
        logger.error(f"Error monitoring insert bytes distribution: {e!s}")
        return f"Error monitoring insert bytes distribution: {format_exception(e)}"


# Parts Merges Tools


@mcp.tool()
@trace_mcp_call
def monitor_current_merges():
    """Get information about currently running merge operations.

    Returns:
        A list of dictionaries with information about current merges.
    """
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
    """Get merge performance statistics.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with merge statistics.
    """
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
    """Get recent part log events.

    Args:
        days: Number of days to look back in history (default: 1).
        limit: Maximum number of events to return (default: 100).

    Returns:
        A list of dictionaries with part log events.
    """
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
    """Get partition statistics for a specific table.

    Args:
        database: Database name.
        table: Table name.

    Returns:
        A list of dictionaries with partition statistics.
    """
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
    """Get parts analysis for a specific table.

    Args:
        database: Database name.
        table: Table name.

    Returns:
        A list of dictionaries with parts analysis.
    """
    logger.info(f"Monitoring parts analysis for {database}.{table}")
    client = create_clickhouse_client()
    try:
        return get_parts_analysis(client, database, table)
    except Exception as e:
        logger.error(f"Error monitoring parts analysis: {e!s}")
        return f"Error monitoring parts analysis: {format_exception(e)}"


# System Components Tools


@mcp.tool()
@trace_mcp_call
def monitor_blob_storage_stats(days: int = 7):
    """Get statistics for blob storage operations.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with blob storage statistics.
    """
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
    """Get statistics for materialized view queries.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with materialized view query statistics.
    """
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
    """Get statistics for S3 Queue operations.

    Args:
        days: Number of days to look back in history (default: 7).

    Returns:
        A list of dictionaries with S3 Queue statistics.
    """
    logger.info(f"Monitoring S3 Queue stats over the past {days} days")
    client = create_clickhouse_client()
    try:
        return get_s3queue_stats(client, days)
    except Exception as e:
        logger.error(f"Error monitoring S3 Queue stats: {e!s}")
        return f"Error monitoring S3 Queue stats: {format_exception(e)}"


# Table Statistics Tools


@mcp.tool()
@trace_mcp_call
def monitor_table_stats(database: str, table: str = None):
    """Get detailed statistics for tables.

    Args:
        database: Database name.
        table: Table name (optional). If not provided, stats for all tables in the database are returned.

    Returns:
        A list of dictionaries with table statistics.
    """
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
    """Get information about inactive parts for a table.

    Args:
        database: Database name.
        table: Table name.

    Returns:
        A list of dictionaries with inactive parts information.
    """
    logger.info(f"Monitoring inactive parts for {database}.{table}")
    client = create_clickhouse_client()
    try:
        return get_table_inactive_parts(client, database, table)
    except Exception as e:
        logger.error(f"Error monitoring table inactive parts: {e!s}")
        return f"Error monitoring table inactive parts: {format_exception(e)}"


# Utility Tools


@mcp.tool()
@trace_mcp_call
def generate_table_drop_script(database: str):
    """Generate a script to drop all tables in a database.

    Args:
        database: Database name.

    Returns:
        A string containing SQL commands to drop all tables.
    """
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
    """Get information about user-defined functions.

    Returns:
        A list of dictionaries with user-defined function information.
    """
    logger.info("Listing user-defined functions")
    client = create_clickhouse_client()
    try:
        return get_user_defined_functions(client)
    except Exception as e:
        logger.error(f"Error listing user-defined functions: {e!s}")
        return f"Error listing user-defined functions: {format_exception(e)}"


# Customize run method to support our configurations
_original_run = mcp.run


def run(
    host: str = "127.0.0.1",
    port: int = 8505,
    ssl_config: Dict[str, Any] = None,
    server_config: Optional[ServerConfig] = None,
    _uvicorn_run=None,
    _setup_monitoring=None,
):
    """Run the MCP server with the specified configuration.

    Args:
        host: Host to bind to
        port: Port to bind to
        ssl_config: SSL configuration dictionary
        server_config: Server configuration instance
        _uvicorn_run: Function to use for running uvicorn (for testing)
        _setup_monitoring: Function to use for setting up monitoring (for testing)
    """
    # Extract SSL arguments for uvicorn
    ssl_args = {}
    if ssl_config:
        if "certfile" in ssl_config:
            ssl_args["ssl_certfile"] = ssl_config["certfile"]
        if "keyfile" in ssl_config:
            ssl_args["ssl_keyfile"] = ssl_config["keyfile"]

    # Use injected dependencies or import them
    uvicorn_run = _uvicorn_run or __import__("uvicorn").run
    setup_monitoring_endpoints_func = _setup_monitoring

    if setup_monitoring_endpoints_func is None:
        # Only import if not provided (for testing)
        from agent_zero.monitoring_endpoints import setup_monitoring_endpoints as setup_func

        setup_monitoring_endpoints_func = setup_func

    # Setup monitoring endpoints if server_config is provided and app exists
    if hasattr(mcp, "app"):
        if server_config or not ssl_config:  # Setup if server_config provided or no SSL
            setup_monitoring_endpoints_func(mcp.app, create_clickhouse_client)

    # Run with uvicorn directly
    return uvicorn_run("agent_zero.mcp_server:mcp.app", host=host, port=port, **ssl_args)


# Replace the original run method with our customized version
mcp.run = run

# Set up monitoring endpoints for the MCP server
if hasattr(mcp, "app"):
    setup_monitoring_endpoints(mcp.app, create_clickhouse_client)
