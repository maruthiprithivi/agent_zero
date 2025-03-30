import atexit
import concurrent.futures
import logging
from collections.abc import Sequence

import clickhouse_connect
from clickhouse_connect.driver.binding import format_query_value, quote_identifier
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agent_one.mcp_env import config
from agent_one.monitoring import (
    # Utility
    get_cpu_usage,
    get_current_processes,
    get_memory_usage,
    get_normalized_query_stats,
    get_query_duration_stats,
    get_query_kind_breakdown,
    # Error Analysis
    get_server_sizing,
    get_uptime,
)
from agent_one.utils import format_exception

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

deps = [
    "clickhouse-connect",
    "python-dotenv",
    "uvicorn",
    "pip-system-certs",
]

mcp = FastMCP(MCP_SERVER_NAME, dependencies=deps)


@mcp.tool()
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
        table_comments_query = f"SELECT name, comment FROM system.tables WHERE database = {format_query_value(database)}"
        table_comments_result = client.query(table_comments_query)
        table_comments = {row[0]: row[1] for row in table_comments_result.result_rows}

        # Get all column comments in one query
        column_comments_query = f"SELECT table, name, comment FROM system.columns WHERE database = {format_query_value(database)}"
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
        res = client.query(query, settings={"readonly": 1})
        column_names = res.column_names
        rows = []
        for row in res.result_rows:
            row_dict = {}
            for i, col_name in enumerate(column_names):
                row_dict[col_name] = row[i]
            rows.append(row_dict)
        logger.info(f"Query returned {len(rows)} rows")
        return rows
    except Exception as err:
        logger.error(f"Error executing query: {err}")
        return f"error running query: {format_exception(err)}"


@mcp.tool()
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
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ClickHouse: {e!s}")
        raise


# ===== MONITORING TOOLS =====

# Query Performance Tools


@mcp.tool()
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
