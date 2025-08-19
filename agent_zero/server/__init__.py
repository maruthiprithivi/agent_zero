"""MCP server components for Agent Zero.

This module contains the refactored MCP server implementation split into
focused, maintainable components following the standards in CLAUDE.md.
"""

from .backup import BackupManager, get_backup_manager
from .client import create_clickhouse_client
from .core import run

# Production operations components
from .health import HealthCheckManager, get_health_manager
from .logging import get_logger, set_correlation_id
from .metrics import MetricsManager, get_metrics_manager
from .performance import PerformanceMonitor, get_performance_monitor
from .production import ProductionMCPServer, run_production_server
from .tools import register_all_tools
from .tracing import TracingManager, get_tracing_manager

# Import query module for compatibility
from . import query


# Import individual MCP tool functions that tests depend on
# These are exported from tools.py but need to be available at server level
def list_databases():
    """List all databases in the ClickHouse server."""
    from .tools import create_clickhouse_client

    client = create_clickhouse_client()
    try:
        result = client.command("SHOW DATABASES")
        databases = result.split("\n") if isinstance(result, str) else [row[0] for row in result]
        return {"databases": [db for db in databases if db.strip()]}
    except Exception as e:
        return {"error": f"Failed to list databases: {e!s}"}


def monitor_blob_storage_stats(days: int = 7):
    """Get statistics for blob storage operations."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.system_components import get_blob_storage_stats

    client = create_clickhouse_client()
    try:
        return get_blob_storage_stats(client, days)
    except Exception as e:
        return {"error": f"Failed to get blob storage stats: {e!s}"}


def monitor_materialized_view_stats(days: int = 7):
    """Get statistics for materialized view queries."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.system_components import get_mv_query_stats

    client = create_clickhouse_client()
    try:
        return get_mv_query_stats(client, days)
    except Exception as e:
        return {"error": f"Failed to get materialized view stats: {e!s}"}


def monitor_s3queue_stats(days: int = 7):
    """Get statistics for S3 Queue operations."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.system_components import get_s3_queue_stats

    client = create_clickhouse_client()
    try:
        return get_s3_queue_stats(client, days)
    except Exception as e:
        return {"error": f"Failed to get S3 queue stats: {e!s}"}


def monitor_table_inactive_parts(database: str, table: str):
    """Get information about inactive parts for a table."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.table_statistics import get_table_inactive_parts

    client = create_clickhouse_client()
    try:
        return get_table_inactive_parts(client, database, table)
    except Exception as e:
        return {"error": f"Failed to get table inactive parts: {e!s}"}


def monitor_table_stats(database: str, table: str = None):
    """Get detailed statistics for tables."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.table_statistics import get_table_stats

    client = create_clickhouse_client()
    try:
        return get_table_stats(client, database, table)
    except Exception as e:
        return {"error": f"Failed to get table stats: {e!s}"}


def generate_table_drop_script(database: str):
    """Generate a script to drop all tables in a database."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.utility_tools import generate_drop_tables_script

    client = create_clickhouse_client()
    try:
        return generate_drop_tables_script(client, database)
    except Exception as e:
        return {"error": f"Failed to generate drop script: {e!s}"}


def list_user_defined_functions():
    """Get information about user-defined functions."""
    from .tools import create_clickhouse_client
    from agent_zero.monitoring.utility_tools import get_user_defined_functions

    client = create_clickhouse_client()
    try:
        return get_user_defined_functions(client)
    except Exception as e:
        return {"error": f"Failed to list user-defined functions: {e!s}"}


__all__ = [
    # Core components
    "create_clickhouse_client",
    "register_all_tools",
    "run",
    # Production operations
    "get_health_manager",
    "HealthCheckManager",
    "get_metrics_manager",
    "MetricsManager",
    "get_logger",
    "set_correlation_id",
    "get_tracing_manager",
    "TracingManager",
    "get_performance_monitor",
    "PerformanceMonitor",
    "get_backup_manager",
    "BackupManager",
    "ProductionMCPServer",
    "run_production_server",
    # MCP tool functions for test compatibility
    "list_databases",
    "monitor_blob_storage_stats",
    "monitor_materialized_view_stats",
    "monitor_s3queue_stats",
    "monitor_table_inactive_parts",
    "monitor_table_stats",
    "generate_table_drop_script",
    "list_user_defined_functions",
    # Query module
    "query",
]
