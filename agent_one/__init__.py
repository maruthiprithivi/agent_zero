"""Agent One package for ClickHouse database management."""

__version__ = "0.1.0"

from agent_one.mcp_server import (
    create_clickhouse_client,
    list_databases,
    list_tables,
    run_select_query,
)

__all__ = ["create_clickhouse_client", "list_databases", "list_tables", "run_select_query"]
