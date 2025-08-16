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
]
