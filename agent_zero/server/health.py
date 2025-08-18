"""Production-grade health check system for Agent Zero MCP Server.

This module implements comprehensive health checks following 2025 best practices
for production monitoring, including dependency health, readiness checks, and
detailed status reporting.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""

    name: str
    check_func: Callable[[], dict[str, Any]]
    timeout: float = 5.0
    interval: float = 30.0
    critical: bool = True
    description: str = ""
    last_check: datetime | None = None
    last_result: dict[str, Any] | None = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class ServiceHealth:
    """Health status for a service component."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OverallHealth:
    """Overall system health status."""

    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str
    deployment_mode: str
    services: list[ServiceHealth]
    metrics: dict[str, Any] = field(default_factory=dict)


class HealthCheckManager:
    """Manages all health checks for the MCP server."""

    def __init__(self, clickhouse_client_factory: Callable | None = None):
        """Initialize health check manager.

        Args:
            clickhouse_client_factory: Factory function to create ClickHouse clients
        """
        self.start_time = datetime.now()
        self.checks: dict[str, HealthCheck] = {}
        self.clickhouse_client_factory = clickhouse_client_factory
        self._register_default_checks()
        self._background_task: asyncio.Task | None = None
        self._running = False

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # System health check
        self.register_check(
            "system",
            self._check_system_health,
            description="Basic system health indicators",
            critical=True,
            timeout=2.0,
        )

        # ClickHouse connectivity check
        if self.clickhouse_client_factory:
            self.register_check(
                "clickhouse",
                self._check_clickhouse_health,
                description="ClickHouse database connectivity and basic operations",
                critical=True,
                timeout=10.0,
            )

        # Memory and resource check
        self.register_check(
            "resources",
            self._check_resource_health,
            description="Memory and CPU resource utilization",
            critical=False,
            timeout=3.0,
        )

    def register_check(
        self,
        name: str,
        check_func: Callable[[], dict[str, Any]],
        description: str = "",
        critical: bool = True,
        timeout: float = 5.0,
        interval: float = 30.0,
        max_failures: int = 3,
    ) -> None:
        """Register a new health check.

        Args:
            name: Unique name for the health check
            check_func: Function that performs the health check
            description: Description of what the check validates
            critical: Whether this check affects overall health status
            timeout: Maximum time to wait for check completion
            interval: How often to run the check in background
            max_failures: Number of consecutive failures before marking unhealthy
        """
        self.checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            description=description,
            critical=critical,
            timeout=timeout,
            interval=interval,
            max_failures=max_failures,
        )

    async def start_background_checks(self) -> None:
        """Start background health check monitoring."""
        if self._background_task and not self._background_task.done():
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_check_loop())
        logger.info("Started background health check monitoring")

    async def stop_background_checks(self) -> None:
        """Stop background health check monitoring."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background health check monitoring")

    async def _background_check_loop(self) -> None:
        """Background loop for periodic health checks."""
        try:
            while self._running:
                for check in self.checks.values():
                    if not self._running:
                        break

                    # Check if it's time to run this check
                    if check.last_check is None or datetime.now() - check.last_check >= timedelta(
                        seconds=check.interval
                    ):
                        try:
                            await self._run_single_check(check)
                        except Exception as e:
                            logger.error(f"Error in background health check {check.name}: {e}")

                # Wait a bit before next iteration
                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            logger.debug("Background health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in background health check loop: {e}")

    async def _run_single_check(self, check: HealthCheck) -> ServiceHealth:
        """Run a single health check with timeout."""
        start_time = time.time()

        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(check.check_func), timeout=check.timeout
            )

            response_time = (time.time() - start_time) * 1000

            # Update check state
            check.last_check = datetime.now()
            check.last_result = result
            check.consecutive_failures = 0

            # Determine status from result
            status = HealthStatus(result.get("status", "healthy"))
            message = result.get("message", "Check completed successfully")

            return ServiceHealth(
                name=check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=result.get("details", {}),
            )

        except TimeoutError:
            check.consecutive_failures += 1
            response_time = check.timeout * 1000

            return ServiceHealth(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout}s",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={"timeout": True, "consecutive_failures": check.consecutive_failures},
            )

        except Exception as e:
            check.consecutive_failures += 1
            response_time = (time.time() - start_time) * 1000

            return ServiceHealth(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={"error": str(e), "consecutive_failures": check.consecutive_failures},
            )

    async def get_health_status(self, include_details: bool = True) -> OverallHealth:
        """Get current overall health status.

        Args:
            include_details: Whether to include detailed service health information

        Returns:
            Overall health status with service details
        """
        services = []
        overall_status = HealthStatus.HEALTHY

        # Run all health checks
        for check in self.checks.values():
            service_health = await self._run_single_check(check)
            services.append(service_health)

            # Determine overall status based on critical checks
            if check.critical:
                if service_health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    service_health.status == HealthStatus.DEGRADED
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()

        # Gather system metrics
        metrics = {
            "checks_total": len(self.checks),
            "critical_checks": sum(1 for c in self.checks.values() if c.critical),
            "healthy_checks": sum(1 for s in services if s.status == HealthStatus.HEALTHY),
            "degraded_checks": sum(1 for s in services if s.status == HealthStatus.DEGRADED),
            "unhealthy_checks": sum(1 for s in services if s.status == HealthStatus.UNHEALTHY),
        }

        return OverallHealth(
            status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            version=__version__,
            deployment_mode="production",  # TODO: Get from config
            services=services if include_details else [],
            metrics=metrics,
        )

    async def get_readiness_status(self) -> dict[str, Any]:
        """Get readiness status for load balancer health checks.

        Returns:
            Simple readiness status suitable for load balancer probes
        """
        health = await self.get_health_status(include_details=False)

        # Ready if overall status is healthy or degraded
        ready = health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        return {
            "ready": ready,
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
        }

    async def get_liveness_status(self) -> dict[str, Any]:
        """Get liveness status for orchestrator health checks.

        Returns:
            Simple liveness status for orchestrator probes
        """
        # Liveness is based on the service being able to respond
        # and having been running for a reasonable time
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "alive": True,
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
        }

    def _check_system_health(self) -> dict[str, Any]:
        """Check basic system health indicators."""
        try:
            import psutil

            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Determine status based on resource usage
            status = "healthy"
            message = "System resources within normal limits"

            if cpu_percent > 90 or memory.percent > 90:
                status = "unhealthy"
                message = "High resource utilization detected"
            elif cpu_percent > 75 or memory.percent > 75:
                status = "degraded"
                message = "Elevated resource utilization"

            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available // (1024 * 1024),
                    "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
                },
            }

        except ImportError:
            # psutil not available, use basic check
            return {
                "status": "healthy",
                "message": "Basic system check completed (psutil not available)",
                "details": {"psutil_available": False},
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"System health check failed: {e!s}",
                "details": {"error": str(e)},
            }

    def _check_clickhouse_health(self) -> dict[str, Any]:
        """Check ClickHouse database connectivity and basic operations."""
        if not self.clickhouse_client_factory:
            return {
                "status": "unknown",
                "message": "ClickHouse client factory not configured",
                "details": {},
            }

        try:
            # Get a client and test basic connectivity
            client = self.clickhouse_client_factory()

            # Test basic query
            result = client.query("SELECT 1 as test").result_rows

            if result and result[0][0] == 1:
                # Test server version and status
                version_result = client.query("SELECT version()").result_rows
                version = version_result[0][0] if version_result else "unknown"

                # Test server health
                uptime_result = client.query("SELECT uptime()").result_rows
                uptime = uptime_result[0][0] if uptime_result else 0

                return {
                    "status": "healthy",
                    "message": "ClickHouse connectivity verified",
                    "details": {
                        "server_version": version,
                        "server_uptime_seconds": uptime,
                        "test_query_successful": True,
                    },
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "ClickHouse test query returned unexpected result",
                    "details": {"test_result": result},
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"ClickHouse health check failed: {e!s}",
                "details": {"error": str(e), "error_type": type(e).__name__},
            }

    def _check_resource_health(self) -> dict[str, Any]:
        """Check memory and resource utilization."""
        try:
            import gc

            import psutil

            # Get detailed memory information
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            # Get Python garbage collection stats
            gc_stats = gc.get_stats()

            # Determine status
            status = "healthy"
            message = "Resource utilization normal"

            if memory.percent > 85:
                status = "degraded"
                message = "High memory utilization"

            if process_memory.rss > 1024 * 1024 * 1024:  # 1GB
                if status == "healthy":
                    status = "degraded"
                    message = "High process memory usage"

            return {
                "status": status,
                "message": message,
                "details": {
                    "system_memory_percent": memory.percent,
                    "system_memory_available_mb": memory.available // (1024 * 1024),
                    "process_memory_rss_mb": process_memory.rss // (1024 * 1024),
                    "process_memory_vms_mb": process_memory.vms // (1024 * 1024),
                    "gc_collections": (
                        [stat["collections"] for stat in gc_stats] if gc_stats else []
                    ),
                    "gc_collected": [stat["collected"] for stat in gc_stats] if gc_stats else [],
                },
            }

        except ImportError:
            return {
                "status": "unknown",
                "message": "Resource monitoring unavailable (psutil not installed)",
                "details": {"psutil_available": False},
            }
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"Resource check partially failed: {e!s}",
                "details": {"error": str(e)},
            }


# Global health check manager instance
health_manager: HealthCheckManager | None = None


def get_health_manager(clickhouse_client_factory: Callable | None = None) -> HealthCheckManager:
    """Get or create the global health check manager.

    Args:
        clickhouse_client_factory: Factory function to create ClickHouse clients

    Returns:
        Global health check manager instance
    """
    global health_manager
    if health_manager is None:
        health_manager = HealthCheckManager(clickhouse_client_factory)
    return health_manager


def reset_health_manager() -> None:
    """Reset the global health check manager (useful for testing)."""
    global health_manager
    if health_manager:
        # Stop background checks if running
        if health_manager._background_task and not health_manager._background_task.done():
            health_manager._running = False
    health_manager = None
