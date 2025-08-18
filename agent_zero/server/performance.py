"""Production-grade performance monitoring and resource management for Agent Zero MCP Server.

This module implements comprehensive performance monitoring, resource management,
connection pooling, caching, and auto-scaling recommendations following 2025 best practices.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports for performance monitoring
try:
    import psutil

    psutil_available = True
except ImportError:
    logger.warning("psutil not available. System metrics will be limited.")
    psutil_available = False
    psutil = None


class ResourceType(Enum):
    """Resource type enumeration."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE = "cache"


class AlertLevel(Enum):
    """Alert level enumeration."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_mb: float
    network_io_mb: float
    active_connections: int
    response_time_ms: float
    error_rate: float


@dataclass
class PerformanceAlert:
    """Performance alert information."""

    timestamp: datetime
    level: AlertLevel
    resource_type: ResourceType
    message: str
    current_value: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration."""

    min_size: int = 5
    max_size: int = 50
    acquire_timeout: float = 30.0
    max_idle_time: float = 300.0
    validate_on_acquire: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ConnectionPool:
    """Async connection pool with resource management."""

    def __init__(self, connection_factory: Callable, config: ConnectionPoolConfig | None = None):
        """Initialize connection pool.

        Args:
            connection_factory: Function to create new connections
            config: Pool configuration
        """
        self.connection_factory = connection_factory
        self.config = config or ConnectionPoolConfig()

        self._pool: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_size)
        self._active_connections: set[Any] = set()
        self._connection_times: dict[Any, datetime] = {}
        self._lock = asyncio.Lock()
        self._closed = False

        # Initialize minimum connections
        asyncio.create_task(self._initialize_pool())

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

    async def _initialize_pool(self) -> None:
        """Initialize minimum pool connections."""
        try:
            for _ in range(self.config.min_size):
                connection = await self._create_connection()
                if connection:
                    await self._pool.put(connection)
                    self._connection_times[connection] = datetime.now()
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")

    async def _create_connection(self) -> Any | None:
        """Create a new connection with retries."""
        for attempt in range(self.config.retry_attempts):
            try:
                connection = await asyncio.to_thread(self.connection_factory)
                logger.debug(f"Created new connection (attempt {attempt + 1})")
                return connection
            except Exception as e:
                logger.warning(f"Connection creation failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        logger.error("Failed to create connection after all retries")
        return None

    async def _validate_connection(self, connection: Any) -> bool:
        """Validate a connection.

        Args:
            connection: Connection to validate

        Returns:
            True if connection is valid
        """
        if not self.config.validate_on_acquire:
            return True

        try:
            # Basic validation - this should be customized per connection type
            if hasattr(connection, "ping"):
                await asyncio.to_thread(connection.ping)
            elif hasattr(connection, "execute"):
                await asyncio.to_thread(connection.execute, "SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        connection = None
        try:
            # Try to get from pool with timeout
            try:
                connection = await asyncio.wait_for(
                    self._pool.get(), timeout=self.config.acquire_timeout
                )
            except TimeoutError:
                logger.warning("Connection pool timeout, creating new connection")
                connection = await self._create_connection()
                if not connection:
                    raise RuntimeError("Failed to acquire connection")

            # Validate connection
            if not await self._validate_connection(connection):
                # Try to get another connection
                await self._close_connection(connection)
                connection = await self._create_connection()
                if not connection or not await self._validate_connection(connection):
                    raise RuntimeError("Failed to acquire valid connection")

            # Track active connection
            async with self._lock:
                self._active_connections.add(connection)
                self._connection_times[connection] = datetime.now()

            yield connection

        finally:
            if connection:
                async with self._lock:
                    self._active_connections.discard(connection)

                # Return to pool if space available
                try:
                    await self._pool.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool is full, close connection
                    await self._close_connection(connection)

    async def _close_connection(self, connection: Any) -> None:
        """Close a connection safely.

        Args:
            connection: Connection to close
        """
        try:
            if hasattr(connection, "close"):
                if asyncio.iscoroutinefunction(connection.close):
                    await connection.close()
                else:
                    await asyncio.to_thread(connection.close)
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self._connection_times.pop(connection, None)

    async def _cleanup_task(self) -> None:
        """Background task to clean up idle connections."""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Run cleanup every minute

                cutoff_time = datetime.now() - timedelta(seconds=self.config.max_idle_time)
                connections_to_close = []

                async with self._lock:
                    # Find idle connections
                    for conn, last_used in self._connection_times.items():
                        if (
                            conn not in self._active_connections
                            and last_used < cutoff_time
                            and self._pool.qsize() > self.config.min_size
                        ):
                            connections_to_close.append(conn)

                # Close idle connections
                for conn in connections_to_close:
                    try:
                        # Remove from queue if present
                        temp_queue = asyncio.Queue()
                        while not self._pool.empty():
                            item = await self._pool.get()
                            if item != conn:
                                await temp_queue.put(item)

                        while not temp_queue.empty():
                            await self._pool.put(await temp_queue.get())

                        await self._close_connection(conn)
                        logger.debug("Closed idle connection")
                    except Exception as e:
                        logger.warning(f"Error during connection cleanup: {e}")

            except Exception as e:
                logger.error(f"Error in connection pool cleanup: {e}")

    async def close(self) -> None:
        """Close the connection pool."""
        self._closed = True

        # Close all connections in pool
        while not self._pool.empty():
            try:
                connection = await self._pool.get()
                await self._close_connection(connection)
            except Exception as e:
                logger.warning(f"Error closing pool connection: {e}")

        # Close active connections
        active_connections = list(self._active_connections)
        for connection in active_connections:
            await self._close_connection(connection)

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Pool statistics
        """
        return {
            "pool_size": self._pool.qsize(),
            "active_connections": len(self._active_connections),
            "max_size": self.config.max_size,
            "min_size": self.config.min_size,
            "total_connections": self._pool.qsize() + len(self._active_connections),
        }


class CacheManager:
    """In-memory cache with TTL and size limits."""

    def __init__(
        self, max_size: int = 1000, default_ttl: float = 300.0, cleanup_interval: float = 60.0
    ):
        """Initialize cache manager.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_times: dict[str, float] = {}
        self._lock = threading.RLock()

        # Hit/miss tracking
        self._hits = 0
        self._misses = 0

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry_time = self._cache[key]

            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                self._access_times.pop(key, None)
                self._misses += 1
                return None

            # Update access time and record hit
            self._access_times[key] = time.time()
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = (value, expiry_time)
            self._access_times[key] = time.time()

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                current_time = time.time()
                expired_keys = []

                with self._lock:
                    for key, (_, expiry_time) in self._cache.items():
                        if current_time > expiry_time:
                            expired_keys.append(key)

                    for key in expired_keys:
                        self._cache.pop(key, None)
                        self._access_times.pop(key, None)

                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_ratio = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_ratio": hit_ratio,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
            }


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize performance monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

        # Metrics storage
        self._metrics_history: deque = deque(maxlen=1000)
        self._alerts: list[PerformanceAlert] = []
        self._thresholds = self._get_default_thresholds()

        # Performance tracking
        self._operation_times: defaultdict = defaultdict(list)
        self._error_counts: defaultdict = defaultdict(int)
        self._request_counts: defaultdict = defaultdict(int)

        # Resource managers
        self._connection_pools: dict[str, ConnectionPool] = {}
        self._cache_manager = CacheManager()

        # Start monitoring task
        if self.enabled:
            asyncio.create_task(self._monitoring_loop())

    def _get_default_thresholds(self) -> dict[str, dict[str, float]]:
        """Get default performance thresholds."""
        return {
            "cpu": {"warning": 75.0, "critical": 90.0},
            "memory": {"warning": 80.0, "critical": 95.0},
            "disk": {"warning": 85.0, "critical": 95.0},
            "response_time": {"warning": 1000.0, "critical": 5000.0},
            "error_rate": {"warning": 0.05, "critical": 0.10},
        }

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.enabled:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Collect metrics
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)

                # Check thresholds
                await self._check_thresholds(metrics)

                # Auto-scaling recommendations
                await self._generate_scaling_recommendations(metrics)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        current_time = datetime.now()

        # System metrics
        if psutil_available:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network I/O (simplified)
            network_io = 0.0
        else:
            cpu_percent = 0.0
            memory = type("obj", (object,), {"percent": 0.0, "available": 0})()
            disk = type("obj", (object,), {"percent": 0.0, "free": 0})()
            network_io = 0.0

        # Application metrics
        active_connections = sum(
            pool.get_stats()["active_connections"] for pool in self._connection_pools.values()
        )

        # Calculate response time and error rate
        response_time_ms = self._calculate_avg_response_time()
        error_rate = self._calculate_error_rate()

        return ResourceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=(
                memory.available // (1024 * 1024) if hasattr(memory, "available") else 0
            ),
            disk_usage_percent=disk.percent,
            disk_free_mb=disk.free // (1024 * 1024) if hasattr(disk, "free") else 0,
            network_io_mb=network_io,
            active_connections=active_connections,
            response_time_ms=response_time_ms,
            error_rate=error_rate,
        )

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        all_times = []
        for operation_times in self._operation_times.values():
            all_times.extend(operation_times[-100:])  # Last 100 operations

        return sum(all_times) / len(all_times) if all_times else 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total_requests = sum(self._request_counts.values())
        total_errors = sum(self._error_counts.values())

        return total_errors / total_requests if total_requests > 0 else 0.0

    async def _check_thresholds(self, metrics: ResourceMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        checks = [
            ("cpu", metrics.cpu_percent, ResourceType.CPU),
            ("memory", metrics.memory_percent, ResourceType.MEMORY),
            ("disk", metrics.disk_usage_percent, ResourceType.DISK),
            ("response_time", metrics.response_time_ms, ResourceType.NETWORK),
            ("error_rate", metrics.error_rate, ResourceType.NETWORK),
        ]

        for metric_name, value, resource_type in checks:
            thresholds = self._thresholds.get(metric_name, {})

            if value >= thresholds.get("critical", float("inf")):
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL,
                    resource_type=resource_type,
                    message=f"Critical {metric_name} usage: {value:.1f}%",
                    current_value=value,
                    threshold=thresholds["critical"],
                )
                self._alerts.append(alert)
                logger.critical(f"Performance alert: {alert.message}")

            elif value >= thresholds.get("warning", float("inf")):
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    resource_type=resource_type,
                    message=f"High {metric_name} usage: {value:.1f}%",
                    current_value=value,
                    threshold=thresholds["warning"],
                )
                self._alerts.append(alert)
                logger.warning(f"Performance alert: {alert.message}")

    async def _generate_scaling_recommendations(self, metrics: ResourceMetrics) -> None:
        """Generate auto-scaling recommendations."""
        recommendations = []

        # CPU-based recommendations
        if metrics.cpu_percent > 80:
            recommendations.append(
                {
                    "type": "scale_up",
                    "reason": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    "suggested_action": "Add more CPU cores or scale horizontally",
                }
            )
        elif metrics.cpu_percent < 20 and len(self._metrics_history) > 10:
            avg_cpu = sum(m.cpu_percent for m in list(self._metrics_history)[-10:]) / 10
            if avg_cpu < 20:
                recommendations.append(
                    {
                        "type": "scale_down",
                        "reason": f"Low CPU usage: {avg_cpu:.1f}%",
                        "suggested_action": "Consider reducing resources",
                    }
                )

        # Memory-based recommendations
        if metrics.memory_percent > 85:
            recommendations.append(
                {
                    "type": "scale_up",
                    "reason": f"High memory usage: {metrics.memory_percent:.1f}%",
                    "suggested_action": "Add more memory or optimize caching",
                }
            )

        # Connection pool recommendations
        for pool_name, pool in self._connection_pools.items():
            stats = pool.get_stats()
            if stats["active_connections"] / stats["max_size"] > 0.8:
                recommendations.append(
                    {
                        "type": "tune_pool",
                        "reason": f"High connection pool usage for {pool_name}",
                        "suggested_action": "Increase max pool size",
                    }
                )

        # Log recommendations
        for rec in recommendations:
            logger.info(f"Scaling recommendation: {rec}")

    def record_operation_time(self, operation: str, duration_ms: float) -> None:
        """Record operation execution time.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        self._operation_times[operation].append(duration_ms)
        # Keep only last 1000 measurements per operation
        if len(self._operation_times[operation]) > 1000:
            self._operation_times[operation] = self._operation_times[operation][-1000:]

    def record_request(self, operation: str, success: bool) -> None:
        """Record request outcome.

        Args:
            operation: Operation name
            success: Whether request was successful
        """
        self._request_counts[operation] += 1
        if not success:
            self._error_counts[operation] += 1

    def create_connection_pool(
        self, name: str, connection_factory: Callable, config: ConnectionPoolConfig | None = None
    ) -> ConnectionPool:
        """Create a managed connection pool.

        Args:
            name: Pool name
            connection_factory: Function to create connections
            config: Pool configuration

        Returns:
            Connection pool instance
        """
        pool = ConnectionPool(connection_factory, config)
        self._connection_pools[name] = pool
        return pool

    def get_cache_manager(self) -> CacheManager:
        """Get the cache manager.

        Returns:
            Cache manager instance
        """
        return self._cache_manager

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary.

        Returns:
            Performance summary data
        """
        if not self._metrics_history:
            return {"error": "No metrics available"}

        latest_metrics = self._metrics_history[-1]

        # Calculate averages over last 10 minutes
        recent_metrics = [
            m for m in self._metrics_history if (datetime.now() - m.timestamp).total_seconds() < 600
        ]

        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(
                recent_metrics
            )
        else:
            avg_cpu = avg_memory = avg_response_time = 0.0

        return {
            "current": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "response_time_ms": latest_metrics.response_time_ms,
                "error_rate": latest_metrics.error_rate,
                "active_connections": latest_metrics.active_connections,
            },
            "averages_10min": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "response_time_ms": avg_response_time,
            },
            "alerts": len(
                [a for a in self._alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]
            ),
            "connection_pools": {
                name: pool.get_stats() for name, pool in self._connection_pools.items()
            },
            "cache": self._cache_manager.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup performance monitor resources."""
        self.enabled = False

        # Close connection pools
        for pool in self._connection_pools.values():
            await pool.close()


# Global performance monitor
performance_monitor: PerformanceMonitor | None = None


def get_performance_monitor(config: dict[str, Any] | None = None) -> PerformanceMonitor:
    """Get or create the global performance monitor.

    Args:
        config: Monitor configuration

    Returns:
        Global performance monitor instance
    """
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor(config)
    return performance_monitor


def reset_performance_monitor() -> None:
    """Reset the global performance monitor (useful for testing)."""
    global performance_monitor
    if performance_monitor:
        asyncio.create_task(performance_monitor.cleanup())
    performance_monitor = None
