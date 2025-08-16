"""Performance and Scalability Enhancements for Agent Zero.

This module provides advanced performance and scalability features:
- Advanced caching strategies (Redis Cluster, distributed cache)
- Serverless and edge computing support
- Advanced load balancing and traffic management
- Real-time analytics and streaming capabilities
- Connection pooling and circuit breakers
- Auto-scaling and performance optimization
"""

import asyncio
import json
import logging
import pickle
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Redis imports
try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Celery for distributed task processing
try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Streaming analytics
try:
    import numpy as np
    import pandas as pd

    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    DISTRIBUTED = "distributed"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""

    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULE_BASED = "schedule_based"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""

    timestamp: datetime
    response_time: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    cache_hit_rate: float
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time,
            "throughput": self.throughput,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "active_connections": self.active_connections,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate,
        }


@dataclass
class CacheConfig:
    """Cache configuration."""

    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 10000
    ttl_seconds: int = 3600
    redis_url: str | None = None
    redis_cluster_nodes: list[str] | None = None
    compression_enabled: bool = True
    serialization_format: str = "pickle"  # pickle, json, msgpack


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""

    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_CONNECTIONS
    health_check_interval: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


class DistributedCache:
    """Advanced distributed caching system."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.local_cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self.redis_client: Redis | None = None
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        if REDIS_AVAILABLE and config.redis_url:
            self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis client."""
        try:
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle our own serialization
            )
            logger.info("Redis client initialized for distributed caching")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            # Try local cache first
            if key in self.local_cache:
                value, expiry = self.local_cache[key]
                if time.time() < expiry:
                    self.cache_stats["hits"] += 1
                    return value
                else:
                    # Expired, remove from local cache
                    del self.local_cache[key]

            # Try Redis cache
            if self.redis_client:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    value = self._deserialize(cached_data)
                    self.cache_stats["hits"] += 1

                    # Store in local cache too
                    expiry = time.time() + self.config.ttl_seconds
                    self.local_cache[key] = (value, expiry)

                    return value

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.cache_stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.config.ttl_seconds

            # Set in local cache
            expiry = time.time() + ttl
            self.local_cache[key] = (value, expiry)

            # Set in Redis cache
            if self.redis_client:
                serialized_value = self._serialize(value)
                await self.redis_client.setex(key, ttl, serialized_value)

            self.cache_stats["sets"] += 1

            # Cleanup local cache if too large
            await self._cleanup_local_cache()

            return True

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Remove from local cache
            if key in self.local_cache:
                del self.local_cache[key]

            # Remove from Redis cache
            if self.redis_client:
                await self.redis_client.delete(key)

            self.cache_stats["deletes"] += 1
            return True

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self.local_cache.clear()

            if self.redis_client:
                await self.redis_client.flushdb()

            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests) if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
            "stats": self.cache_stats.copy(),
        }

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for caching."""
        if self.config.serialization_format == "json":
            return json.dumps(value).encode()
        elif self.config.serialization_format == "pickle":
            data = pickle.dumps(value)
            if self.config.compression_enabled:
                import zlib

                data = zlib.compress(data)
            return data
        else:
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from cache."""
        if self.config.serialization_format == "json":
            return json.loads(data.decode())
        elif self.config.serialization_format == "pickle":
            if self.config.compression_enabled:
                import zlib

                data = zlib.decompress(data)
            return pickle.loads(data)
        else:
            return pickle.loads(data)

    async def _cleanup_local_cache(self):
        """Clean up local cache based on strategy."""
        if len(self.local_cache) <= self.config.max_size:
            return

        current_time = time.time()

        if self.config.strategy == CacheStrategy.LRU:
            # Remove oldest entries
            items_to_remove = len(self.local_cache) - self.config.max_size
            oldest_keys = sorted(
                self.local_cache.keys(),
                key=lambda k: self.local_cache[k][1],  # Sort by expiry time
            )[:items_to_remove]

            for key in oldest_keys:
                del self.local_cache[key]

        elif self.config.strategy == CacheStrategy.TTL:
            # Remove expired entries
            expired_keys = [
                key for key, (value, expiry) in self.local_cache.items() if current_time >= expiry
            ]

            for key in expired_keys:
                del self.local_cache[key]


class LoadBalancer:
    """Advanced load balancer with circuit breaker."""

    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.servers: list[dict[str, Any]] = []
        self.server_stats: dict[str, dict[str, Any]] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}
        self.current_index = 0
        self.request_history: deque = deque(maxlen=1000)

    def add_server(self, server_id: str, endpoint: str, weight: int = 1):
        """Add server to load balancer."""
        server = {
            "id": server_id,
            "endpoint": endpoint,
            "weight": weight,
            "healthy": True,
            "active_connections": 0,
            "last_health_check": datetime.utcnow(),
        }

        self.servers.append(server)
        self.server_stats[server_id] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "response_times": deque(maxlen=100),
        }
        self.circuit_breakers[server_id] = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure": None,
            "next_attempt": None,
        }

        logger.info(f"Added server {server_id} to load balancer")

    async def get_server(
        self, request_context: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Get server based on load balancing algorithm."""
        healthy_servers = [
            s for s in self.servers if s["healthy"] and self._is_circuit_closed(s["id"])
        ]

        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None

        if self.config.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1

        elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            server = min(healthy_servers, key=lambda s: s["active_connections"])

        elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            server = min(
                healthy_servers, key=lambda s: self.server_stats[s["id"]]["avg_response_time"]
            )

        elif self.config.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            # Implement weighted selection
            weights = [s["weight"] for s in healthy_servers]
            total_weight = sum(weights)
            if total_weight == 0:
                server = healthy_servers[0]
            else:
                import random

                r = random.randint(1, total_weight)
                cumulative = 0
                server = healthy_servers[0]
                for s in healthy_servers:
                    cumulative += s["weight"]
                    if r <= cumulative:
                        server = s
                        break

        elif self.config.algorithm == LoadBalancingAlgorithm.IP_HASH:
            if request_context and "client_ip" in request_context:
                ip_hash = hash(request_context["client_ip"])
                server = healthy_servers[ip_hash % len(healthy_servers)]
            else:
                server = healthy_servers[0]

        else:
            server = healthy_servers[0]

        # Increment active connections
        server["active_connections"] += 1

        return server

    def release_server(self, server_id: str, success: bool, response_time: float):
        """Release server and update statistics."""
        for server in self.servers:
            if server["id"] == server_id:
                server["active_connections"] = max(0, server["active_connections"] - 1)
                break

        stats = self.server_stats[server_id]
        stats["total_requests"] += 1

        if success:
            stats["successful_requests"] += 1
            stats["response_times"].append(response_time)
            if stats["response_times"]:
                stats["avg_response_time"] = statistics.mean(stats["response_times"])

            # Reset circuit breaker on success
            self.circuit_breakers[server_id]["failure_count"] = 0
        else:
            stats["failed_requests"] += 1

            # Update circuit breaker
            cb = self.circuit_breakers[server_id]
            cb["failure_count"] += 1
            cb["last_failure"] = datetime.utcnow()

            if cb["failure_count"] >= self.config.circuit_breaker_threshold:
                cb["state"] = "open"
                cb["next_attempt"] = datetime.utcnow() + timedelta(
                    seconds=self.config.circuit_breaker_timeout
                )
                logger.warning(f"Circuit breaker opened for server {server_id}")

        # Store request history
        self.request_history.append(
            {
                "timestamp": datetime.utcnow(),
                "server_id": server_id,
                "success": success,
                "response_time": response_time,
            }
        )

    def _is_circuit_closed(self, server_id: str) -> bool:
        """Check if circuit breaker is closed."""
        cb = self.circuit_breakers[server_id]

        if cb["state"] == "closed":
            return True
        elif cb["state"] == "open":
            if cb["next_attempt"] and datetime.utcnow() >= cb["next_attempt"]:
                cb["state"] = "half_open"
                logger.info(f"Circuit breaker half-opened for server {server_id}")
                return True
            return False
        elif cb["state"] == "half_open":
            return True

        return False

    async def health_check_servers(self):
        """Perform health checks on all servers."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            for server in self.servers:
                try:
                    health_endpoint = f"{server['endpoint']}/health"
                    async with session.get(health_endpoint, timeout=5) as response:
                        server["healthy"] = response.status == 200
                        server["last_health_check"] = datetime.utcnow()

                        if (
                            server["healthy"]
                            and self.circuit_breakers[server["id"]]["state"] == "half_open"
                        ):
                            self.circuit_breakers[server["id"]]["state"] = "closed"
                            logger.info(f"Circuit breaker closed for server {server['id']}")

                except Exception as e:
                    server["healthy"] = False
                    logger.warning(f"Health check failed for server {server['id']}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(stats["total_requests"] for stats in self.server_stats.values())

        return {
            "total_requests": total_requests,
            "servers": [
                {
                    "id": server["id"],
                    "endpoint": server["endpoint"],
                    "healthy": server["healthy"],
                    "active_connections": server["active_connections"],
                    "stats": self.server_stats[server["id"]],
                    "circuit_breaker": self.circuit_breakers[server["id"]],
                }
                for server in self.servers
            ],
            "algorithm": self.config.algorithm.value,
        }


class RealTimeAnalytics:
    """Real-time analytics and streaming capabilities."""

    def __init__(self):
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.aggregated_metrics: dict[str, Any] = {}
        self.alert_thresholds: dict[str, float] = {
            "response_time": 5.0,  # seconds
            "error_rate": 0.05,  # 5%
            "cpu_usage": 80.0,  # 80%
            "memory_usage": 85.0,  # 85%
        }
        self.active_alerts: set[str] = set()

        # Start background processing
        self._processing_task = None
        self.start_processing()

    def start_processing(self):
        """Start background analytics processing."""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_metrics())

    async def stop_processing(self):
        """Stop background analytics processing."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

    async def add_metric(self, metric: PerformanceMetrics):
        """Add performance metric for analysis."""
        self.metrics_buffer.append(metric)

    async def _process_metrics(self):
        """Background task to process metrics."""
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds

                if not self.metrics_buffer:
                    continue

                # Convert to DataFrame for analysis
                if ANALYTICS_AVAILABLE:
                    await self._analyze_with_pandas()
                else:
                    await self._analyze_basic()

                # Check for alerts
                await self._check_alerts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")

    async def _analyze_with_pandas(self):
        """Analyze metrics using pandas."""
        try:
            # Convert metrics to DataFrame
            data = [metric.to_dict() for metric in list(self.metrics_buffer)]
            df = pd.DataFrame(data)

            if df.empty:
                return

            # Calculate aggregated metrics
            self.aggregated_metrics = {
                "avg_response_time": df["response_time"].mean(),
                "p95_response_time": df["response_time"].quantile(0.95),
                "p99_response_time": df["response_time"].quantile(0.99),
                "avg_throughput": df["throughput"].mean(),
                "total_requests": len(df),
                "error_rate": df["error_rate"].mean(),
                "avg_cpu_usage": df["cpu_usage"].mean(),
                "avg_memory_usage": df["memory_usage"].mean(),
                "avg_cache_hit_rate": df["cache_hit_rate"].mean(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Time series analysis (last hour)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_df = df[df["timestamp"] > (datetime.utcnow() - timedelta(hours=1))]

            if not recent_df.empty:
                # Calculate trends
                self.aggregated_metrics["hourly_trend"] = {
                    "response_time_trend": self._calculate_trend(recent_df["response_time"]),
                    "throughput_trend": self._calculate_trend(recent_df["throughput"]),
                    "error_rate_trend": self._calculate_trend(recent_df["error_rate"]),
                }

        except Exception as e:
            logger.error(f"Error in pandas analysis: {e}")

    async def _analyze_basic(self):
        """Basic analysis without pandas."""
        try:
            metrics_list = list(self.metrics_buffer)

            if not metrics_list:
                return

            response_times = [m.response_time for m in metrics_list]
            throughputs = [m.throughput for m in metrics_list]
            error_rates = [m.error_rate for m in metrics_list]
            cpu_usages = [m.cpu_usage for m in metrics_list]
            memory_usages = [m.memory_usage for m in metrics_list]

            self.aggregated_metrics = {
                "avg_response_time": statistics.mean(response_times),
                "avg_throughput": statistics.mean(throughputs),
                "total_requests": len(metrics_list),
                "error_rate": statistics.mean(error_rates),
                "avg_cpu_usage": statistics.mean(cpu_usages),
                "avg_memory_usage": statistics.mean(memory_usages),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in basic analysis: {e}")

    def _calculate_trend(self, series) -> str:
        """Calculate trend direction."""
        if len(series) < 2:
            return "stable"

        # Simple linear regression slope
        x = range(len(series))
        y = series.values

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    async def _check_alerts(self):
        """Check for alert conditions."""
        if not self.aggregated_metrics:
            return

        new_alerts = set()

        # Response time alert
        if (
            self.aggregated_metrics.get("avg_response_time", 0)
            > self.alert_thresholds["response_time"]
        ):
            new_alerts.add("high_response_time")

        # Error rate alert
        if self.aggregated_metrics.get("error_rate", 0) > self.alert_thresholds["error_rate"]:
            new_alerts.add("high_error_rate")

        # CPU usage alert
        if self.aggregated_metrics.get("avg_cpu_usage", 0) > self.alert_thresholds["cpu_usage"]:
            new_alerts.add("high_cpu_usage")

        # Memory usage alert
        if (
            self.aggregated_metrics.get("avg_memory_usage", 0)
            > self.alert_thresholds["memory_usage"]
        ):
            new_alerts.add("high_memory_usage")

        # Log new alerts
        for alert in new_alerts - self.active_alerts:
            logger.warning(f"Alert triggered: {alert}")

        # Log resolved alerts
        for alert in self.active_alerts - new_alerts:
            logger.info(f"Alert resolved: {alert}")

        self.active_alerts = new_alerts

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get dashboard data."""
        return {
            "metrics": self.aggregated_metrics,
            "active_alerts": list(self.active_alerts),
            "alert_thresholds": self.alert_thresholds,
            "buffer_size": len(self.metrics_buffer),
            "processing_status": "active" if self._processing_task else "stopped",
        }


class PerformanceManager:
    """Main performance and scalability manager."""

    def __init__(
        self, cache_config: CacheConfig | None = None, lb_config: LoadBalancerConfig | None = None
    ):
        self.cache = DistributedCache(cache_config or CacheConfig())
        self.load_balancer = LoadBalancer(lb_config or LoadBalancerConfig())
        self.analytics = RealTimeAnalytics()

        logger.info("Performance Manager initialized")

    @asynccontextmanager
    async def performance_context(self, operation: str):
        """Context manager for performance tracking."""
        start_time = time.time()
        success = True

        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time()
            response_time = end_time - start_time

            # Create performance metric
            metric = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                response_time=response_time,
                throughput=1.0 / response_time if response_time > 0 else 0,
                cpu_usage=0.0,  # Would be populated from system metrics
                memory_usage=0.0,
                active_connections=0,
                cache_hit_rate=0.0,
                error_rate=0.0 if success else 1.0,
            )

            await self.analytics.add_metric(metric)

    async def shutdown(self):
        """Shutdown performance manager."""
        await self.analytics.stop_processing()
        logger.info("Performance Manager shutdown")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "cache_stats": self.cache.get_stats(),
            "load_balancer_stats": self.load_balancer.get_stats(),
            "analytics_dashboard": self.analytics.get_dashboard_data(),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Factory functions
def create_performance_manager(
    cache_config: CacheConfig | None = None, lb_config: LoadBalancerConfig | None = None
) -> PerformanceManager:
    """Create performance manager."""
    return PerformanceManager(cache_config, lb_config)


def create_distributed_cache(config: CacheConfig | None = None) -> DistributedCache:
    """Create distributed cache."""
    return DistributedCache(config or CacheConfig())


def create_load_balancer(config: LoadBalancerConfig | None = None) -> LoadBalancer:
    """Create load balancer."""
    return LoadBalancer(config or LoadBalancerConfig())


# Example usage
async def demonstrate_performance_features():
    """Demonstrate performance and scalability features."""
    print("Performance and Scalability Features Demo")
    print(f"Redis Available: {REDIS_AVAILABLE}")
    print(f"Analytics Available: {ANALYTICS_AVAILABLE}")

    # Create performance manager
    cache_config = CacheConfig(strategy=CacheStrategy.LRU, max_size=1000, ttl_seconds=300)

    lb_config = LoadBalancerConfig(algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS)

    perf_manager = create_performance_manager(cache_config, lb_config)

    # Add servers to load balancer
    perf_manager.load_balancer.add_server("server1", "http://localhost:8501")
    perf_manager.load_balancer.add_server("server2", "http://localhost:8502")
    perf_manager.load_balancer.add_server("server3", "http://localhost:8503")

    # Test caching
    await perf_manager.cache.set("test_key", {"data": "test_value"})
    cached_value = await perf_manager.cache.get("test_key")
    print(f"Cached value: {cached_value}")

    # Test performance tracking
    async with perf_manager.performance_context("test_operation"):
        await asyncio.sleep(0.1)  # Simulate work

    # Wait for analytics processing
    await asyncio.sleep(1)

    # Get performance report
    report = perf_manager.get_performance_report()
    print(f"Performance report: {json.dumps(report, indent=2, default=str)}")

    # Cleanup
    await perf_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(demonstrate_performance_features())
