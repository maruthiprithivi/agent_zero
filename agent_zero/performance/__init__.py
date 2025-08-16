"""Performance and Scalability Module for Agent Zero.

This module provides advanced performance optimization and scalability features:
- Distributed caching with Redis cluster support
- Advanced load balancing with circuit breakers
- Real-time analytics and streaming capabilities
- Performance monitoring and auto-scaling
- Connection pooling and resource management
"""

from .scalability import (
    CacheConfig,
    CacheStrategy,
    DistributedCache,
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancingAlgorithm,
    PerformanceManager,
    PerformanceMetrics,
    RealTimeAnalytics,
    ScalingStrategy,
    create_distributed_cache,
    create_load_balancer,
    create_performance_manager,
)

__all__ = [
    "CacheConfig",
    "CacheStrategy",
    "DistributedCache",
    "LoadBalancer",
    "LoadBalancerConfig",
    "LoadBalancingAlgorithm",
    "PerformanceManager",
    "PerformanceMetrics",
    "RealTimeAnalytics",
    "ScalingStrategy",
    "create_distributed_cache",
    "create_load_balancer",
    "create_performance_manager",
]
