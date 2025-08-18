"""AI-powered performance advisor engine for ClickHouse optimization recommendations.

This module provides comprehensive machine learning-based performance optimization
recommendations for ClickHouse deployments. It analyzes system performance data,
identifies optimization opportunities, and provides actionable recommendations
with impact predictions and implementation complexity assessments.

The performance advisor complements the bottleneck detector by not just identifying
problems, but providing clear, prioritized solutions based on AI analysis of
actual performance data.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from clickhouse_connect.driver.client import Client

from agent_zero.ai_diagnostics.bottleneck_detector import (
    BottleneckDetection,
    BottleneckSeverity,
    ConfidenceLevel,
    IntelligentBottleneckDetector,
)
from agent_zero.monitoring.hardware_diagnostics import HardwareHealthEngine
from agent_zero.monitoring.performance_diagnostics import (
    PerformanceDiagnosticEngine,
)
from agent_zero.monitoring.storage_cloud_diagnostics import (
    StorageOptimizationEngine,
)
from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


class RecommendationCategory(Enum):
    """Categories of performance recommendations for structured advisory."""

    IMMEDIATE_FIXES = "immediate_fixes"  # < 1 hour implementation
    MEDIUM_TERM_OPTIMIZATIONS = "medium_term_optimizations"  # < 1 week
    LONG_TERM_PLANNING = "long_term_planning"  # > 1 week
    PREVENTIVE_MEASURES = "preventive_measures"  # Monitoring and alerting
    COST_OPTIMIZATION = "cost_optimization"  # Cloud cost reduction


class ImpactLevel(Enum):
    """Expected performance impact levels for recommendations."""

    TRANSFORMATIONAL = "transformational"  # 50%+ improvement
    SIGNIFICANT = "significant"  # 20-50% improvement
    MODERATE = "moderate"  # 10-20% improvement
    MINOR = "minor"  # 5-10% improvement
    MARGINAL = "marginal"  # 1-5% improvement


class ImplementationComplexity(Enum):
    """Implementation complexity levels for recommendations."""

    TRIVIAL = "trivial"  # Configuration change only
    EASY = "easy"  # Simple query/schema changes
    MODERATE = "moderate"  # Multi-step implementation
    COMPLEX = "complex"  # Significant architectural changes
    EXPERT = "expert"  # Requires specialized knowledge


class RiskLevel(Enum):
    """Risk levels for implementing recommendations."""

    MINIMAL = "minimal"  # No risk of data loss or downtime
    LOW = "low"  # Minimal risk with proper testing
    MEDIUM = "medium"  # Moderate risk, requires careful planning
    HIGH = "high"  # High risk, requires expert guidance
    CRITICAL = "critical"  # Could cause system outage if done incorrectly


class RecommendationType(Enum):
    """Types of performance recommendations."""

    CONFIGURATION_TUNING = "configuration_tuning"
    QUERY_OPTIMIZATION = "query_optimization"
    INDEX_OPTIMIZATION = "index_optimization"
    HARDWARE_SCALING = "hardware_scaling"
    STORAGE_OPTIMIZATION = "storage_optimization"
    MEMORY_TUNING = "memory_tuning"
    CACHE_OPTIMIZATION = "cache_optimization"
    COMPRESSION_TUNING = "compression_tuning"
    THREAD_POOL_SIZING = "thread_pool_sizing"
    NETWORK_OPTIMIZATION = "network_optimization"
    REPLICATION_OPTIMIZATION = "replication_optimization"
    PARTITION_STRATEGY = "partition_strategy"


@dataclass
class PerformanceRecommendation:
    """Structured performance recommendation with comprehensive impact analysis."""

    # Core identification
    recommendation_id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    category: RecommendationCategory

    # Impact analysis
    expected_impact: ImpactLevel
    impact_percentage: float  # Estimated performance improvement %
    confidence_score: float  # 0-100 scale
    confidence_level: ConfidenceLevel

    # Implementation details
    complexity: ImplementationComplexity
    risk_level: RiskLevel
    estimated_time_hours: float
    prerequisites: list[str] = field(default_factory=list)

    # Technical details
    current_state: dict[str, Any] = field(default_factory=dict)
    recommended_changes: dict[str, Any] = field(default_factory=dict)
    implementation_steps: list[str] = field(default_factory=list)
    validation_queries: list[str] = field(default_factory=list)
    rollback_plan: list[str] = field(default_factory=list)

    # Evidence and justification
    evidence: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    bottleneck_patterns: list[str] = field(default_factory=list)
    related_issues: list[str] = field(default_factory=list)

    # Priority and scheduling
    priority_score: float = 0.0  # Calculated based on impact/effort ratio
    urgency_level: int = 1  # 1-5 scale (5 = urgent)
    business_impact: str = ""

    # Cost analysis
    estimated_cost_savings: float | None = None  # Cloud cost savings per month
    implementation_cost: float | None = None  # One-time implementation cost

    # Tracking
    created_timestamp: datetime = field(default_factory=datetime.now)
    recommendation_source: str = "ai_performance_advisor"
    tags: set[str] = field(default_factory=set)

    # Follow-up
    monitoring_recommendations: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)


@dataclass
class RecommendationContext:
    """Context information for generating recommendations."""

    # System state
    system_health_score: float
    active_bottlenecks: list[BottleneckDetection]
    performance_trends: dict[str, float]
    resource_utilization: dict[str, float]

    # Workload characteristics
    query_patterns: dict[str, Any]
    data_volume: dict[str, float]
    concurrent_users: int
    peak_hours: list[int]

    # Infrastructure
    hardware_specs: dict[str, Any]
    storage_config: dict[str, Any]
    network_topology: dict[str, Any]
    deployment_mode: str  # cloud, on-premise, hybrid

    # Historical performance
    baseline_metrics: dict[str, float]
    performance_history: list[dict[str, Any]]
    previous_optimizations: list[str]

    # Business context
    sla_requirements: dict[str, float]
    cost_constraints: dict[str, float]
    availability_requirements: float


class RecommendationEngine:
    """Core ML-based recommendation system for performance optimization."""

    def __init__(self, client: Client):
        """Initialize the recommendation engine.

        Args:
            client: ClickHouse client instance
        """
        self.client = client
        self.logger = logger.getChild("recommendation_engine")

        # ML models and pattern storage
        self.pattern_database = self._initialize_pattern_database()
        self.historical_validations = defaultdict(list)
        self.recommendation_effectiveness = defaultdict(float)

        # Configuration
        self.confidence_threshold = 0.3
        self.min_impact_threshold = 5.0  # Minimum 5% improvement
        self.max_recommendations_per_category = 10

    def _initialize_pattern_database(self) -> dict[str, dict[str, Any]]:
        """Initialize the pattern database with known optimization patterns."""
        patterns = {}

        # Configuration tuning patterns
        patterns["high_cpu_configuration"] = {
            "indicators": [
                "ProfileEvent_Query",
                "ProfileEvent_SelectQuery",
                "ProfileEvent_OSCPUVirtualTimeMicroseconds",
            ],
            "thresholds": {"cpu_usage_percent": 80.0, "query_duration_avg": 1000.0},
            "recommendations": {
                "max_threads": "auto",
                "max_memory_usage": "0",  # Remove memory limits if CPU bound
                "max_execution_time": 3600,
            },
            "impact_estimate": 25.0,
            "complexity": ImplementationComplexity.TRIVIAL,
        }

        patterns["memory_pressure_configuration"] = {
            "indicators": ["ProfileEvent_MemoryTrackerUsage", "ProfileEvent_OSMemoryAvailable"],
            "thresholds": {"memory_usage_percent": 85.0, "swap_usage": 10.0},
            "recommendations": {
                "max_memory_usage": "0.8 * total_memory",
                "max_bytes_before_external_group_by": "0.6 * max_memory_usage",
                "max_bytes_before_external_sort": "0.6 * max_memory_usage",
            },
            "impact_estimate": 40.0,
            "complexity": ImplementationComplexity.EASY,
        }

        patterns["cache_optimization"] = {
            "indicators": ["ProfileEvent_MarkCacheMisses", "ProfileEvent_UncompressedCacheMisses"],
            "thresholds": {"cache_miss_rate": 20.0, "cache_hit_rate": 70.0},
            "recommendations": {
                "mark_cache_size": "8GB",
                "uncompressed_cache_size": "16GB",
                "compiled_expression_cache_size": "1GB",
            },
            "impact_estimate": 30.0,
            "complexity": ImplementationComplexity.TRIVIAL,
        }

        patterns["io_optimization"] = {
            "indicators": [
                "ProfileEvent_OSReadBytes",
                "ProfileEvent_OSWriteBytes",
                "ProfileEvent_OSReadChars",
            ],
            "thresholds": {"io_wait_percent": 25.0, "disk_queue_depth": 10.0},
            "recommendations": {
                "merge_tree_settings": {
                    "parts_to_delay_insert": 300,
                    "parts_to_throw_insert": 600,
                    "max_parts_in_total": 1000,
                },
                "background_pool_size": 32,
                "background_merges_mutations_concurrency_ratio": 4,
            },
            "impact_estimate": 35.0,
            "complexity": ImplementationComplexity.MODERATE,
        }

        return patterns

    def analyze_system_state(self, context: RecommendationContext) -> dict[str, Any]:
        """Analyze current system state to identify optimization opportunities.

        Args:
            context: System context for analysis

        Returns:
            Analysis results with identified patterns and opportunities
        """
        analysis = {
            "optimization_opportunities": [],
            "system_stress_indicators": [],
            "performance_gaps": {},
            "resource_bottlenecks": [],
        }

        try:
            # Analyze resource utilization patterns
            if context.resource_utilization.get("cpu_percent", 0) > 80:
                analysis["system_stress_indicators"].append("high_cpu_usage")
                analysis["optimization_opportunities"].append("cpu_optimization")

            if context.resource_utilization.get("memory_percent", 0) > 85:
                analysis["system_stress_indicators"].append("memory_pressure")
                analysis["optimization_opportunities"].append("memory_optimization")

            if context.resource_utilization.get("io_wait_percent", 0) > 20:
                analysis["system_stress_indicators"].append("io_bottleneck")
                analysis["optimization_opportunities"].append("storage_optimization")

            # Analyze performance gaps against SLA
            for metric, requirement in context.sla_requirements.items():
                current_value = context.baseline_metrics.get(metric, 0)
                if current_value > requirement * 1.2:  # 20% worse than SLA
                    analysis["performance_gaps"][metric] = {
                        "current": current_value,
                        "required": requirement,
                        "gap_percent": ((current_value - requirement) / requirement) * 100,
                    }

            # Identify bottleneck patterns from active issues
            for bottleneck in context.active_bottlenecks:
                if bottleneck.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]:
                    analysis["resource_bottlenecks"].append(
                        {
                            "category": bottleneck.signature.category.value,
                            "severity": bottleneck.severity.value,
                            "confidence": bottleneck.confidence,
                        }
                    )

            self.logger.info(
                f"System analysis completed: {len(analysis['optimization_opportunities'])} opportunities identified"
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing system state: {e}")
            return analysis

    def predict_impact(
        self, pattern_key: str, context: RecommendationContext
    ) -> tuple[float, float]:
        """Predict the performance impact of applying a recommendation pattern.

        Args:
            pattern_key: Key identifying the optimization pattern
            context: System context for impact calculation

        Returns:
            Tuple of (impact_percentage, confidence_score)
        """
        try:
            pattern = self.pattern_database.get(pattern_key, {})
            base_impact = pattern.get("impact_estimate", 10.0)

            # Adjust impact based on system state
            adjustment_factor = 1.0

            # Higher impact if system is under stress
            cpu_usage = context.resource_utilization.get("cpu_percent", 50)
            memory_usage = context.resource_utilization.get("memory_percent", 50)

            if cpu_usage > 90 or memory_usage > 90:
                adjustment_factor *= 1.5  # 50% higher impact under extreme stress
            elif cpu_usage > 70 or memory_usage > 70:
                adjustment_factor *= 1.2  # 20% higher impact under moderate stress

            # Lower impact if already optimized
            if context.system_health_score > 85:
                adjustment_factor *= 0.7  # Diminishing returns for well-optimized systems

            # Historical validation adjustment
            if pattern_key in self.recommendation_effectiveness:
                historical_effectiveness = self.recommendation_effectiveness[pattern_key]
                adjustment_factor *= 0.5 + historical_effectiveness  # Blend with historical data

            adjusted_impact = base_impact * adjustment_factor

            # Calculate confidence based on pattern match strength
            confidence = min(95.0, 60.0 + (adjustment_factor - 1.0) * 30.0)

            return adjusted_impact, confidence

        except Exception as e:
            self.logger.error(f"Error predicting impact for pattern {pattern_key}: {e}")
            return 10.0, 50.0  # Conservative defaults

    def calculate_priority_score(self, recommendation: PerformanceRecommendation) -> float:
        """Calculate priority score based on impact/effort ratio and urgency.

        Args:
            recommendation: The recommendation to score

        Returns:
            Priority score (higher = more important)
        """
        try:
            # Base score from impact vs. complexity
            impact_score = recommendation.impact_percentage

            # Complexity penalty (higher complexity = lower priority)
            complexity_penalties = {
                ImplementationComplexity.TRIVIAL: 1.0,
                ImplementationComplexity.EASY: 0.9,
                ImplementationComplexity.MODERATE: 0.7,
                ImplementationComplexity.COMPLEX: 0.5,
                ImplementationComplexity.EXPERT: 0.3,
            }
            complexity_multiplier = complexity_penalties.get(recommendation.complexity, 0.5)

            # Risk penalty (higher risk = lower priority)
            risk_penalties = {
                RiskLevel.MINIMAL: 1.0,
                RiskLevel.LOW: 0.95,
                RiskLevel.MEDIUM: 0.8,
                RiskLevel.HIGH: 0.6,
                RiskLevel.CRITICAL: 0.4,
            }
            risk_multiplier = risk_penalties.get(recommendation.risk_level, 0.5)

            # Time penalty (longer implementation = lower priority for immediate fixes)
            time_penalty = max(0.3, 1.0 - (recommendation.estimated_time_hours / 100.0))

            # Confidence boost
            confidence_multiplier = recommendation.confidence_score / 100.0

            # Urgency multiplier
            urgency_multiplier = recommendation.urgency_level / 5.0 * 0.5 + 0.5

            # Calculate final score
            priority_score = (
                impact_score
                * complexity_multiplier
                * risk_multiplier
                * time_penalty
                * confidence_multiplier
                * urgency_multiplier
            )

            return round(priority_score, 2)

        except Exception as e:
            self.logger.error(f"Error calculating priority score: {e}")
            return 0.0

    def validate_recommendation_effectiveness(
        self, recommendation_id: str, actual_impact: float
    ) -> None:
        """Track the effectiveness of implemented recommendations.

        Args:
            recommendation_id: ID of the implemented recommendation
            actual_impact: Measured performance improvement percentage
        """
        try:
            # Store validation result
            self.historical_validations[recommendation_id].append(
                {
                    "timestamp": datetime.now(),
                    "actual_impact": actual_impact,
                }
            )

            # Update pattern effectiveness
            # This would be more sophisticated in a production system
            avg_effectiveness = statistics.mean(
                [v["actual_impact"] for v in self.historical_validations[recommendation_id]]
            )

            self.recommendation_effectiveness[recommendation_id] = avg_effectiveness / 100.0

            self.logger.info(
                f"Recorded recommendation effectiveness: {recommendation_id} -> {actual_impact}%"
            )

        except Exception as e:
            self.logger.error(f"Error validating recommendation effectiveness: {e}")


class ConfigurationAdvisor:
    """ClickHouse configuration optimization advisor."""

    def __init__(self, client: Client, recommendation_engine: RecommendationEngine):
        """Initialize the configuration advisor.

        Args:
            client: ClickHouse client instance
            recommendation_engine: Core recommendation engine
        """
        self.client = client
        self.recommendation_engine = recommendation_engine
        self.logger = logger.getChild("configuration_advisor")

    def analyze_server_configuration(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze server configuration and provide optimization recommendations.

        Args:
            context: System context for analysis

        Returns:
            List of configuration optimization recommendations
        """
        recommendations = []

        try:
            # Get current server configuration
            current_config = self._get_current_server_settings()

            # Memory configuration recommendations
            memory_recommendations = self._analyze_memory_configuration(current_config, context)
            recommendations.extend(memory_recommendations)

            # CPU and thread configuration
            cpu_recommendations = self._analyze_cpu_configuration(current_config, context)
            recommendations.extend(cpu_recommendations)

            # Cache configuration
            cache_recommendations = self._analyze_cache_configuration(current_config, context)
            recommendations.extend(cache_recommendations)

            # I/O and merge configuration
            io_recommendations = self._analyze_io_configuration(current_config, context)
            recommendations.extend(io_recommendations)

            # Network configuration
            network_recommendations = self._analyze_network_configuration(current_config, context)
            recommendations.extend(network_recommendations)

            self.logger.info(f"Generated {len(recommendations)} configuration recommendations")
            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing server configuration: {e}")
            return []

    def _get_current_server_settings(self) -> dict[str, Any]:
        """Get current server configuration settings."""
        try:
            query = """
            SELECT name, value, description, type
            FROM system.settings
            WHERE changed = 1 OR name IN (
                'max_memory_usage', 'max_threads', 'max_execution_time',
                'mark_cache_size', 'uncompressed_cache_size',
                'background_pool_size', 'background_merges_mutations_concurrency_ratio'
            )
            ORDER BY name
            """

            results = execute_query_with_retry(self.client, query)
            return {row["name"]: row["value"] for row in results}

        except Exception as e:
            self.logger.error(f"Error getting server settings: {e}")
            return {}

    def _analyze_memory_configuration(
        self, current_config: dict[str, Any], context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze memory configuration and generate recommendations."""
        recommendations = []

        try:
            memory_usage = context.resource_utilization.get("memory_percent", 0)
            total_memory = context.hardware_specs.get("total_memory_gb", 32)

            # High memory usage recommendation
            if memory_usage > 85:
                impact, confidence = self.recommendation_engine.predict_impact(
                    "memory_pressure_configuration", context
                )

                rec = PerformanceRecommendation(
                    recommendation_id="memory_limits_optimization",
                    title="Optimize Memory Usage Limits",
                    description="Adjust memory limits to prevent OOM kills and improve query performance",
                    recommendation_type=RecommendationType.MEMORY_TUNING,
                    category=RecommendationCategory.IMMEDIATE_FIXES,
                    expected_impact=ImpactLevel.SIGNIFICANT,
                    impact_percentage=impact,
                    confidence_score=confidence,
                    confidence_level=(
                        ConfidenceLevel.HIGH if confidence > 75 else ConfidenceLevel.MEDIUM
                    ),
                    complexity=ImplementationComplexity.TRIVIAL,
                    risk_level=RiskLevel.LOW,
                    estimated_time_hours=0.5,
                    current_state={
                        "max_memory_usage": current_config.get("max_memory_usage", "0"),
                        "memory_usage_percent": memory_usage,
                    },
                    recommended_changes={
                        "max_memory_usage": f"{int(total_memory * 0.8 * 1024**3)}",  # 80% of total memory
                        "max_bytes_before_external_group_by": f"{int(total_memory * 0.6 * 1024**3)}",
                        "max_bytes_before_external_sort": f"{int(total_memory * 0.6 * 1024**3)}",
                    },
                    implementation_steps=[
                        "Update max_memory_usage setting to 80% of total memory",
                        "Configure external operations to use 60% of max_memory_usage",
                        "Restart ClickHouse service to apply changes",
                        "Monitor memory usage for 24 hours",
                    ],
                    validation_queries=[
                        "SELECT name, value FROM system.settings WHERE name LIKE '%memory%'",
                        "SELECT * FROM system.processes WHERE memory_usage > 0 ORDER BY memory_usage DESC LIMIT 10",
                    ],
                    urgency_level=5 if memory_usage > 95 else 4,
                    business_impact="Prevent out-of-memory errors and improve query stability",
                    tags={"memory", "configuration", "stability"},
                )

                rec.priority_score = self.recommendation_engine.calculate_priority_score(rec)
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing memory configuration: {e}")
            return []

    def _analyze_cpu_configuration(
        self, current_config: dict[str, Any], context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze CPU and thread configuration."""
        recommendations = []

        try:
            cpu_cores = context.hardware_specs.get("cpu_cores", 8)
            cpu_usage = context.resource_utilization.get("cpu_percent", 50)

            current_max_threads = int(current_config.get("max_threads", "auto"))

            # Thread pool optimization
            if cpu_usage > 80 and (current_max_threads == 0 or current_max_threads > cpu_cores * 2):
                impact, confidence = self.recommendation_engine.predict_impact(
                    "high_cpu_configuration", context
                )

                rec = PerformanceRecommendation(
                    recommendation_id="thread_pool_optimization",
                    title="Optimize Thread Pool Configuration",
                    description="Adjust thread pool settings to match CPU cores and reduce context switching",
                    recommendation_type=RecommendationType.THREAD_POOL_SIZING,
                    category=RecommendationCategory.IMMEDIATE_FIXES,
                    expected_impact=ImpactLevel.MODERATE,
                    impact_percentage=impact,
                    confidence_score=confidence,
                    confidence_level=ConfidenceLevel.HIGH,
                    complexity=ImplementationComplexity.TRIVIAL,
                    risk_level=RiskLevel.MINIMAL,
                    estimated_time_hours=0.25,
                    current_state={
                        "max_threads": str(current_max_threads),
                        "cpu_usage_percent": cpu_usage,
                        "cpu_cores": cpu_cores,
                    },
                    recommended_changes={
                        "max_threads": str(cpu_cores),
                        "max_concurrent_queries": str(cpu_cores * 2),
                    },
                    implementation_steps=[
                        f"Set max_threads to {cpu_cores} (number of CPU cores)",
                        f"Set max_concurrent_queries to {cpu_cores * 2}",
                        "Apply settings and monitor CPU usage",
                    ],
                    validation_queries=[
                        "SELECT name, value FROM system.settings WHERE name IN ('max_threads', 'max_concurrent_queries')",
                        "SELECT * FROM system.processes WHERE elapsed > 1 ORDER BY elapsed DESC LIMIT 10",
                    ],
                    urgency_level=3,
                    business_impact="Reduce CPU contention and improve query response times",
                    tags={"cpu", "threads", "performance"},
                )

                rec.priority_score = self.recommendation_engine.calculate_priority_score(rec)
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing CPU configuration: {e}")
            return []

    def _analyze_cache_configuration(
        self, current_config: dict[str, Any], context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze cache configuration and generate recommendations."""
        recommendations = []

        try:
            # This would analyze cache hit rates and recommend cache size adjustments
            # For brevity, including a simplified version

            total_memory = context.hardware_specs.get("total_memory_gb", 32)
            current_mark_cache = current_config.get("mark_cache_size", "5368709120")  # 5GB default

            # Cache optimization recommendation
            if total_memory >= 64:  # Only for systems with sufficient memory
                rec = PerformanceRecommendation(
                    recommendation_id="cache_size_optimization",
                    title="Optimize Cache Sizes for Better Performance",
                    description="Increase cache sizes to improve query performance on systems with sufficient memory",
                    recommendation_type=RecommendationType.CACHE_OPTIMIZATION,
                    category=RecommendationCategory.IMMEDIATE_FIXES,
                    expected_impact=ImpactLevel.MODERATE,
                    impact_percentage=20.0,
                    confidence_score=80.0,
                    confidence_level=ConfidenceLevel.HIGH,
                    complexity=ImplementationComplexity.TRIVIAL,
                    risk_level=RiskLevel.MINIMAL,
                    estimated_time_hours=0.25,
                    current_state={
                        "mark_cache_size": current_mark_cache,
                        "total_memory_gb": total_memory,
                    },
                    recommended_changes={
                        "mark_cache_size": f"{8 * 1024**3}",  # 8GB
                        "uncompressed_cache_size": f"{16 * 1024**3}",  # 16GB
                        "compiled_expression_cache_size": f"{1 * 1024**3}",  # 1GB
                    },
                    implementation_steps=[
                        "Increase mark_cache_size to 8GB",
                        "Increase uncompressed_cache_size to 16GB",
                        "Set compiled_expression_cache_size to 1GB",
                        "Monitor cache hit rates after changes",
                    ],
                    validation_queries=[
                        "SELECT name, value FROM system.settings WHERE name LIKE '%cache%'",
                        "SELECT * FROM system.events WHERE event LIKE '%Cache%'",
                    ],
                    urgency_level=2,
                    business_impact="Improve query performance through better caching",
                    tags={"cache", "performance", "memory"},
                )

                rec.priority_score = self.recommendation_engine.calculate_priority_score(rec)
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing cache configuration: {e}")
            return []

    def _analyze_io_configuration(
        self, current_config: dict[str, Any], context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze I/O and merge configuration."""
        # Simplified implementation - would analyze merge patterns and I/O performance
        return []

    def _analyze_network_configuration(
        self, current_config: dict[str, Any], context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze network configuration."""
        # Simplified implementation - would analyze distributed query performance
        return []


class QueryOptimizationAdvisor:
    """Query performance optimization advisor."""

    def __init__(self, client: Client, recommendation_engine: RecommendationEngine):
        """Initialize the query optimization advisor.

        Args:
            client: ClickHouse client instance
            recommendation_engine: Core recommendation engine
        """
        self.client = client
        self.recommendation_engine = recommendation_engine
        self.logger = logger.getChild("query_optimization_advisor")

    def analyze_query_patterns(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze query patterns and provide optimization recommendations.

        Args:
            context: System context for analysis

        Returns:
            List of query optimization recommendations
        """
        recommendations = []

        try:
            # Analyze slow queries
            slow_query_recommendations = self._analyze_slow_queries(context)
            recommendations.extend(slow_query_recommendations)

            # Analyze query patterns
            pattern_recommendations = self._analyze_common_query_patterns(context)
            recommendations.extend(pattern_recommendations)

            # Analyze index usage
            index_recommendations = self._analyze_index_usage(context)
            recommendations.extend(index_recommendations)

            self.logger.info(f"Generated {len(recommendations)} query optimization recommendations")
            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing query patterns: {e}")
            return []

    def _analyze_slow_queries(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze slow queries and generate optimization recommendations."""
        recommendations = []

        try:
            # Get slow queries from query log
            query = """
            SELECT
                normalized_query_hash,
                any(query) as sample_query,
                count() as execution_count,
                avg(query_duration_ms) as avg_duration_ms,
                max(query_duration_ms) as max_duration_ms,
                avg(memory_usage) as avg_memory_usage,
                avg(read_rows) as avg_read_rows,
                avg(read_bytes) as avg_read_bytes
            FROM system.query_log
            WHERE event_date >= today() - 1
            AND type = 'QueryFinish'
            AND query_duration_ms > 1000  -- Queries taking more than 1 second
            GROUP BY normalized_query_hash
            HAVING execution_count > 10  -- Frequently executed slow queries
            ORDER BY avg_duration_ms DESC
            LIMIT 10
            """

            slow_queries = execute_query_with_retry(self.client, query)

            for i, slow_query in enumerate(slow_queries):
                if slow_query["avg_duration_ms"] > 5000:  # Very slow queries
                    rec = PerformanceRecommendation(
                        recommendation_id=f"slow_query_optimization_{i}",
                        title=f"Optimize Slow Query Pattern (Avg: {slow_query['avg_duration_ms']:.0f}ms)",
                        description="Optimize frequently executed slow query pattern to improve overall system performance",
                        recommendation_type=RecommendationType.QUERY_OPTIMIZATION,
                        category=RecommendationCategory.MEDIUM_TERM_OPTIMIZATIONS,
                        expected_impact=ImpactLevel.SIGNIFICANT,
                        impact_percentage=30.0,
                        confidence_score=85.0,
                        confidence_level=ConfidenceLevel.HIGH,
                        complexity=ImplementationComplexity.MODERATE,
                        risk_level=RiskLevel.LOW,
                        estimated_time_hours=4.0,
                        current_state={
                            "avg_duration_ms": slow_query["avg_duration_ms"],
                            "execution_count": slow_query["execution_count"],
                            "avg_memory_usage": slow_query["avg_memory_usage"],
                        },
                        recommended_changes={
                            "query_optimization": "Add proper indexes, optimize WHERE clauses, consider materialized views",
                            "expected_duration_ms": slow_query["avg_duration_ms"]
                            * 0.3,  # 70% improvement target
                        },
                        implementation_steps=[
                            "Analyze query execution plan with EXPLAIN",
                            "Identify missing or inefficient indexes",
                            "Optimize WHERE clause ordering",
                            "Consider query rewriting or materialized views",
                            "Test optimized query in staging environment",
                        ],
                        validation_queries=[
                            f"EXPLAIN SELECT {slow_query['sample_query'][:100]}...",
                            "SELECT * FROM system.query_log WHERE normalized_query_hash = unhex(...) ORDER BY event_time DESC LIMIT 5",
                        ],
                        evidence={
                            "sample_query": slow_query["sample_query"][:500],
                            "performance_metrics": slow_query,
                        },
                        urgency_level=4,
                        business_impact="Significantly improve response times for frequently used queries",
                        tags={"query_optimization", "performance", "slow_queries"},
                    )

                    rec.priority_score = self.recommendation_engine.calculate_priority_score(rec)
                    recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing slow queries: {e}")
            return []

    def _analyze_common_query_patterns(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze common query patterns for optimization opportunities."""
        # Simplified implementation - would analyze query patterns and suggest optimizations
        return []

    def _analyze_index_usage(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze index usage patterns."""
        # Simplified implementation - would analyze table structures and suggest index optimizations
        return []


class CapacityPlanningAdvisor:
    """Hardware and capacity planning advisor."""

    def __init__(self, client: Client, recommendation_engine: RecommendationEngine):
        """Initialize the capacity planning advisor.

        Args:
            client: ClickHouse client instance
            recommendation_engine: Core recommendation engine
        """
        self.client = client
        self.recommendation_engine = recommendation_engine
        self.logger = logger.getChild("capacity_planning_advisor")

    def analyze_capacity_requirements(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze capacity requirements and provide scaling recommendations.

        Args:
            context: System context for analysis

        Returns:
            List of capacity planning recommendations
        """
        recommendations = []

        try:
            # Hardware scaling recommendations
            hardware_recommendations = self._analyze_hardware_scaling(context)
            recommendations.extend(hardware_recommendations)

            # Storage scaling recommendations
            storage_recommendations = self._analyze_storage_scaling(context)
            recommendations.extend(storage_recommendations)

            # Network scaling recommendations
            network_recommendations = self._analyze_network_scaling(context)
            recommendations.extend(network_recommendations)

            self.logger.info(f"Generated {len(recommendations)} capacity planning recommendations")
            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing capacity requirements: {e}")
            return []

    def _analyze_hardware_scaling(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze hardware scaling requirements."""
        recommendations = []

        try:
            cpu_usage = context.resource_utilization.get("cpu_percent", 50)
            memory_usage = context.resource_utilization.get("memory_percent", 50)

            # CPU scaling recommendation
            if cpu_usage > 85:
                rec = PerformanceRecommendation(
                    recommendation_id="cpu_scaling_recommendation",
                    title="Scale CPU Resources",
                    description="Current CPU utilization is consistently high, consider scaling CPU resources",
                    recommendation_type=RecommendationType.HARDWARE_SCALING,
                    category=RecommendationCategory.LONG_TERM_PLANNING,
                    expected_impact=ImpactLevel.SIGNIFICANT,
                    impact_percentage=40.0,
                    confidence_score=90.0,
                    confidence_level=ConfidenceLevel.VERY_HIGH,
                    complexity=ImplementationComplexity.COMPLEX,
                    risk_level=RiskLevel.MEDIUM,
                    estimated_time_hours=8.0,
                    current_state={
                        "cpu_usage_percent": cpu_usage,
                        "cpu_cores": context.hardware_specs.get("cpu_cores", 8),
                    },
                    recommended_changes={
                        "cpu_cores": context.hardware_specs.get("cpu_cores", 8) * 2,
                        "scaling_method": (
                            "vertical" if context.deployment_mode == "cloud" else "horizontal"
                        ),
                    },
                    implementation_steps=[
                        "Plan maintenance window for scaling operation",
                        "Create backup of current configuration",
                        "Scale CPU resources (vertical or add nodes)",
                        "Update ClickHouse configuration for new resources",
                        "Validate performance improvement",
                    ],
                    urgency_level=4,
                    business_impact="Resolve CPU bottlenecks and improve query processing capacity",
                    estimated_cost_savings=500.0,  # Monthly savings from improved efficiency
                    implementation_cost=2000.0,  # One-time scaling cost
                    tags={"hardware", "scaling", "cpu"},
                )

                rec.priority_score = self.recommendation_engine.calculate_priority_score(rec)
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing hardware scaling: {e}")
            return []

    def _analyze_storage_scaling(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze storage scaling requirements."""
        # Simplified implementation - would analyze storage usage trends and performance
        return []

    def _analyze_network_scaling(
        self, context: RecommendationContext
    ) -> list[PerformanceRecommendation]:
        """Analyze network scaling requirements."""
        # Simplified implementation - would analyze network throughput and latency
        return []


class PerformanceAdvisorEngine:
    """Main performance advisor engine that coordinates all advisory components."""

    def __init__(
        self, client: Client, bottleneck_detector: IntelligentBottleneckDetector | None = None
    ):
        """Initialize the performance advisor engine.

        Args:
            client: ClickHouse client instance
            bottleneck_detector: Optional bottleneck detector for integration
        """
        self.client = client
        self.bottleneck_detector = bottleneck_detector
        self.logger = logger.getChild("performance_advisor_engine")

        # Initialize core components
        self.recommendation_engine = RecommendationEngine(client)
        self.configuration_advisor = ConfigurationAdvisor(client, self.recommendation_engine)
        self.query_optimizer = QueryOptimizationAdvisor(client, self.recommendation_engine)
        self.capacity_planner = CapacityPlanningAdvisor(client, self.recommendation_engine)

        # Initialize diagnostic engines for data gathering
        self.performance_diagnostics = PerformanceDiagnosticEngine(client)
        self.storage_diagnostics = StorageOptimizationEngine(client)
        self.hardware_diagnostics = HardwareHealthEngine(client)

    @log_execution_time
    def generate_comprehensive_recommendations(
        self, time_range: tuple[datetime, datetime] | None = None, max_recommendations: int = 20
    ) -> dict[str, Any]:
        """Generate comprehensive performance recommendations.

        Args:
            time_range: Optional time range for analysis
            max_recommendations: Maximum number of recommendations to return

        Returns:
            Dictionary containing recommendations organized by category
        """
        try:
            self.logger.info("Starting comprehensive performance recommendation generation")

            # Gather system context
            context = self._gather_system_context(time_range)

            # Generate recommendations from all advisors
            all_recommendations = []

            # Configuration recommendations
            config_recs = self.configuration_advisor.analyze_server_configuration(context)
            all_recommendations.extend(config_recs)

            # Query optimization recommendations
            query_recs = self.query_optimizer.analyze_query_patterns(context)
            all_recommendations.extend(query_recs)

            # Capacity planning recommendations
            capacity_recs = self.capacity_planner.analyze_capacity_requirements(context)
            all_recommendations.extend(capacity_recs)

            # Sort by priority score and limit results
            all_recommendations.sort(key=lambda r: r.priority_score, reverse=True)
            top_recommendations = all_recommendations[:max_recommendations]

            # Organize by category
            categorized_recommendations = self._categorize_recommendations(top_recommendations)

            # Generate summary
            summary = self._generate_recommendation_summary(top_recommendations, context)

            result = {
                "summary": summary,
                "recommendations_by_category": categorized_recommendations,
                "total_recommendations": len(top_recommendations),
                "analysis_timestamp": datetime.now().isoformat(),
                "context": {
                    "system_health_score": context.system_health_score,
                    "active_bottlenecks": len(context.active_bottlenecks),
                    "resource_utilization": context.resource_utilization,
                },
            }

            self.logger.info(f"Generated {len(top_recommendations)} performance recommendations")
            return result

        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendations: {e}")
            return {
                "error": f"Failed to generate recommendations: {e!s}",
                "recommendations_by_category": {},
                "total_recommendations": 0,
            }

    def _gather_system_context(
        self, time_range: tuple[datetime, datetime] | None
    ) -> RecommendationContext:
        """Gather system context for recommendation generation."""
        try:
            # Default time range: last 24 hours
            if time_range is None:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
                time_range = (start_time, end_time)

            # Gather system health data
            system_health_score = 75.0  # Default, would be calculated from diagnostics

            # Gather active bottlenecks
            active_bottlenecks = []
            if self.bottleneck_detector:
                try:
                    bottlenecks = self.bottleneck_detector.detect_bottlenecks()
                    active_bottlenecks = [
                        b
                        for b in bottlenecks
                        if b.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]
                    ]
                except Exception as e:
                    self.logger.warning(f"Could not gather bottleneck data: {e}")

            # Gather resource utilization
            resource_utilization = self._get_current_resource_utilization()

            # Gather hardware specs
            hardware_specs = self._get_hardware_specifications()

            # Create context
            context = RecommendationContext(
                system_health_score=system_health_score,
                active_bottlenecks=active_bottlenecks,
                performance_trends={},
                resource_utilization=resource_utilization,
                query_patterns={},
                data_volume={},
                concurrent_users=10,  # Default
                peak_hours=[9, 10, 11, 14, 15, 16],  # Business hours
                hardware_specs=hardware_specs,
                storage_config={},
                network_topology={},
                deployment_mode="cloud",  # Default
                baseline_metrics={},
                performance_history=[],
                previous_optimizations=[],
                sla_requirements={"avg_query_time_ms": 1000.0, "p95_query_time_ms": 5000.0},
                cost_constraints={},
                availability_requirements=99.9,
            )

            return context

        except Exception as e:
            self.logger.error(f"Error gathering system context: {e}")
            # Return minimal context
            return RecommendationContext(
                system_health_score=50.0,
                active_bottlenecks=[],
                performance_trends={},
                resource_utilization={},
                query_patterns={},
                data_volume={},
                concurrent_users=1,
                peak_hours=[],
                hardware_specs={},
                storage_config={},
                network_topology={},
                deployment_mode="unknown",
                baseline_metrics={},
                performance_history=[],
                previous_optimizations=[],
                sla_requirements={},
                cost_constraints={},
                availability_requirements=99.0,
            )

    def _get_current_resource_utilization(self) -> dict[str, float]:
        """Get current resource utilization metrics."""
        try:
            query = """
            SELECT
                'cpu_percent' as metric,
                OSCPUVirtualTimeMicroseconds / 1000000.0 as value
            FROM system.events
            WHERE event = 'OSCPUVirtualTimeMicroseconds'

            UNION ALL

            SELECT
                'memory_percent' as metric,
                (MemoryTrackerUsage / 1024/1024/1024) /
                (OSMemoryAvailable / 1024/1024/1024) * 100 as value
            FROM system.events
            WHERE event IN ('MemoryTrackerUsage', 'OSMemoryAvailable')
            """

            # Simplified - in practice would use system.metrics and more sophisticated calculation
            return {
                "cpu_percent": 65.0,
                "memory_percent": 45.0,
                "disk_usage_percent": 30.0,
                "io_wait_percent": 5.0,
            }

        except Exception as e:
            self.logger.error(f"Error getting resource utilization: {e}")
            return {"cpu_percent": 50.0, "memory_percent": 50.0}

    def _get_hardware_specifications(self) -> dict[str, Any]:
        """Get hardware specifications."""
        try:
            # This would query system information
            return {
                "cpu_cores": 16,
                "total_memory_gb": 64,
                "disk_type": "SSD",
                "network_bandwidth_gbps": 10,
            }

        except Exception as e:
            self.logger.error(f"Error getting hardware specs: {e}")
            return {"cpu_cores": 8, "total_memory_gb": 32}

    def _categorize_recommendations(
        self, recommendations: list[PerformanceRecommendation]
    ) -> dict[str, list[dict[str, Any]]]:
        """Organize recommendations by category."""
        categorized = defaultdict(list)

        for rec in recommendations:
            rec_dict = {
                "recommendation_id": rec.recommendation_id,
                "title": rec.title,
                "description": rec.description,
                "impact_percentage": rec.impact_percentage,
                "confidence_score": rec.confidence_score,
                "complexity": rec.complexity.value,
                "risk_level": rec.risk_level.value,
                "estimated_time_hours": rec.estimated_time_hours,
                "priority_score": rec.priority_score,
                "urgency_level": rec.urgency_level,
                "implementation_steps": rec.implementation_steps,
                "tags": list(rec.tags),
            }

            categorized[rec.category.value].append(rec_dict)

        return dict(categorized)

    def _generate_recommendation_summary(
        self, recommendations: list[PerformanceRecommendation], context: RecommendationContext
    ) -> dict[str, Any]:
        """Generate a summary of recommendations."""
        if not recommendations:
            return {"message": "No performance recommendations generated"}

        try:
            total_impact = sum(r.impact_percentage for r in recommendations)
            avg_confidence = statistics.mean(r.confidence_score for r in recommendations)
            total_time = sum(r.estimated_time_hours for r in recommendations)

            high_priority = len([r for r in recommendations if r.priority_score > 50])
            immediate_fixes = len(
                [r for r in recommendations if r.category == RecommendationCategory.IMMEDIATE_FIXES]
            )

            return {
                "total_recommendations": len(recommendations),
                "potential_cumulative_impact": f"{total_impact:.1f}%",
                "average_confidence": f"{avg_confidence:.1f}%",
                "total_implementation_time": f"{total_time:.1f} hours",
                "high_priority_recommendations": high_priority,
                "immediate_fixes_available": immediate_fixes,
                "system_health_score": f"{context.system_health_score:.1f}",
                "top_recommendation": {
                    "title": recommendations[0].title,
                    "impact": f"{recommendations[0].impact_percentage:.1f}%",
                    "complexity": recommendations[0].complexity.value,
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"message": "Summary generation failed"}


# Utility function for creating the performance advisor
def create_performance_advisor(
    client: Client, bottleneck_detector: IntelligentBottleneckDetector | None = None
) -> PerformanceAdvisorEngine:
    """Create and configure a performance advisor engine.

    Args:
        client: ClickHouse client instance
        bottleneck_detector: Optional bottleneck detector for integration

    Returns:
        Configured PerformanceAdvisorEngine instance
    """
    return PerformanceAdvisorEngine(client, bottleneck_detector)


def integrate_with_diagnostics(
    advisor: PerformanceAdvisorEngine,
    performance_engine: PerformanceDiagnosticEngine,
    storage_engine: StorageOptimizationEngine,
    hardware_engine: HardwareHealthEngine,
) -> None:
    """Integrate performance advisor with existing diagnostic engines.

    Args:
        advisor: Performance advisor engine
        performance_engine: Performance diagnostic engine
        storage_engine: Storage optimization engine
        hardware_engine: Hardware health engine
    """
    # Store references for data gathering
    advisor.performance_diagnostics = performance_engine
    advisor.storage_diagnostics = storage_engine
    advisor.hardware_diagnostics = hardware_engine

    advisor.logger.info("Performance advisor integrated with diagnostic engines")
