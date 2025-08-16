"""Hardware performance diagnostics suite for ClickHouse.

This module provides comprehensive analysis of hardware performance characteristics
including CPU efficiency, memory management, and thread pool operations using
ClickHouse ProfileEvents. It delivers deep insights into low-level system performance,
identifies hardware bottlenecks, and provides optimization recommendations for
maximizing ClickHouse performance on modern multi-core systems.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from clickhouse_connect.driver.client import Client

from agent_zero.monitoring.profile_events_core import (
    ProfileEventAggregation,
    ProfileEventsAnalyzer,
)
from agent_zero.utils import log_execution_time

logger = logging.getLogger("mcp-clickhouse")


class HardwareBottleneckType(Enum):
    """Types of hardware bottlenecks that can be detected."""

    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    CACHE_BOUND = "cache_bound"
    THREAD_CONTENTION = "thread_contention"
    CONTEXT_SWITCHING = "context_switching"
    NUMA_INEFFICIENCY = "numa_inefficiency"
    BRANCH_MISPREDICTION = "branch_misprediction"
    TLB_THRASHING = "tlb_thrashing"
    MEMORY_ALLOCATION = "memory_allocation"


class HardwareSeverity(Enum):
    """Severity levels for hardware performance issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreadPoolType(Enum):
    """Types of thread pools in ClickHouse."""

    GLOBAL = "global"
    LOCAL = "local"
    BACKGROUND_PROCESSING = "background_processing"
    BACKGROUND_MOVE = "background_move"
    BACKGROUND_FETCHES = "background_fetches"
    BACKGROUND_COMMON = "background_common"


@dataclass
class HardwareBottleneck:
    """Represents a detected hardware performance bottleneck."""

    type: HardwareBottleneckType
    severity: HardwareSeverity
    description: str
    efficiency_score: float  # 0-100 scale, higher is better
    impact_percentage: float  # Estimated performance impact
    affected_components: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    optimization_potential: float = 0.0  # Estimated improvement potential


@dataclass
class CPUAnalysis:
    """Analysis results for CPU performance."""

    efficiency_score: float  # 0-100 scale
    instructions_per_cycle: float
    cache_hit_rate: float
    branch_prediction_accuracy: float
    context_switch_overhead: float
    cpu_utilization: dict[str, Any]
    performance_counters: dict[str, Any]
    bottlenecks: list[HardwareBottleneck]
    recommendations: list[str]


@dataclass
class MemoryAnalysis:
    """Analysis results for memory performance."""

    efficiency_score: float  # 0-100 scale
    allocation_pattern_score: float
    page_fault_analysis: dict[str, Any]
    memory_pressure_indicators: dict[str, Any]
    swap_usage_analysis: dict[str, Any]
    memory_fragmentation: dict[str, Any]
    overcommit_analysis: dict[str, Any]
    bottlenecks: list[HardwareBottleneck]
    recommendations: list[str]


@dataclass
class ThreadPoolAnalysis:
    """Analysis results for thread pool performance."""

    efficiency_score: float  # 0-100 scale
    thread_utilization: dict[str, Any]
    contention_analysis: dict[str, Any]
    queue_efficiency: dict[str, Any]
    scaling_analysis: dict[str, Any]
    lock_contention: dict[str, Any]
    thread_migration: dict[str, Any]
    bottlenecks: list[HardwareBottleneck]
    recommendations: list[str]


@dataclass
class HardwareHealthReport:
    """Comprehensive hardware health assessment report."""

    overall_health_score: float  # 0-100 scale
    cpu_analysis: CPUAnalysis
    memory_analysis: MemoryAnalysis
    thread_pool_analysis: ThreadPoolAnalysis
    system_efficiency: dict[str, Any]
    capacity_planning: dict[str, Any]
    critical_bottlenecks: list[HardwareBottleneck]
    optimization_priorities: list[str]
    performance_trends: dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)


class CPUAnalyzer:
    """Analyzer for CPU performance and efficiency."""

    def __init__(self, profile_analyzer: ProfileEventsAnalyzer):
        """Initialize CPU analyzer.

        Args:
            profile_analyzer: ProfileEventsAnalyzer instance
        """
        self.profile_analyzer = profile_analyzer
        self.cpu_events = self._get_cpu_profile_events()

    def _get_cpu_profile_events(self) -> list[str]:
        """Get list of CPU-related ProfileEvents."""
        return [
            "PerfCPUCycles",
            "PerfInstructions",
            "PerfCacheMisses",
            "PerfBranchMisses",
            "PerfContextSwitches",
            "PerfCPUMigrations",
            "PerfAlignmentFaults",
            "PerfEmulationFaults",
            "PerfDataTLBMisses",
            "PerfInstructionTLBMisses",
            "UserTimeMicroseconds",
            "SystemTimeMicroseconds",
            "OSCPUVirtualTimeMicroseconds",
            "OSCPUWaitMicroseconds",
            "OSIOWaitMicroseconds",
        ]

    @log_execution_time
    def analyze_cpu_performance(
        self, start_time: datetime, end_time: datetime, query_filter: str | None = None
    ) -> CPUAnalysis:
        """Analyze CPU performance characteristics.

        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            query_filter: Optional query filter

        Returns:
            CPUAnalysis with performance metrics and recommendations
        """
        logger.info("Analyzing CPU performance characteristics")

        # Get aggregated ProfileEvents data
        aggregations = self.profile_analyzer.aggregate_profile_events(
            self.cpu_events, start_time, end_time, query_filter
        )

        # Create metrics dictionary for analysis
        metrics = {agg.event_name: agg for agg in aggregations}

        # Calculate CPU efficiency metrics
        efficiency_score = self._calculate_cpu_efficiency_score(metrics)
        instructions_per_cycle = self._calculate_instructions_per_cycle(metrics)
        cache_hit_rate = self._calculate_cache_hit_rate(metrics)
        branch_prediction_accuracy = self._calculate_branch_prediction_accuracy(metrics)
        context_switch_overhead = self._calculate_context_switch_overhead(metrics)

        # Analyze CPU utilization patterns
        cpu_utilization = self._analyze_cpu_utilization(metrics)

        # Extract performance counter data
        performance_counters = self._extract_performance_counters(metrics)

        # Identify CPU bottlenecks
        bottlenecks = self._identify_cpu_bottlenecks(metrics, efficiency_score)

        # Generate optimization recommendations
        recommendations = self._generate_cpu_recommendations(metrics, efficiency_score, bottlenecks)

        return CPUAnalysis(
            efficiency_score=efficiency_score,
            instructions_per_cycle=instructions_per_cycle,
            cache_hit_rate=cache_hit_rate,
            branch_prediction_accuracy=branch_prediction_accuracy,
            context_switch_overhead=context_switch_overhead,
            cpu_utilization=cpu_utilization,
            performance_counters=performance_counters,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _calculate_cpu_efficiency_score(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate overall CPU efficiency score (0-100)."""
        try:
            score_factors = []

            # Instructions per cycle efficiency (higher is better)
            ipc = self._calculate_instructions_per_cycle(metrics)
            if ipc > 0:
                ipc_score = min(ipc * 25, 100)  # Scale IPC to 0-100
                score_factors.append(ipc_score)

            # Cache hit rate (higher is better)
            cache_hit_rate = self._calculate_cache_hit_rate(metrics)
            score_factors.append(cache_hit_rate)

            # Branch prediction accuracy (higher is better)
            branch_accuracy = self._calculate_branch_prediction_accuracy(metrics)
            score_factors.append(branch_accuracy)

            # Context switching overhead (lower is better)
            context_overhead = self._calculate_context_switch_overhead(metrics)
            context_score = max(0, 100 - context_overhead)
            score_factors.append(context_score)

            # TLB efficiency (lower miss rate is better)
            tlb_efficiency = self._calculate_tlb_efficiency(metrics)
            score_factors.append(tlb_efficiency)

            if score_factors:
                return statistics.mean(score_factors)
            else:
                return 50.0  # Default neutral score

        except Exception as e:
            logger.warning(f"Error calculating CPU efficiency score: {e}")
            return 50.0

    def _calculate_instructions_per_cycle(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate instructions per cycle (IPC) ratio."""
        try:
            instructions = metrics.get("PerfInstructions")
            cycles = metrics.get("PerfCPUCycles")

            if instructions and cycles and cycles.avg_value > 0:
                return instructions.avg_value / cycles.avg_value
            return 0.0
        except Exception:
            return 0.0

    def _calculate_cache_hit_rate(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate cache hit rate percentage."""
        try:
            cache_misses = metrics.get("PerfCacheMisses")
            instructions = metrics.get("PerfInstructions")

            if cache_misses and instructions and instructions.avg_value > 0:
                miss_rate = cache_misses.avg_value / instructions.avg_value
                hit_rate = max(0, 1 - miss_rate) * 100
                return min(hit_rate, 100)
            return 95.0  # Default good hit rate
        except Exception:
            return 95.0

    def _calculate_branch_prediction_accuracy(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate branch prediction accuracy percentage."""
        try:
            branch_misses = metrics.get("PerfBranchMisses")
            instructions = metrics.get("PerfInstructions")

            if branch_misses and instructions and instructions.avg_value > 0:
                # Estimate total branches as ~15% of instructions
                estimated_branches = instructions.avg_value * 0.15
                if estimated_branches > 0:
                    miss_rate = branch_misses.avg_value / estimated_branches
                    accuracy = max(0, 1 - miss_rate) * 100
                    return min(accuracy, 100)
            return 95.0  # Default good accuracy
        except Exception:
            return 95.0

    def _calculate_context_switch_overhead(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate context switch overhead as percentage of total time."""
        try:
            context_switches = metrics.get("PerfContextSwitches")
            cpu_time = metrics.get("OSCPUVirtualTimeMicroseconds")

            if context_switches and cpu_time and cpu_time.avg_value > 0:
                # Estimate context switch cost (typical 3-5 microseconds)
                switch_cost_us = 4.0
                switch_overhead_us = context_switches.avg_value * switch_cost_us
                overhead_percentage = (switch_overhead_us / cpu_time.avg_value) * 100
                return min(overhead_percentage, 100)
            return 5.0  # Default low overhead
        except Exception:
            return 5.0

    def _calculate_tlb_efficiency(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate TLB (Translation Lookaside Buffer) efficiency."""
        try:
            data_tlb_misses = metrics.get("PerfDataTLBMisses")
            instruction_tlb_misses = metrics.get("PerfInstructionTLBMisses")
            instructions = metrics.get("PerfInstructions")

            if instructions and instructions.avg_value > 0:
                total_tlb_misses = 0
                if data_tlb_misses:
                    total_tlb_misses += data_tlb_misses.avg_value
                if instruction_tlb_misses:
                    total_tlb_misses += instruction_tlb_misses.avg_value

                tlb_miss_rate = total_tlb_misses / instructions.avg_value
                efficiency = max(0, 1 - tlb_miss_rate) * 100
                return min(efficiency, 100)
            return 98.0  # Default high efficiency
        except Exception:
            return 98.0

    def _analyze_cpu_utilization(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze CPU utilization patterns."""
        try:
            user_time = metrics.get("UserTimeMicroseconds")
            system_time = metrics.get("SystemTimeMicroseconds")
            cpu_virtual_time = metrics.get("OSCPUVirtualTimeMicroseconds")
            cpu_wait_time = metrics.get("OSCPUWaitMicroseconds")
            io_wait_time = metrics.get("OSIOWaitMicroseconds")

            utilization = {
                "user_time_us": user_time.avg_value if user_time else 0,
                "system_time_us": system_time.avg_value if system_time else 0,
                "virtual_time_us": cpu_virtual_time.avg_value if cpu_virtual_time else 0,
                "cpu_wait_us": cpu_wait_time.avg_value if cpu_wait_time else 0,
                "io_wait_us": io_wait_time.avg_value if io_wait_time else 0,
            }

            # Calculate utilization percentages
            total_time = (
                utilization["user_time_us"]
                + utilization["system_time_us"]
                + utilization["cpu_wait_us"]
                + utilization["io_wait_us"]
            )

            if total_time > 0:
                utilization["user_percentage"] = (utilization["user_time_us"] / total_time) * 100
                utilization["system_percentage"] = (
                    utilization["system_time_us"] / total_time
                ) * 100
                utilization["wait_percentage"] = (utilization["cpu_wait_us"] / total_time) * 100
                utilization["io_wait_percentage"] = (utilization["io_wait_us"] / total_time) * 100
            else:
                utilization.update(
                    {
                        "user_percentage": 0,
                        "system_percentage": 0,
                        "wait_percentage": 0,
                        "io_wait_percentage": 0,
                    }
                )

            # Determine utilization pattern
            if utilization["io_wait_percentage"] > 30:
                utilization["pattern"] = "IO_BOUND"
            elif utilization["system_percentage"] > 40:
                utilization["pattern"] = "SYSTEM_INTENSIVE"
            elif utilization["user_percentage"] > 70:
                utilization["pattern"] = "CPU_INTENSIVE"
            else:
                utilization["pattern"] = "BALANCED"

            return utilization

        except Exception as e:
            logger.warning(f"Error analyzing CPU utilization: {e}")
            return {"pattern": "UNKNOWN", "error": str(e)}

    def _extract_performance_counters(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Extract hardware performance counter data."""
        counters = {}

        for event_name, agg in metrics.items():
            if event_name.startswith("Perf"):
                counters[event_name] = {
                    "average": agg.avg_value,
                    "total": agg.sum_value,
                    "max": agg.max_value,
                    "p99": agg.p99_value,
                    "unit": (
                        "count" if "Misses" in event_name or "Faults" in event_name else "cycles"
                    ),
                }

        return counters

    def _identify_cpu_bottlenecks(
        self, metrics: dict[str, ProfileEventAggregation], efficiency_score: float
    ) -> list[HardwareBottleneck]:
        """Identify CPU performance bottlenecks."""
        bottlenecks = []

        try:
            # High cache miss rate
            cache_hit_rate = self._calculate_cache_hit_rate(metrics)
            if cache_hit_rate < 85:
                severity = HardwareSeverity.HIGH if cache_hit_rate < 75 else HardwareSeverity.MEDIUM
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.CACHE_BOUND,
                        severity=severity,
                        description=f"Low cache hit rate of {cache_hit_rate:.1f}%",
                        efficiency_score=cache_hit_rate,
                        impact_percentage=max(0, 90 - cache_hit_rate),
                        affected_components=["L1 Cache", "L2 Cache", "L3 Cache"],
                        recommendations=[
                            "Optimize data access patterns for better cache locality",
                            "Consider data structure layout optimizations",
                            "Increase cache sizes if possible",
                            "Use cache-friendly algorithms",
                        ],
                        metrics={"cache_hit_rate": cache_hit_rate},
                        optimization_potential=min(30, 90 - cache_hit_rate),
                    )
                )

            # High branch misprediction
            branch_accuracy = self._calculate_branch_prediction_accuracy(metrics)
            if branch_accuracy < 90:
                severity = (
                    HardwareSeverity.HIGH if branch_accuracy < 85 else HardwareSeverity.MEDIUM
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.BRANCH_MISPREDICTION,
                        severity=severity,
                        description=f"Low branch prediction accuracy of {branch_accuracy:.1f}%",
                        efficiency_score=branch_accuracy,
                        impact_percentage=max(0, 95 - branch_accuracy),
                        affected_components=["Branch Predictor", "Pipeline"],
                        recommendations=[
                            "Reduce complex branching in hot code paths",
                            "Use branchless programming techniques where possible",
                            "Profile and optimize conditional logic",
                            "Consider lookup tables for complex conditions",
                        ],
                        metrics={"branch_accuracy": branch_accuracy},
                    )
                )

            # High context switching overhead
            context_overhead = self._calculate_context_switch_overhead(metrics)
            if context_overhead > 15:
                severity = (
                    HardwareSeverity.HIGH if context_overhead > 25 else HardwareSeverity.MEDIUM
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.CONTEXT_SWITCHING,
                        severity=severity,
                        description=f"High context switching overhead of {context_overhead:.1f}%",
                        efficiency_score=max(0, 100 - context_overhead),
                        impact_percentage=min(context_overhead, 50),
                        affected_components=["CPU Scheduler", "Thread Management"],
                        recommendations=[
                            "Reduce number of active threads",
                            "Use thread pools more efficiently",
                            "Minimize thread synchronization points",
                            "Consider CPU affinity settings",
                        ],
                        metrics={"context_overhead_percentage": context_overhead},
                    )
                )

            # Low instructions per cycle
            ipc = self._calculate_instructions_per_cycle(metrics)
            if ipc < 1.0 and ipc > 0:
                severity = HardwareSeverity.HIGH if ipc < 0.5 else HardwareSeverity.MEDIUM
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.CPU_BOUND,
                        severity=severity,
                        description=f"Low instructions per cycle ratio of {ipc:.2f}",
                        efficiency_score=min(ipc * 50, 100),
                        impact_percentage=max(0, (2.0 - ipc) * 25),
                        affected_components=["CPU Pipeline", "Execution Units"],
                        recommendations=[
                            "Optimize algorithm complexity",
                            "Reduce memory stalls through better data locality",
                            "Use CPU-specific optimizations",
                            "Profile for instruction-level bottlenecks",
                        ],
                        metrics={"instructions_per_cycle": ipc},
                    )
                )

            # TLB thrashing
            tlb_efficiency = self._calculate_tlb_efficiency(metrics)
            if tlb_efficiency < 95:
                severity = HardwareSeverity.HIGH if tlb_efficiency < 90 else HardwareSeverity.MEDIUM
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.TLB_THRASHING,
                        severity=severity,
                        description=f"Low TLB efficiency of {tlb_efficiency:.1f}%",
                        efficiency_score=tlb_efficiency,
                        impact_percentage=max(0, 98 - tlb_efficiency),
                        affected_components=["TLB", "Memory Management"],
                        recommendations=[
                            "Use huge pages where appropriate",
                            "Optimize memory access patterns",
                            "Reduce memory fragmentation",
                            "Consider NUMA topology optimizations",
                        ],
                        metrics={"tlb_efficiency": tlb_efficiency},
                    )
                )

        except Exception as e:
            logger.warning(f"Error identifying CPU bottlenecks: {e}")

        return bottlenecks

    def _generate_cpu_recommendations(
        self,
        metrics: dict[str, ProfileEventAggregation],
        efficiency_score: float,
        bottlenecks: list[HardwareBottleneck],
    ) -> list[str]:
        """Generate CPU optimization recommendations."""
        recommendations = []

        try:
            # General efficiency recommendations
            if efficiency_score < 70:
                recommendations.append(
                    "Overall CPU efficiency is low - consider comprehensive performance profiling"
                )

            # Specific recommendations based on bottlenecks
            bottleneck_types = {b.type for b in bottlenecks}

            if HardwareBottleneckType.CACHE_BOUND in bottleneck_types:
                recommendations.extend(
                    [
                        "Optimize data structures for cache efficiency",
                        "Use cache-oblivious algorithms where possible",
                        "Consider prefetching strategies for predictable access patterns",
                    ]
                )

            if HardwareBottleneckType.BRANCH_MISPREDICTION in bottleneck_types:
                recommendations.extend(
                    [
                        "Profile branch prediction and optimize hot paths",
                        "Use profile-guided optimization (PGO) if available",
                        "Consider branchless implementations for critical code",
                    ]
                )

            if HardwareBottleneckType.CONTEXT_SWITCHING in bottleneck_types:
                recommendations.extend(
                    [
                        "Review thread pool configuration and sizing",
                        "Minimize synchronization overhead",
                        "Consider CPU affinity for I/O intensive threads",
                    ]
                )

            # CPU utilization pattern recommendations
            utilization = self._analyze_cpu_utilization(metrics)
            if utilization.get("pattern") == "IO_BOUND":
                recommendations.append(
                    "System appears I/O bound - focus on storage and network optimizations"
                )
            elif utilization.get("pattern") == "SYSTEM_INTENSIVE":
                recommendations.append(
                    "High system time indicates potential kernel/driver bottlenecks"
                )

            # Hardware-specific recommendations
            recommendations.extend(
                [
                    "Monitor CPU frequency scaling and thermal throttling",
                    "Ensure optimal NUMA node allocation for memory-intensive operations",
                    "Consider CPU-specific compiler optimizations and instruction sets",
                ]
            )

        except Exception as e:
            logger.warning(f"Error generating CPU recommendations: {e}")
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return recommendations


class MemoryAnalyzer:
    """Analyzer for memory performance and allocation patterns."""

    def __init__(self, profile_analyzer: ProfileEventsAnalyzer):
        """Initialize memory analyzer.

        Args:
            profile_analyzer: ProfileEventsAnalyzer instance
        """
        self.profile_analyzer = profile_analyzer
        self.memory_events = self._get_memory_profile_events()

    def _get_memory_profile_events(self) -> list[str]:
        """Get list of memory-related ProfileEvents."""
        return [
            "MemoryOvercommitWaitTimeMicroseconds",
            "SoftPageFaults",
            "HardPageFaults",
            "ArenaAllocChunks",
            "ArenaAllocBytes",
            "MemoryTrackingInBackgroundProcessingPoolAllocated",
            "MemoryTrackingInBackgroundMoveProcessingPoolAllocated",
            "MemoryTrackingForMerges",
            "ContextLock",
            "RWLockAcquiredReadLocks",
            "RWLockAcquiredWriteLocks",
            "OSReadBytes",
            "OSWriteBytes",
            "OSReadChars",
            "OSWriteChars",
        ]

    @log_execution_time
    def analyze_memory_performance(
        self, start_time: datetime, end_time: datetime, query_filter: str | None = None
    ) -> MemoryAnalysis:
        """Analyze memory performance characteristics.

        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            query_filter: Optional query filter

        Returns:
            MemoryAnalysis with performance metrics and recommendations
        """
        logger.info("Analyzing memory performance characteristics")

        # Get aggregated ProfileEvents data
        aggregations = self.profile_analyzer.aggregate_profile_events(
            self.memory_events, start_time, end_time, query_filter
        )

        # Create metrics dictionary for analysis
        metrics = {agg.event_name: agg for agg in aggregations}

        # Calculate memory efficiency metrics
        efficiency_score = self._calculate_memory_efficiency_score(metrics)
        allocation_pattern_score = self._calculate_allocation_pattern_score(metrics)

        # Analyze memory subsystems
        page_fault_analysis = self._analyze_page_faults(metrics)
        memory_pressure_indicators = self._analyze_memory_pressure(metrics)
        swap_usage_analysis = self._analyze_swap_usage(metrics)
        memory_fragmentation = self._analyze_memory_fragmentation(metrics)
        overcommit_analysis = self._analyze_memory_overcommit(metrics)

        # Identify memory bottlenecks
        bottlenecks = self._identify_memory_bottlenecks(metrics, efficiency_score)

        # Generate optimization recommendations
        recommendations = self._generate_memory_recommendations(
            metrics, efficiency_score, bottlenecks
        )

        return MemoryAnalysis(
            efficiency_score=efficiency_score,
            allocation_pattern_score=allocation_pattern_score,
            page_fault_analysis=page_fault_analysis,
            memory_pressure_indicators=memory_pressure_indicators,
            swap_usage_analysis=swap_usage_analysis,
            memory_fragmentation=memory_fragmentation,
            overcommit_analysis=overcommit_analysis,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _calculate_memory_efficiency_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate overall memory efficiency score (0-100)."""
        try:
            score_factors = []

            # Page fault efficiency (lower is better)
            page_fault_score = self._calculate_page_fault_score(metrics)
            score_factors.append(page_fault_score)

            # Memory allocation efficiency
            allocation_efficiency = self._calculate_allocation_efficiency(metrics)
            score_factors.append(allocation_efficiency)

            # Memory pressure score
            pressure_score = self._calculate_memory_pressure_score(metrics)
            score_factors.append(pressure_score)

            # Lock contention score (lower contention is better)
            lock_score = self._calculate_lock_contention_score(metrics)
            score_factors.append(lock_score)

            if score_factors:
                return statistics.mean(score_factors)
            else:
                return 50.0  # Default neutral score

        except Exception as e:
            logger.warning(f"Error calculating memory efficiency score: {e}")
            return 50.0

    def _calculate_allocation_pattern_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate allocation pattern efficiency score."""
        try:
            arena_chunks = metrics.get("ArenaAllocChunks")
            arena_bytes = metrics.get("ArenaAllocBytes")

            if arena_chunks and arena_bytes and arena_chunks.avg_value > 0:
                avg_chunk_size = arena_bytes.avg_value / arena_chunks.avg_value

                # Optimal chunk sizes are typically 4KB-64KB
                if 4096 <= avg_chunk_size <= 65536:
                    return 90.0
                elif 1024 <= avg_chunk_size <= 131072:
                    return 75.0
                else:
                    return 50.0

            return 70.0  # Default moderate score

        except Exception:
            return 70.0

    def _calculate_page_fault_score(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate page fault efficiency score."""
        try:
            soft_faults = metrics.get("SoftPageFaults")
            hard_faults = metrics.get("HardPageFaults")

            total_faults = 0
            if soft_faults:
                total_faults += soft_faults.avg_value
            if hard_faults:
                total_faults += hard_faults.avg_value * 10  # Hard faults are more expensive

            # Score based on fault rate (lower is better)
            if total_faults < 100:
                return 95.0
            elif total_faults < 1000:
                return 85.0
            elif total_faults < 10000:
                return 70.0
            else:
                return 50.0

        except Exception:
            return 80.0

    def _calculate_allocation_efficiency(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate memory allocation efficiency."""
        try:
            # Check various allocation pools
            bg_pool = metrics.get("MemoryTrackingInBackgroundProcessingPoolAllocated")
            bg_move_pool = metrics.get("MemoryTrackingInBackgroundMoveProcessingPoolAllocated")
            merge_memory = metrics.get("MemoryTrackingForMerges")

            allocation_pools = [bg_pool, bg_move_pool, merge_memory]
            active_pools = [pool for pool in allocation_pools if pool and pool.avg_value > 0]

            if active_pools:
                # Check for balanced allocation across pools
                values = [pool.avg_value for pool in active_pools]
                if len(values) > 1:
                    std_dev = statistics.stdev(values)
                    mean_val = statistics.mean(values)
                    if mean_val > 0:
                        coefficient_of_variation = std_dev / mean_val
                        # Lower CV indicates more balanced allocation
                        return max(50, 100 - (coefficient_of_variation * 100))

                return 75.0  # Default good score for active allocation

            return 60.0  # Default score when no significant allocation detected

        except Exception:
            return 60.0

    def _calculate_memory_pressure_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate memory pressure score."""
        try:
            overcommit_wait = metrics.get("MemoryOvercommitWaitTimeMicroseconds")

            if overcommit_wait and overcommit_wait.avg_value > 0:
                # High wait times indicate memory pressure
                wait_time_ms = overcommit_wait.avg_value / 1000
                if wait_time_ms > 1000:  # > 1 second
                    return 20.0
                elif wait_time_ms > 100:  # > 100ms
                    return 50.0
                elif wait_time_ms > 10:  # > 10ms
                    return 75.0
                else:
                    return 90.0

            return 85.0  # Default good score when no overcommit waits

        except Exception:
            return 85.0

    def _calculate_lock_contention_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate lock contention score."""
        try:
            context_locks = metrics.get("ContextLock")
            read_locks = metrics.get("RWLockAcquiredReadLocks")
            write_locks = metrics.get("RWLockAcquiredWriteLocks")

            total_locks = 0
            if context_locks:
                total_locks += context_locks.avg_value
            if read_locks:
                total_locks += read_locks.avg_value
            if write_locks:
                total_locks += write_locks.avg_value * 2  # Write locks are more expensive

            # Score based on lock frequency (lower is better for contention)
            if total_locks < 1000:
                return 95.0
            elif total_locks < 10000:
                return 85.0
            elif total_locks < 100000:
                return 70.0
            else:
                return 50.0

        except Exception:
            return 80.0

    def _analyze_page_faults(self, metrics: dict[str, ProfileEventAggregation]) -> dict[str, Any]:
        """Analyze page fault patterns."""
        try:
            soft_faults = metrics.get("SoftPageFaults")
            hard_faults = metrics.get("HardPageFaults")

            analysis = {
                "soft_faults": soft_faults.avg_value if soft_faults else 0,
                "hard_faults": hard_faults.avg_value if hard_faults else 0,
                "total_faults": 0,
                "fault_ratio": 0.0,
                "severity": "LOW",
            }

            analysis["total_faults"] = analysis["soft_faults"] + analysis["hard_faults"]

            if analysis["total_faults"] > 0:
                analysis["fault_ratio"] = analysis["hard_faults"] / analysis["total_faults"]

            # Determine severity
            if analysis["hard_faults"] > 1000:
                analysis["severity"] = "HIGH"
            elif analysis["hard_faults"] > 100 or analysis["total_faults"] > 10000:
                analysis["severity"] = "MEDIUM"

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing page faults: {e}")
            return {"error": str(e), "severity": "UNKNOWN"}

    def _analyze_memory_pressure(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze memory pressure indicators."""
        try:
            overcommit_wait = metrics.get("MemoryOvercommitWaitTimeMicroseconds")

            pressure = {
                "overcommit_wait_us": overcommit_wait.avg_value if overcommit_wait else 0,
                "overcommit_wait_ms": 0,
                "pressure_level": "NONE",
            }

            if pressure["overcommit_wait_us"] > 0:
                pressure["overcommit_wait_ms"] = pressure["overcommit_wait_us"] / 1000

                if pressure["overcommit_wait_ms"] > 1000:
                    pressure["pressure_level"] = "CRITICAL"
                elif pressure["overcommit_wait_ms"] > 100:
                    pressure["pressure_level"] = "HIGH"
                elif pressure["overcommit_wait_ms"] > 10:
                    pressure["pressure_level"] = "MEDIUM"
                else:
                    pressure["pressure_level"] = "LOW"

            return pressure

        except Exception as e:
            logger.warning(f"Error analyzing memory pressure: {e}")
            return {"error": str(e), "pressure_level": "UNKNOWN"}

    def _analyze_swap_usage(self, metrics: dict[str, ProfileEventAggregation]) -> dict[str, Any]:
        """Analyze swap usage patterns."""
        try:
            hard_faults = metrics.get("HardPageFaults")

            swap_analysis = {
                "estimated_swap_activity": hard_faults.avg_value if hard_faults else 0,
                "swap_pressure": "NONE",
            }

            # Hard page faults often indicate swap activity
            if swap_analysis["estimated_swap_activity"] > 1000:
                swap_analysis["swap_pressure"] = "HIGH"
            elif swap_analysis["estimated_swap_activity"] > 100:
                swap_analysis["swap_pressure"] = "MEDIUM"
            elif swap_analysis["estimated_swap_activity"] > 10:
                swap_analysis["swap_pressure"] = "LOW"

            return swap_analysis

        except Exception as e:
            logger.warning(f"Error analyzing swap usage: {e}")
            return {"error": str(e), "swap_pressure": "UNKNOWN"}

    def _analyze_memory_fragmentation(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze memory fragmentation indicators."""
        try:
            arena_chunks = metrics.get("ArenaAllocChunks")
            arena_bytes = metrics.get("ArenaAllocBytes")

            fragmentation = {
                "total_chunks": arena_chunks.avg_value if arena_chunks else 0,
                "total_bytes": arena_bytes.avg_value if arena_bytes else 0,
                "avg_chunk_size": 0,
                "fragmentation_score": 0.0,
                "fragmentation_level": "UNKNOWN",
            }

            if fragmentation["total_chunks"] > 0:
                fragmentation["avg_chunk_size"] = (
                    fragmentation["total_bytes"] / fragmentation["total_chunks"]
                )

                # Calculate fragmentation score based on chunk size distribution
                # Smaller average chunk sizes may indicate fragmentation
                if fragmentation["avg_chunk_size"] < 1024:  # < 1KB
                    fragmentation["fragmentation_score"] = 80.0
                    fragmentation["fragmentation_level"] = "HIGH"
                elif fragmentation["avg_chunk_size"] < 4096:  # < 4KB
                    fragmentation["fragmentation_score"] = 60.0
                    fragmentation["fragmentation_level"] = "MEDIUM"
                elif fragmentation["avg_chunk_size"] < 16384:  # < 16KB
                    fragmentation["fragmentation_score"] = 40.0
                    fragmentation["fragmentation_level"] = "LOW"
                else:
                    fragmentation["fragmentation_score"] = 20.0
                    fragmentation["fragmentation_level"] = "MINIMAL"

            return fragmentation

        except Exception as e:
            logger.warning(f"Error analyzing memory fragmentation: {e}")
            return {"error": str(e), "fragmentation_level": "UNKNOWN"}

    def _analyze_memory_overcommit(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze memory overcommit behavior."""
        try:
            overcommit_wait = metrics.get("MemoryOvercommitWaitTimeMicroseconds")

            overcommit = {
                "wait_time_us": overcommit_wait.avg_value if overcommit_wait else 0,
                "wait_time_ms": 0,
                "max_wait_us": overcommit_wait.max_value if overcommit_wait else 0,
                "overcommit_frequency": overcommit_wait.count if overcommit_wait else 0,
                "overcommit_severity": "NONE",
            }

            if overcommit["wait_time_us"] > 0:
                overcommit["wait_time_ms"] = overcommit["wait_time_us"] / 1000

                # Determine severity based on wait times and frequency
                if overcommit["wait_time_ms"] > 1000 or overcommit["overcommit_frequency"] > 100:
                    overcommit["overcommit_severity"] = "CRITICAL"
                elif overcommit["wait_time_ms"] > 100 or overcommit["overcommit_frequency"] > 50:
                    overcommit["overcommit_severity"] = "HIGH"
                elif overcommit["wait_time_ms"] > 10 or overcommit["overcommit_frequency"] > 10:
                    overcommit["overcommit_severity"] = "MEDIUM"
                else:
                    overcommit["overcommit_severity"] = "LOW"

            return overcommit

        except Exception as e:
            logger.warning(f"Error analyzing memory overcommit: {e}")
            return {"error": str(e), "overcommit_severity": "UNKNOWN"}

    def _identify_memory_bottlenecks(
        self, metrics: dict[str, ProfileEventAggregation], efficiency_score: float
    ) -> list[HardwareBottleneck]:
        """Identify memory performance bottlenecks."""
        bottlenecks = []

        try:
            # High page fault rate
            page_fault_analysis = self._analyze_page_faults(metrics)
            if page_fault_analysis.get("severity") in ["HIGH", "CRITICAL"]:
                severity = (
                    HardwareSeverity.CRITICAL
                    if page_fault_analysis["severity"] == "CRITICAL"
                    else HardwareSeverity.HIGH
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.MEMORY_BOUND,
                        severity=severity,
                        description=f"High page fault rate: {page_fault_analysis.get('total_faults', 0)} total faults",
                        efficiency_score=max(
                            0, 100 - page_fault_analysis.get("total_faults", 0) / 100
                        ),
                        impact_percentage=min(50, page_fault_analysis.get("total_faults", 0) / 200),
                        affected_components=["Virtual Memory", "Page Cache"],
                        recommendations=[
                            "Increase available physical memory",
                            "Optimize memory access patterns",
                            "Use memory mapping more efficiently",
                            "Consider memory-aware data structures",
                        ],
                        metrics=page_fault_analysis,
                    )
                )

            # Memory pressure from overcommit
            pressure_analysis = self._analyze_memory_pressure(metrics)
            if pressure_analysis.get("pressure_level") in ["HIGH", "CRITICAL"]:
                severity = (
                    HardwareSeverity.CRITICAL
                    if pressure_analysis["pressure_level"] == "CRITICAL"
                    else HardwareSeverity.HIGH
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.MEMORY_BOUND,
                        severity=severity,
                        description=f"Memory pressure detected: {pressure_analysis.get('overcommit_wait_ms', 0):.1f}ms avg wait",
                        efficiency_score=max(
                            0, 100 - pressure_analysis.get("overcommit_wait_ms", 0) / 10
                        ),
                        impact_percentage=min(
                            60, pressure_analysis.get("overcommit_wait_ms", 0) / 20
                        ),
                        affected_components=["Memory Allocator", "Virtual Memory"],
                        recommendations=[
                            "Increase memory limits or available RAM",
                            "Optimize memory allocation patterns",
                            "Implement memory pooling strategies",
                            "Monitor and tune memory overcommit settings",
                        ],
                        metrics=pressure_analysis,
                    )
                )

            # High memory fragmentation
            fragmentation_analysis = self._analyze_memory_fragmentation(metrics)
            if fragmentation_analysis.get("fragmentation_level") in ["HIGH", "MEDIUM"]:
                severity = (
                    HardwareSeverity.HIGH
                    if fragmentation_analysis["fragmentation_level"] == "HIGH"
                    else HardwareSeverity.MEDIUM
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.MEMORY_ALLOCATION,
                        severity=severity,
                        description=f"Memory fragmentation detected: {fragmentation_analysis.get('avg_chunk_size', 0):.0f} byte avg chunks",
                        efficiency_score=max(
                            0, 100 - fragmentation_analysis.get("fragmentation_score", 0)
                        ),
                        impact_percentage=fragmentation_analysis.get("fragmentation_score", 0) / 2,
                        affected_components=["Memory Allocator", "Heap Management"],
                        recommendations=[
                            "Use memory pools for frequent allocations",
                            "Implement custom allocators for specific use cases",
                            "Reduce allocation/deallocation frequency",
                            "Consider memory compaction strategies",
                        ],
                        metrics=fragmentation_analysis,
                    )
                )

            # Swap pressure
            swap_analysis = self._analyze_swap_usage(metrics)
            if swap_analysis.get("swap_pressure") in ["HIGH", "MEDIUM"]:
                severity = (
                    HardwareSeverity.HIGH
                    if swap_analysis["swap_pressure"] == "HIGH"
                    else HardwareSeverity.MEDIUM
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.IO_BOUND,
                        severity=severity,
                        description=f"Swap activity detected: {swap_analysis.get('estimated_swap_activity', 0)} hard page faults",
                        efficiency_score=max(
                            0, 100 - swap_analysis.get("estimated_swap_activity", 0) / 50
                        ),
                        impact_percentage=min(
                            70, swap_analysis.get("estimated_swap_activity", 0) / 20
                        ),
                        affected_components=["Swap Space", "Storage I/O"],
                        recommendations=[
                            "Increase physical memory to reduce swapping",
                            "Optimize memory usage patterns",
                            "Use faster storage for swap if swapping is unavoidable",
                            "Consider disabling swap for performance-critical applications",
                        ],
                        metrics=swap_analysis,
                    )
                )

        except Exception as e:
            logger.warning(f"Error identifying memory bottlenecks: {e}")

        return bottlenecks

    def _generate_memory_recommendations(
        self,
        metrics: dict[str, ProfileEventAggregation],
        efficiency_score: float,
        bottlenecks: list[HardwareBottleneck],
    ) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        try:
            # General efficiency recommendations
            if efficiency_score < 70:
                recommendations.append(
                    "Overall memory efficiency is low - consider comprehensive memory profiling"
                )

            # Specific recommendations based on bottlenecks
            bottleneck_types = {b.type for b in bottlenecks}

            if HardwareBottleneckType.MEMORY_BOUND in bottleneck_types:
                recommendations.extend(
                    [
                        "Increase available physical memory",
                        "Optimize data structures for memory efficiency",
                        "Implement memory-conscious algorithms",
                    ]
                )

            if HardwareBottleneckType.MEMORY_ALLOCATION in bottleneck_types:
                recommendations.extend(
                    [
                        "Implement custom memory allocators for hot paths",
                        "Use object pools for frequently allocated objects",
                        "Reduce allocation frequency through better memory management",
                    ]
                )

            if HardwareBottleneckType.IO_BOUND in bottleneck_types:
                recommendations.extend(
                    [
                        "Address swap usage by increasing RAM or optimizing memory usage",
                        "Use memory-mapped files for large datasets",
                        "Implement data compression to reduce memory footprint",
                    ]
                )

            # General memory optimization recommendations
            recommendations.extend(
                [
                    "Monitor memory usage patterns and identify optimization opportunities",
                    "Use huge pages for large memory allocations where appropriate",
                    "Consider NUMA-aware memory allocation strategies",
                    "Implement memory prefetching for predictable access patterns",
                ]
            )

        except Exception as e:
            logger.warning(f"Error generating memory recommendations: {e}")
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return recommendations


class ThreadPoolAnalyzer:
    """Analyzer for thread pool performance and contention."""

    def __init__(self, profile_analyzer: ProfileEventsAnalyzer):
        """Initialize thread pool analyzer.

        Args:
            profile_analyzer: ProfileEventsAnalyzer instance
        """
        self.profile_analyzer = profile_analyzer
        self.thread_events = self._get_thread_profile_events()

    def _get_thread_profile_events(self) -> list[str]:
        """Get list of thread pool related ProfileEvents."""
        return [
            "GlobalThreadPoolExpansions",
            "GlobalThreadPoolShrinks",
            "GlobalThreadPoolJobs",
            "LocalThreadPoolExpansions",
            "LocalThreadPoolShrinks",
            "LocalThreadPoolJobs",
            "GlobalThreadPoolLockWaitMicroseconds",
            "LocalThreadPoolLockWaitMicroseconds",
            "GlobalThreadPoolJobWaitTimeMicroseconds",
            "LocalThreadPoolJobWaitTimeMicroseconds",
            "BackgroundPoolTask",
            "BackgroundMovePoolTask",
            "BackgroundFetchesPoolTask",
            "BackgroundCommonPoolTask",
            "QueryThread",
            "CreatedHTTPConnections",
            "CannotScheduleTask",
            "PerfContextSwitches",
            "PerfCPUMigrations",
        ]

    @log_execution_time
    def analyze_thread_pool_performance(
        self, start_time: datetime, end_time: datetime, query_filter: str | None = None
    ) -> ThreadPoolAnalysis:
        """Analyze thread pool performance characteristics.

        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            query_filter: Optional query filter

        Returns:
            ThreadPoolAnalysis with performance metrics and recommendations
        """
        logger.info("Analyzing thread pool performance characteristics")

        # Get aggregated ProfileEvents data
        aggregations = self.profile_analyzer.aggregate_profile_events(
            self.thread_events, start_time, end_time, query_filter
        )

        # Create metrics dictionary for analysis
        metrics = {agg.event_name: agg for agg in aggregations}

        # Calculate thread pool efficiency metrics
        efficiency_score = self._calculate_thread_pool_efficiency_score(metrics)

        # Analyze thread pool subsystems
        thread_utilization = self._analyze_thread_utilization(metrics)
        contention_analysis = self._analyze_thread_contention(metrics)
        queue_efficiency = self._analyze_queue_efficiency(metrics)
        scaling_analysis = self._analyze_thread_scaling(metrics)
        lock_contention = self._analyze_lock_contention(metrics)
        thread_migration = self._analyze_thread_migration(metrics)

        # Identify thread pool bottlenecks
        bottlenecks = self._identify_thread_pool_bottlenecks(metrics, efficiency_score)

        # Generate optimization recommendations
        recommendations = self._generate_thread_pool_recommendations(
            metrics, efficiency_score, bottlenecks
        )

        return ThreadPoolAnalysis(
            efficiency_score=efficiency_score,
            thread_utilization=thread_utilization,
            contention_analysis=contention_analysis,
            queue_efficiency=queue_efficiency,
            scaling_analysis=scaling_analysis,
            lock_contention=lock_contention,
            thread_migration=thread_migration,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _calculate_thread_pool_efficiency_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate overall thread pool efficiency score (0-100)."""
        try:
            score_factors = []

            # Job wait time efficiency (lower is better)
            job_wait_score = self._calculate_job_wait_score(metrics)
            score_factors.append(job_wait_score)

            # Lock contention efficiency (lower is better)
            lock_wait_score = self._calculate_lock_wait_score(metrics)
            score_factors.append(lock_wait_score)

            # Thread scaling efficiency
            scaling_score = self._calculate_scaling_efficiency_score(metrics)
            score_factors.append(scaling_score)

            # Task scheduling efficiency
            scheduling_score = self._calculate_scheduling_efficiency_score(metrics)
            score_factors.append(scheduling_score)

            # Context switching efficiency
            context_switch_score = self._calculate_context_switch_score(metrics)
            score_factors.append(context_switch_score)

            if score_factors:
                return statistics.mean(score_factors)
            else:
                return 50.0  # Default neutral score

        except Exception as e:
            logger.warning(f"Error calculating thread pool efficiency score: {e}")
            return 50.0

    def _calculate_job_wait_score(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate job wait time score."""
        try:
            global_wait = metrics.get("GlobalThreadPoolJobWaitTimeMicroseconds")
            local_wait = metrics.get("LocalThreadPoolJobWaitTimeMicroseconds")

            total_wait_us = 0
            if global_wait:
                total_wait_us += global_wait.avg_value
            if local_wait:
                total_wait_us += local_wait.avg_value

            wait_time_ms = total_wait_us / 1000

            # Score based on average wait time (lower is better)
            if wait_time_ms < 1:
                return 95.0
            elif wait_time_ms < 10:
                return 85.0
            elif wait_time_ms < 100:
                return 70.0
            elif wait_time_ms < 1000:
                return 50.0
            else:
                return 30.0

        except Exception:
            return 75.0

    def _calculate_lock_wait_score(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate lock wait time score."""
        try:
            global_lock_wait = metrics.get("GlobalThreadPoolLockWaitMicroseconds")
            local_lock_wait = metrics.get("LocalThreadPoolLockWaitMicroseconds")

            total_lock_wait_us = 0
            if global_lock_wait:
                total_lock_wait_us += global_lock_wait.avg_value
            if local_lock_wait:
                total_lock_wait_us += local_lock_wait.avg_value

            lock_wait_ms = total_lock_wait_us / 1000

            # Score based on lock wait time (lower is better)
            if lock_wait_ms < 0.1:
                return 95.0
            elif lock_wait_ms < 1:
                return 85.0
            elif lock_wait_ms < 10:
                return 70.0
            elif lock_wait_ms < 100:
                return 50.0
            else:
                return 30.0

        except Exception:
            return 80.0

    def _calculate_scaling_efficiency_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate thread pool scaling efficiency score."""
        try:
            global_expansions = metrics.get("GlobalThreadPoolExpansions")
            global_shrinks = metrics.get("GlobalThreadPoolShrinks")
            local_expansions = metrics.get("LocalThreadPoolExpansions")
            local_shrinks = metrics.get("LocalThreadPoolShrinks")

            total_expansions = 0
            total_shrinks = 0

            if global_expansions:
                total_expansions += global_expansions.avg_value
            if local_expansions:
                total_expansions += local_expansions.avg_value
            if global_shrinks:
                total_shrinks += global_shrinks.avg_value
            if local_shrinks:
                total_shrinks += local_shrinks.avg_value

            total_scaling_events = total_expansions + total_shrinks

            # Lower scaling frequency generally indicates better stability
            if total_scaling_events < 10:
                return 90.0
            elif total_scaling_events < 100:
                return 80.0
            elif total_scaling_events < 1000:
                return 70.0
            else:
                return 60.0

        except Exception:
            return 75.0

    def _calculate_scheduling_efficiency_score(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> float:
        """Calculate task scheduling efficiency score."""
        try:
            cannot_schedule = metrics.get("CannotScheduleTask")
            global_jobs = metrics.get("GlobalThreadPoolJobs")
            local_jobs = metrics.get("LocalThreadPoolJobs")

            total_jobs = 0
            if global_jobs:
                total_jobs += global_jobs.avg_value
            if local_jobs:
                total_jobs += local_jobs.avg_value

            if cannot_schedule and total_jobs > 0:
                scheduling_failure_rate = cannot_schedule.avg_value / total_jobs
                scheduling_success_rate = max(0, 1 - scheduling_failure_rate)
                return scheduling_success_rate * 100

            return 90.0  # Default good score if no scheduling failures

        except Exception:
            return 85.0

    def _calculate_context_switch_score(self, metrics: dict[str, ProfileEventAggregation]) -> float:
        """Calculate context switch efficiency score."""
        try:
            context_switches = metrics.get("PerfContextSwitches")
            cpu_migrations = metrics.get("PerfCPUMigrations")

            if context_switches:
                switches_per_second = context_switches.avg_value / 3600  # Assuming 1-hour window

                # Score based on context switching frequency
                if switches_per_second < 1000:
                    switch_score = 95.0
                elif switches_per_second < 10000:
                    switch_score = 80.0
                elif switches_per_second < 100000:
                    switch_score = 65.0
                else:
                    switch_score = 50.0
            else:
                switch_score = 80.0

            if cpu_migrations:
                # CPU migrations should be minimized
                migrations_per_second = cpu_migrations.avg_value / 3600
                if migrations_per_second < 100:
                    migration_score = 95.0
                elif migrations_per_second < 1000:
                    migration_score = 80.0
                else:
                    migration_score = 60.0
            else:
                migration_score = 85.0

            return statistics.mean([switch_score, migration_score])

        except Exception:
            return 80.0

    def _analyze_thread_utilization(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze thread utilization patterns."""
        try:
            utilization = {}

            # Analyze different thread pool types
            thread_pools = {
                "global": {
                    "jobs": metrics.get("GlobalThreadPoolJobs"),
                    "expansions": metrics.get("GlobalThreadPoolExpansions"),
                    "shrinks": metrics.get("GlobalThreadPoolShrinks"),
                },
                "local": {
                    "jobs": metrics.get("LocalThreadPoolJobs"),
                    "expansions": metrics.get("LocalThreadPoolExpansions"),
                    "shrinks": metrics.get("LocalThreadPoolShrinks"),
                },
                "background": {
                    "pool_tasks": metrics.get("BackgroundPoolTask"),
                    "move_tasks": metrics.get("BackgroundMovePoolTask"),
                    "fetch_tasks": metrics.get("BackgroundFetchesPoolTask"),
                    "common_tasks": metrics.get("BackgroundCommonPoolTask"),
                },
            }

            for pool_name, pool_metrics in thread_pools.items():
                pool_util = {}

                if pool_name in ["global", "local"]:
                    jobs = pool_metrics["jobs"]
                    expansions = pool_metrics["expansions"]
                    shrinks = pool_metrics["shrinks"]

                    pool_util["total_jobs"] = jobs.avg_value if jobs else 0
                    pool_util["expansions"] = expansions.avg_value if expansions else 0
                    pool_util["shrinks"] = shrinks.avg_value if shrinks else 0
                    pool_util["net_scaling"] = pool_util["expansions"] - pool_util["shrinks"]

                    # Determine utilization pattern
                    if pool_util["expansions"] > pool_util["shrinks"] * 2:
                        pool_util["pattern"] = "GROWING"
                    elif pool_util["shrinks"] > pool_util["expansions"] * 2:
                        pool_util["pattern"] = "SHRINKING"
                    elif pool_util["expansions"] + pool_util["shrinks"] > 100:
                        pool_util["pattern"] = "VOLATILE"
                    else:
                        pool_util["pattern"] = "STABLE"

                else:  # background pools
                    total_bg_tasks = 0
                    for task_type, task_metric in pool_metrics.items():
                        if task_metric:
                            task_count = task_metric.avg_value
                            pool_util[task_type] = task_count
                            total_bg_tasks += task_count

                    pool_util["total_background_tasks"] = total_bg_tasks

                    if total_bg_tasks > 1000:
                        pool_util["pattern"] = "HIGH_ACTIVITY"
                    elif total_bg_tasks > 100:
                        pool_util["pattern"] = "MODERATE_ACTIVITY"
                    else:
                        pool_util["pattern"] = "LOW_ACTIVITY"

                utilization[pool_name] = pool_util

            return utilization

        except Exception as e:
            logger.warning(f"Error analyzing thread utilization: {e}")
            return {"error": str(e)}

    def _analyze_thread_contention(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze thread contention patterns."""
        try:
            global_lock_wait = metrics.get("GlobalThreadPoolLockWaitMicroseconds")
            local_lock_wait = metrics.get("LocalThreadPoolLockWaitMicroseconds")
            global_job_wait = metrics.get("GlobalThreadPoolJobWaitTimeMicroseconds")
            local_job_wait = metrics.get("LocalThreadPoolJobWaitTimeMicroseconds")

            contention = {
                "global_lock_wait_us": global_lock_wait.avg_value if global_lock_wait else 0,
                "local_lock_wait_us": local_lock_wait.avg_value if local_lock_wait else 0,
                "global_job_wait_us": global_job_wait.avg_value if global_job_wait else 0,
                "local_job_wait_us": local_job_wait.avg_value if local_job_wait else 0,
                "total_contention_us": 0,
                "contention_level": "NONE",
            }

            contention["total_contention_us"] = (
                contention["global_lock_wait_us"]
                + contention["local_lock_wait_us"]
                + contention["global_job_wait_us"]
                + contention["local_job_wait_us"]
            )

            contention_ms = contention["total_contention_us"] / 1000

            # Determine contention level
            if contention_ms > 1000:
                contention["contention_level"] = "CRITICAL"
            elif contention_ms > 100:
                contention["contention_level"] = "HIGH"
            elif contention_ms > 10:
                contention["contention_level"] = "MEDIUM"
            elif contention_ms > 1:
                contention["contention_level"] = "LOW"

            # Calculate contention ratios
            if contention["total_contention_us"] > 0:
                contention["lock_wait_ratio"] = (
                    contention["global_lock_wait_us"] + contention["local_lock_wait_us"]
                ) / contention["total_contention_us"]
                contention["job_wait_ratio"] = (
                    contention["global_job_wait_us"] + contention["local_job_wait_us"]
                ) / contention["total_contention_us"]
            else:
                contention["lock_wait_ratio"] = 0.0
                contention["job_wait_ratio"] = 0.0

            return contention

        except Exception as e:
            logger.warning(f"Error analyzing thread contention: {e}")
            return {"error": str(e), "contention_level": "UNKNOWN"}

    def _analyze_queue_efficiency(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze thread pool queue efficiency."""
        try:
            global_jobs = metrics.get("GlobalThreadPoolJobs")
            local_jobs = metrics.get("LocalThreadPoolJobs")
            global_job_wait = metrics.get("GlobalThreadPoolJobWaitTimeMicroseconds")
            local_job_wait = metrics.get("LocalThreadPoolJobWaitTimeMicroseconds")
            cannot_schedule = metrics.get("CannotScheduleTask")

            efficiency = {
                "total_jobs": 0,
                "total_job_wait_us": 0,
                "avg_job_wait_us": 0,
                "scheduling_failures": cannot_schedule.avg_value if cannot_schedule else 0,
                "scheduling_success_rate": 100.0,
                "queue_efficiency_score": 0.0,
                "efficiency_level": "UNKNOWN",
            }

            if global_jobs:
                efficiency["total_jobs"] += global_jobs.avg_value
            if local_jobs:
                efficiency["total_jobs"] += local_jobs.avg_value

            if global_job_wait:
                efficiency["total_job_wait_us"] += global_job_wait.avg_value
            if local_job_wait:
                efficiency["total_job_wait_us"] += local_job_wait.avg_value

            if efficiency["total_jobs"] > 0:
                efficiency["avg_job_wait_us"] = (
                    efficiency["total_job_wait_us"] / efficiency["total_jobs"]
                )

                if efficiency["scheduling_failures"] > 0:
                    efficiency["scheduling_success_rate"] = (
                        (efficiency["total_jobs"] - efficiency["scheduling_failures"])
                        / efficiency["total_jobs"]
                        * 100
                    )

            # Calculate efficiency score
            wait_score = max(0, 100 - efficiency["avg_job_wait_us"] / 1000)  # Penalize long waits
            success_score = efficiency["scheduling_success_rate"]
            efficiency["queue_efficiency_score"] = statistics.mean([wait_score, success_score])

            # Determine efficiency level
            if efficiency["queue_efficiency_score"] > 90:
                efficiency["efficiency_level"] = "EXCELLENT"
            elif efficiency["queue_efficiency_score"] > 80:
                efficiency["efficiency_level"] = "GOOD"
            elif efficiency["queue_efficiency_score"] > 70:
                efficiency["efficiency_level"] = "FAIR"
            else:
                efficiency["efficiency_level"] = "POOR"

            return efficiency

        except Exception as e:
            logger.warning(f"Error analyzing queue efficiency: {e}")
            return {"error": str(e), "efficiency_level": "UNKNOWN"}

    def _analyze_thread_scaling(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze thread pool scaling behavior."""
        try:
            scaling = {
                "global_expansions": 0,
                "global_shrinks": 0,
                "local_expansions": 0,
                "local_shrinks": 0,
                "total_scaling_events": 0,
                "scaling_volatility": 0.0,
                "scaling_pattern": "STABLE",
            }

            global_exp = metrics.get("GlobalThreadPoolExpansions")
            global_shr = metrics.get("GlobalThreadPoolShrinks")
            local_exp = metrics.get("LocalThreadPoolExpansions")
            local_shr = metrics.get("LocalThreadPoolShrinks")

            if global_exp:
                scaling["global_expansions"] = global_exp.avg_value
            if global_shr:
                scaling["global_shrinks"] = global_shr.avg_value
            if local_exp:
                scaling["local_expansions"] = local_exp.avg_value
            if local_shr:
                scaling["local_shrinks"] = local_shr.avg_value

            scaling["total_scaling_events"] = (
                scaling["global_expansions"]
                + scaling["global_shrinks"]
                + scaling["local_expansions"]
                + scaling["local_shrinks"]
            )

            # Calculate scaling volatility
            if scaling["total_scaling_events"] > 0:
                expansion_ratio = (
                    scaling["global_expansions"] + scaling["local_expansions"]
                ) / scaling["total_scaling_events"]
                shrink_ratio = (scaling["global_shrinks"] + scaling["local_shrinks"]) / scaling[
                    "total_scaling_events"
                ]

                # Volatility is higher when expansions and shrinks are more balanced
                scaling["scaling_volatility"] = 1 - abs(expansion_ratio - shrink_ratio)

            # Determine scaling pattern
            total_exp = scaling["global_expansions"] + scaling["local_expansions"]
            total_shr = scaling["global_shrinks"] + scaling["local_shrinks"]

            if scaling["total_scaling_events"] < 10:
                scaling["scaling_pattern"] = "STABLE"
            elif total_exp > total_shr * 2:
                scaling["scaling_pattern"] = "GROWING"
            elif total_shr > total_exp * 2:
                scaling["scaling_pattern"] = "SHRINKING"
            elif scaling["scaling_volatility"] > 0.7:
                scaling["scaling_pattern"] = "VOLATILE"
            else:
                scaling["scaling_pattern"] = "ADAPTIVE"

            return scaling

        except Exception as e:
            logger.warning(f"Error analyzing thread scaling: {e}")
            return {"error": str(e), "scaling_pattern": "UNKNOWN"}

    def _analyze_lock_contention(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze lock contention in thread pools."""
        try:
            global_lock_wait = metrics.get("GlobalThreadPoolLockWaitMicroseconds")
            local_lock_wait = metrics.get("LocalThreadPoolLockWaitMicroseconds")

            lock_analysis = {
                "global_lock_wait_us": global_lock_wait.avg_value if global_lock_wait else 0,
                "local_lock_wait_us": local_lock_wait.avg_value if local_lock_wait else 0,
                "total_lock_wait_us": 0,
                "lock_contention_level": "NONE",
                "dominant_contention_source": "NONE",
            }

            lock_analysis["total_lock_wait_us"] = (
                lock_analysis["global_lock_wait_us"] + lock_analysis["local_lock_wait_us"]
            )

            lock_wait_ms = lock_analysis["total_lock_wait_us"] / 1000

            # Determine contention level
            if lock_wait_ms > 1000:
                lock_analysis["lock_contention_level"] = "CRITICAL"
            elif lock_wait_ms > 100:
                lock_analysis["lock_contention_level"] = "HIGH"
            elif lock_wait_ms > 10:
                lock_analysis["lock_contention_level"] = "MEDIUM"
            elif lock_wait_ms > 1:
                lock_analysis["lock_contention_level"] = "LOW"

            # Identify dominant contention source
            if lock_analysis["total_lock_wait_us"] > 0:
                global_ratio = (
                    lock_analysis["global_lock_wait_us"] / lock_analysis["total_lock_wait_us"]
                )
                if global_ratio > 0.7:
                    lock_analysis["dominant_contention_source"] = "GLOBAL_POOL"
                elif global_ratio < 0.3:
                    lock_analysis["dominant_contention_source"] = "LOCAL_POOL"
                else:
                    lock_analysis["dominant_contention_source"] = "BALANCED"

            return lock_analysis

        except Exception as e:
            logger.warning(f"Error analyzing lock contention: {e}")
            return {"error": str(e), "lock_contention_level": "UNKNOWN"}

    def _analyze_thread_migration(
        self, metrics: dict[str, ProfileEventAggregation]
    ) -> dict[str, Any]:
        """Analyze thread migration patterns."""
        try:
            cpu_migrations = metrics.get("PerfCPUMigrations")
            context_switches = metrics.get("PerfContextSwitches")

            migration = {
                "cpu_migrations": cpu_migrations.avg_value if cpu_migrations else 0,
                "context_switches": context_switches.avg_value if context_switches else 0,
                "migration_rate": 0.0,
                "migration_level": "UNKNOWN",
            }

            if migration["context_switches"] > 0:
                migration["migration_rate"] = (
                    migration["cpu_migrations"] / migration["context_switches"]
                )

            # Determine migration level
            if migration["cpu_migrations"] > 10000:
                migration["migration_level"] = "HIGH"
            elif migration["cpu_migrations"] > 1000:
                migration["migration_level"] = "MEDIUM"
            elif migration["cpu_migrations"] > 100:
                migration["migration_level"] = "LOW"
            else:
                migration["migration_level"] = "MINIMAL"

            return migration

        except Exception as e:
            logger.warning(f"Error analyzing thread migration: {e}")
            return {"error": str(e), "migration_level": "UNKNOWN"}

    def _identify_thread_pool_bottlenecks(
        self, metrics: dict[str, ProfileEventAggregation], efficiency_score: float
    ) -> list[HardwareBottleneck]:
        """Identify thread pool performance bottlenecks."""
        bottlenecks = []

        try:
            # High lock contention
            lock_analysis = self._analyze_lock_contention(metrics)
            if lock_analysis.get("lock_contention_level") in ["HIGH", "CRITICAL"]:
                severity = (
                    HardwareSeverity.CRITICAL
                    if lock_analysis["lock_contention_level"] == "CRITICAL"
                    else HardwareSeverity.HIGH
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.THREAD_CONTENTION,
                        severity=severity,
                        description=f"High thread pool lock contention: {lock_analysis.get('total_lock_wait_us', 0)/1000:.1f}ms avg wait",
                        efficiency_score=max(
                            0, 100 - lock_analysis.get("total_lock_wait_us", 0) / 10000
                        ),
                        impact_percentage=min(
                            50, lock_analysis.get("total_lock_wait_us", 0) / 20000
                        ),
                        affected_components=["Thread Pool Locks", "Resource Synchronization"],
                        recommendations=[
                            "Reduce lock scope and duration in thread pools",
                            "Consider lock-free data structures where possible",
                            "Optimize thread pool sizing to reduce contention",
                            "Use thread-local storage for frequently accessed data",
                        ],
                        metrics=lock_analysis,
                    )
                )

            # Poor queue efficiency
            queue_analysis = self._analyze_queue_efficiency(metrics)
            if queue_analysis.get("efficiency_level") in ["POOR", "FAIR"]:
                severity = (
                    HardwareSeverity.HIGH
                    if queue_analysis["efficiency_level"] == "POOR"
                    else HardwareSeverity.MEDIUM
                )
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.THREAD_CONTENTION,
                        severity=severity,
                        description=f"Poor thread pool queue efficiency: {queue_analysis.get('queue_efficiency_score', 0):.1f}% score",
                        efficiency_score=queue_analysis.get("queue_efficiency_score", 50),
                        impact_percentage=max(
                            0, 100 - queue_analysis.get("queue_efficiency_score", 50)
                        ),
                        affected_components=["Thread Pool Queues", "Task Scheduling"],
                        recommendations=[
                            "Optimize task distribution across thread pools",
                            "Consider work-stealing algorithms",
                            "Adjust thread pool sizes based on workload",
                            "Minimize task scheduling overhead",
                        ],
                        metrics=queue_analysis,
                    )
                )

            # High thread migration
            migration_analysis = self._analyze_thread_migration(metrics)
            if migration_analysis.get("migration_level") == "HIGH":
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.CONTEXT_SWITCHING,
                        severity=HardwareSeverity.MEDIUM,
                        description=f"High thread CPU migration rate: {migration_analysis.get('cpu_migrations', 0)} migrations",
                        efficiency_score=max(
                            0, 100 - migration_analysis.get("cpu_migrations", 0) / 1000
                        ),
                        impact_percentage=min(
                            30, migration_analysis.get("cpu_migrations", 0) / 5000
                        ),
                        affected_components=["CPU Scheduler", "Thread Affinity"],
                        recommendations=[
                            "Set CPU affinity for performance-critical threads",
                            "Optimize NUMA node allocation",
                            "Reduce unnecessary context switching",
                            "Consider CPU topology in thread pool design",
                        ],
                        metrics=migration_analysis,
                    )
                )

            # Excessive thread pool scaling
            scaling_analysis = self._analyze_thread_scaling(metrics)
            if scaling_analysis.get("scaling_pattern") == "VOLATILE":
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.THREAD_CONTENTION,
                        severity=HardwareSeverity.MEDIUM,
                        description=f"Volatile thread pool scaling: {scaling_analysis.get('total_scaling_events', 0)} events",
                        efficiency_score=max(
                            0, 100 - scaling_analysis.get("scaling_volatility", 0) * 50
                        ),
                        impact_percentage=scaling_analysis.get("scaling_volatility", 0) * 25,
                        affected_components=["Thread Pool Management", "Resource Allocation"],
                        recommendations=[
                            "Tune thread pool sizing parameters",
                            "Implement more stable scaling policies",
                            "Monitor workload patterns for better sizing",
                            "Consider pre-warming thread pools for predictable loads",
                        ],
                        metrics=scaling_analysis,
                    )
                )

            # High scheduling failures
            cannot_schedule = metrics.get("CannotScheduleTask")
            if cannot_schedule and cannot_schedule.avg_value > 100:
                bottlenecks.append(
                    HardwareBottleneck(
                        type=HardwareBottleneckType.THREAD_CONTENTION,
                        severity=HardwareSeverity.HIGH,
                        description=f"High task scheduling failures: {cannot_schedule.avg_value} failures",
                        efficiency_score=max(0, 100 - cannot_schedule.avg_value / 50),
                        impact_percentage=min(60, cannot_schedule.avg_value / 20),
                        affected_components=["Task Scheduler", "Thread Pool Capacity"],
                        recommendations=[
                            "Increase thread pool capacity",
                            "Optimize task submission patterns",
                            "Implement task prioritization",
                            "Monitor and adjust queue sizes",
                        ],
                        metrics={"scheduling_failures": cannot_schedule.avg_value},
                    )
                )

        except Exception as e:
            logger.warning(f"Error identifying thread pool bottlenecks: {e}")

        return bottlenecks

    def _generate_thread_pool_recommendations(
        self,
        metrics: dict[str, ProfileEventAggregation],
        efficiency_score: float,
        bottlenecks: list[HardwareBottleneck],
    ) -> list[str]:
        """Generate thread pool optimization recommendations."""
        recommendations = []

        try:
            # General efficiency recommendations
            if efficiency_score < 70:
                recommendations.append(
                    "Overall thread pool efficiency is low - consider comprehensive thread profiling"
                )

            # Specific recommendations based on bottlenecks
            bottleneck_types = {b.type for b in bottlenecks}

            if HardwareBottleneckType.THREAD_CONTENTION in bottleneck_types:
                recommendations.extend(
                    [
                        "Optimize thread pool configuration to reduce contention",
                        "Implement lock-free or wait-free algorithms where possible",
                        "Use thread-local storage for frequently accessed shared data",
                    ]
                )

            if HardwareBottleneckType.CONTEXT_SWITCHING in bottleneck_types:
                recommendations.extend(
                    [
                        "Set CPU affinity for performance-critical threads",
                        "Optimize thread pool sizes to match CPU core count",
                        "Minimize unnecessary thread creation and destruction",
                    ]
                )

            # Utilization pattern recommendations
            utilization = self._analyze_thread_utilization(metrics)
            for pool_name, pool_data in utilization.items():
                if isinstance(pool_data, dict) and pool_data.get("pattern") == "VOLATILE":
                    recommendations.append(
                        f"Stabilize {pool_name} thread pool scaling through better configuration"
                    )
                elif isinstance(pool_data, dict) and pool_data.get("pattern") == "GROWING":
                    recommendations.append(
                        f"Monitor {pool_name} thread pool growth and adjust limits if needed"
                    )

            # General thread pool optimization recommendations
            recommendations.extend(
                [
                    "Monitor thread pool metrics and adjust sizes based on workload patterns",
                    "Implement thread pool warming strategies for predictable workloads",
                    "Use work-stealing algorithms to improve load balancing",
                    "Consider NUMA-aware thread allocation for multi-socket systems",
                ]
            )

        except Exception as e:
            logger.warning(f"Error generating thread pool recommendations: {e}")
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return recommendations


class HardwareHealthEngine:
    """Unified hardware health assessment engine."""

    def __init__(self, client: Client):
        """Initialize hardware health engine.

        Args:
            client: ClickHouse client instance
        """
        self.client = client
        self.profile_analyzer = ProfileEventsAnalyzer(client)
        self.cpu_analyzer = CPUAnalyzer(self.profile_analyzer)
        self.memory_analyzer = MemoryAnalyzer(self.profile_analyzer)
        self.thread_pool_analyzer = ThreadPoolAnalyzer(self.profile_analyzer)

    @log_execution_time
    def generate_hardware_health_report(
        self, start_time: datetime, end_time: datetime, query_filter: str | None = None
    ) -> HardwareHealthReport:
        """Generate comprehensive hardware health report.

        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            query_filter: Optional query filter

        Returns:
            HardwareHealthReport with comprehensive analysis
        """
        logger.info("Generating comprehensive hardware health report")

        try:
            # Run all component analyses
            cpu_analysis = self.cpu_analyzer.analyze_cpu_performance(
                start_time, end_time, query_filter
            )

            memory_analysis = self.memory_analyzer.analyze_memory_performance(
                start_time, end_time, query_filter
            )

            thread_pool_analysis = self.thread_pool_analyzer.analyze_thread_pool_performance(
                start_time, end_time, query_filter
            )

            # Calculate overall health score
            overall_health_score = self._calculate_overall_health_score(
                cpu_analysis, memory_analysis, thread_pool_analysis
            )

            # Analyze system efficiency
            system_efficiency = self._analyze_system_efficiency(
                cpu_analysis, memory_analysis, thread_pool_analysis
            )

            # Generate capacity planning insights
            capacity_planning = self._generate_capacity_planning_insights(
                cpu_analysis, memory_analysis, thread_pool_analysis
            )

            # Identify critical bottlenecks
            critical_bottlenecks = self._identify_critical_bottlenecks(
                cpu_analysis, memory_analysis, thread_pool_analysis
            )

            # Generate optimization priorities
            optimization_priorities = self._generate_optimization_priorities(
                critical_bottlenecks, cpu_analysis, memory_analysis, thread_pool_analysis
            )

            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(
                cpu_analysis, memory_analysis, thread_pool_analysis
            )

            return HardwareHealthReport(
                overall_health_score=overall_health_score,
                cpu_analysis=cpu_analysis,
                memory_analysis=memory_analysis,
                thread_pool_analysis=thread_pool_analysis,
                system_efficiency=system_efficiency,
                capacity_planning=capacity_planning,
                critical_bottlenecks=critical_bottlenecks,
                optimization_priorities=optimization_priorities,
                performance_trends=performance_trends,
            )

        except Exception as e:
            logger.error(f"Error generating hardware health report: {e}")
            raise

    def _calculate_overall_health_score(
        self,
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> float:
        """Calculate overall hardware health score."""
        try:
            # Weight the different components
            cpu_weight = 0.4
            memory_weight = 0.35
            thread_pool_weight = 0.25

            overall_score = (
                cpu_analysis.efficiency_score * cpu_weight
                + memory_analysis.efficiency_score * memory_weight
                + thread_pool_analysis.efficiency_score * thread_pool_weight
            )

            # Apply penalties for critical bottlenecks
            critical_bottlenecks = (
                len(
                    [b for b in cpu_analysis.bottlenecks if b.severity == HardwareSeverity.CRITICAL]
                )
                + len(
                    [
                        b
                        for b in memory_analysis.bottlenecks
                        if b.severity == HardwareSeverity.CRITICAL
                    ]
                )
                + len(
                    [
                        b
                        for b in thread_pool_analysis.bottlenecks
                        if b.severity == HardwareSeverity.CRITICAL
                    ]
                )
            )

            penalty = min(critical_bottlenecks * 10, 30)  # Max 30 point penalty
            overall_score = max(0, overall_score - penalty)

            return overall_score

        except Exception as e:
            logger.warning(f"Error calculating overall health score: {e}")
            return 50.0

    def _analyze_system_efficiency(
        self,
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> dict[str, Any]:
        """Analyze overall system efficiency."""
        try:
            efficiency = {
                "component_scores": {
                    "cpu": cpu_analysis.efficiency_score,
                    "memory": memory_analysis.efficiency_score,
                    "thread_pools": thread_pool_analysis.efficiency_score,
                },
                "bottleneck_distribution": {},
                "system_balance": "UNKNOWN",
                "efficiency_rating": "UNKNOWN",
            }

            # Analyze bottleneck distribution
            all_bottlenecks = (
                cpu_analysis.bottlenecks
                + memory_analysis.bottlenecks
                + thread_pool_analysis.bottlenecks
            )

            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_type = bottleneck.type.value
                bottleneck_counts[bottleneck_type] = bottleneck_counts.get(bottleneck_type, 0) + 1

            efficiency["bottleneck_distribution"] = bottleneck_counts

            # Determine system balance
            scores = list(efficiency["component_scores"].values())
            if len(scores) > 1:
                score_std = statistics.stdev(scores)
                if score_std < 10:
                    efficiency["system_balance"] = "WELL_BALANCED"
                elif score_std < 20:
                    efficiency["system_balance"] = "MODERATELY_BALANCED"
                else:
                    efficiency["system_balance"] = "IMBALANCED"

            # Overall efficiency rating
            avg_score = statistics.mean(scores)
            if avg_score > 85:
                efficiency["efficiency_rating"] = "EXCELLENT"
            elif avg_score > 75:
                efficiency["efficiency_rating"] = "GOOD"
            elif avg_score > 65:
                efficiency["efficiency_rating"] = "FAIR"
            else:
                efficiency["efficiency_rating"] = "POOR"

            return efficiency

        except Exception as e:
            logger.warning(f"Error analyzing system efficiency: {e}")
            return {"error": str(e)}

    def _generate_capacity_planning_insights(
        self,
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> dict[str, Any]:
        """Generate capacity planning insights."""
        try:
            insights = {
                "scaling_recommendations": [],
                "resource_constraints": [],
                "optimization_potential": {},
                "capacity_headroom": {},
            }

            # CPU capacity insights
            if cpu_analysis.efficiency_score < 70:
                insights["scaling_recommendations"].append(
                    "Consider CPU upgrade or optimization for better performance"
                )
                if cpu_analysis.instructions_per_cycle < 1.0:
                    insights["resource_constraints"].append("CPU instruction throughput limited")

            insights["capacity_headroom"]["cpu"] = cpu_analysis.efficiency_score
            insights["optimization_potential"]["cpu"] = max(0, 100 - cpu_analysis.efficiency_score)

            # Memory capacity insights
            if memory_analysis.efficiency_score < 70:
                insights["scaling_recommendations"].append(
                    "Consider memory expansion or optimization strategies"
                )

                # Check for memory pressure indicators
                if any(
                    b.type == HardwareBottleneckType.MEMORY_BOUND
                    for b in memory_analysis.bottlenecks
                ):
                    insights["resource_constraints"].append("Memory capacity limited")

            insights["capacity_headroom"]["memory"] = memory_analysis.efficiency_score
            insights["optimization_potential"]["memory"] = max(
                0, 100 - memory_analysis.efficiency_score
            )

            # Thread pool capacity insights
            if thread_pool_analysis.efficiency_score < 70:
                insights["scaling_recommendations"].append(
                    "Optimize thread pool configuration for better concurrency"
                )

                if any(
                    b.type == HardwareBottleneckType.THREAD_CONTENTION
                    for b in thread_pool_analysis.bottlenecks
                ):
                    insights["resource_constraints"].append(
                        "Thread pool contention limiting scalability"
                    )

            insights["capacity_headroom"]["thread_pools"] = thread_pool_analysis.efficiency_score
            insights["optimization_potential"]["thread_pools"] = max(
                0, 100 - thread_pool_analysis.efficiency_score
            )

            # Overall capacity assessment
            min_headroom = min(insights["capacity_headroom"].values())
            if min_headroom < 50:
                insights["scaling_recommendations"].append(
                    "System approaching capacity limits - consider scaling soon"
                )
            elif min_headroom < 70:
                insights["scaling_recommendations"].append(
                    "Monitor capacity trends and plan for future scaling"
                )

            return insights

        except Exception as e:
            logger.warning(f"Error generating capacity planning insights: {e}")
            return {"error": str(e)}

    def _identify_critical_bottlenecks(
        self,
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> list[HardwareBottleneck]:
        """Identify critical bottlenecks across all components."""
        try:
            all_bottlenecks = (
                cpu_analysis.bottlenecks
                + memory_analysis.bottlenecks
                + thread_pool_analysis.bottlenecks
            )

            # Filter for critical and high severity bottlenecks
            critical_bottlenecks = [
                b
                for b in all_bottlenecks
                if b.severity in [HardwareSeverity.CRITICAL, HardwareSeverity.HIGH]
            ]

            # Sort by impact percentage descending
            critical_bottlenecks.sort(key=lambda x: x.impact_percentage, reverse=True)

            return critical_bottlenecks[:10]  # Return top 10 critical bottlenecks

        except Exception as e:
            logger.warning(f"Error identifying critical bottlenecks: {e}")
            return []

    def _generate_optimization_priorities(
        self,
        critical_bottlenecks: list[HardwareBottleneck],
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> list[str]:
        """Generate prioritized optimization recommendations."""
        try:
            priorities = []

            # Priority 1: Address critical bottlenecks
            if critical_bottlenecks:
                for bottleneck in critical_bottlenecks[:3]:  # Top 3 critical issues
                    priorities.append(
                        f"CRITICAL: Address {bottleneck.type.value} - {bottleneck.description}"
                    )

            # Priority 2: Component-specific optimizations
            scores = [
                ("CPU", cpu_analysis.efficiency_score),
                ("Memory", memory_analysis.efficiency_score),
                ("Thread Pools", thread_pool_analysis.efficiency_score),
            ]

            # Sort by efficiency score (lowest first)
            scores.sort(key=lambda x: x[1])

            for component, score in scores:
                if score < 70:
                    priorities.append(
                        f"HIGH: Optimize {component} performance (current score: {score:.1f}%)"
                    )

            # Priority 3: System balance improvements
            score_values = [score for _, score in scores]
            if len(score_values) > 1 and statistics.stdev(score_values) > 20:
                priorities.append("MEDIUM: Improve system balance across components")

            # Priority 4: Capacity planning
            min_score = min(score_values)
            if min_score < 50:
                priorities.append("MEDIUM: Plan for capacity scaling to avoid future bottlenecks")

            return priorities[:8]  # Return top 8 priorities

        except Exception as e:
            logger.warning(f"Error generating optimization priorities: {e}")
            return ["Unable to generate priorities due to analysis error"]

    def _analyze_performance_trends(
        self,
        cpu_analysis: CPUAnalysis,
        memory_analysis: MemoryAnalysis,
        thread_pool_analysis: ThreadPoolAnalysis,
    ) -> dict[str, Any]:
        """Analyze performance trends and patterns."""
        try:
            trends = {
                "component_trends": {},
                "bottleneck_trends": {},
                "efficiency_trajectory": "STABLE",
                "risk_indicators": [],
            }

            # Analyze component efficiency trends
            component_scores = {
                "cpu": cpu_analysis.efficiency_score,
                "memory": memory_analysis.efficiency_score,
                "thread_pools": thread_pool_analysis.efficiency_score,
            }

            for component, score in component_scores.items():
                if score < 50:
                    trends["component_trends"][component] = "DECLINING"
                    trends["risk_indicators"].append(
                        f"{component.upper()} performance below acceptable threshold"
                    )
                elif score < 70:
                    trends["component_trends"][component] = "AT_RISK"
                else:
                    trends["component_trends"][component] = "HEALTHY"

            # Analyze bottleneck patterns
            all_bottlenecks = (
                cpu_analysis.bottlenecks
                + memory_analysis.bottlenecks
                + thread_pool_analysis.bottlenecks
            )

            bottleneck_types = [b.type.value for b in all_bottlenecks]
            bottleneck_counts = {bt: bottleneck_types.count(bt) for bt in set(bottleneck_types)}
            trends["bottleneck_trends"] = bottleneck_counts

            # Determine overall efficiency trajectory
            avg_efficiency = statistics.mean(component_scores.values())
            critical_count = len(
                [b for b in all_bottlenecks if b.severity == HardwareSeverity.CRITICAL]
            )

            if critical_count > 2 or avg_efficiency < 50:
                trends["efficiency_trajectory"] = "DECLINING"
                trends["risk_indicators"].append("Multiple critical bottlenecks detected")
            elif critical_count > 0 or avg_efficiency < 70:
                trends["efficiency_trajectory"] = "AT_RISK"
            elif avg_efficiency > 85:
                trends["efficiency_trajectory"] = "IMPROVING"

            return trends

        except Exception as e:
            logger.warning(f"Error analyzing performance trends: {e}")
            return {"error": str(e)}
