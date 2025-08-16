"""ProfileEvents core framework for ClickHouse monitoring.

This module provides a comprehensive framework for analyzing ClickHouse ProfileEvents,
including categorization, aggregation, comparison, and anomaly detection capabilities.
It serves as the foundation for all ProfileEvents-based analysis and troubleshooting.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")


class ProfileEventsCategory(Enum):
    """Comprehensive categorization of ClickHouse ProfileEvents.

    Based on the analysis of 400+ ProfileEvents from ClickHouse source code,
    these categories group related events for easier analysis and monitoring.
    """

    # Query execution and processing
    QUERY_EXECUTION = "query_execution"
    QUERY_PLANNING = "query_planning"
    QUERY_OPTIMIZATION = "query_optimization"

    # I/O operations
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    FILE_IO = "file_io"

    # Memory management
    MEMORY_ALLOCATION = "memory_allocation"
    MEMORY_DEALLOCATION = "memory_deallocation"
    MEMORY_TRACKING = "memory_tracking"

    # Cache operations
    MARK_CACHE = "mark_cache"
    UNCOMPRESSED_CACHE = "uncompressed_cache"
    COMPILED_EXPRESSION_CACHE = "compiled_expression_cache"
    QUERY_CACHE = "query_cache"

    # Storage and data processing
    DATA_COMPRESSION = "data_compression"
    DATA_PARTS = "data_parts"
    DATA_SKIPPING = "data_skipping"
    MERGES = "merges"

    # Replication and distributed operations
    REPLICATION = "replication"
    DISTRIBUTED_QUERIES = "distributed_queries"
    INTER_SERVER_COMMUNICATION = "inter_server_communication"

    # Threading and concurrency
    THREAD_POOL = "thread_pool"
    BACKGROUND_TASKS = "background_tasks"
    PARALLEL_PROCESSING = "parallel_processing"

    # External integrations
    S3_OPERATIONS = "s3_operations"
    HDFS_OPERATIONS = "hdfs_operations"
    KAFKA_OPERATIONS = "kafka_operations"
    MYSQL_OPERATIONS = "mysql_operations"
    POSTGRESQL_OPERATIONS = "postgresql_operations"

    # System resources
    CPU_USAGE = "cpu_usage"
    SYSTEM_CALLS = "system_calls"
    OS_OPERATIONS = "os_operations"

    # Error handling and exceptions
    EXCEPTIONS = "exceptions"
    RETRIES = "retries"
    TIMEOUTS = "timeouts"

    # Specialized operations
    AGGREGATE_FUNCTIONS = "aggregate_functions"
    TABLE_FUNCTIONS = "table_functions"
    DICTIONARIES = "dictionaries"
    MATERIALIZED_VIEWS = "materialized_views"

    # Performance monitoring
    PROFILING = "profiling"
    METRICS = "metrics"
    DEBUGGING = "debugging"

    # Uncategorized or misc
    MISCELLANEOUS = "miscellaneous"


@dataclass
class ProfileEventDefinition:
    """Definition of a ProfileEvent with its metadata."""

    name: str
    category: ProfileEventsCategory
    description: str
    unit: str = "count"
    is_cumulative: bool = True
    threshold_warning: float | None = None
    threshold_critical: float | None = None


@dataclass
class ProfileEventAggregation:
    """Aggregated ProfileEvent data over a time period."""

    event_name: str
    category: ProfileEventsCategory
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    p50_value: float
    p90_value: float
    p99_value: float
    stddev_value: float
    time_range_start: datetime
    time_range_end: datetime
    sample_queries: list[str] = field(default_factory=list)


@dataclass
class ProfileEventComparison:
    """Comparison result between two ProfileEvent datasets."""

    event_name: str
    category: ProfileEventsCategory
    baseline_stats: ProfileEventAggregation
    comparison_stats: ProfileEventAggregation
    change_percentage: float
    change_absolute: float
    significance_score: float
    is_anomaly: bool
    anomaly_reason: str | None = None


@dataclass
class ProfileEventThreshold:
    """Threshold configuration for ProfileEvent monitoring."""

    event_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_type: str  # "greater_than", "less_than", "absolute_change", "percentage_change"
    time_window_minutes: int = 15
    consecutive_violations: int = 3


class ProfileEventsAnalyzer:
    """Core analyzer for ClickHouse ProfileEvents data."""

    def __init__(self, client: Client):
        """Initialize the ProfileEvents analyzer.

        Args:
            client: ClickHouse client instance
        """
        self.client = client
        self.event_definitions = self._initialize_event_definitions()
        self.thresholds: dict[str, ProfileEventThreshold] = {}

    def _initialize_event_definitions(self) -> dict[str, ProfileEventDefinition]:
        """Initialize ProfileEvent definitions with categories and metadata."""
        definitions = {}

        # Query execution events
        query_events = [
            ("Query", "query_execution", "Number of queries executed"),
            ("SelectQuery", "query_execution", "Number of SELECT queries"),
            ("InsertQuery", "query_execution", "Number of INSERT queries"),
            ("DeleteQuery", "query_execution", "Number of DELETE queries"),
            ("UpdateQuery", "query_execution", "Number of UPDATE queries"),
            (
                "QueryTimeMicroseconds",
                "query_execution",
                "Total query execution time",
                "microseconds",
            ),
            (
                "SelectQueryTimeMicroseconds",
                "query_execution",
                "SELECT query execution time",
                "microseconds",
            ),
            (
                "InsertQueryTimeMicroseconds",
                "query_execution",
                "INSERT query execution time",
                "microseconds",
            ),
        ]

        # I/O events
        io_events = [
            ("ReadBufferFromFileDescriptorRead", "disk_io", "File descriptor reads"),
            (
                "ReadBufferFromFileDescriptorReadBytes",
                "disk_io",
                "Bytes read from file descriptors",
                "bytes",
            ),
            ("WriteBufferFromFileDescriptorWrite", "disk_io", "File descriptor writes"),
            (
                "WriteBufferFromFileDescriptorWriteBytes",
                "disk_io",
                "Bytes written to file descriptors",
                "bytes",
            ),
            (
                "NetworkReceiveElapsedMicroseconds",
                "network_io",
                "Network receive time",
                "microseconds",
            ),
            ("NetworkSendElapsedMicroseconds", "network_io", "Network send time", "microseconds"),
            ("NetworkReceiveBytes", "network_io", "Bytes received from network", "bytes"),
            ("NetworkSendBytes", "network_io", "Bytes sent to network", "bytes"),
        ]

        # Memory events
        memory_events = [
            (
                "MemoryTrackingInBackgroundProcessingPoolAllocated",
                "memory_allocation",
                "Background pool memory allocated",
                "bytes",
            ),
            (
                "MemoryTrackingInBackgroundMoveProcessingPoolAllocated",
                "memory_allocation",
                "Background move pool memory allocated",
                "bytes",
            ),
            (
                "MemoryTrackingForMerges",
                "memory_allocation",
                "Memory allocated for merges",
                "bytes",
            ),
            ("ArenaAllocChunks", "memory_allocation", "Arena allocation chunks"),
            ("ArenaAllocBytes", "memory_allocation", "Arena allocation bytes", "bytes"),
            ("ContextLock", "memory_tracking", "Context lock acquisitions"),
            ("RWLockAcquiredReadLocks", "memory_tracking", "Read locks acquired"),
            ("RWLockAcquiredWriteLocks", "memory_tracking", "Write locks acquired"),
        ]

        # Cache events
        cache_events = [
            ("MarkCacheHits", "mark_cache", "Mark cache hits"),
            ("MarkCacheMisses", "mark_cache", "Mark cache misses"),
            ("UncompressedCacheHits", "uncompressed_cache", "Uncompressed cache hits"),
            ("UncompressedCacheMisses", "uncompressed_cache", "Uncompressed cache misses"),
            ("UncompressedCacheWeightLost", "uncompressed_cache", "Uncompressed cache weight lost"),
            (
                "CompiledExpressionCacheHits",
                "compiled_expression_cache",
                "Compiled expression cache hits",
            ),
            (
                "CompiledExpressionCacheMisses",
                "compiled_expression_cache",
                "Compiled expression cache misses",
            ),
            ("QueryCacheHits", "query_cache", "Query cache hits"),
            ("QueryCacheMisses", "query_cache", "Query cache misses"),
        ]

        # CPU and system events
        system_events = [
            ("OSCPUVirtualTimeMicroseconds", "cpu_usage", "CPU virtual time", "microseconds"),
            ("OSCPUWaitMicroseconds", "cpu_usage", "CPU wait time", "microseconds"),
            ("OSIOWaitMicroseconds", "system_calls", "I/O wait time", "microseconds"),
            ("OSReadChars", "system_calls", "Characters read by OS"),
            ("OSWriteChars", "system_calls", "Characters written by OS"),
            ("OSReadBytes", "system_calls", "Bytes read by OS", "bytes"),
            ("OSWriteBytes", "system_calls", "Bytes written by OS", "bytes"),
            ("UserTimeMicroseconds", "cpu_usage", "User CPU time", "microseconds"),
            ("SystemTimeMicroseconds", "cpu_usage", "System CPU time", "microseconds"),
        ]

        # Replication events
        replication_events = [
            ("ReplicatedPartFetches", "replication", "Replicated part fetches"),
            ("ReplicatedPartFailedFetches", "replication", "Failed replicated part fetches"),
            ("ReplicatedPartMerges", "replication", "Replicated part merges"),
            ("ReplicatedPartFetchesOfMerged", "replication", "Fetches of merged parts"),
            ("ReplicatedPartChecks", "replication", "Replicated part checks"),
            ("ReplicatedPartChecksFailed", "replication", "Failed replicated part checks"),
            ("ReplicatedDataLoss", "replication", "Replicated data loss events"),
        ]

        # S3 and external storage events
        s3_events = [
            ("S3ReadMicroseconds", "s3_operations", "S3 read time", "microseconds"),
            ("S3WriteMicroseconds", "s3_operations", "S3 write time", "microseconds"),
            ("S3ReadBytes", "s3_operations", "Bytes read from S3", "bytes"),
            ("S3WriteBytes", "s3_operations", "Bytes written to S3", "bytes"),
            ("S3ReadRequestsCount", "s3_operations", "S3 read requests"),
            ("S3WriteRequestsCount", "s3_operations", "S3 write requests"),
            ("S3ReadRequestsErrors", "s3_operations", "S3 read request errors"),
            ("S3WriteRequestsErrors", "s3_operations", "S3 write request errors"),
        ]

        # Merge and parts events
        merge_events = [
            ("MergedRows", "merges", "Rows merged"),
            ("MergedUncompressedBytes", "merges", "Uncompressed bytes merged", "bytes"),
            ("MergesTimeMilliseconds", "merges", "Merge time", "milliseconds"),
            ("SelectedParts", "data_parts", "Parts selected for processing"),
            ("SelectedRanges", "data_parts", "Ranges selected for processing"),
            ("SelectedMarks", "data_parts", "Marks selected for processing"),
            ("SelectedRows", "data_parts", "Rows selected for processing"),
            ("SelectedBytes", "data_parts", "Bytes selected for processing", "bytes"),
        ]

        # Compression events
        compression_events = [
            ("CompressedReadBufferBlocks", "data_compression", "Compressed blocks read"),
            ("CompressedReadBufferBytes", "data_compression", "Compressed bytes read", "bytes"),
            ("CompressedWriteBufferBlocks", "data_compression", "Compressed blocks written"),
            ("CompressedWriteBufferBytes", "data_compression", "Compressed bytes written", "bytes"),
            ("AIOWrite", "disk_io", "Asynchronous I/O writes"),
            ("AIORead", "disk_io", "Asynchronous I/O reads"),
        ]

        # Exception and error events
        exception_events = [
            ("ZooKeeperExceptions", "exceptions", "ZooKeeper exceptions"),
            ("DistributedConnectionFailTry", "exceptions", "Distributed connection failures"),
            ("DistributedConnectionMissingTable", "exceptions", "Distributed missing table errors"),
            ("DistributedConnectionStaleReplica", "exceptions", "Distributed stale replica errors"),
            ("CannotWriteToWriteBufferDiscard", "exceptions", "Write buffer discard errors"),
            ("QueryMemoryLimitExceeded", "exceptions", "Query memory limit exceeded"),
        ]

        # Thread pool events
        thread_events = [
            ("CreatedHTTPConnections", "thread_pool", "HTTP connections created"),
            ("CannotScheduleTask", "thread_pool", "Tasks that cannot be scheduled"),
            ("QueryThread", "parallel_processing", "Query threads created"),
            ("BackgroundPoolTask", "background_tasks", "Background pool tasks"),
            ("BackgroundMovePoolTask", "background_tasks", "Background move pool tasks"),
            ("BackgroundFetchesPoolTask", "background_tasks", "Background fetches pool tasks"),
            ("BackgroundCommonPoolTask", "background_tasks", "Background common pool tasks"),
        ]

        # All event categories combined
        all_events = (
            query_events
            + io_events
            + memory_events
            + cache_events
            + system_events
            + replication_events
            + s3_events
            + merge_events
            + compression_events
            + exception_events
            + thread_events
        )

        # Create ProfileEventDefinition objects
        for event_data in all_events:
            name = event_data[0]
            category = ProfileEventsCategory(event_data[1])
            description = event_data[2]
            unit = event_data[3] if len(event_data) > 3 else "count"

            definitions[name] = ProfileEventDefinition(
                name=name, category=category, description=description, unit=unit
            )

        return definitions

    @log_execution_time
    def get_available_profile_events(self, days: int = 1) -> list[str]:
        """Get list of ProfileEvents available in the system over the specified period.

        Args:
            days: Number of days to look back

        Returns:
            List of available ProfileEvent names
        """
        query = f"""
        SELECT DISTINCT arrayJoin(ProfileEvents.Names) AS event_name
        FROM clusterAllReplicas(default, system.query_log)
        WHERE event_time >= now() - toIntervalDay({days})
          AND ProfileEvents.Names IS NOT NULL
          AND length(ProfileEvents.Names) > 0
        ORDER BY event_name
        """

        logger.info(f"Retrieving available ProfileEvents for the past {days} days")

        try:
            result = execute_query_with_retry(self.client, query)
            return [row["event_name"] for row in result]
        except ClickHouseError as e:
            logger.error(f"Error retrieving available ProfileEvents: {e!s}")
            # Fallback to local query
            fallback_query = f"""
            SELECT DISTINCT arrayJoin(ProfileEvents.Names) AS event_name
            FROM system.query_log
            WHERE event_time >= now() - toIntervalDay({days})
              AND ProfileEvents.Names IS NOT NULL
              AND length(ProfileEvents.Names) > 0
            ORDER BY event_name
            """
            logger.info("Falling back to local query_log query")
            result = execute_query_with_retry(self.client, fallback_query)
            return [row["event_name"] for row in result]

    @log_execution_time
    def aggregate_profile_events(
        self,
        event_names: list[str],
        start_time: datetime,
        end_time: datetime,
        query_filter: str | None = None,
    ) -> list[ProfileEventAggregation]:
        """Aggregate ProfileEvents data over a time period.

        Args:
            event_names: List of ProfileEvent names to aggregate
            start_time: Start of the time range
            end_time: End of the time range
            query_filter: Optional WHERE clause filter

        Returns:
            List of ProfileEventAggregation objects
        """
        # Build the ProfileEvents extraction query
        profile_events_selects = []
        for event_name in event_names:
            profile_events_selects.append(f"ProfileEvents['{event_name}'] AS {event_name}")

        profile_events_clause = ",\n        ".join(profile_events_selects)
        where_filter = f"AND ({query_filter})" if query_filter else ""

        query = f"""
        SELECT
            '{event_names[0]}' AS event_name,
            count() AS query_count,
            sum({event_names[0]}) AS sum_value,
            min({event_names[0]}) AS min_value,
            max({event_names[0]}) AS max_value,
            avg({event_names[0]}) AS avg_value,
            quantile(0.5)({event_names[0]}) AS p50_value,
            quantile(0.9)({event_names[0]}) AS p90_value,
            quantile(0.99)({event_names[0]}) AS p99_value,
            stddevPop({event_names[0]}) AS stddev_value,
            groupArray(query_id) AS sample_query_ids
        FROM (
            SELECT
                query_id,
                {profile_events_clause}
            FROM clusterAllReplicas(default, system.query_log)
            WHERE event_time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'
              AND event_time <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
              AND type != 'QueryStart'
              AND user NOT ILIKE '%internal%'
              {where_filter}
              AND ProfileEvents.Names IS NOT NULL
        )
        """

        logger.info(f"Aggregating ProfileEvents data from {start_time} to {end_time}")

        aggregations = []
        for event_name in event_names:
            try:
                # Execute query for each event individually to handle missing events gracefully
                single_event_query = query.replace(
                    f"'{event_names[0]}'", f"'{event_name}'"
                ).replace(f"{event_names[0]}", event_name)

                result = execute_query_with_retry(self.client, single_event_query)

                if result and len(result) > 0:
                    row = result[0]
                    category = self.event_definitions.get(event_name, {}).get(
                        "category", ProfileEventsCategory.MISCELLANEOUS
                    )

                    aggregation = ProfileEventAggregation(
                        event_name=event_name,
                        category=category,
                        count=row["query_count"],
                        sum_value=float(row["sum_value"] or 0),
                        min_value=float(row["min_value"] or 0),
                        max_value=float(row["max_value"] or 0),
                        avg_value=float(row["avg_value"] or 0),
                        p50_value=float(row["p50_value"] or 0),
                        p90_value=float(row["p90_value"] or 0),
                        p99_value=float(row["p99_value"] or 0),
                        stddev_value=float(row["stddev_value"] or 0),
                        time_range_start=start_time,
                        time_range_end=end_time,
                        sample_queries=(
                            row["sample_query_ids"][:10] if row["sample_query_ids"] else []
                        ),
                    )
                    aggregations.append(aggregation)

            except ClickHouseError as e:
                logger.warning(f"Failed to aggregate ProfileEvent '{event_name}': {e!s}")
                continue

        return aggregations

    @log_execution_time
    def get_profile_events_by_query(
        self, query_id: str, include_thread_log: bool = True
    ) -> dict[str, int | float]:
        """Get ProfileEvents data for a specific query.

        Args:
            query_id: The query ID to analyze
            include_thread_log: Whether to include thread-level ProfileEvents

        Returns:
            Dictionary mapping ProfileEvent names to their values
        """
        base_query = """
        SELECT
            ProfileEvents.Names AS event_names,
            ProfileEvents.Values AS event_values
        FROM clusterAllReplicas(default, {table})
        WHERE query_id = '{query_id}'
          AND type != 'QueryStart'
          AND ProfileEvents.Names IS NOT NULL
        ORDER BY event_time DESC
        LIMIT 1
        """

        logger.info(f"Retrieving ProfileEvents for query {query_id}")

        profile_events = {}

        try:
            # Get ProfileEvents from query_log
            query_log_query = base_query.format(table="system.query_log", query_id=query_id)
            result = execute_query_with_retry(self.client, query_log_query)

            if result and len(result) > 0:
                names = result[0]["event_names"]
                values = result[0]["event_values"]

                if names and values:
                    for name, value in zip(names, values, strict=False):
                        profile_events[name] = value

            # If requested and no data found, try thread log
            if include_thread_log and not profile_events:
                thread_log_query = base_query.format(
                    table="system.query_thread_log", query_id=query_id
                )
                result = execute_query_with_retry(self.client, thread_log_query)

                if result:
                    # Aggregate thread-level events
                    thread_events = {}
                    for row in result:
                        names = row["event_names"]
                        values = row["event_values"]

                        if names and values:
                            for name, value in zip(names, values, strict=False):
                                thread_events[name] = thread_events.get(name, 0) + value

                    profile_events.update(thread_events)

        except ClickHouseError as e:
            logger.error(f"Error retrieving ProfileEvents for query {query_id}: {e!s}")
            # Try fallback to local queries
            try:
                local_query = base_query.replace("clusterAllReplicas(default, ", "").replace(
                    ")", "", 1
                )
                local_query = local_query.format(table="system.query_log", query_id=query_id)
                result = execute_query_with_retry(self.client, local_query)

                if result and len(result) > 0:
                    names = result[0]["event_names"]
                    values = result[0]["event_values"]

                    if names and values:
                        for name, value in zip(names, values, strict=False):
                            profile_events[name] = value

            except ClickHouseError as e2:
                logger.error(f"Fallback query also failed for query {query_id}: {e2!s}")

        return profile_events

    @log_execution_time
    def get_top_profile_events(
        self, category: ProfileEventsCategory | None = None, days: int = 1, limit: int = 20
    ) -> list[tuple[str, float, ProfileEventsCategory]]:
        """Get the top ProfileEvents by total value over a time period.

        Args:
            category: Optional category filter
            days: Number of days to look back
            limit: Maximum number of events to return

        Returns:
            List of tuples (event_name, total_value, category)
        """
        # Get available events and filter by category if specified
        available_events = self.get_available_profile_events(days)

        if category:
            filtered_events = []
            for event_name in available_events:
                event_def = self.event_definitions.get(event_name)
                if event_def and event_def.category == category:
                    filtered_events.append(event_name)
            available_events = filtered_events

        if not available_events:
            logger.warning(f"No ProfileEvents found for category {category}")
            return []

        # Build dynamic query for all events
        event_sums = []
        for event_name in available_events[:100]:  # Limit to prevent query complexity
            event_sums.append(f"sum(ProfileEvents['{event_name}']) AS {event_name}")

        query = f"""
        SELECT {', '.join(event_sums)}
        FROM clusterAllReplicas(default, system.query_log)
        WHERE event_time >= now() - toIntervalDay({days})
          AND type != 'QueryStart'
          AND user NOT ILIKE '%internal%'
          AND ProfileEvents.Names IS NOT NULL
        """

        logger.info(f"Retrieving top ProfileEvents for the past {days} days")

        try:
            result = execute_query_with_retry(self.client, query)

            if not result or len(result) == 0:
                return []

            # Extract and sort events by value
            event_totals = []
            row = result[0]

            for event_name in available_events[:100]:
                total_value = row.get(event_name, 0)
                if total_value and total_value > 0:
                    event_def = self.event_definitions.get(event_name)
                    event_category = (
                        event_def.category if event_def else ProfileEventsCategory.MISCELLANEOUS
                    )
                    event_totals.append((event_name, float(total_value), event_category))

            # Sort by total value descending
            event_totals.sort(key=lambda x: x[1], reverse=True)
            return event_totals[:limit]

        except ClickHouseError as e:
            logger.error(f"Error retrieving top ProfileEvents: {e!s}")
            return []

    @log_execution_time
    def analyze_comprehensive(self, hours: int = 24) -> dict[str, Any]:
        """Perform comprehensive ProfileEvents analysis by category.

        Args:
            hours: Number of hours to analyze

        Returns:
            Comprehensive analysis results by category
        """
        try:
            results = {}

            # Get all available events
            available_events = self.get_available_profile_events(days=1)

            # Group events by category
            events_by_category = {}
            for event in available_events:
                event_def = self.event_definitions.get(event)
                category = event_def.category if event_def else ProfileEventsCategory.MISCELLANEOUS
                if category not in events_by_category:
                    events_by_category[category] = []
                events_by_category[category].append(event)

            # Analyze each category
            for category, events in events_by_category.items():
                if events:
                    category_analysis = self.aggregate_profile_events(
                        event_names=events, hours=hours, aggregation_type="hourly"
                    )
                    results[category.value] = category_analysis

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive ProfileEvents analysis: {e!s}")
            return {"error": str(e)}

    @log_execution_time
    def analyze_zookeeper_operations(self, hours: int = 24) -> dict[str, Any]:
        """Analyze ZooKeeper-related ProfileEvents.

        Args:
            hours: Number of hours to analyze

        Returns:
            ZooKeeper operations analysis
        """
        try:
            zk_events = [
                "ZooKeeperInit",
                "ZooKeeperTransactions",
                "ZooKeeperList",
                "ZooKeeperCreate",
                "ZooKeeperRemove",
                "ZooKeeperExists",
                "ZooKeeperGet",
                "ZooKeeperSet",
                "ZooKeeperMulti",
                "ZooKeeperCheck",
                "ZooKeeperClose",
                "ZooKeeperWatchResponse",
                "ZooKeeperUserExceptions",
                "ZooKeeperHardwareExceptions",
                "ZooKeeperOtherExceptions",
                "ZooKeeperWaitMicroseconds",
            ]

            # Filter events that actually exist
            available_events = self.get_available_profile_events(days=1)
            existing_zk_events = [e for e in zk_events if e in available_events]

            if not existing_zk_events:
                return {"message": "No ZooKeeper ProfileEvents found in the system"}

            return self.aggregate_profile_events(
                event_names=existing_zk_events, hours=hours, aggregation_type="hourly"
            )

        except Exception as e:
            logger.error(f"Error analyzing ZooKeeper operations: {e!s}")
            return {"error": str(e)}

    @log_execution_time
    def analyze_replication_performance(self, hours: int = 24) -> dict[str, Any]:
        """Analyze replication-related ProfileEvents.

        Args:
            hours: Number of hours to analyze

        Returns:
            Replication performance analysis
        """
        try:
            replication_events = [
                "ReplicatedPartFetches",
                "ReplicatedPartFailedFetches",
                "ReplicatedDataLoss",
                "ReplicatedPartMerges",
                "ReplicatedPartFetchesOfMerged",
                "ReplicatedPartMutations",
                "ReplicatedPartChecks",
                "ReplicatedPartChecksFailed",
                "ReplicatedPartSends",
                "ReplicatedPartSendsRejectedByThrottler",
                "DataAfterMergeDiffersFromReplica",
                "DataAfterMutationDiffersFromReplica",
                "PolygonsInPoolAllocated",
                "PolygonsInPoolAllocatedBytes",
            ]

            # Filter events that actually exist
            available_events = self.get_available_profile_events(days=1)
            existing_repl_events = [e for e in replication_events if e in available_events]

            if not existing_repl_events:
                return {"message": "No replication ProfileEvents found in the system"}

            return self.aggregate_profile_events(
                event_names=existing_repl_events, hours=hours, aggregation_type="hourly"
            )

        except Exception as e:
            logger.error(f"Error analyzing replication performance: {e!s}")
            return {"error": str(e)}

    @log_execution_time
    def analyze_distributed_queries(self, hours: int = 24) -> dict[str, Any]:
        """Analyze distributed query ProfileEvents.

        Args:
            hours: Number of hours to analyze

        Returns:
            Distributed query performance analysis
        """
        try:
            distributed_events = [
                "DistributedConnectionTries",
                "DistributedConnectionUsable",
                "DistributedConnectionFailTries",
                "DistributedConnectionMissingTable",
                "DistributedConnectionStaleReplica",
                "DistributedConnectionFailAtAll",
                "DistributedSyncInsertionTimeoutExceeded",
                "DistributedAsyncInsertionFailures",
                "CompileAttempt",
                "CompileSuccess",
                "CompileExpressionsMicroseconds",
                "CompileExpressionsBytes",
                "CompiledFunctionExecute",
            ]

            # Filter events that actually exist
            available_events = self.get_available_profile_events(days=1)
            existing_dist_events = [e for e in distributed_events if e in available_events]

            if not existing_dist_events:
                return {"message": "No distributed query ProfileEvents found in the system"}

            return self.aggregate_profile_events(
                event_names=existing_dist_events, hours=hours, aggregation_type="hourly"
            )

        except Exception as e:
            logger.error(f"Error analyzing distributed queries: {e!s}")
            return {"error": str(e)}


class ProfileEventsComparator:
    """Compares ProfileEvents between different time periods or query sets."""

    def __init__(self, analyzer: ProfileEventsAnalyzer):
        """Initialize the comparator with a ProfileEvents analyzer.

        Args:
            analyzer: ProfileEventsAnalyzer instance
        """
        self.analyzer = analyzer

    @log_execution_time
    def compare_time_periods(
        self,
        event_names: list[str],
        baseline_start: datetime,
        baseline_end: datetime,
        comparison_start: datetime,
        comparison_end: datetime,
        baseline_filter: str | None = None,
        comparison_filter: str | None = None,
    ) -> list[ProfileEventComparison]:
        """Compare ProfileEvents between two time periods.

        Args:
            event_names: List of ProfileEvent names to compare
            baseline_start: Start of baseline period
            baseline_end: End of baseline period
            comparison_start: Start of comparison period
            comparison_end: End of comparison period
            baseline_filter: Optional filter for baseline period
            comparison_filter: Optional filter for comparison period

        Returns:
            List of ProfileEventComparison objects
        """
        logger.info(
            f"Comparing ProfileEvents between periods: "
            f"{baseline_start}-{baseline_end} vs {comparison_start}-{comparison_end}"
        )

        # Get aggregations for both periods
        baseline_aggs = self.analyzer.aggregate_profile_events(
            event_names, baseline_start, baseline_end, baseline_filter
        )
        comparison_aggs = self.analyzer.aggregate_profile_events(
            event_names, comparison_start, comparison_end, comparison_filter
        )

        # Create lookup dictionaries
        baseline_dict = {agg.event_name: agg for agg in baseline_aggs}
        comparison_dict = {agg.event_name: agg for agg in comparison_aggs}

        comparisons = []

        for event_name in event_names:
            baseline_agg = baseline_dict.get(event_name)
            comparison_agg = comparison_dict.get(event_name)

            if not baseline_agg or not comparison_agg:
                logger.warning(
                    f"Missing data for ProfileEvent '{event_name}' in one or both periods"
                )
                continue

            # Calculate change metrics
            baseline_value = baseline_agg.avg_value
            comparison_value = comparison_agg.avg_value

            if baseline_value == 0:
                change_percentage = float("inf") if comparison_value > 0 else 0
            else:
                change_percentage = ((comparison_value - baseline_value) / baseline_value) * 100

            change_absolute = comparison_value - baseline_value

            # Calculate significance score (simplified version)
            significance_score = self._calculate_significance_score(baseline_agg, comparison_agg)

            # Determine if this is an anomaly
            is_anomaly, anomaly_reason = self._is_anomaly(
                event_name, baseline_agg, comparison_agg, change_percentage, significance_score
            )

            comparison = ProfileEventComparison(
                event_name=event_name,
                category=baseline_agg.category,
                baseline_stats=baseline_agg,
                comparison_stats=comparison_agg,
                change_percentage=change_percentage,
                change_absolute=change_absolute,
                significance_score=significance_score,
                is_anomaly=is_anomaly,
                anomaly_reason=anomaly_reason,
            )

            comparisons.append(comparison)

        # Sort by significance score descending
        comparisons.sort(key=lambda x: x.significance_score, reverse=True)
        return comparisons

    def _calculate_significance_score(
        self, baseline: ProfileEventAggregation, comparison: ProfileEventAggregation
    ) -> float:
        """Calculate a significance score for the change between two aggregations.

        Args:
            baseline: Baseline aggregation
            comparison: Comparison aggregation

        Returns:
            Significance score (higher = more significant)
        """
        try:
            # Use coefficient of variation and magnitude of change
            baseline_cv = (
                baseline.stddev_value / baseline.avg_value if baseline.avg_value > 0 else 0
            )
            comparison_cv = (
                comparison.stddev_value / comparison.avg_value if comparison.avg_value > 0 else 0
            )

            # Average coefficient of variation (measure of relative variability)
            avg_cv = (baseline_cv + comparison_cv) / 2

            # Magnitude of change relative to baseline
            if baseline.avg_value == 0:
                magnitude = abs(comparison.avg_value) if comparison.avg_value > 0 else 0
            else:
                magnitude = abs(comparison.avg_value - baseline.avg_value) / baseline.avg_value

            # Significance score: higher magnitude and lower variability = more significant
            significance = magnitude / max(avg_cv, 0.1)  # Prevent division by zero

            return min(significance, 100.0)  # Cap at 100

        except (ZeroDivisionError, ValueError):
            return 0.0

    def _is_anomaly(
        self,
        event_name: str,
        baseline: ProfileEventAggregation,
        comparison: ProfileEventAggregation,
        change_percentage: float,
        significance_score: float,
    ) -> tuple[bool, str | None]:
        """Determine if a ProfileEvent change represents an anomaly.

        Args:
            event_name: Name of the ProfileEvent
            baseline: Baseline aggregation
            comparison: Comparison aggregation
            change_percentage: Percentage change
            significance_score: Calculated significance score

        Returns:
            Tuple of (is_anomaly, reason)
        """
        # Check configured thresholds first
        threshold = self.analyzer.thresholds.get(event_name)
        if threshold:
            if threshold.comparison_type == "percentage_change":
                if abs(change_percentage) > threshold.warning_threshold:
                    severity = (
                        "critical"
                        if abs(change_percentage) > threshold.critical_threshold
                        else "warning"
                    )
                    return (
                        True,
                        f"Percentage change {change_percentage:.1f}% exceeds {severity} threshold",
                    )
            elif threshold.comparison_type == "absolute_change":
                abs_change = abs(comparison.avg_value - baseline.avg_value)
                if abs_change > threshold.warning_threshold:
                    severity = (
                        "critical" if abs_change > threshold.critical_threshold else "warning"
                    )
                    return True, f"Absolute change {abs_change:.2f} exceeds {severity} threshold"

        # Default anomaly detection rules
        # High significance score with substantial change
        if significance_score > 10 and abs(change_percentage) > 50:
            return (
                True,
                f"High significance score ({significance_score:.1f}) with {change_percentage:.1f}% change",
            )

        # Very large percentage changes
        if abs(change_percentage) > 200:
            return True, f"Very large percentage change: {change_percentage:.1f}%"

        # Large absolute changes for high-volume events
        if baseline.avg_value > 1000 and abs(comparison.avg_value - baseline.avg_value) > 5000:
            return True, "Large absolute change for high-volume event"

        # Check for events that went from zero to non-zero or vice versa
        if baseline.avg_value == 0 and comparison.avg_value > 100:
            return True, "Event went from zero to significant activity"

        if baseline.avg_value > 100 and comparison.avg_value == 0:
            return True, "Event went from significant activity to zero"

        return False, None


# Utility functions
def format_profile_event_value(value: int | float, unit: str = "count") -> str:
    """Format a ProfileEvent value for display.

    Args:
        value: The numeric value
        unit: The unit of measurement

    Returns:
        Formatted string representation
    """
    if unit == "bytes":
        if value >= 1024**4:
            return f"{value / (1024**4):.2f} TB"
        elif value >= 1024**3:
            return f"{value / (1024**3):.2f} GB"
        elif value >= 1024**2:
            return f"{value / (1024**2):.2f} MB"
        elif value >= 1024:
            return f"{value / 1024:.2f} KB"
        else:
            return f"{value:.0f} B"
    elif unit == "microseconds":
        if value >= 1000000:
            return f"{value / 1000000:.2f} s"
        elif value >= 1000:
            return f"{value / 1000:.2f} ms"
        else:
            return f"{value:.0f} μs"
    elif unit == "milliseconds":
        if value >= 1000:
            return f"{value / 1000:.2f} s"
        else:
            return f"{value:.0f} ms"
    else:
        if value >= 1000000:
            return f"{value / 1000000:.2f}M"
        elif value >= 1000:
            return f"{value / 1000:.2f}K"
        else:
            return f"{value:.0f}"


def get_profile_events_by_category(
    analyzer: ProfileEventsAnalyzer, category: ProfileEventsCategory
) -> list[str]:
    """Get all ProfileEvent names for a specific category.

    Args:
        analyzer: ProfileEventsAnalyzer instance
        category: The category to filter by

    Returns:
        List of ProfileEvent names in the specified category
    """
    return [
        name
        for name, definition in analyzer.event_definitions.items()
        if definition.category == category
    ]


def detect_profile_event_anomalies(
    comparisons: list[ProfileEventComparison], min_significance_score: float = 5.0
) -> list[ProfileEventComparison]:
    """Filter ProfileEvent comparisons to only return anomalies.

    Args:
        comparisons: List of ProfileEventComparison objects
        min_significance_score: Minimum significance score to consider

    Returns:
        List of ProfileEventComparison objects that represent anomalies
    """
    anomalies = []

    for comparison in comparisons:
        if comparison.is_anomaly or comparison.significance_score >= min_significance_score:
            anomalies.append(comparison)

    return anomalies


def create_profile_events_summary(
    aggregations: list[ProfileEventAggregation],
) -> dict[ProfileEventsCategory, dict[str, Any]]:
    """Create a summary of ProfileEvents aggregations grouped by category.

    Args:
        aggregations: List of ProfileEventAggregation objects

    Returns:
        Dictionary mapping categories to summary statistics
    """
    summary = {}

    # Group by category
    category_groups = {}
    for agg in aggregations:
        if agg.category not in category_groups:
            category_groups[agg.category] = []
        category_groups[agg.category].append(agg)

    # Calculate summary for each category
    for category, aggs in category_groups.items():
        total_events = len(aggs)
        total_sum = sum(agg.sum_value for agg in aggs)
        avg_avg = statistics.mean(agg.avg_value for agg in aggs)
        max_max = max(agg.max_value for agg in aggs)
        top_events = sorted(aggs, key=lambda x: x.sum_value, reverse=True)[:5]

        summary[category] = {
            "total_events": total_events,
            "total_sum": total_sum,
            "average_value": avg_avg,
            "maximum_value": max_max,
            "top_events": [(e.event_name, e.sum_value) for e in top_events],
        }

    return summary
