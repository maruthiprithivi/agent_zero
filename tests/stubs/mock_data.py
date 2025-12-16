"""Mock data generator for ClickHouse tests.

This module provides realistic mock data for various ClickHouse system tables
and query results to support comprehensive unit testing without a real database.
"""

from datetime import datetime, timedelta
import random

def generate_mock_system_tables(database_name="default", count=10):
    """Generate mock data for system.tables."""
    tables = []
    engines = ["MergeTree", "ReplicatedMergeTree", "ReplacingMergeTree", "SummingMergeTree", "AggregatingMergeTree"]

    for i in range(count):
        tables.append({
            "database": database_name,
            "name": f"table_{i}",
            "engine": random.choice(engines),
            "total_rows": random.randint(1000, 10000000),
            "total_bytes": random.randint(1024 * 1024, 1024 * 1024 * 1024),
            "metadata_modification_time": datetime.now() - timedelta(days=random.randint(0, 30))
        })
    return tables

def generate_mock_profile_events(count=20):
    """Generate mock data for system.profile_events (or query log analysis)."""
    events = []
    event_names = [
        "Query", "SelectQuery", "InsertQuery", "FailedQuery",
        "ReadCompressedBytes", "ReadBufferFromFileDescriptorRead",
        "MarkCacheHits", "MarkCacheMisses", "Merge",
        "OSCPUVirtualTimeMicroseconds", "OSCPUWaitMicroseconds"
    ]

    for i in range(count):
        events.append({
            "event_name": random.choice(event_names),
            "hostname": f"host-{i % 3}",
            "value": random.uniform(10, 1000),
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 60))
        })
    return events

def generate_mock_query_log(count=50):
    """Generate mock data for system.query_log."""
    logs = []
    query_types = ["Select", "Insert", "Create", "Alter"]

    for i in range(count):
        logs.append({
            "type": random.choice(query_types),
            "event_time": datetime.now() - timedelta(minutes=random.randint(0, 120)),
            "query_duration_ms": random.randint(1, 5000),
            "read_rows": random.randint(100, 10000),
            "read_bytes": random.randint(1000, 1000000),
            "memory_usage": random.randint(10000, 10000000),
            "query": f"SELECT * FROM table_{i % 5}",
            "exception": "" if random.random() > 0.1 else "Code: 123. DB::Exception: ..."
        })
    return logs

class MockClickHouseResult:
    """Mock for ClickHouse client query result."""

    def __init__(self, data, columns=None):
        self.result_rows = [list(d.values()) if isinstance(d, dict) else d for d in data]
        if columns:
            self.column_names = columns
        elif data and isinstance(data[0], dict):
            self.column_names = list(data[0].keys())
        else:
            self.column_names = [f"col_{i}" for i in range(len(self.result_rows[0]))] if self.result_rows else []

    def __iter__(self):
        return iter(self.result_rows)

    def __len__(self):
        return len(self.result_rows)

    def __getitem__(self, item):
        return self.result_rows[item]
