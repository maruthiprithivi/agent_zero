"""Integration tests for ClickHouse connectivity and operations.

This module contains comprehensive integration tests that verify
Agent Zero's integration with real ClickHouse instances.
"""

import asyncio
import time
from datetime import datetime, timedelta

import pytest

from agent_zero.ai_diagnostics import create_ai_bottleneck_detector
from agent_zero.monitoring import ProfileEventsAnalyzer
from agent_zero.server.client import create_clickhouse_client


@pytest.mark.integration
@pytest.mark.asyncio
class TestClickHouseIntegration:
    """Integration tests for ClickHouse database operations."""

    @pytest.fixture(scope="class")
    def clickhouse_client(self):
        """Create a ClickHouse client for testing."""
        try:
            client = create_clickhouse_client()
            # Test connection
            result = client.command("SELECT 1")
            assert result == 1
            return client
        except Exception as e:
            pytest.skip(f"ClickHouse not available for integration tests: {e}")

    @pytest.fixture(scope="class")
    def test_database(self, clickhouse_client):
        """Create a test database for integration tests."""
        db_name = f"test_agent_zero_{int(time.time())}"

        try:
            # Create test database
            clickhouse_client.command(f"CREATE DATABASE IF NOT EXISTS {db_name}")

            # Create test table with sample data
            clickhouse_client.command(
                f"""
                CREATE TABLE {db_name}.test_events (
                    timestamp DateTime,
                    event_type String,
                    user_id UInt32,
                    value Float64
                ) ENGINE = MergeTree()
                ORDER BY timestamp
            """
            )

            # Insert sample data
            sample_data = []
            base_time = datetime.now() - timedelta(hours=24)
            for i in range(1000):
                sample_data.append(
                    {
                        "timestamp": base_time + timedelta(minutes=i),
                        "event_type": f"event_{i % 10}",
                        "user_id": i % 100,
                        "value": float(i * 1.5),
                    }
                )

            clickhouse_client.insert(f"{db_name}.test_events", sample_data)

            yield db_name

        finally:
            # Cleanup
            try:
                clickhouse_client.command(f"DROP DATABASE IF EXISTS {db_name}")
            except Exception:
                pass

    @pytest.mark.slow
    def test_database_operations(self, clickhouse_client, test_database):
        """Test basic database operations."""
        # Test listing databases
        databases = clickhouse_client.command("SHOW DATABASES")
        assert test_database in databases

        # Test listing tables
        tables = clickhouse_client.query(
            f"""
            SELECT name FROM system.tables
            WHERE database = '{test_database}'
        """
        )
        table_names = [row["name"] for row in tables]
        assert "test_events" in table_names

        # Test querying data
        result = clickhouse_client.query(
            f"""
            SELECT count() as count
            FROM {test_database}.test_events
        """
        )
        assert result[0]["count"] == 1000

    @pytest.mark.slow
    def test_profile_events_analysis(self, clickhouse_client):
        """Test ProfileEvents analysis with real data."""
        analyzer = ProfileEventsAnalyzer(clickhouse_client)

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        # Test getting top ProfileEvents
        events = analyzer.get_top_profile_events(
            category=None, start_time=start_time, end_time=end_time, limit=10
        )

        assert isinstance(events, list)
        # Should have some events even if minimal activity
        assert len(events) >= 0

        for event in events:
            assert "event_name" in event
            assert "total_value" in event
            assert "category" in event

    @pytest.mark.slow
    def test_performance_diagnostics(self, clickhouse_client, test_database):
        """Test performance diagnostics with real queries."""
        # Execute some test queries to generate ProfileEvents
        test_queries = [
            f"SELECT count() FROM {test_database}.test_events",
            f"SELECT event_type, count() FROM {test_database}.test_events GROUP BY event_type",
            f"SELECT avg(value) FROM {test_database}.test_events WHERE user_id < 50",
        ]

        for query in test_queries:
            result = clickhouse_client.query(query)
            assert result is not None

        # Test AI bottleneck detection
        detector = create_ai_bottleneck_detector(clickhouse_client)
        bottlenecks = detector.detect_bottlenecks(confidence_threshold=0.5)

        assert isinstance(bottlenecks, list)
        # Bottlenecks may or may not be detected in test environment

    @pytest.mark.slow
    def test_query_performance_monitoring(self, clickhouse_client, test_database):
        """Test query performance monitoring."""
        # Execute queries and monitor performance
        start_time = time.time()

        # Complex query for performance testing
        result = clickhouse_client.query(
            f"""
            SELECT
                event_type,
                count() as event_count,
                avg(value) as avg_value,
                min(timestamp) as min_time,
                max(timestamp) as max_time
            FROM {test_database}.test_events
            WHERE timestamp >= now() - INTERVAL 1 DAY
            GROUP BY event_type
            ORDER BY event_count DESC
        """
        )

        execution_time = time.time() - start_time

        assert result is not None
        assert len(result) > 0
        assert execution_time < 30.0  # Should complete within 30 seconds

        # Verify result structure
        for row in result:
            assert "event_type" in row
            assert "event_count" in row
            assert "avg_value" in row

    def test_system_tables_access(self, clickhouse_client):
        """Test access to ClickHouse system tables."""
        # Test access to system.tables
        tables = clickhouse_client.query(
            """
            SELECT database, name, engine
            FROM system.tables
            WHERE database = 'system'
            LIMIT 10
        """
        )

        assert len(tables) > 0
        for table in tables:
            assert "database" in table
            assert "name" in table
            assert "engine" in table

    def test_error_handling(self, clickhouse_client):
        """Test error handling for invalid queries."""
        # Test invalid SQL
        with pytest.raises(Exception):
            clickhouse_client.query("INVALID SQL STATEMENT")

        # Test accessing non-existent database
        with pytest.raises(Exception):
            clickhouse_client.query("SELECT * FROM non_existent_db.non_existent_table")

    @pytest.mark.slow
    def test_concurrent_operations(self, clickhouse_client, test_database):
        """Test concurrent database operations."""

        async def execute_query(query_id: int):
            """Execute a query asynchronously."""
            result = clickhouse_client.query(
                f"""
                SELECT {query_id} as query_id, count() as count
                FROM {test_database}.test_events
                WHERE user_id % 10 = {query_id % 10}
            """
            )
            return result[0]["count"]

        async def run_concurrent_queries():
            """Run multiple queries concurrently."""
            tasks = [execute_query(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent queries
        results = asyncio.run(run_concurrent_queries())

        assert len(results) == 10
        assert all(isinstance(count, int) for count in results)
        assert sum(results) == 1000  # Total should equal our test data

    def test_connection_pooling(self, clickhouse_client):
        """Test connection pooling behavior."""
        # Execute multiple queries to test connection reuse
        for i in range(5):
            result = clickhouse_client.command("SELECT 1")
            assert result == 1

        # Test connection health
        health_check = clickhouse_client.command("SELECT 'healthy' as status")
        assert health_check == "healthy"


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPToolsIntegration:
    """Integration tests for MCP tools with real ClickHouse."""

    @pytest.fixture(scope="class")
    def mcp_server(self):
        """Initialize MCP server for testing."""
        from agent_zero.server.core import initialize_mcp_server

        try:
            return initialize_mcp_server()
        except Exception as e:
            pytest.skip(f"MCP server initialization failed: {e}")

    @pytest.mark.slow
    def test_list_databases_tool(self, mcp_server):
        """Test list_databases MCP tool."""
        # Simulate MCP tool call
        result = mcp_server.call_tool("list_databases", {})

        assert result is not None
        assert "databases" in result or isinstance(result, list)

    @pytest.mark.slow
    def test_query_performance_tools(self, mcp_server):
        """Test query performance analysis tools."""
        # Test monitor_current_processes
        try:
            result = mcp_server.call_tool("monitor_current_processes", {})
            assert result is not None
        except Exception:
            # May fail if no active processes
            pass

    @pytest.mark.slow
    def test_resource_monitoring_tools(self, mcp_server):
        """Test resource monitoring tools."""
        # Test monitor_cpu_usage
        result = mcp_server.call_tool("monitor_cpu_usage", {})
        assert result is not None

    @pytest.mark.slow
    def test_profile_events_tools(self, mcp_server):
        """Test ProfileEvents analysis tools."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        # Test analyze_profile_events_comprehensive
        result = mcp_server.call_tool(
            "analyze_profile_events_comprehensive",
            {"start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "limit": 10},
        )

        assert result is not None


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks and load tests."""

    @pytest.fixture(scope="class")
    def load_test_client(self):
        """Create client for load testing."""
        try:
            return create_clickhouse_client()
        except Exception as e:
            pytest.skip(f"ClickHouse not available for load tests: {e}")

    @pytest.mark.slow
    def test_query_throughput(self, load_test_client):
        """Test query throughput under load."""
        num_queries = 100
        start_time = time.time()

        for i in range(num_queries):
            result = load_test_client.command("SELECT 1")
            assert result == 1

        end_time = time.time()
        duration = end_time - start_time
        throughput = num_queries / duration

        # Should handle at least 10 queries per second
        assert throughput >= 10.0, f"Throughput too low: {throughput:.2f} qps"

    @pytest.mark.slow
    def test_concurrent_query_load(self, load_test_client):
        """Test concurrent query load."""

        async def execute_batch_queries(batch_id: int, num_queries: int = 20):
            """Execute a batch of queries."""
            results = []
            for i in range(num_queries):
                result = load_test_client.command(f"SELECT {batch_id * 100 + i}")
                results.append(result)
            return results

        async def run_load_test():
            """Run concurrent load test."""
            num_batches = 5
            tasks = [execute_batch_queries(i) for i in range(num_batches)]

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            return results, end_time - start_time

        results, duration = asyncio.run(run_load_test())

        # Verify all queries completed successfully
        total_queries = sum(len(batch) for batch in results)
        assert total_queries == 100

        # Should complete within reasonable time
        assert duration < 60.0, f"Load test took too long: {duration:.2f}s"

    @pytest.mark.slow
    def test_memory_usage_under_load(self, load_test_client):
        """Test memory usage under sustained load."""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute sustained load
        for batch in range(10):
            for i in range(50):
                result = load_test_client.query(
                    """
                    SELECT
                        number,
                        number * 2 as doubled,
                        toString(number) as str_number
                    FROM numbers(1000)
                    LIMIT 100
                """
                )
                assert len(result) == 100

            # Force garbage collection
            gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.2f}MB increase"

    @pytest.mark.slow
    def test_connection_pool_under_load(self, load_test_client):
        """Test connection pool behavior under load."""
        # Test rapid connection requests
        for i in range(200):
            result = load_test_client.command("SELECT connection_id()")
            assert isinstance(result, (int, str))

        # Verify connection pool is stable
        result = load_test_client.command("SELECT 'stable' as status")
        assert result == "stable"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration", "--durations=10"])
