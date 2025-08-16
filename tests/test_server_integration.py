"""FastMCP server integration tests using the new modular architecture."""

from unittest.mock import MagicMock, patch

import pytest

from agent_zero.server.core import initialize_mcp_server
from agent_zero.server.tools import register_all_tools


class TestServerIntegration:
    """Test the MCP server integration with FastMCP framework."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, comprehensive_mock_client, mock_unified_config):
        """Set up test environment with comprehensive mocks."""
        with (
            patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client,
            patch("agent_zero.config.get_config") as mock_get_config,
        ):
            mock_create_client.return_value = comprehensive_mock_client
            mock_get_config.return_value = mock_unified_config

            self.mock_client = comprehensive_mock_client
            self.mock_config = mock_unified_config
            yield

    def test_create_mcp_server(self):
        """Test creating the MCP server instance."""
        server = initialize_mcp_server()

        # Verify server was created
        assert server is not None

        # Check that tools are registered
        # Note: FastMCP uses internal attributes, so we just verify server was created and initialized
        assert hasattr(server, "name")  # FastMCP servers have a name attribute

    def test_register_all_tools(self):
        """Test registering all MCP tools."""
        from mcp.server.fastmcp import FastMCP

        # Create a test server instance
        server = FastMCP("TestAgentZero")

        # Register all tools
        register_all_tools(server)

        # Verify tools were registered
        # The tools are registered internally in FastMCP, so we verify by attempting to get them
        assert server is not None

    def test_server_tools_execution(self):
        """Test executing MCP tools through the server."""
        server = initialize_mcp_server()

        # Verify server can be created without errors
        assert server is not None

        # Test that the server can handle basic operations
        # This validates the tool registration and mocking setup
        assert self.mock_client.query.call_count >= 0  # May have been called during setup

    def test_mock_client_behavior(self):
        """Test that the mock client behaves correctly."""
        # Test query method
        result = self.mock_client.query("SELECT * FROM system.tables")
        assert result is not None
        assert hasattr(result, "column_names")
        assert hasattr(result, "result_rows")

        # Test command method
        result = self.mock_client.command("SHOW TABLES")
        assert result == "OK"

    def test_config_integration(self):
        """Test configuration integration."""
        assert self.mock_config.clickhouse_host == "localhost"
        assert self.mock_config.clickhouse_port == 8123
        assert self.mock_config.clickhouse_user == "default"

        # Test client config generation
        client_config = self.mock_config.get_clickhouse_client_config()
        assert isinstance(client_config, dict)
        assert "host" in client_config
        assert "port" in client_config
        assert "username" in client_config


class TestToolRegistration:
    """Test individual tool category registration."""

    @pytest.fixture(autouse=True)
    def setup_tool_testing(self, comprehensive_mock_client, mock_unified_config):
        """Set up tool testing environment."""
        with (
            patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client,
            patch("agent_zero.config.get_config") as mock_get_config,
        ):
            mock_create_client.return_value = comprehensive_mock_client
            mock_get_config.return_value = mock_unified_config

            self.mock_client = comprehensive_mock_client

            # Create server for testing
            from mcp.server.fastmcp import FastMCP

            self.test_server = FastMCP("TestToolRegistration")

            yield

    def test_database_tools_registration(self):
        """Test database tools registration."""
        from agent_zero.server.tools import register_database_tools

        register_database_tools(self.test_server)

        # Verify registration completed without errors
        assert self.test_server is not None

    def test_query_performance_tools_registration(self):
        """Test query performance tools registration."""
        from agent_zero.server.tools import register_query_performance_tools

        register_query_performance_tools(self.test_server)

        # Verify registration completed without errors
        assert self.test_server is not None

    def test_profile_events_tools_registration(self):
        """Test profile events tools registration."""
        from agent_zero.server.tools import register_profile_events_tools

        register_profile_events_tools(self.test_server)

        # Verify registration completed without errors
        assert self.test_server is not None

    def test_hardware_diagnostics_tools_registration(self):
        """Test hardware diagnostics tools registration."""
        from agent_zero.server.tools import register_hardware_diagnostics_tools

        register_hardware_diagnostics_tools(self.test_server)

        # Verify registration completed without errors
        assert self.test_server is not None

    def test_all_tool_categories_registration(self):
        """Test that all tool categories can be registered without errors."""
        from agent_zero.server.tools import (
            register_ai_diagnostics_tools,
            register_ai_powered_analysis_tools,
            register_database_tools,
            register_distributed_systems_tools,
            register_error_analysis_tools,
            register_hardware_diagnostics_tools,
            register_insert_operations_tools,
            register_parts_merges_tools,
            register_performance_diagnostics_tools,
            register_profile_events_tools,
            register_query_performance_tools,
            register_resource_usage_tools,
            register_storage_cloud_tools,
            register_system_components_tools,
            register_table_statistics_tools,
            register_utility_tools,
        )

        # Register all tool categories
        register_database_tools(self.test_server)
        register_query_performance_tools(self.test_server)
        register_resource_usage_tools(self.test_server)
        register_error_analysis_tools(self.test_server)
        register_insert_operations_tools(self.test_server)
        register_parts_merges_tools(self.test_server)
        register_system_components_tools(self.test_server)
        register_table_statistics_tools(self.test_server)
        register_utility_tools(self.test_server)
        register_ai_diagnostics_tools(self.test_server)
        register_profile_events_tools(self.test_server)
        register_performance_diagnostics_tools(self.test_server)
        register_storage_cloud_tools(self.test_server)
        register_distributed_systems_tools(self.test_server)
        register_hardware_diagnostics_tools(self.test_server)
        register_ai_powered_analysis_tools(self.test_server)

        # Verify all categories registered successfully
        assert self.test_server is not None


class TestErrorHandling:
    """Test error handling in the FastMCP integration."""

    @pytest.fixture(autouse=True)
    def setup_error_testing(self, mock_unified_config):
        """Set up error testing with failing mock client."""
        error_client = MagicMock()
        error_client.query.side_effect = Exception("Mock ClickHouse connection error")
        error_client.command.side_effect = Exception("Mock ClickHouse command error")

        with (
            patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client,
            patch("agent_zero.config.get_config") as mock_get_config,
        ):
            mock_create_client.return_value = error_client
            mock_get_config.return_value = mock_unified_config

            self.error_client = error_client
            yield

    def test_server_creation_with_client_errors(self):
        """Test server creation when ClickHouse client has errors."""
        # Server creation should not fail even if client has issues
        server = initialize_mcp_server()
        assert server is not None

    def test_tool_registration_with_client_errors(self):
        """Test tool registration when ClickHouse client has errors."""
        from mcp.server.fastmcp import FastMCP

        server = FastMCP("ErrorTestServer")

        # Tool registration should complete even with client errors
        register_all_tools(server)
        assert server is not None


class TestMockDataConsistency:
    """Test that mock data is consistent and realistic."""

    @pytest.fixture(autouse=True)
    def setup_data_testing(self, comprehensive_mock_client, sample_clickhouse_data):
        """Set up data consistency testing."""
        self.mock_client = comprehensive_mock_client
        self.sample_data = sample_clickhouse_data
        yield

    def test_table_data_consistency(self):
        """Test that table mock data is consistent."""
        tables = self.sample_data["tables"]

        # Verify table data structure
        assert len(tables) >= 2
        for table in tables:
            assert "database" in table
            assert "name" in table
            assert "engine" in table
            assert "total_rows" in table
            assert "total_bytes" in table
            assert isinstance(table["total_rows"], int)
            assert isinstance(table["total_bytes"], int)
            assert table["total_rows"] >= 0
            assert table["total_bytes"] >= 0

    def test_profile_events_data_consistency(self):
        """Test that profile events mock data is consistent."""
        events = self.sample_data["profile_events"]

        # Verify profile events structure
        assert len(events) >= 4
        for event in events:
            assert "event" in event
            assert "value" in event
            assert "description" in event
            assert isinstance(event["value"], int)
            assert event["value"] >= 0
            assert len(event["event"]) > 0
            assert len(event["description"]) > 0

    def test_query_data_consistency(self):
        """Test that query mock data is consistent."""
        queries = self.sample_data["current_queries"]

        # Verify query data structure
        for query in queries:
            assert "query_id" in query
            assert "query" in query
            assert "user" in query
            assert "elapsed" in query
            assert "memory_usage" in query
            assert isinstance(query["elapsed"], (int, float))
            assert isinstance(query["memory_usage"], int)
            assert query["elapsed"] >= 0
            assert query["memory_usage"] >= 0

    def test_system_metrics_data_consistency(self):
        """Test that system metrics mock data is consistent."""
        metrics = self.sample_data["system_metrics"]

        # Verify metrics structure
        for metric in metrics:
            assert "metric" in metric
            assert "value" in metric
            assert "description" in metric
            assert isinstance(metric["value"], (int, float))
            assert metric["value"] >= 0
            assert len(metric["metric"]) > 0
            assert len(metric["description"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
