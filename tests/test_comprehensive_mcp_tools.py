"""Test suite for comprehensive MCP tools integration.

This module tests all the new ProfileEvents analysis MCP tools to ensure they
work correctly and integrate with the existing system.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from agent_zero.server.tools import (
    register_ai_powered_analysis_tools,
    register_distributed_systems_tools,
    register_hardware_diagnostics_tools,
    register_performance_diagnostics_tools,
    register_profile_events_tools,
    register_storage_cloud_tools,
)


class TestComprehensiveMCPTools:
    """Test comprehensive MCP tools integration."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ClickHouse client."""
        client = Mock()
        client.query.return_value = Mock()
        client.command.return_value = []
        return client

    @pytest.fixture
    def mock_mcp(self):
        """Create a mock MCP instance."""
        mcp = Mock()
        mcp.tool = Mock(return_value=lambda func: func)
        return mcp

    @pytest.fixture
    def mock_profile_analyzer(self):
        """Create a mock ProfileEventsAnalyzer."""
        analyzer = Mock()
        analyzer.get_available_profile_events.return_value = [
            "Query",
            "SelectQuery",
            "MarkCacheHits",
            "MarkCacheMisses",
        ]
        analyzer.analyze_comprehensive.return_value = {
            "query_execution": {"total_queries": 100},
            "memory_allocation": {"total_allocations": 50},
        }
        return analyzer

    def test_profile_events_tools_registration(self, mock_mcp, mock_client):
        """Test ProfileEvents tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_profile_events_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 4  # Should have registered at least 4 tools

    def test_performance_diagnostics_tools_registration(self, mock_mcp, mock_client):
        """Test performance diagnostics tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_performance_diagnostics_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 4  # Should have registered at least 4 tools

    def test_storage_cloud_tools_registration(self, mock_mcp, mock_client):
        """Test storage & cloud tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_storage_cloud_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 4  # Should have registered at least 4 tools

    def test_distributed_systems_tools_registration(self, mock_mcp, mock_client):
        """Test distributed systems tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_distributed_systems_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 3  # Should have registered at least 3 tools

    def test_hardware_diagnostics_tools_registration(self, mock_mcp, mock_client):
        """Test hardware diagnostics tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_hardware_diagnostics_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 4  # Should have registered at least 4 tools

    def test_ai_powered_analysis_tools_registration(self, mock_mcp, mock_client):
        """Test AI-powered analysis tools registration."""
        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_ai_powered_analysis_tools(mock_mcp)

            # Verify that tools were registered
            assert mock_mcp.tool.call_count >= 4  # Should have registered at least 4 tools

    def test_analyze_profile_events_comprehensive(self, mock_client, mock_profile_analyzer):
        """Test comprehensive ProfileEvents analysis tool."""
        from agent_zero.server.tools import register_profile_events_tools

        mock_mcp = Mock()
        registered_functions = []

        def mock_tool_decorator():
            def decorator(func):
                registered_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        with (
            patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client),
            patch(
                "agent_zero.server.tools.ProfileEventsAnalyzer", return_value=mock_profile_analyzer
            ),
        ):
            register_profile_events_tools(mock_mcp)

            # Find the comprehensive analysis function
            comprehensive_func = None
            for func in registered_functions:
                if "comprehensive" in func.__name__:
                    comprehensive_func = func
                    break

            assert comprehensive_func is not None

            # Test the function
            result = comprehensive_func(24)
            assert isinstance(result, dict)
            mock_profile_analyzer.analyze_comprehensive.assert_called_once_with(24)

    @patch.dict(
        "os.environ",
        {
            "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
            "AGENT_ZERO_CLICKHOUSE_USER": "default",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
            "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
        },
    )
    def test_error_handling_in_tools(self, mock_client):
        """Test error handling in MCP tools."""
        from agent_zero.server.tools import register_profile_events_tools

        # Make the client raise an exception
        mock_client.query.side_effect = Exception("Test database error")

        mock_mcp = Mock()
        registered_functions = []

        def mock_tool_decorator():
            def decorator(func):
                registered_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_profile_events_tools(mock_mcp)

            # Test error handling
            for func in registered_functions:
                try:
                    # Some functions don't require parameters
                    result = func()
                except TypeError as e:
                    # Functions that require parameters - try with default values
                    if "missing" in str(e) and "positional argument" in str(e):
                        # Try with a simple string parameter for category-based functions
                        if "category" in str(e):
                            result = func("query_execution")
                        else:
                            # Skip functions that require other complex parameters
                            continue
                    else:
                        raise

                # Functions should return either strings with "Error" or dict with "error" key
                if isinstance(result, str):
                    assert "Error" in result
                elif isinstance(result, dict):
                    assert "error" in result
                else:
                    raise AssertionError(f"Unexpected result type: {type(result)}")

    def test_tool_parameter_validation(self, mock_client, mock_profile_analyzer):
        """Test tool parameter validation."""
        from agent_zero.server.tools import register_profile_events_tools

        mock_mcp = Mock()
        registered_functions = []

        def mock_tool_decorator():
            def decorator(func):
                registered_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        with (
            patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client),
            patch(
                "agent_zero.server.tools.ProfileEventsAnalyzer", return_value=mock_profile_analyzer
            ),
        ):
            register_profile_events_tools(mock_mcp)

            # Find a function that takes parameters
            category_func = None
            for func in registered_functions:
                if "category" in func.__name__:
                    category_func = func
                    break

            if category_func:
                # Test with invalid category
                result = category_func("invalid_category")
                assert isinstance(result, str)
                assert "Invalid category" in result

    def test_integration_with_existing_monitoring(self, mock_client):
        """Test that new tools integrate correctly with existing monitoring."""
        from agent_zero.server.tools import register_all_tools

        mock_mcp = Mock()
        mock_mcp.tool = Mock(return_value=lambda func: func)

        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            # This should not raise any exceptions
            register_all_tools(mock_mcp)

            # Verify that a significant number of tools were registered
            # (existing tools + new comprehensive tools)
            assert mock_mcp.tool.call_count >= 50

    def test_datetime_handling_in_tools(self, mock_client, mock_profile_analyzer):
        """Test datetime handling in tools."""
        from agent_zero.server.tools import register_performance_diagnostics_tools

        mock_mcp = Mock()
        registered_functions = []

        def mock_tool_decorator():
            def decorator(func):
                registered_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        # Mock the analyzers and their methods
        mock_query_analyzer = Mock()
        mock_query_analyzer.analyze_query_execution.return_value = {
            "analysis_period": {
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "hours": 24,
            },
            "function_performance": {},
            "memory_allocation": {},
            "primary_key_usage": {},
            "null_handling": {},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

        with (
            patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client),
            patch(
                "agent_zero.server.tools.ProfileEventsAnalyzer", return_value=mock_profile_analyzer
            ),
            patch(
                "agent_zero.server.tools.QueryExecutionAnalyzer", return_value=mock_query_analyzer
            ),
        ):
            register_performance_diagnostics_tools(mock_mcp)

            # Find and test query execution analysis
            query_func = None
            for func in registered_functions:
                if "query_execution" in func.__name__:
                    query_func = func
                    break

            if query_func:
                result = query_func(24)
                assert isinstance(result, dict)
                assert "analysis_timestamp" in result

    @pytest.mark.parametrize("hours", [1, 24, 168])
    def test_different_time_ranges(self, hours, mock_client, mock_profile_analyzer):
        """Test tools with different time ranges."""
        from agent_zero.server.tools import register_profile_events_tools

        mock_mcp = Mock()
        registered_functions = []

        def mock_tool_decorator():
            def decorator(func):
                registered_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        with (
            patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client),
            patch(
                "agent_zero.server.tools.ProfileEventsAnalyzer", return_value=mock_profile_analyzer
            ),
        ):
            register_profile_events_tools(mock_mcp)

            # Test with different hour values
            for func in registered_functions:
                try:
                    # Try to call with the hours parameter
                    result = func(hours)
                    # Should either return a dict or an error string
                    assert isinstance(result, (dict, str))
                except TypeError:
                    # Some functions might not take hours parameter
                    pass

    def test_tool_return_format_consistency(self, mock_client):
        """Test that all tools return consistent formats."""
        from agent_zero.server.tools import (
            register_performance_diagnostics_tools,
            register_profile_events_tools,
        )

        mock_mcp = Mock()
        all_functions = []

        def mock_tool_decorator():
            def decorator(func):
                all_functions.append(func)
                return func

            return decorator

        mock_mcp.tool = mock_tool_decorator

        with patch("agent_zero.server.tools.create_clickhouse_client", return_value=mock_client):
            register_profile_events_tools(mock_mcp)
            register_performance_diagnostics_tools(mock_mcp)

            # Test that all functions handle errors consistently
            mock_client.query.side_effect = Exception("Test error")

            for func in all_functions:
                try:
                    result = func()
                    # All error responses should be strings containing "Error"
                    if isinstance(result, str):
                        assert "Error" in result
                except TypeError:
                    # Some functions require parameters
                    pass


if __name__ == "__main__":
    pytest.main([__file__])
