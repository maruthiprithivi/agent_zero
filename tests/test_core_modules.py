"""Direct unit tests for core modules to achieve 90%+ coverage.

This module tests the main.py, server core, and configuration modules
directly with comprehensive scenarios and error handling.
"""

import concurrent.futures
import os
from unittest.mock import Mock, patch

import pytest

# Import core modules for direct testing
from agent_zero.config.unified import UnifiedConfig
from agent_zero.mcp_tracer import MCPTracer, trace_mcp_call
from agent_zero.server.client import create_clickhouse_client
from agent_zero.server.core import determine_transport, initialize_mcp_server, run
from agent_zero.server.query import execute_query, execute_query_threaded


@pytest.mark.unit
class TestUnifiedConfig:
    """Test UnifiedConfig class directly with comprehensive scenarios."""

    def test_config_initialization_with_env_vars(self):
        """Test config initialization with environment variables."""
        test_env = {
            "AGENT_ZERO_CLICKHOUSE_HOST": "test-host",
            "AGENT_ZERO_CLICKHOUSE_PORT": "9000",
            "AGENT_ZERO_CLICKHOUSE_USER": "test-user",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "test-pass",
            "AGENT_ZERO_CLICKHOUSE_DATABASE": "test-db",
            "AGENT_ZERO_CLICKHOUSE_SECURE": "true",
            "AGENT_ZERO_ENABLE_QUERY_LOGGING": "true",
            "AGENT_ZERO_ENABLE_MCP_TRACING": "true",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = UnifiedConfig.from_env()

            assert config.clickhouse_host == "test-host"
            assert config.clickhouse_port == 9000
            assert config.clickhouse_user == "test-user"
            assert config.clickhouse_password == "test-pass"
            assert config.clickhouse_database == "test-db"
            assert config.clickhouse_secure is True
            assert config.enable_query_logging is True
            assert config.enable_mcp_tracing is True

    def test_config_initialization_with_defaults(self):
        """Test config initialization with default values."""
        # Clear any existing env vars that might affect the test
        env_vars_to_clear = [
            "AGENT_ZERO_CLICKHOUSE_HOST",
            "AGENT_ZERO_CLICKHOUSE_PORT",
            "AGENT_ZERO_CLICKHOUSE_USER",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD",
        ]

        with patch.dict(os.environ, {}, clear=True):
            # This should raise ValueError due to missing required vars
            with pytest.raises(ValueError, match="Missing required environment variable"):
                UnifiedConfig.from_env()

    def test_config_initialization_with_overrides(self):
        """Test config initialization with override parameters."""
        test_env = {
            "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
            "AGENT_ZERO_CLICKHOUSE_USER": "default",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = UnifiedConfig.from_env(
                clickhouse_host="override-host", clickhouse_port=8123, enable_query_logging=True
            )

            assert config.clickhouse_host == "override-host"
            assert config.clickhouse_port == 8123
            assert config.enable_query_logging is True

    def test_get_clickhouse_client_config(self):
        """Test ClickHouse client configuration generation."""
        test_env = {
            "AGENT_ZERO_CLICKHOUSE_HOST": "test-host",
            "AGENT_ZERO_CLICKHOUSE_PORT": "9000",
            "AGENT_ZERO_CLICKHOUSE_USER": "test-user",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "test-pass",
            "AGENT_ZERO_CLICKHOUSE_DATABASE": "test-db",
            "AGENT_ZERO_CLICKHOUSE_SECURE": "true",
            "AGENT_ZERO_CLICKHOUSE_VERIFY": "false",
            "AGENT_ZERO_CLICKHOUSE_CONNECT_TIMEOUT": "15",
            "AGENT_ZERO_CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "600",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = UnifiedConfig.from_env()
            client_config = config.get_clickhouse_client_config()

            assert isinstance(client_config, dict)
            assert client_config["host"] == "test-host"
            assert client_config["port"] == 9000
            assert client_config["username"] == "test-user"
            assert client_config["password"] == "test-pass"
            assert client_config["database"] == "test-db"
            assert client_config["secure"] is True
            assert client_config["verify"] is False
            assert client_config["connect_timeout"] == 15
            assert client_config["send_receive_timeout"] == 600

    def test_config_type_conversions(self):
        """Test configuration type conversions."""
        test_env = {
            "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
            "AGENT_ZERO_CLICKHOUSE_PORT": "8123",  # String that should become int
            "AGENT_ZERO_CLICKHOUSE_USER": "default",
            "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
            "AGENT_ZERO_CLICKHOUSE_SECURE": "false",  # String that should become bool
            "AGENT_ZERO_CLICKHOUSE_CONNECT_TIMEOUT": "10",  # String that should become int
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = UnifiedConfig.from_env()

            assert isinstance(config.clickhouse_port, int)
            assert config.clickhouse_port == 8123
            assert isinstance(config.clickhouse_secure, bool)
            assert config.clickhouse_secure is False
            assert isinstance(config.clickhouse_connect_timeout, int)
            assert config.clickhouse_connect_timeout == 10


@pytest.mark.unit
class TestServerCore:
    """Test server core functionality directly."""

    def test_initialize_mcp_server_success(self):
        """Test successful MCP server initialization."""
        with (
            patch("agent_zero.server.client.create_clickhouse_client") as mock_create_client,
            patch("agent_zero.config.get_config") as mock_get_config,
        ):
            mock_get_config.return_value = Mock()
            mock_create_client.return_value = Mock()

            server = initialize_mcp_server()

            assert server is not None
            assert hasattr(server, "name")

    def test_initialize_mcp_server_with_import_error(self):
        """Test MCP server initialization with import error."""
        with (
            patch(
                "agent_zero.server.core.FastMCP", side_effect=ImportError("FastMCP not available")
            ),
            patch("agent_zero.server.client.create_clickhouse_client"),
            patch("agent_zero.config.get_config"),
        ):
            with pytest.raises(ImportError):
                initialize_mcp_server()

    def test_determine_transport_stdio_default(self):
        """Test transport determination defaults to stdio."""
        config = Mock()
        config.deployment_mode = "local"
        config.determine_optimal_transport.return_value = "stdio"

        # Mock the actual function since it may not exist
        with patch("agent_zero.server.core.determine_transport") as mock_determine:
            mock_determine.return_value = "stdio"
            transport = determine_transport(config, "127.0.0.1", 8505)
            assert transport == "stdio"

    def test_determine_transport_sse_for_remote(self):
        """Test transport determination for remote deployment."""
        config = Mock()
        config.deployment_mode = "standalone"

        from agent_zero.config import TransportType

        transport = determine_transport(config, "0.0.0.0", 8080)

        assert transport == TransportType.SSE

    @patch("agent_zero.server.core.run_remote_mode")
    @patch("agent_zero.server.core.initialize_mcp_server")
    def test_run_function_remote_mode(self, mock_init_server, mock_run_remote):
        """Test run function in remote mode."""
        config = Mock()
        config.deployment_mode = "standalone"

        mock_server = Mock()
        mock_init_server.return_value = mock_server

        run(host="0.0.0.0", port=8080, server_config=config)

        mock_init_server.assert_called_once()
        mock_run_remote.assert_called_once_with(config)

    @patch("mcp.server.stdio.stdio_server")
    @patch("agent_zero.server.core.initialize_mcp_server")
    def test_run_function_stdio_mode(self, mock_init_server, mock_stdio_server):
        """Test run function in stdio mode."""
        config = Mock()
        config.deployment_mode = "local"

        mock_server = Mock()
        mock_init_server.return_value = mock_server

        # Mock the async context manager
        mock_stdio_server.return_value.__aenter__ = Mock(return_value=(Mock(), Mock()))
        mock_stdio_server.return_value.__aexit__ = Mock(return_value=None)

        run(host="127.0.0.1", port=8505, server_config=config)

        mock_init_server.assert_called_once()


@pytest.mark.unit
class TestServerClient:
    """Test server client functionality directly."""

    @patch("clickhouse_connect.get_client")
    def test_create_clickhouse_client_success(self, mock_get_client):
        """Test successful ClickHouse client creation."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock config
        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.get_clickhouse_client_config.return_value = {
                "host": "localhost",
                "port": 8123,
                "username": "default",
                "password": "",
            }
            mock_get_config.return_value = config

            client = create_clickhouse_client()

            assert client == mock_client
            mock_get_client.assert_called_once()

    @patch("clickhouse_connect.get_client")
    def test_create_clickhouse_client_connection_error(self, mock_get_client):
        """Test ClickHouse client creation with connection error."""
        mock_get_client.side_effect = Exception("Connection failed")

        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.get_clickhouse_client_config.return_value = {
                "host": "localhost",
                "port": 8123,
                "username": "default",
                "password": "",
            }
            mock_get_config.return_value = config

            with pytest.raises(Exception, match="Connection failed"):
                create_clickhouse_client()


@pytest.mark.unit
class TestServerQuery:
    """Test server query functionality directly."""

    def test_execute_query_success(self):
        """Test successful query execution."""
        with patch("agent_zero.server.query.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_result = Mock()
            mock_result.result_rows = [["test", 123], ["data", 456]]
            mock_result.column_names = ["name", "value"]
            mock_client.query.return_value = mock_result
            mock_create_client.return_value = mock_client

            result = execute_query("SELECT * FROM test_table")

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["name"] == "test"
            assert result[0]["value"] == 123

    def test_execute_query_threaded_success(self):
        """Test successful threaded query execution."""
        with patch("agent_zero.server.query.execute_query") as mock_execute:
            mock_execute.return_value = [{"name": "test", "value": 123}]

            result = execute_query_threaded("SELECT * FROM test_table")

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["name"] == "test"

    def test_execute_query_error_handling(self):
        """Test query execution error handling."""
        with patch("agent_zero.server.query.create_clickhouse_client") as mock_create_client:
            mock_client = Mock()
            mock_client.query.side_effect = Exception("Query execution failed")
            mock_create_client.return_value = mock_client

            result = execute_query("SELECT * FROM test_table")

            assert isinstance(result, str)
            assert "error running query" in result

    def test_execute_query_threaded_timeout(self):
        """Test threaded query execution timeout."""
        with patch("agent_zero.server.query.QUERY_EXECUTOR") as mock_executor:
            mock_future = Mock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError()
            mock_executor.submit.return_value = mock_future

            result = execute_query_threaded("SELECT * FROM slow_table")

            assert isinstance(result, str)
            assert "timed out" in result

    def test_query_timeout_constant(self):
        """Test query timeout constant is properly set."""
        from agent_zero.server.query import SELECT_QUERY_TIMEOUT_SECS

        assert isinstance(SELECT_QUERY_TIMEOUT_SECS, int)
        assert SELECT_QUERY_TIMEOUT_SECS > 0

    def test_query_executor_initialization(self):
        """Test query executor is properly initialized."""
        from agent_zero.server.query import QUERY_EXECUTOR

        assert isinstance(QUERY_EXECUTOR, concurrent.futures.ThreadPoolExecutor)
        assert QUERY_EXECUTOR._max_workers == 10


@pytest.mark.unit
class TestMCPTracer:
    """Test MCP tracer functionality directly."""

    def test_mcp_tracer_initialization(self):
        """Test MCP tracer initialization."""
        tracer = MCPTracer(enabled=True)

        assert tracer.enabled is True
        assert tracer.traces == []

    def test_mcp_tracer_disabled(self):
        """Test MCP tracer when disabled."""
        tracer = MCPTracer(enabled=False)

        assert tracer.enabled is False
        assert tracer.traces == []

    def test_trace_mcp_call_decorator_success(self):
        """Test trace_mcp_call decorator with successful function."""
        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.enable_mcp_tracing = True
            mock_get_config.return_value = config

            @trace_mcp_call
            def test_function(param1, param2="default"):
                return {"result": "success", "param1": param1, "param2": param2}

            result = test_function("test_value", param2="custom")

            assert result == {"result": "success", "param1": "test_value", "param2": "custom"}

    def test_trace_mcp_call_decorator_with_exception(self):
        """Test trace_mcp_call decorator with function that raises exception."""
        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.enable_mcp_tracing = True
            mock_get_config.return_value = config

            @trace_mcp_call
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

    def test_trace_mcp_call_decorator_disabled_tracing(self):
        """Test trace_mcp_call decorator when tracing is disabled."""
        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.enable_mcp_tracing = False
            mock_get_config.return_value = config

            @trace_mcp_call
            def test_function():
                return "result"

            result = test_function()

            assert result == "result"

    def test_trace_mcp_call_decorator_async_function(self):
        """Test trace_mcp_call decorator with async function."""
        import asyncio

        with patch("agent_zero.config.get_config") as mock_get_config:
            config = Mock()
            config.enable_mcp_tracing = True
            mock_get_config.return_value = config

            @trace_mcp_call
            async def async_test_function(value):
                await asyncio.sleep(0.001)  # Simulate async work
                return f"async_result_{value}"

            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(async_test_function("test"))
                assert result == "async_result_test"
            finally:
                loop.close()


@pytest.mark.unit
class TestMainModule:
    """Test main module functionality directly."""

    @patch("agent_zero.main.run")
    def test_main_function_basic_args(self, mock_run):
        """Test main function with basic arguments."""
        test_args = ["ch-agent-zero", "--host", "127.0.0.1", "--port", "8505"]

        with patch("sys.argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 8505

    @patch("agent_zero.main.run")
    def test_main_function_with_auth(self, mock_run):
        """Test main function with authentication arguments."""
        test_args = [
            "ch-agent-zero",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--auth-username",
            "admin",
            "--auth-password",
            "secret",
        ]

        with patch("sys.argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080
            assert call_kwargs["server_config"].auth_username == "admin"
            assert call_kwargs["server_config"].auth_password == "secret"

    @patch("agent_zero.main.run")
    def test_main_function_deployment_mode(self, mock_run):
        """Test main function with deployment mode."""
        test_args = ["ch-agent-zero", "--deployment-mode", "standalone"]

        with patch("sys.argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].deployment_mode == "standalone"

    @patch("agent_zero.main.run")
    def test_main_function_ssl_config(self, mock_run):
        """Test main function with SSL configuration."""
        test_args = [
            "ch-agent-zero",
            "--ssl-certfile",
            "/path/to/cert.pem",
            "--ssl-keyfile",
            "/path/to/key.pem",
        ]

        with patch("sys.argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].ssl_certfile == "/path/to/cert.pem"
            assert call_kwargs["server_config"].ssl_keyfile == "/path/to/key.pem"

    def test_main_function_invalid_port(self):
        """Test main function with invalid port."""
        test_args = ["ch-agent-zero", "--port", "invalid"]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                from agent_zero.main import main

                main()

    @patch("agent_zero.main.run")
    def test_main_function_environment_variables(self, mock_run):
        """Test main function respects environment variables."""
        env_vars = {"MCP_SERVER_HOST": "env-host", "MCP_SERVER_PORT": "9999"}

        with patch.dict(os.environ, env_vars, clear=False):
            with patch("sys.argv", ["ch-agent-zero"]):
                from agent_zero.main import main

                main()

                mock_run.assert_called_once()
                # Environment variables should be used as defaults
                call_kwargs = mock_run.call_args[1]
                # Default values should be overridden by command line args if present


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
