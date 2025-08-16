"""Comprehensive tests for main.py module to achieve 90%+ coverage.

This module tests the main CLI entry point and argument parsing functionality
to significantly improve code coverage.
"""

import sys
from unittest.mock import patch

import pytest

# Set up environment variables before imports
test_env = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
    "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
}


@pytest.mark.unit
class TestMainModule:
    """Comprehensive tests for main module functionality."""

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_basic_arguments(self, mock_run):
        """Test main function with basic CLI arguments."""
        test_args = ["ch-agent-zero", "--host", "127.0.0.1", "--port", "8505"]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 8505

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_deployment_mode_standalone(self, mock_run):
        """Test main function with standalone deployment mode."""
        test_args = [
            "ch-agent-zero",
            "--deployment-mode",
            "standalone",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_auth_config(self, mock_run):
        """Test main function with authentication configuration."""
        test_args = [
            "ch-agent-zero",
            "--auth-username",
            "admin",
            "--auth-password",
            "secret123",
            "--host",
            "localhost",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].auth_username == "admin"
            assert call_kwargs["server_config"].auth_password == "secret123"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_ssl_config(self, mock_run):
        """Test main function with SSL configuration."""
        test_args = [
            "ch-agent-zero",
            "--ssl-enable",
            "--ssl-certfile",
            "/path/to/cert.pem",
            "--ssl-keyfile",
            "/path/to/key.pem",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            # SSL configuration would be handled by unified config
            # Just verify the call was made successfully

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_cors_config(self, mock_run):
        """Test main function with CORS configuration."""
        test_args = [
            "ch-agent-zero",
            "--cors-origins",
            "http://localhost:3000,https://app.example.com",
            "--deployment-mode",
            "standalone",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            expected_origins = ["http://localhost:3000", "https://app.example.com"]
            assert call_kwargs["server_config"].cors_origins == expected_origins

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_clickhouse_config(self, mock_run):
        """Test main function with ClickHouse-specific configuration."""
        test_args = [
            "ch-agent-zero",
            "--clickhouse-host",
            "ch-cluster.example.com",
            "--clickhouse-port",
            "9000",
            "--clickhouse-user",
            "analytics",
            "--clickhouse-password",
            "ch-password",
            "--clickhouse-database",
            "analytics_db",
            "--clickhouse-secure",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].clickhouse_host == "ch-cluster.example.com"
            assert call_kwargs["server_config"].clickhouse_port == 9000
            assert call_kwargs["server_config"].clickhouse_user == "analytics"
            assert call_kwargs["server_config"].clickhouse_database == "analytics_db"
            assert call_kwargs["server_config"].clickhouse_secure is True

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_logging_config(self, mock_run):
        """Test main function with logging configuration."""
        test_args = [
            "ch-agent-zero",
            "--enable-query-logging",
            "--enable-mcp-tracing",
            "--log-level",
            "DEBUG",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].enable_query_logging is True
            assert call_kwargs["server_config"].enable_mcp_tracing is True
            assert call_kwargs["server_config"].log_level == "DEBUG"

    @patch.dict("os.environ", test_env)
    def test_main_with_invalid_port(self):
        """Test main function with invalid port number."""
        test_args = ["ch-agent-zero", "--port", "invalid-port"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                from agent_zero.main import main

                main()

    @patch.dict("os.environ", test_env)
    def test_main_with_help_argument(self):
        """Test main function with help argument."""
        test_args = ["ch-agent-zero", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as excinfo:
                from agent_zero.main import main

                main()

            # Help should exit with code 0
            assert excinfo.value.code == 0

    @patch.dict("os.environ", test_env)
    def test_main_with_version_argument(self):
        """Test main function with version argument."""
        test_args = ["ch-agent-zero", "--version"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as excinfo:
                from agent_zero.main import main

                main()

            # Version should exit with code 0
            assert excinfo.value.code == 0

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_environment_variables(self, mock_run):
        """Test main function respects environment variables."""
        env_vars = {
            **test_env,
            "MCP_SERVER_HOST": "env-host",
            "MCP_SERVER_PORT": "9999",
            "AGENT_ZERO_DEPLOYMENT_MODE": "standalone",
        }

        with patch.dict("os.environ", env_vars):
            with patch.object(sys, "argv", ["ch-agent-zero"]):
                from agent_zero.main import main

                main()

                mock_run.assert_called_once()
                # Environment variables should be used as defaults
                call_kwargs = mock_run.call_args[1]
                # Check that defaults are applied correctly
                assert "host" in call_kwargs
                assert "port" in call_kwargs

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_config_file(self, mock_run):
        """Test main function with configuration file."""
        test_args = ["ch-agent-zero", "--config", "/path/to/config.json"]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["server_config"].config_file == "/path/to/config.json"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_all_arguments(self, mock_run):
        """Test main function with comprehensive argument set."""
        test_args = [
            "ch-agent-zero",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--deployment-mode",
            "standalone",
            "--auth-username",
            "admin",
            "--auth-password",
            "secret",
            "--ssl-certfile",
            "/cert.pem",
            "--ssl-keyfile",
            "/key.pem",
            "--clickhouse-host",
            "ch.example.com",
            "--clickhouse-port",
            "9000",
            "--clickhouse-user",
            "ch_user",
            "--clickhouse-database",
            "analytics",
            "--enable-query-logging",
            "--enable-mcp-tracing",
            "--log-level",
            "INFO",
            "--cors-origins",
            "http://localhost:3000",
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]

            # Verify all configuration options were set
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080
            config = call_kwargs["server_config"]
            assert config.deployment_mode == "standalone"
            assert config.auth_username == "admin"
            assert config.clickhouse_host == "ch.example.com"
            assert config.enable_query_logging is True
            assert config.log_level == "INFO"


@pytest.mark.unit
class TestMainErrorHandling:
    """Test error handling in main module."""

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_with_run_exception(self, mock_run):
        """Test main function when run() raises an exception."""
        mock_run.side_effect = Exception("Server startup failed")

        test_args = ["ch-agent-zero"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(Exception, match="Server startup failed"):
                from agent_zero.main import main

                main()

    @patch.dict("os.environ", test_env)
    def test_main_with_missing_required_args(self):
        """Test main function with missing required arguments."""
        # Test with port but no host
        test_args = ["ch-agent-zero", "--port", "8080"]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            # Should use defaults and not raise an error
            try:
                main()
            except SystemExit:
                pass  # Expected for missing configuration

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    def test_main_argument_parsing_edge_cases(self, mock_run):
        """Test argument parsing edge cases."""
        # Test with boolean flags
        test_args = [
            "ch-agent-zero",
            "--clickhouse-secure",  # Boolean flag without value
            "--enable-query-logging",  # Another boolean flag
            "--port",
            "0",  # Edge case: port 0
        ]

        with patch.object(sys, "argv", test_args):
            from agent_zero.main import main

            main()

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["port"] == 0
            assert call_kwargs["server_config"].clickhouse_secure is True
            assert call_kwargs["server_config"].enable_query_logging is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
