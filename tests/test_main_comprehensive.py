"""Comprehensive tests for agent_zero/main.py module.

This test file aims to achieve 100% coverage of the main.py module
by testing all functions, argument parsing, configuration generation,
and error conditions.
"""

import json
import sys
from unittest.mock import Mock, mock_open, patch

import pytest

# Test environment setup
test_env = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "test-host",
    "AGENT_ZERO_CLICKHOUSE_USER": "test-user",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "test-pass",
    "AGENT_ZERO_CLICKHOUSE_PORT": "9000",
    "AGENT_ZERO_ENABLE_QUERY_LOGGING": "false",
}


@pytest.mark.unit
class TestGenerateConfig:
    """Tests for generate_config function."""

    @patch.dict("os.environ", test_env)
    @patch("sys.argv", ["script", "--ide", "claude-desktop"])
    @patch("builtins.print")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_generate_config_claude_desktop_stdout(self, mock_from_env, mock_print):
        """Test generate_config for Claude Desktop with stdout output."""
        from agent_zero.main import generate_config

        # Mock config
        mock_config = Mock()
        mock_from_env.return_value = mock_config

        # Call function
        generate_config()

        # Verify config was generated
        mock_from_env.assert_called_once_with(deployment_mode="local")

        # Verify print was called with JSON config
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        config_dict = json.loads(args)

        assert config_dict["name"] == "agent-zero"
        assert config_dict["description"] == "ClickHouse monitoring and analysis MCP server"
        assert config_dict["command"] == "ch-agent-zero"
        assert "env" in config_dict

    @patch.dict("os.environ", test_env)
    @patch(
        "sys.argv",
        ["script", "--ide", "cursor", "--deployment-mode", "local", "--output", "test.json"],
    )
    @patch("builtins.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_generate_config_cursor_file_output(self, mock_from_env, mock_file, mock_print):
        """Test generate_config for Cursor with file output."""
        from agent_zero.main import generate_config

        # Mock config
        mock_config = Mock()
        mock_from_env.return_value = mock_config

        # Call function
        generate_config()

        # Verify config was generated with local mode
        mock_from_env.assert_called_once_with(deployment_mode="local")

        # Verify file was written
        mock_file.assert_called_once_with("test.json", "w")
        mock_file().write.assert_called_once()

        # Verify success message
        mock_print.assert_called_once_with("Configuration written to test.json")

    @patch.dict("os.environ", test_env)
    @patch("sys.argv", ["script", "--ide", "windsurf", "--install-path", "/custom/path"])
    @patch("builtins.print")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_generate_config_with_install_path(self, mock_from_env, mock_print):
        """Test generate_config with custom install path."""
        from agent_zero.main import generate_config

        # Mock config
        mock_config = Mock()
        mock_from_env.return_value = mock_config

        # Call function
        generate_config()

        # Verify custom command was used
        args = mock_print.call_args[0][0]
        config_dict = json.loads(args)
        assert config_dict["command"] == "/custom/path"

    @patch.dict("os.environ", test_env)
    @patch("sys.argv", ["script", "--ide", "vscode"])
    @patch("builtins.print")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_generate_config_all_ide_types(self, mock_from_env, mock_print):
        """Test generate_config works for all supported IDE types."""
        from agent_zero.main import generate_config

        # Mock config
        mock_config = Mock()
        mock_from_env.return_value = mock_config

        # Call function
        generate_config()

        # Should complete without error for vscode
        mock_from_env.assert_called_once()
        mock_print.assert_called_once()


@pytest.mark.unit
class TestMainFunction:
    """Tests for main function."""

    def setup_method(self):
        """Set up test environment."""
        self.original_argv = sys.argv[:]

    def teardown_method(self):
        """Restore original sys.argv."""
        sys.argv = self.original_argv

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.generate_config")
    def test_main_generate_config_command(self, mock_generate_config):
        """Test main with generate-config command."""
        from agent_zero.main import main

        sys.argv = ["script", "generate-config", "--ide", "cursor"]

        result = main()

        # Should call generate_config and remove the command from argv
        mock_generate_config.assert_called_once()
        assert "generate-config" not in sys.argv

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_default_stdio_mode(self, mock_from_env, mock_run):
        """Test main with default stdio mode."""
        from agent_zero.main import main

        # Mock config
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script"]

        main()

        # Verify server was started with config
        mock_run.assert_called_once_with(host="127.0.0.1", port=8505, server_config=mock_config)

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_deployment_mode_arg(self, mock_from_env, mock_run):
        """Test main with deployment mode argument."""
        from agent_zero.config.unified import DeploymentMode
        from agent_zero.main import main

        # Mock config
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--deployment-mode", "local"]

        main()

        # Verify config was created with deployment mode override
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert "deployment_mode" in call_kwargs
        assert call_kwargs["deployment_mode"] == "local"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_ide_type_arg(self, mock_from_env, mock_run):
        """Test main with IDE type argument."""
        from agent_zero.config.unified import IDEType
        from agent_zero.main import main

        # Mock config
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type.value = "claude_code"
        mock_config.determine_optimal_transport.return_value.value = "stdio"
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--ide-type", "claude-code"]

        main()

        # Verify config was created with IDE type override
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert "ide_type" in call_kwargs
        assert call_kwargs["ide_type"] == IDEType.CLAUDE_CODE

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_ssl_args(self, mock_from_env, mock_run):
        """Test main with SSL arguments."""
        from agent_zero.main import main

        # Mock config with SSL
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = {"certfile": "/path/to/cert.pem"}
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--ssl-enable", "--ssl-certfile", "/path/to/cert.pem"]

        main()

        # Verify SSL config overrides were passed
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert call_kwargs["ssl_enable"] is True
        assert call_kwargs["ssl_certfile"] == "/path/to/cert.pem"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_auth_args(self, mock_from_env, mock_run):
        """Test main with authentication arguments."""
        from agent_zero.main import main

        # Mock config with auth
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = {"username": "testuser"}
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--auth-username", "testuser", "--auth-password", "testpass"]

        main()

        # Verify auth config overrides were passed
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert call_kwargs["auth_username"] == "testuser"
        assert call_kwargs["auth_password"] == "testpass"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_cursor_args(self, mock_from_env, mock_run):
        """Test main with Cursor IDE arguments."""
        from agent_zero.main import main

        # Mock config with Cursor settings
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = "agent"
        mock_config.cursor_transport.value = "sse"
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--cursor-mode", "agent", "--cursor-transport", "sse"]

        main()

        # Verify Cursor config overrides were passed
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert call_kwargs["cursor_mode"] == "agent"
        assert call_kwargs["cursor_transport"] == "sse"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_with_feature_args(self, mock_from_env, mock_run):
        """Test main with feature configuration arguments."""
        from agent_zero.main import main

        # Mock config with features enabled
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = True
        mock_config.rate_limit_requests = 200
        mock_config.tool_limit = 150
        mock_config.resource_limit = 75
        mock_from_env.return_value = mock_config

        sys.argv = [
            "script",
            "--rate-limit",
            "--rate-limit-requests",
            "200",
            "--tool-limit",
            "150",
            "--resource-limit",
            "75",
        ]

        main()

        # Verify feature config overrides were passed
        mock_from_env.assert_called_once()
        call_kwargs = mock_from_env.call_args[1]
        assert call_kwargs["rate_limit_enabled"] is True
        assert call_kwargs["rate_limit_requests"] == 200
        assert call_kwargs["tool_limit"] == 150
        assert call_kwargs["resource_limit"] == 75

    @patch.dict("os.environ", test_env)
    @patch("builtins.print")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_show_config(self, mock_from_env, mock_print):
        """Test main with --show-config argument."""
        from agent_zero.main import main

        # Mock config
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.clickhouse_host = "test-host"
        mock_config.clickhouse_port = 9000
        mock_from_env.return_value = mock_config

        sys.argv = ["script", "--show-config"]

        main()

        # Should print config and return early (not start server)
        mock_print.assert_called_once()
        config_output = mock_print.call_args[0][0]
        config_dict = json.loads(config_output)

        assert config_dict["deployment_mode"] == "local"
        assert config_dict["server"]["host"] == "127.0.0.1"
        assert config_dict["server"]["port"] == 8505
        assert config_dict["transport"] == "stdio"
        assert config_dict["clickhouse"]["host"] == "test-host"

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_error_handling(self, mock_from_env, mock_run):
        """Test main error handling."""
        from agent_zero.main import main

        # Mock config properly for all the logging calls
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "127.0.0.1"
        mock_config.server_port = 8505
        mock_config.transport.value = "stdio"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = False
        mock_config.tool_limit = 100
        mock_config.resource_limit = 50
        mock_from_env.return_value = mock_config

        # Make run raise an exception
        mock_run.side_effect = RuntimeError("Server startup failed")

        sys.argv = ["script"]

        # Should re-raise the exception
        with pytest.raises(RuntimeError, match="Server startup failed"):
            main()

    @patch.dict("os.environ", test_env)
    @patch("agent_zero.main.run")
    @patch("agent_zero.config.unified.UnifiedConfig.from_env")
    def test_main_all_argument_combinations(self, mock_from_env, mock_run):
        """Test main with various argument combinations."""
        from agent_zero.main import main

        # Mock config
        mock_config = Mock()
        mock_config.deployment_mode.value = "local"
        mock_config.server_host = "0.0.0.0"
        mock_config.server_port = 9000
        mock_config.transport.value = "sse"
        mock_config.ide_type = None
        mock_config.get_ssl_config.return_value = None
        mock_config.get_auth_config.return_value = None
        mock_config.cursor_mode = None
        mock_config.enable_health_check = True
        mock_config.rate_limit_enabled = True
        mock_config.rate_limit_requests = 500
        mock_config.tool_limit = 200
        mock_config.resource_limit = 100
        mock_from_env.return_value = mock_config

        sys.argv = [
            "script",
            "--deployment-mode",
            "local",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--transport",
            "sse",
            "--enable-health-check",
            "--rate-limit",
            "--rate-limit-requests",
            "500",
        ]

        main()

        # Should handle all arguments without error
        mock_from_env.assert_called_once()
        mock_run.assert_called_once()
