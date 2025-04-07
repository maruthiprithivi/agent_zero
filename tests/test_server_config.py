"""Tests for server_config.py."""

import os
import tempfile
from unittest.mock import patch


from agent_zero.server_config import ServerConfig


class TestServerConfig:
    """Tests for the ServerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8505
        assert config.ssl_certfile is None
        assert config.ssl_keyfile is None
        assert config.auth_username is None
        assert config.auth_password is None
        assert config.auth_password_file is None

    def test_environment_variables(self):
        """Test that environment variables are read correctly."""
        with patch.dict(
            os.environ,
            {
                "MCP_SERVER_HOST": "0.0.0.0",
                "MCP_SERVER_PORT": "9000",
                "MCP_SSL_CERTFILE": "/path/to/cert.pem",
                "MCP_SSL_KEYFILE": "/path/to/key.pem",
                "MCP_AUTH_USERNAME": "testuser",
                "MCP_AUTH_PASSWORD": "testpass",
                "MCP_AUTH_PASSWORD_FILE": "/path/to/password.txt",
            },
        ):
            config = ServerConfig()
            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.ssl_certfile == "/path/to/cert.pem"
            assert config.ssl_keyfile == "/path/to/key.pem"
            assert config.auth_username == "testuser"
            assert config.auth_password == "testpass"
            assert config.auth_password_file == "/path/to/password.txt"

    def test_override_values(self):
        """Test that override values take precedence over environment variables."""
        with patch.dict(os.environ, {"MCP_SERVER_HOST": "0.0.0.0", "MCP_SERVER_PORT": "9000"}):
            config = ServerConfig(host="localhost", port="8080")
            assert config.host == "localhost"
            assert config.port == 8080

    def test_port_type_conversion(self):
        """Test that port is converted to int."""
        config = ServerConfig(port="8080")
        assert isinstance(config.port, int)
        assert config.port == 8080

    def test_get_ssl_config_none(self):
        """Test that get_ssl_config returns None when SSL is not configured."""
        config = ServerConfig()
        assert config.get_ssl_config() is None

    def test_get_ssl_config_partial(self):
        """Test that get_ssl_config returns None when only one SSL file is configured."""
        config = ServerConfig(ssl_certfile="/path/to/cert.pem")
        assert config.get_ssl_config() is None

    def test_get_ssl_config_complete(self):
        """Test that get_ssl_config returns a dict when SSL is fully configured."""
        config = ServerConfig(ssl_certfile="/path/to/cert.pem", ssl_keyfile="/path/to/key.pem")
        ssl_config = config.get_ssl_config()
        assert ssl_config is not None
        assert ssl_config["certfile"] == "/path/to/cert.pem"
        assert ssl_config["keyfile"] == "/path/to/key.pem"

    def test_get_auth_config_none(self):
        """Test that get_auth_config returns None when auth is not configured."""
        config = ServerConfig()
        assert config.get_auth_config() is None

    def test_get_auth_config_username_only(self):
        """Test that get_auth_config returns None when only username is configured."""
        config = ServerConfig(auth_username="testuser")
        assert config.get_auth_config() is None

    def test_get_auth_config_complete(self):
        """Test that get_auth_config returns a dict when auth is fully configured."""
        config = ServerConfig(auth_username="testuser", auth_password="testpass")
        auth_config = config.get_auth_config()
        assert auth_config is not None
        assert auth_config["username"] == "testuser"
        assert auth_config["password"] == "testpass"

    def test_auth_config_with_password_file(self):
        """Test that get_auth_config reads password from file when configured."""
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("file_password")
            temp_file.flush()

            config = ServerConfig(auth_username="testuser", auth_password_file=temp_file.name)
            auth_config = config.get_auth_config()
            assert auth_config is not None
            assert auth_config["username"] == "testuser"
            assert auth_config["password"] == "file_password"

    def test_auth_config_password_precedence(self):
        """Test that auth_password takes precedence over auth_password_file."""
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("file_password")
            temp_file.flush()

            config = ServerConfig(
                auth_username="testuser",
                auth_password="direct_password",
                auth_password_file=temp_file.name,
            )
            auth_config = config.get_auth_config()
            assert auth_config is not None
            assert auth_config["username"] == "testuser"
            assert auth_config["password"] == "direct_password"

    def test_auth_config_file_not_found(self):
        """Test that get_auth_config returns None when password file is not found."""
        config = ServerConfig(auth_username="testuser", auth_password_file="/nonexistent/file.txt")
        assert config.get_auth_config() is None
