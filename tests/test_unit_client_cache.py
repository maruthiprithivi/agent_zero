import os
from unittest.mock import Mock, patch

REQUIRED_ENV = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
}


@patch.dict(os.environ, REQUIRED_ENV | {"AGENT_ZERO_ENABLE_CLIENT_CACHE": "true"}, clear=False)
@patch("clickhouse_connect.get_client")
def test_client_cache_reuse(mock_get_client):
    from agent_zero.server.client import create_clickhouse_client

    cli = Mock()
    type(cli).server_version = Mock(return_value="24.1")
    mock_get_client.return_value = cli

    c1 = create_clickhouse_client()
    c2 = create_clickhouse_client()
    assert c1 is c2


@patch.dict(os.environ, REQUIRED_ENV | {"AGENT_ZERO_ENABLE_CLIENT_CACHE": "false"}, clear=False)
@patch("clickhouse_connect.get_client")
def test_client_cache_disabled(mock_get_client):
    from agent_zero.server.client import create_clickhouse_client

    cli = Mock()
    type(cli).server_version = Mock(return_value="24.1")
    mock_get_client.return_value = cli

    c1 = create_clickhouse_client()
    c2 = create_clickhouse_client()
    assert c1 is not c2
