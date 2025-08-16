import os
from unittest.mock import Mock, patch

# Ensure required env for UnifiedConfig
REQUIRED_ENV = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
}


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_determine_transport_defaults_stdio():
    from agent_zero.config import TransportType, UnifiedConfig
    from agent_zero.server.core import determine_transport

    cfg = UnifiedConfig.from_env()
    t = determine_transport(cfg, host="127.0.0.1", port=8505)
    # For UnifiedConfig, return type is TransportType
    assert isinstance(t, TransportType)
    assert t == TransportType.STDIO


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_determine_transport_remote_sse():
    from agent_zero.config import DeploymentMode, TransportType, UnifiedConfig
    from agent_zero.server.core import determine_transport

    cfg = UnifiedConfig.from_env(deployment_mode=DeploymentMode.REMOTE)
    t = determine_transport(cfg, host="0.0.0.0", port=9000)
    assert isinstance(t, TransportType)
    assert t == TransportType.SSE


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_run_stdio_calls_mcp_run_without_transport():
    from agent_zero.config import UnifiedConfig
    from agent_zero.server.core import run

    # Mock FastMCP instance returned by initialize_mcp_server
    mock_mcp = Mock()

    with patch("agent_zero.server.core.initialize_mcp_server", return_value=mock_mcp):
        cfg = UnifiedConfig.from_env()  # default local/stdio
        run(host="127.0.0.1", port=8505, server_config=cfg)
        # Should be called without 'transport' when stdio
        assert mock_mcp.run.called
        kwargs = mock_mcp.run.call_args.kwargs
        assert "transport" not in kwargs
