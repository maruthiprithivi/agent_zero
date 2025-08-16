import os
from unittest.mock import Mock, patch

REQUIRED_ENV = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
}


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_health_check_tool_runs():
    # Build a minimal FastMCP-like registrar
    registered = {}

    class DummyMCP:
        def tool(self, **_):
            def dec(fn):
                registered[fn.__name__] = fn
                return fn

            return dec

    from agent_zero.server.tools import register_utility_tools

    mcp = DummyMCP()
    register_utility_tools(mcp)

    assert "health_check" in registered

    # Mock client and version
    with patch("agent_zero.server.client.create_clickhouse_client") as mk:
        cli = Mock()
        type(cli).server_version = Mock(return_value="24.1")
        mk.return_value = cli
        res = registered["health_check"]()
        assert res["status"] in ("ok", "degraded")


@patch.dict(
    os.environ, REQUIRED_ENV | {"AGENT_ZERO_ENABLE_STRUCTURED_TOOL_OUTPUT": "true"}, clear=False
)
def test_server_info_tool_runs():
    registered = {}

    class DummyMCP:
        def tool(self, **_):
            def dec(fn):
                registered[fn.__name__] = fn
                return fn

            return dec

    from agent_zero.server.tools import register_utility_tools

    mcp = DummyMCP()
    register_utility_tools(mcp)
    assert "server_info" in registered
    res = registered["server_info"]()
    assert res["name"] == "mcp-clickhouse"
    assert "features" in res
