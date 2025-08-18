import asyncio
import os
from unittest.mock import Mock, patch

REQUIRED_ENV = {
    "AGENT_ZERO_CLICKHOUSE_HOST": "localhost",
    "AGENT_ZERO_CLICKHOUSE_USER": "default",
    "AGENT_ZERO_CLICKHOUSE_PASSWORD": "",
}


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_trace_mcp_call_sync_enabled():
    from agent_zero.mcp_tracer import trace_mcp_call

    with patch("agent_zero.config.get_config") as mk:
        cfg = Mock()
        cfg.enable_mcp_tracing = True
        mk.return_value = cfg

        @trace_mcp_call
        def f(x):
            return x + 1

        assert f(1) == 2


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_trace_mcp_call_async_enabled():
    from agent_zero.mcp_tracer import trace_mcp_call

    with patch("agent_zero.config.get_config") as mk:
        cfg = Mock()
        cfg.enable_mcp_tracing = True
        mk.return_value = cfg

        @trace_mcp_call
        async def g(x):
            await asyncio.sleep(0)
            return x * 2

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            assert loop.run_until_complete(g(3)) == 6
        finally:
            loop.close()


@patch.dict(os.environ, REQUIRED_ENV, clear=False)
def test_trace_mcp_call_sync_disabled():
    from agent_zero.mcp_tracer import trace_mcp_call

    with patch("agent_zero.config.get_config") as mk:
        cfg = Mock()
        cfg.enable_mcp_tracing = False
        mk.return_value = cfg

        @trace_mcp_call
        def f(x):
            return x + 1

        assert f(1) == 2
