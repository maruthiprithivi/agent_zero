# Python SDK Documentation

## Installation

```bash
pip install agent-zero-sdk
```

## Quick Start

```python
from agent_zero import AgentZeroClient

# Initialize the client
client = AgentZeroClient(
    host="localhost",
    port=8500,
    transport="sse"
)

# Connect to the server
await client.connect()

# Get available tools
tools = await client.list_tools()
print(f"Available tools: {len(tools)}")

# Execute a tool
result = await client.execute_tool(
    "get_clickhouse_metrics",
    {"metric_type": "performance"}
)

print(f"Metrics: {result}")
```

## Advanced Usage

### Performance Monitoring

```python
import asyncio
from agent_zero import AgentZeroClient

async def monitor_performance():
    client = AgentZeroClient()
    await client.connect()

    # Get performance metrics
    metrics = await client.execute_tool(
        "analyze_clickhouse_performance",
        {
            "time_range": "1h",
            "include_ai_insights": True
        }
    )

    # Display results
    for metric in metrics["data"]:
        print(f"{metric['name']}: {metric['value']}")

    # Get AI recommendations
    if "ai_insights" in metrics:
        for insight in metrics["ai_insights"]:
            print(f"💡 {insight['recommendation']}")

# Run the monitoring
asyncio.run(monitor_performance())
```

### Error Handling

```python
from agent_zero import AgentZeroClient, AgentZeroError

async def safe_execution():
    client = AgentZeroClient()

    try:
        await client.connect()
        result = await client.execute_tool("some_tool", {})
        return result

    except AgentZeroError as e:
        print(f"Agent Zero error: {e.message}")
        print(f"Error code: {e.code}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        await client.disconnect()
```

## API Reference

### AgentZeroClient

#### Methods

- `connect()` - Connect to the Agent Zero server
- `disconnect()` - Disconnect from the server
- `list_tools()` - Get list of available MCP tools
- `execute_tool(name, params)` - Execute a specific tool
- `get_health()` - Check server health status

#### Configuration

```python
client = AgentZeroClient(
    host="localhost",          # Server host
    port=8500,                # Server port
    transport="sse",          # Transport protocol (sse, websocket)
    timeout=30,               # Request timeout in seconds
    retry_attempts=3,         # Number of retry attempts
    api_key="your-api-key"    # Authentication key (if required)
)
```
