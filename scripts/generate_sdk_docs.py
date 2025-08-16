#!/usr/bin/env python3
"""
Generate SDK documentation and code examples.

This script generates SDK documentation, code examples, and usage guides
for different programming languages.
"""

from pathlib import Path


def generate_python_sdk_docs() -> str:
    """Generate Python SDK documentation."""
    return """# Python SDK Documentation

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
"""


def generate_javascript_sdk_docs() -> str:
    """Generate JavaScript SDK documentation."""
    return """# JavaScript SDK Documentation

## Installation

```bash
npm install agent-zero-sdk
```

## Quick Start

```javascript
import { AgentZeroClient } from 'agent-zero-sdk';

// Initialize the client
const client = new AgentZeroClient({
    host: 'localhost',
    port: 8500,
    transport: 'sse'
});

// Connect to the server
await client.connect();

// Get available tools
const tools = await client.listTools();
console.log(`Available tools: ${tools.length}`);

// Execute a tool
const result = await client.executeTool(
    'get_clickhouse_metrics',
    { metric_type: 'performance' }
);

console.log('Metrics:', result);
```

## Advanced Usage

### Performance Monitoring

```javascript
import { AgentZeroClient } from 'agent-zero-sdk';

async function monitorPerformance() {
    const client = new AgentZeroClient();
    await client.connect();

    try {
        // Get performance metrics
        const metrics = await client.executeTool(
            'analyze_clickhouse_performance',
            {
                time_range: '1h',
                include_ai_insights: true
            }
        );

        // Display results
        metrics.data.forEach(metric => {
            console.log(`${metric.name}: ${metric.value}`);
        });

        // Show AI recommendations
        if (metrics.ai_insights) {
            metrics.ai_insights.forEach(insight => {
                console.log(`💡 ${insight.recommendation}`);
            });
        }

    } finally {
        await client.disconnect();
    }
}

// Run monitoring
monitorPerformance().catch(console.error);
```

### Error Handling

```javascript
import { AgentZeroClient, AgentZeroError } from 'agent-zero-sdk';

async function safeExecution() {
    const client = new AgentZeroClient();

    try {
        await client.connect();
        const result = await client.executeTool('some_tool', {});
        return result;

    } catch (error) {
        if (error instanceof AgentZeroError) {
            console.error(`Agent Zero error: ${error.message}`);
            console.error(`Error code: ${error.code}`);
        } else {
            console.error(`Unexpected error: ${error.message}`);
        }

    } finally {
        await client.disconnect();
    }
}
```

## API Reference

### AgentZeroClient

#### Constructor

```javascript
new AgentZeroClient({
    host: 'localhost',        // Server host
    port: 8500,              // Server port
    transport: 'sse',        // Transport protocol
    timeout: 30000,          // Request timeout in milliseconds
    retryAttempts: 3,        // Number of retry attempts
    apiKey: 'your-api-key'   // Authentication key
});
```

#### Methods

- `connect()` - Connect to the Agent Zero server
- `disconnect()` - Disconnect from the server
- `listTools()` - Get list of available MCP tools
- `executeTool(name, params)` - Execute a specific tool
- `getHealth()` - Check server health status

#### Events

```javascript
client.on('connected', () => {
    console.log('Connected to Agent Zero server');
});

client.on('disconnected', () => {
    console.log('Disconnected from server');
});

client.on('error', (error) => {
    console.error('Client error:', error);
});
```
"""


def generate_examples() -> dict[str, str]:
    """Generate code examples for different scenarios."""
    return {
        "basic_usage.py": """#!/usr/bin/env python3
\"\"\"Basic usage example for Agent Zero.\"\"\"

import asyncio
from agent_zero import AgentZeroClient

async def main():
    # Create client
    client = AgentZeroClient()

    try:
        # Connect
        await client.connect()
        print("✅ Connected to Agent Zero")

        # List available tools
        tools = await client.list_tools()
        print(f"📋 Available tools: {len(tools)}")

        for tool in tools[:5]:  # Show first 5 tools
            print(f"  • {tool['name']}: {tool['description']}")

        # Get basic metrics
        metrics = await client.execute_tool(
            "get_clickhouse_metrics",
            {"metric_type": "basic"}
        )

        print(f"📊 Current metrics: {metrics}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        print("👋 Disconnected")

if __name__ == "__main__":
    asyncio.run(main())
""",
        "performance_analysis.py": """#!/usr/bin/env python3
\"\"\"Performance analysis example.\"\"\"

import asyncio
from agent_zero import AgentZeroClient

async def analyze_performance():
    client = AgentZeroClient()
    await client.connect()

    try:
        # Get comprehensive performance analysis
        analysis = await client.execute_tool(
            "analyze_clickhouse_performance",
            {
                "time_range": "24h",
                "include_ai_insights": True,
                "detail_level": "comprehensive"
            }
        )

        print("🔍 Performance Analysis Results")
        print("=" * 40)

        # Show key metrics
        if "summary" in analysis:
            summary = analysis["summary"]
            print(f"Query Performance: {summary.get('query_performance', 'N/A')}")
            print(f"Memory Usage: {summary.get('memory_usage', 'N/A')}")
            print(f"Disk I/O: {summary.get('disk_io', 'N/A')}")

        # Show AI insights
        if "ai_insights" in analysis:
            print("\\n🤖 AI Insights:")
            for insight in analysis["ai_insights"]:
                print(f"  💡 {insight['title']}")
                print(f"     {insight['description']}")
                if insight.get('severity') == 'high':
                    print("     ⚠️  High Priority")

        # Show recommendations
        if "recommendations" in analysis:
            print("\\n📋 Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"  ✅ {rec['action']}")
                print(f"     Impact: {rec.get('impact', 'Unknown')}")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(analyze_performance())
""",
        "basic_usage.js": """// Basic usage example for Agent Zero (JavaScript)

import { AgentZeroClient } from 'agent-zero-sdk';

async function main() {
    const client = new AgentZeroClient();

    try {
        // Connect
        await client.connect();
        console.log('✅ Connected to Agent Zero');

        // List available tools
        const tools = await client.listTools();
        console.log(`📋 Available tools: ${tools.length}`);

        tools.slice(0, 5).forEach(tool => {
            console.log(`  • ${tool.name}: ${tool.description}`);
        });

        // Get basic metrics
        const metrics = await client.executeTool(
            'get_clickhouse_metrics',
            { metric_type: 'basic' }
        );

        console.log('📊 Current metrics:', metrics);

    } catch (error) {
        console.error('❌ Error:', error.message);
    } finally {
        await client.disconnect();
        console.log('👋 Disconnected');
    }
}

main().catch(console.error);
""",
    }


def main():
    """Main function to generate SDK documentation."""
    print("Generating SDK documentation...")

    # Create output directory
    output_dir = Path("docs/sdk/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Python SDK docs
    python_docs = generate_python_sdk_docs()
    with open(output_dir / "python.md", "w") as f:
        f.write(python_docs)

    # Generate JavaScript SDK docs
    js_docs = generate_javascript_sdk_docs()
    with open(output_dir / "javascript.md", "w") as f:
        f.write(js_docs)

    # Generate code examples
    examples = generate_examples()
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    for filename, content in examples.items():
        with open(examples_dir / filename, "w") as f:
            f.write(content)

    # Generate index file
    index_content = """# SDK Documentation

## Available SDKs

- [Python SDK](python.md) - Full-featured Python client
- [JavaScript SDK](javascript.md) - Browser and Node.js client

## Code Examples

- [Basic Usage (Python)](examples/basic_usage.py)
- [Performance Analysis (Python)](examples/performance_analysis.py)
- [Basic Usage (JavaScript)](examples/basic_usage.js)

## Quick Start

Choose your preferred language and follow the installation and usage guides.
All SDKs provide the same core functionality with language-specific conventions.
"""

    with open(output_dir / "index.md", "w") as f:
        f.write(index_content)

    print(f"SDK documentation generated in {output_dir}")
    print("Files created:")
    print(f"  • {output_dir}/python.md")
    print(f"  • {output_dir}/javascript.md")
    print(f"  • {output_dir}/index.md")
    print(f"  • {examples_dir}/ (code examples)")


if __name__ == "__main__":
    main()
