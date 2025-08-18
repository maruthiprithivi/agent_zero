# JavaScript SDK Documentation

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
