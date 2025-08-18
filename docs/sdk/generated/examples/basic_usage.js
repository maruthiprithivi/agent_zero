// Basic usage example for Agent Zero (JavaScript)

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
