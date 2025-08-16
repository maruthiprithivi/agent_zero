#!/usr/bin/env python3
"""Basic usage example for Agent Zero."""

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
        metrics = await client.execute_tool("get_clickhouse_metrics", {"metric_type": "basic"})

        print(f"📊 Current metrics: {metrics}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()
        print("👋 Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
