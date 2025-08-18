#!/usr/bin/env python3
"""Performance analysis example."""

import asyncio

from agent_zero import AgentZeroClient


async def analyze_performance():
    client = AgentZeroClient()
    await client.connect()

    try:
        # Get comprehensive performance analysis
        analysis = await client.execute_tool(
            "analyze_clickhouse_performance",
            {"time_range": "24h", "include_ai_insights": True, "detail_level": "comprehensive"},
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
            print("\n🤖 AI Insights:")
            for insight in analysis["ai_insights"]:
                print(f"  💡 {insight['title']}")
                print(f"     {insight['description']}")
                if insight.get("severity") == "high":
                    print("     ⚠️  High Priority")

        # Show recommendations
        if "recommendations" in analysis:
            print("\n📋 Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"  ✅ {rec['action']}")
                print(f"     Impact: {rec.get('impact', 'Unknown')}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(analyze_performance())
