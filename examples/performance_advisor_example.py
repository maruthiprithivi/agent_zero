#!/usr/bin/env python3
"""
Example usage of the Performance Advisor Engine for ClickHouse optimization.

This example demonstrates how to use the AI-powered performance advisor to:
1. Generate comprehensive performance recommendations
2. Get specific configuration optimization advice
3. Analyze query performance patterns
4. Plan capacity scaling
5. Detect performance bottlenecks

The performance advisor provides actionable recommendations with impact predictions,
implementation complexity assessments, and priority rankings.
"""

import os
import sys

# Add the parent directory to the path to import agent_zero
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.ai_diagnostics import (
    create_ai_bottleneck_detector,
    create_performance_advisor,
)
from agent_zero.server.client import create_clickhouse_client


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")


def print_recommendations(recommendations: list, title: str):
    """Print formatted recommendations."""
    print(f"\n{title} ({len(recommendations)} recommendations):")
    print("-" * 60)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(
            f"   Impact: {rec['impact_percentage']:.1f}% | "
            f"Confidence: {rec['confidence_score']:.1f}% | "
            f"Priority: {rec['priority_score']:.1f}"
        )
        print(f"   Complexity: {rec['complexity']} | " f"Time: {rec['estimated_time_hours']:.1f}h")
        print(f"   Description: {rec['description']}")

        if rec.get("implementation_steps"):
            print(f"   Steps: {len(rec['implementation_steps'])} implementation steps")


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive performance analysis."""
    print_section_header("COMPREHENSIVE PERFORMANCE ANALYSIS")

    try:
        # Create ClickHouse client
        client = create_clickhouse_client()
        print("✓ Connected to ClickHouse")

        # Create bottleneck detector for integration
        bottleneck_detector = create_ai_bottleneck_detector(client)
        print("✓ Created AI bottleneck detector")

        # Create performance advisor
        advisor = create_performance_advisor(client, bottleneck_detector)
        print("✓ Created performance advisor engine")

        # Generate comprehensive recommendations
        print("\n🔍 Analyzing system performance and generating recommendations...")
        recommendations = advisor.generate_comprehensive_recommendations(max_recommendations=15)

        # Display summary
        summary = recommendations.get("summary", {})
        print("\n📊 SYSTEM HEALTH SUMMARY:")
        print(f"   Total Recommendations: {recommendations.get('total_recommendations', 0)}")
        print(f"   System Health Score: {summary.get('system_health_score', 'N/A')}")
        print(f"   Potential Impact: {summary.get('potential_cumulative_impact', 'N/A')}")
        print(f"   Average Confidence: {summary.get('average_confidence', 'N/A')}")
        print(f"   Implementation Time: {summary.get('total_implementation_time', 'N/A')}")

        # Display top recommendation
        if "top_recommendation" in summary:
            top_rec = summary["top_recommendation"]
            print("\n🎯 TOP RECOMMENDATION:")
            print(f"   {top_rec['title']}")
            print(f"   Expected Impact: {top_rec['impact']}")
            print(f"   Complexity: {top_rec['complexity']}")

        # Display recommendations by category
        recommendations_by_category = recommendations.get("recommendations_by_category", {})

        for category, recs in recommendations_by_category.items():
            if recs:
                category_title = category.replace("_", " ").title()
                print_recommendations(recs, f"{category_title}")

        return True

    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        return False


def demonstrate_specific_advisors():
    """Demonstrate specific advisor functionality."""
    print_section_header("SPECIFIC ADVISOR DEMONSTRATIONS")

    try:
        client = create_clickhouse_client()
        bottleneck_detector = create_ai_bottleneck_detector(client)
        advisor = create_performance_advisor(client, bottleneck_detector)

        # Configuration recommendations
        print("\n🔧 CONFIGURATION OPTIMIZATION ANALYSIS:")
        context = advisor._gather_system_context()
        config_recs = advisor.configuration_advisor.analyze_server_configuration(context)

        if config_recs:
            print(f"Found {len(config_recs)} configuration optimization opportunities:")
            for rec in config_recs[:3]:  # Show top 3
                print(f"  • {rec.title} (Impact: {rec.impact_percentage:.1f}%)")
                print(f"    {rec.description}")
        else:
            print("  No configuration optimizations needed at this time.")

        # Query optimization
        print("\n🔍 QUERY PERFORMANCE ANALYSIS:")
        query_recs = advisor.query_optimizer.analyze_query_patterns(context)

        if query_recs:
            print(f"Found {len(query_recs)} query optimization opportunities:")
            for rec in query_recs[:3]:  # Show top 3
                print(f"  • {rec.title} (Impact: {rec.impact_percentage:.1f}%)")
                print(f"    {rec.description}")
        else:
            print("  No query optimizations identified at this time.")

        # Capacity planning
        print("\n📈 CAPACITY PLANNING ANALYSIS:")
        capacity_recs = advisor.capacity_planner.analyze_capacity_requirements(context)

        if capacity_recs:
            print(f"Found {len(capacity_recs)} capacity planning recommendations:")
            for rec in capacity_recs[:3]:  # Show top 3
                print(f"  • {rec.title} (Impact: {rec.impact_percentage:.1f}%)")
                print(f"    {rec.description}")
                if rec.estimated_cost_savings:
                    print(f"    Potential monthly savings: ${rec.estimated_cost_savings:.2f}")
        else:
            print("  Current capacity appears adequate for the workload.")

        return True

    except Exception as e:
        print(f"❌ Error in specific advisor analysis: {e}")
        return False


def demonstrate_bottleneck_detection():
    """Demonstrate AI bottleneck detection."""
    print_section_header("AI BOTTLENECK DETECTION")

    try:
        client = create_clickhouse_client()
        bottleneck_detector = create_ai_bottleneck_detector(client)

        print("\n🤖 Running AI bottleneck detection...")
        bottlenecks = bottleneck_detector.detect_bottlenecks()

        if bottlenecks:
            print(f"\n⚠️  DETECTED {len(bottlenecks)} PERFORMANCE BOTTLENECKS:")

            for i, bottleneck in enumerate(bottlenecks[:5], 1):  # Show top 5
                print(f"\n{i}. {bottleneck.signature.name}")
                print(f"   Category: {bottleneck.signature.category.value}")
                print(f"   Severity: {bottleneck.severity.value}")
                print(f"   Confidence: {bottleneck.confidence:.1f}%")
                print(f"   Impact: {bottleneck.estimated_performance_impact:.1f}%")
                print(f"   Trend: {bottleneck.trend_direction.value}")

                if bottleneck.immediate_actions:
                    print(f"   Immediate Actions: {len(bottleneck.immediate_actions)} recommended")
        else:
            print("\n✅ No significant performance bottlenecks detected!")

        # System health score
        health_score = bottleneck_detector.calculate_system_health_score()
        print(f"\n💚 SYSTEM HEALTH SCORE: {health_score.overall_score:.1f}/100")

        if health_score.component_scores:
            print("   Component Health:")
            for component, score in health_score.component_scores.items():
                status = "✅" if score > 80 else "⚠️" if score > 60 else "❌"
                print(f"   {status} {component}: {score:.1f}")

        return True

    except Exception as e:
        print(f"❌ Error in bottleneck detection: {e}")
        return False


def demonstrate_recommendation_categories():
    """Demonstrate different recommendation categories."""
    print_section_header("RECOMMENDATION CATEGORIES EXPLAINED")

    categories = {
        "Immediate Fixes": "Quick wins that can be implemented in under 1 hour",
        "Medium-term Optimizations": "Changes requiring up to 1 week of planning",
        "Long-term Planning": "Strategic improvements requiring >1 week",
        "Preventive Measures": "Monitoring and alerting setup",
        "Cost Optimization": "Cloud cost reduction opportunities",
    }

    impact_levels = {
        "Transformational": "50%+ performance improvement",
        "Significant": "20-50% performance improvement",
        "Moderate": "10-20% performance improvement",
        "Minor": "5-10% performance improvement",
        "Marginal": "1-5% performance improvement",
    }

    print("\n📋 RECOMMENDATION CATEGORIES:")
    for category, description in categories.items():
        print(f"  • {category}: {description}")

    print("\n📊 IMPACT LEVELS:")
    for level, description in impact_levels.items():
        print(f"  • {level}: {description}")


def main():
    """Main demonstration function."""
    print("🚀 Agent Zero Performance Advisor Engine Demo")
    print("=" * 80)
    print("This demonstration shows the AI-powered performance optimization")
    print("capabilities for ClickHouse deployments.")

    # Check if we can connect to ClickHouse
    try:
        client = create_clickhouse_client()
        client.command("SELECT 1")
        print("✅ ClickHouse connection successful")
    except Exception as e:
        print(f"❌ Cannot connect to ClickHouse: {e}")
        print("Please ensure ClickHouse is running and properly configured.")
        return

    success_count = 0
    total_tests = 4

    # Run demonstrations
    if demonstrate_comprehensive_analysis():
        success_count += 1

    if demonstrate_specific_advisors():
        success_count += 1

    if demonstrate_bottleneck_detection():
        success_count += 1

    demonstrate_recommendation_categories()
    success_count += 1

    # Summary
    print_section_header("DEMONSTRATION SUMMARY")
    print(f"✅ Completed {success_count}/{total_tests} demonstrations successfully")

    if success_count == total_tests:
        print("\n🎉 All performance advisor features are working correctly!")
        print("\nNext Steps:")
        print("1. Use the MCP tools in your IDE for real-time recommendations")
        print("2. Implement high-priority recommendations from the analysis")
        print("3. Monitor the impact of optimizations over time")
        print("4. Schedule regular performance health checks")
    else:
        print("\n⚠️  Some demonstrations failed. Please check the error messages above.")

    print("\n📝 For more information, see:")
    print("   - Performance Advisor Documentation")
    print("   - AI Diagnostics Module Reference")
    print("   - ClickHouse Optimization Best Practices")


if __name__ == "__main__":
    main()
