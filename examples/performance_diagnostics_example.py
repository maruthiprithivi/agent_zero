#!/usr/bin/env python3
"""Example usage of the Performance Diagnostics Suite.

This script demonstrates how to use the comprehensive performance diagnostics
capabilities of Agent Zero to analyze ClickHouse performance issues.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_zero.monitoring.performance_diagnostics import (
    CacheAnalyzer,
    IOPerformanceAnalyzer,
    PerformanceDiagnosticEngine,
    QueryExecutionAnalyzer,
)
from agent_zero.server.client import create_clickhouse_client


def main():
    """Demonstrate performance diagnostics usage."""
    print("ClickHouse Performance Diagnostics Suite Example")
    print("=" * 50)

    try:
        # Create ClickHouse client
        print("1. Connecting to ClickHouse...")
        client = create_clickhouse_client()
        print("   ✓ Connected successfully")

        # Initialize the diagnostic engine
        print("\n2. Initializing Performance Diagnostic Engine...")
        engine = PerformanceDiagnosticEngine(client)
        print("   ✓ Engine initialized")

        # Define analysis period (last hour)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        print(f"\n3. Analyzing performance for period: {start_time} to {end_time}")

        # Generate comprehensive performance report
        print("\n   Generating comprehensive performance report...")
        report = engine.generate_comprehensive_report(
            start_time=start_time,
            end_time=end_time,
            comparison_period_hours=24,  # Compare with 24 hours ago
        )

        print("   ✓ Report generated successfully")

        # Display report summary
        print("\n4. Performance Report Summary")
        print("-" * 30)
        print(f"Analysis Period: {report.analysis_period_start} to {report.analysis_period_end}")
        print(f"Overall Performance Score: {report.overall_performance_score:.1f}/100")

        # Display cache analysis
        cache_analysis = report.cache_analysis
        print("\nCache Performance:")
        print(f"  Overall Cache Score: {cache_analysis.overall_cache_score:.1f}/100")
        print(
            f"  Mark Cache Efficiency: {cache_analysis.mark_cache_efficiency.get('efficiency', 'N/A')}"
        )
        print(
            f"  Mark Cache Hit Rate: {cache_analysis.mark_cache_efficiency.get('hit_rate', 0):.1f}%"
        )
        print(
            f"  Uncompressed Cache Efficiency: {cache_analysis.uncompressed_cache_efficiency.get('efficiency', 'N/A')}"
        )
        print(
            f"  Query Cache Efficiency: {cache_analysis.query_cache_efficiency.get('efficiency', 'N/A')}"
        )

        # Display critical bottlenecks
        if report.critical_bottlenecks:
            print(f"\n5. Critical Performance Issues ({len(report.critical_bottlenecks)} found)")
            print("-" * 40)
            for i, bottleneck in enumerate(report.critical_bottlenecks[:5], 1):
                print(f"  {i}. {bottleneck.description}")
                print(f"     Type: {bottleneck.type.value}")
                print(f"     Severity: {bottleneck.severity.value}")
                print(f"     Impact Score: {bottleneck.impact_score:.1f}/100")
                if bottleneck.recommendations:
                    print(f"     Recommendations: {bottleneck.recommendations[0]}")
                print()
        else:
            print("\n5. No critical performance issues detected ✓")

        # Display top recommendations
        if report.top_recommendations:
            print(f"\n6. Top Performance Recommendations ({len(report.top_recommendations)})")
            print("-" * 45)
            for i, recommendation in enumerate(report.top_recommendations[:5], 1):
                print(f"  {i}. {recommendation}")
        else:
            print("\n6. No specific recommendations at this time")

        # Display comparative analysis if available
        if report.comparative_analysis:
            comp_analysis = report.comparative_analysis
            if "significant_changes" in comp_analysis:
                print("\n7. Comparative Analysis")
                print("-" * 25)
                print(f"   Comparison Period: {comp_analysis.get('comparison_period', 'N/A')}")
                print(f"   Significant Changes: {len(comp_analysis['significant_changes'])}")
                print(f"   Anomalies Detected: {comp_analysis.get('anomalies_detected', 0)}")

                if comp_analysis["significant_changes"]:
                    print("\n   Notable Changes:")
                    for change in comp_analysis["significant_changes"][:3]:
                        change_type = "↑" if change["change_type"] == "degradation" else "↓"
                        print(
                            f"     {change_type} {change['event']}: {change['change_percentage']:+.1f}%"
                        )

        print("\n" + "=" * 50)
        print("Performance analysis completed successfully!")

        # Demonstrate individual analyzers
        print("\n8. Individual Analyzer Examples")
        print("-" * 35)

        # Query Execution Analysis
        print("   Query Execution Analysis:")
        query_analyzer = QueryExecutionAnalyzer(engine.profile_analyzer)

        try:
            function_perf = query_analyzer.analyze_function_performance(start_time, end_time)
            print(f"     Function Performance Events: {len(function_perf)}")

            null_handling = query_analyzer.analyze_null_handling_efficiency(start_time, end_time)
            print(f"     NULL Handling Status: {null_handling.get('status', 'N/A')}")

            memory_alloc = query_analyzer.analyze_memory_allocation_patterns(start_time, end_time)
            print(f"     Memory Allocation Patterns: {len(memory_alloc)}")
        except Exception as e:
            print(f"     Error in query analysis: {e}")

        # I/O Performance Analysis
        print("\n   I/O Performance Analysis:")
        io_analyzer = IOPerformanceAnalyzer(engine.profile_analyzer)

        try:
            file_ops = io_analyzer.analyze_file_operations(start_time, end_time)
            print(f"     File Operations Analysis: {len(file_ops)} categories")

            network_perf = io_analyzer.analyze_network_performance(start_time, end_time)
            print(f"     Network Performance Analysis: {len(network_perf)} categories")

            disk_perf = io_analyzer.analyze_disk_performance(start_time, end_time)
            print(f"     Disk Performance Analysis: {len(disk_perf)} categories")
        except Exception as e:
            print(f"     Error in I/O analysis: {e}")

        # Cache Analysis
        print("\n   Cache Analysis:")
        cache_analyzer = CacheAnalyzer(engine.profile_analyzer)

        try:
            mark_cache = cache_analyzer.analyze_mark_cache(start_time, end_time)
            print(f"     Mark Cache Status: {mark_cache.get('efficiency', 'N/A')}")

            uncompressed_cache = cache_analyzer.analyze_uncompressed_cache(start_time, end_time)
            print(f"     Uncompressed Cache Status: {uncompressed_cache.get('efficiency', 'N/A')}")

            query_cache = cache_analyzer.analyze_query_cache(start_time, end_time)
            print(f"     Query Cache Status: {query_cache.get('efficiency', 'N/A')}")
        except Exception as e:
            print(f"     Error in cache analysis: {e}")

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure ClickHouse is running and accessible")
        print("2. Check your connection configuration")
        print("3. Verify you have sufficient data for analysis")
        print("4. Make sure ProfileEvents are enabled in ClickHouse")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
