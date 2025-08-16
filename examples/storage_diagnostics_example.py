#!/usr/bin/env python3
"""Example usage of the storage and cloud diagnostics module.

This example demonstrates how to use the comprehensive storage diagnostics
capabilities to analyze S3, Azure, and compression performance in ClickHouse.
"""

import asyncio
import logging

from agent_zero.monitoring.profile_events_core import ProfileEventsAnalyzer
from agent_zero.monitoring.storage_cloud_diagnostics import (
    S3StorageAnalyzer,
    StorageOptimizationEngine,
    analyze_compression_efficiency,
    diagnose_high_storage_latency,
    diagnose_storage_throttling,
    identify_storage_cost_optimizations,
)
from agent_zero.server.client import create_clickhouse_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate storage diagnostics capabilities."""

    try:
        # Create ClickHouse client
        client = create_clickhouse_client()
        logger.info("Connected to ClickHouse")

        # Initialize the storage optimization engine
        storage_engine = StorageOptimizationEngine(client)
        profile_analyzer = ProfileEventsAnalyzer(client)

        print("=" * 80)
        print("ClickHouse Storage and Cloud Diagnostics Example")
        print("=" * 80)

        # Example 1: Comprehensive Storage Optimization Report
        print("\n1. Comprehensive Storage Optimization Report")
        print("-" * 50)

        report = storage_engine.generate_comprehensive_storage_report(
            time_range_hours=24, include_s3=True, include_azure=True, include_compression=True
        )

        print(f"Overall Storage Score: {report.overall_score:.1f}/100")
        print(f"Priority Issues Found: {len(report.priority_issues)}")
        print(f"Cost Savings Opportunities: {len(report.cost_savings_opportunities)}")
        print(f"Performance Improvements: {len(report.performance_improvements)}")

        if report.priority_issues:
            print("\nTop 3 Priority Issues:")
            for i, issue in enumerate(report.priority_issues[:3], 1):
                print(f"  {i}. {issue.type.value}: {issue.description}")
                print(f"     Severity: {issue.severity.value}, Impact: {issue.impact_score:.1f}")

        if report.recommendations:
            print(f"\nTop Recommendations ({len(report.recommendations)} total):")
            for rec in report.recommendations[:5]:
                print(f"  • {rec}")

        # Example 2: S3 Storage Analysis
        print("\n\n2. S3 Storage Performance Analysis")
        print("-" * 50)

        s3_analyzer = S3StorageAnalyzer(profile_analyzer)
        s3_analysis = s3_analyzer.analyze_s3_performance(time_range_hours=4)

        print(f"S3 Issues Found: {len(s3_analysis.issues)}")
        if s3_analysis.operation_performance:
            print("S3 Operation Performance:")
            for op_type, metrics in s3_analysis.operation_performance.items():
                if isinstance(metrics, dict) and "total_requests" in metrics:
                    print(
                        f"  {op_type}: {metrics['total_requests']} requests, "
                        f"{metrics.get('average_latency_ms', 0):.1f}ms avg latency"
                    )

        if s3_analysis.cost_analysis.get("request_costs"):
            costs = s3_analysis.cost_analysis["request_costs"]
            print("Estimated Monthly S3 Costs:")
            print(f"  GET requests: ${costs.get('estimated_get_cost', 0):.2f}")
            print(f"  PUT requests: ${costs.get('estimated_put_cost', 0):.2f}")

        # Example 3: High Storage Latency Diagnosis
        print("\n\n3. High Storage Latency Diagnosis")
        print("-" * 50)

        latency_diagnosis = diagnose_high_storage_latency(client, time_range_hours=4)

        if "error" not in latency_diagnosis:
            latency_issues = latency_diagnosis.get("latency_issues", [])
            print(f"High Latency Issues: {len(latency_issues)}")

            for issue in latency_issues[:3]:  # Show top 3
                print(f"  • {issue.description}")
                print(f"    Recommendations: {', '.join(issue.recommendations[:2])}")
        else:
            print(f"Latency diagnosis error: {latency_diagnosis['error']}")

        # Example 4: Storage Throttling Analysis
        print("\n\n4. Storage Throttling Analysis")
        print("-" * 50)

        throttling_diagnosis = diagnose_storage_throttling(client, time_range_hours=4)

        if "error" not in throttling_diagnosis:
            throttling_detected = throttling_diagnosis.get("throttling_detected", False)
            print(f"Throttling Detected: {throttling_detected}")

            if throttling_detected:
                issues = throttling_diagnosis.get("throttling_issues", [])
                print(f"Throttling Issues: {len(issues)}")
                print("Mitigation Strategies:")
                for strategy in throttling_diagnosis.get("mitigation_strategies", [])[:3]:
                    print(f"  • {strategy}")
            else:
                print("No throttling issues detected.")
        else:
            print(f"Throttling diagnosis error: {throttling_diagnosis['error']}")

        # Example 5: Compression Efficiency Analysis
        print("\n\n5. Compression Efficiency Analysis")
        print("-" * 50)

        compression_analysis = analyze_compression_efficiency(client, time_range_hours=24)

        if "error" not in compression_analysis:
            efficiency = compression_analysis.get("compression_efficiency", {})
            integrity = compression_analysis.get("integrity_analysis", {})

            print(f"Compression Efficiency Score: {efficiency.get('efficiency_score', 0):.1f}")
            print(f"Data Integrity Score: {integrity.get('integrity_score', 100):.1f}")

            if compression_analysis.get("issues"):
                print("Compression Issues:")
                for issue in compression_analysis["issues"][:3]:
                    print(f"  • {issue['description']} (Severity: {issue['severity']})")
        else:
            print(f"Compression analysis error: {compression_analysis['error']}")

        # Example 6: Cost Optimization Analysis
        print("\n\n6. Storage Cost Optimization")
        print("-" * 50)

        cost_optimization = identify_storage_cost_optimizations(client, time_range_hours=24)

        if "error" not in cost_optimization:
            opportunities = cost_optimization.get("cost_savings_opportunities", [])
            estimated_savings = cost_optimization.get("estimated_monthly_savings", 0)

            print(f"Cost Optimization Opportunities: {len(opportunities)}")
            print(f"Estimated Monthly Savings Potential: {estimated_savings}%")

            if opportunities:
                print("Top Cost Optimization Opportunities:")
                for opp in opportunities[:3]:
                    print(f"  • {opp.get('description', 'Unknown')}")
                    print(f"    Potential Savings: {opp.get('potential_savings_percent', 0)}%")
        else:
            print(f"Cost optimization error: {cost_optimization['error']}")

        print("\n" + "=" * 80)
        print("Storage diagnostics completed successfully!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error in storage diagnostics example: {e}")
        print(f"Error: {e}")
        print("This is normal if ClickHouse is not running or properly configured.")
        print("The storage diagnostics module is ready for use when ClickHouse is available.")

    finally:
        if "client" in locals():
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
