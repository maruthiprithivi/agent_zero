#!/usr/bin/env python3
"""Example demonstrating comprehensive pattern analysis and anomaly detection capabilities.

This example shows how to use the PatternAnalysisEngine to:
- Analyze time series patterns in ProfileEvents
- Detect anomalies using multiple statistical methods
- Identify trends, seasonal patterns, and change points
- Discover correlations between different ProfileEvents
- Generate comprehensive analysis reports

Run this example with:
    python examples/pattern_analysis_example.py
"""

import logging
from datetime import datetime
from typing import Any

from agent_zero.ai_diagnostics import (
    create_pattern_analysis_engine,
)
from agent_zero.server.client import create_clickhouse_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_pattern_analysis_results(analysis_result, event_name: str):
    """Display comprehensive pattern analysis results in a readable format."""
    print(f"\n{'=' * 80}")
    print(f"PATTERN ANALYSIS RESULTS FOR: {event_name}")
    print(f"{'=' * 80}")

    # Analysis period
    start_time, end_time = analysis_result.analysis_period
    print(
        f"Analysis Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Baseline metrics
    baseline = analysis_result.baseline_metrics
    print("\n📊 BASELINE METRICS:")
    print(f"  Mean: {baseline.mean:.2f}")
    print(f"  Median: {baseline.median:.2f}")
    print(f"  Std Deviation: {baseline.std_dev:.2f}")
    print(f"  95th Percentile: {baseline.percentile_95:.2f}")
    print(f"  99th Percentile: {baseline.percentile_99:.2f}")
    print(f"  Sample Size: {baseline.sample_size}")
    print(
        f"  Control Limits: [{baseline.lower_control_limit:.2f}, {baseline.upper_control_limit:.2f}]"
    )

    # Trend analysis
    trend = analysis_result.trend_analysis
    print("\n📈 TREND ANALYSIS:")
    print(f"  Trend Type: {trend.trend_type.value}")
    print(f"  Slope: {trend.slope:.6f}")
    print(f"  R-squared: {trend.r_squared:.4f}")
    print(f"  Trend Strength: {trend.trend_strength:.4f}")
    print(f"  Volatility: {trend.volatility:.4f}")
    print(f"  Persistence: {trend.persistence:.4f}")
    if trend.forecast_values:
        print(f"  Next 3 Forecasted Values: {trend.forecast_values[:3]}")

    # Seasonal patterns
    if analysis_result.seasonal_patterns:
        print("\n🌊 SEASONAL PATTERNS:")
        for i, pattern in enumerate(analysis_result.seasonal_patterns[:3]):
            print(
                f"  Pattern {i + 1}: Period={pattern.period}, Amplitude={pattern.amplitude:.2f}, Confidence={pattern.confidence:.4f}"
            )
    else:
        print("\n🌊 SEASONAL PATTERNS: None detected")

    # Change points
    if analysis_result.change_points:
        print("\n⚡ CHANGE POINTS:")
        for i, cp in enumerate(analysis_result.change_points[:5]):
            print(
                f"  Change {i + 1}: {cp.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {cp.change_type.value}"
            )
            print(f"    Magnitude: {cp.magnitude:.2f}, Confidence: {cp.confidence:.4f}")
            print(f"    Before/After Mean: {cp.before_mean:.2f} → {cp.after_mean:.2f}")
    else:
        print("\n⚡ CHANGE POINTS: None detected")

    # Anomalies
    if analysis_result.anomalies:
        print(f"\n🚨 ANOMALIES DETECTED: {len(analysis_result.anomalies)}")

        # Group by severity
        anomalies_by_severity = {}
        for anomaly in analysis_result.anomalies:
            severity = anomaly.severity.value
            if severity not in anomalies_by_severity:
                anomalies_by_severity[severity] = []
            anomalies_by_severity[severity].append(anomaly)

        for severity in ["critical", "high", "medium", "low"]:
            if severity in anomalies_by_severity:
                count = len(anomalies_by_severity[severity])
                print(f"  {severity.upper()}: {count} anomalies")

                # Show details for critical and high severity
                if severity in ["critical", "high"] and count > 0:
                    for anomaly in anomalies_by_severity[severity][:3]:  # Show first 3
                        print(
                            f"    • {anomaly.timestamp.strftime('%H:%M:%S')} - Value: {anomaly.value:.2f}"
                        )
                        print(
                            f"      Z-score: {anomaly.z_score:.2f}, Overall score: {anomaly.overall_score:.4f}"
                        )
                        print(f"      Types: {[t.value for t in anomaly.anomaly_types]}")
    else:
        print("\n🚨 ANOMALIES: None detected")

    # Detected patterns
    if analysis_result.detected_patterns:
        print("\n🔍 DETECTED PATTERNS:")
        pattern_types = {}
        for pattern in analysis_result.detected_patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(pattern)

        for pattern_type, patterns in pattern_types.items():
            print(f"  {pattern_type.upper()}: {len(patterns)} occurrences")
            if patterns:
                avg_similarity = sum(p.similarity_score for p in patterns) / len(patterns)
                print(f"    Average similarity: {avg_similarity:.4f}")
    else:
        print("\n🔍 DETECTED PATTERNS: None detected")

    # Correlations
    if analysis_result.correlations:
        print("\n🔗 CORRELATIONS:")
        for corr in analysis_result.correlations[:5]:  # Show top 5
            print(
                f"  {corr.secondary_event}: {corr.correlation_coefficient:.4f} (p={corr.p_value:.4f})"
            )
            print(
                f"    Relationship: {corr.relationship_type}, Stability: {corr.correlation_stability:.4f}"
            )
    else:
        print("\n🔗 CORRELATIONS: None found")

    # Summary statistics
    print("\n📋 SUMMARY STATISTICS:")
    print(
        f"  Anomaly Rate: {analysis_result.anomaly_rate:.4f} ({analysis_result.anomaly_rate * 100:.2f}%)"
    )
    print(
        f"  Pattern Coverage: {analysis_result.pattern_coverage:.4f} ({analysis_result.pattern_coverage * 100:.2f}%)"
    )
    print(f"  Predictability Score: {analysis_result.predictability_score:.4f}")
    print(f"  Stability Score: {analysis_result.stability_score:.4f}")


def display_anomaly_summary(anomaly_summary: dict[str, Any]):
    """Display anomaly summary across all ProfileEvents."""
    print(f"\n{'=' * 80}")
    print("SYSTEM-WIDE ANOMALY SUMMARY")
    print(f"{'=' * 80}")

    print("📊 Overview:")
    print(f"  Total Events Analyzed: {anomaly_summary['total_events_analyzed']}")
    print(f"  Events with Anomalies: {anomaly_summary['events_with_anomalies']}")
    print(f"  Total Anomalies: {anomaly_summary['total_anomalies']}")
    print(f"  Critical Anomalies: {anomaly_summary['critical_anomalies']}")
    print(f"  High Severity Anomalies: {anomaly_summary['high_anomalies']}")

    if anomaly_summary["most_anomalous_events"]:
        print("\n🔥 Most Anomalous Events:")
        for event_name, stats in anomaly_summary["most_anomalous_events"][:5]:
            print(
                f"  {event_name}: {stats['total']} anomalies ({stats['anomaly_rate'] * 100:.2f}% rate)"
            )
            print(f"    Critical: {stats['critical']}, High: {stats['high']}")

    if anomaly_summary["recent_change_points"]:
        print(f"\n⚡ Recent Change Points ({len(anomaly_summary['recent_change_points'])}):")
        for cp in anomaly_summary["recent_change_points"][:5]:
            age_hours = (datetime.now() - cp.timestamp).total_seconds() / 3600
            print(f"  {cp.event_name}: {cp.change_type.value} ({age_hours:.1f}h ago)")
            print(f"    Magnitude: {cp.magnitude:.2f}, Confidence: {cp.confidence:.4f}")


def main():
    """Main function demonstrating pattern analysis capabilities."""
    print("🔬 ClickHouse Pattern Analysis & Anomaly Detection Example")
    print("=" * 60)

    try:
        # Create ClickHouse client
        print("🔌 Connecting to ClickHouse...")
        client = create_clickhouse_client()

        # Create pattern analysis engine
        print("🧠 Initializing Pattern Analysis Engine...")
        pattern_engine = create_pattern_analysis_engine(client)

        # List of ProfileEvents to analyze
        events_to_analyze = [
            "Query",
            "SelectQuery",
            "InsertQuery",
            "SelectedRows",
            "SelectedBytes",
            "NetworkReceiveElapsedMicroseconds",
            "MemoryTrackingInBackgroundProcessingPool",
            "ReadBufferFromFileDescriptorRead",
            "WriteBufferFromFileDescriptorWrite",
        ]

        print(f"📊 Analyzing patterns for {len(events_to_analyze)} ProfileEvents...")

        # Analyze patterns for individual events
        successful_analyses = 0
        for i, event_name in enumerate(events_to_analyze, 1):
            try:
                print(f"\n[{i}/{len(events_to_analyze)}] Analyzing {event_name}...")

                # Perform comprehensive pattern analysis
                analysis_result = pattern_engine.analyze_patterns(
                    event_name=event_name, lookback_hours=24, force_refresh=True
                )

                # Display results
                display_pattern_analysis_results(analysis_result, event_name)
                successful_analyses += 1

            except Exception as e:
                print(f"❌ Error analyzing {event_name}: {e}")
                logger.error(f"Pattern analysis failed for {event_name}: {e}")
                continue

        print(f"\n✅ Successfully analyzed {successful_analyses}/{len(events_to_analyze)} events")

        # Get system-wide anomaly summary
        print("\n🌍 Generating system-wide anomaly summary...")
        try:
            anomaly_summary = pattern_engine.get_anomaly_summary(lookback_hours=24)
            display_anomaly_summary(anomaly_summary)
        except Exception as e:
            print(f"❌ Error generating anomaly summary: {e}")
            logger.error(f"Anomaly summary generation failed: {e}")

        # Demonstrate multiple event analysis
        print("\n🔄 Demonstrating batch analysis...")
        try:
            batch_results = pattern_engine.analyze_multiple_events(
                event_names=events_to_analyze[:3],  # Analyze first 3 events
                lookback_hours=12,
            )

            print("📊 Batch Analysis Results:")
            for event_name, result in batch_results.items():
                print(
                    f"  {event_name}: {len(result.anomalies)} anomalies, "
                    f"{result.anomaly_rate * 100:.2f}% rate, "
                    f"stability: {result.stability_score:.4f}"
                )

        except Exception as e:
            print(f"❌ Error in batch analysis: {e}")
            logger.error(f"Batch analysis failed: {e}")

        print("\n🎉 Pattern analysis demonstration completed!")
        print("\n💡 Key Features Demonstrated:")
        print("  ✓ Baseline establishment and management")
        print("  ✓ Multi-method anomaly detection")
        print("  ✓ Trend analysis and forecasting")
        print("  ✓ Seasonal pattern recognition")
        print("  ✓ Change point detection")
        print("  ✓ Pattern matching and classification")
        print("  ✓ Correlation analysis")
        print("  ✓ System-wide anomaly monitoring")

    except Exception as e:
        print(f"❌ Failed to connect to ClickHouse or initialize engine: {e}")
        logger.error(f"Example execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
