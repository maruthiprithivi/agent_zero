"""AI-powered diagnostics module for ClickHouse performance analysis.

This module provides sophisticated machine learning capabilities for automated
bottleneck detection, predictive performance analysis, and intelligent system
health assessment. It combines data from all diagnostic modules to provide
holistic insights and actionable recommendations.

Key Components:

Bottleneck Detection:
- IntelligentBottleneckDetector: Main AI engine for comprehensive analysis
- PatternMatcher: ML-based pattern recognition for bottleneck signatures
- PredictiveAnalyzer: Performance degradation forecasting and trend analysis
- BottleneckSignature: Pattern definitions for different bottleneck types

Performance Advisory:
- PerformanceAdvisorEngine: Main advisory system coordinator
- RecommendationEngine: Core ML recommendation system with impact prediction
- ConfigurationAdvisor: ClickHouse configuration optimization recommendations
- QueryOptimizationAdvisor: Query performance improvement suggestions
- CapacityPlanningAdvisor: Hardware and scaling recommendations

Pattern Analysis & Anomaly Detection:
- PatternAnalysisEngine: Main coordinator for all pattern analysis capabilities
- TimeSeriesAnalyzer: Advanced time-series analysis and forecasting
- PerformanceBaselineEngine: Baseline establishment and management
- AnomalyDetectionEngine: Multi-method anomaly detection with statistical scoring
- PatternRecognitionEngine: Pattern matching and classification for recurring issues
- CorrelationAnalyzer: Cross-event correlation analysis and relationship discovery

Example Usage:
    from agent_zero.ai_diagnostics import (
        create_ai_bottleneck_detector, create_performance_advisor,
        create_pattern_analysis_engine, create_anomaly_detection_engine
    )
    from agent_zero.server.client import create_clickhouse_client

    client = create_clickhouse_client()

    # Bottleneck Detection
    detector = create_ai_bottleneck_detector(client)
    bottlenecks = detector.detect_bottlenecks()
    health_score = detector.calculate_system_health_score()

    # Performance Advisory
    advisor = create_performance_advisor(client, detector)
    recommendations = advisor.generate_comprehensive_recommendations()

    # Pattern Analysis & Anomaly Detection
    pattern_engine = create_pattern_analysis_engine(client)
    analysis = pattern_engine.analyze_patterns("Query", lookback_hours=24)
    anomaly_summary = pattern_engine.get_anomaly_summary(lookback_hours=24)

    # Get actionable recommendations by category
    immediate_fixes = recommendations["recommendations_by_category"]["immediate_fixes"]
    long_term_plans = recommendations["recommendations_by_category"]["long_term_planning"]
"""

from .bottleneck_detector import (
    # Enums
    BottleneckCategory,
    BottleneckDetection,
    BottleneckSeverity,
    # Data structures
    BottleneckSignature,
    ConfidenceLevel,
    # Main AI engine
    IntelligentBottleneckDetector,
    # Core analysis components
    PatternMatcher,
    PredictiveAnalyzer,
    PredictiveMetrics,
    SystemHealthScore,
    TrendDirection,
    # Utility functions
    create_ai_bottleneck_detector,
    integrate_with_performance_diagnostics,
)
from .pattern_analyzer import (
    AnomalyDetectionEngine,
    AnomalyScore,
    AnomalySeverity,
    # Enums
    AnomalyType,
    BaselineMetrics,
    ChangePoint,
    ChangePointType,
    CorrelationAnalysis,
    CorrelationAnalyzer,
    # Main pattern analysis engine
    PatternAnalysisEngine,
    PatternAnalysisResult,
    PatternMatch,
    PatternRecognitionEngine,
    PerformanceBaselineEngine,
    SeasonalPattern,
    # Core analysis engines
    TimeSeriesAnalyzer,
    # Data structures
    TimeSeriesPoint,
    TrendAnalysis,
    TrendType,
    create_anomaly_detection_engine,
    # Utility functions
    create_pattern_analysis_engine,
    create_time_series_analyzer,
)
from .performance_advisor import (
    CapacityPlanningAdvisor,
    ConfigurationAdvisor,
    ImpactLevel,
    ImplementationComplexity,
    # Main performance advisor engine
    PerformanceAdvisorEngine,
    # Data structures
    PerformanceRecommendation,
    QueryOptimizationAdvisor,
    # Enums
    RecommendationCategory,
    RecommendationContext,
    # Core advisor components
    RecommendationEngine,
    RecommendationType,
    RiskLevel,
    # Utility functions
    create_performance_advisor,
    integrate_with_diagnostics,
)

__all__ = [
    # Bottleneck Detection - Main classes
    "IntelligentBottleneckDetector",
    "PatternMatcher",
    "PredictiveAnalyzer",
    # Bottleneck Detection - Data structures
    "BottleneckSignature",
    "BottleneckDetection",
    "PredictiveMetrics",
    "SystemHealthScore",
    # Bottleneck Detection - Enums
    "BottleneckCategory",
    "BottleneckSeverity",
    "ConfidenceLevel",
    "TrendDirection",
    # Performance Advisor - Main classes
    "PerformanceAdvisorEngine",
    "RecommendationEngine",
    "ConfigurationAdvisor",
    "QueryOptimizationAdvisor",
    "CapacityPlanningAdvisor",
    # Performance Advisor - Data structures
    "PerformanceRecommendation",
    "RecommendationContext",
    # Performance Advisor - Enums
    "RecommendationCategory",
    "ImpactLevel",
    "ImplementationComplexity",
    "RiskLevel",
    "RecommendationType",
    # Pattern Analysis - Main classes
    "PatternAnalysisEngine",
    "TimeSeriesAnalyzer",
    "PerformanceBaselineEngine",
    "AnomalyDetectionEngine",
    "PatternRecognitionEngine",
    "CorrelationAnalyzer",
    # Pattern Analysis - Data structures
    "TimeSeriesPoint",
    "BaselineMetrics",
    "AnomalyScore",
    "PatternMatch",
    "TrendAnalysis",
    "CorrelationAnalysis",
    "ChangePoint",
    "SeasonalPattern",
    "PatternAnalysisResult",
    # Pattern Analysis - Enums
    "AnomalyType",
    "AnomalySeverity",
    "TrendType",
    "ChangePointType",
    # Utility functions
    "create_ai_bottleneck_detector",
    "integrate_with_performance_diagnostics",
    "create_performance_advisor",
    "integrate_with_diagnostics",
    "create_pattern_analysis_engine",
    "create_time_series_analyzer",
    "create_anomaly_detection_engine",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Agent Zero AI Diagnostics Team"
__description__ = "AI-powered bottleneck detection, performance analysis, and optimization recommendations for ClickHouse"

# Default configuration for AI diagnostics
DEFAULT_CONFIG = {
    "confidence_threshold": 0.3,
    "historical_periods": 7,
    "prediction_horizon_hours": 24,
    "adaptive_learning": True,
    "cross_domain_correlation": True,
    "real_time_analysis": True,
    # Pattern Analysis specific configuration
    "pattern_analysis": {
        "lookback_days": 30,
        "time_series_window_size": 100,
        "seasonal_periods": [24, 168, 720],  # hourly, daily, weekly, monthly
        "anomaly_threshold": 0.25,
        "pattern_similarity_threshold": 0.6,
        "correlation_threshold": 0.3,
        "change_point_threshold": 5.0,
        "baseline_refresh_hours": 6,
    },
}
