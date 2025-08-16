"""AI/ML Integration Module for Agent Zero.

This module provides advanced AI/ML capabilities:
- Performance prediction using machine learning
- Query optimization recommendations
- Anomaly detection for system monitoring
- Vector database integration for RAG systems
- Real-time model inference and training
"""

from .performance_predictor import (
    ModelType,
    PerformanceMetrics,
    PerformancePredictor,
    PredictionResult,
    PredictionType,
    QueryComplexityAnalyzer,
    VectorDatabaseManager,
    create_performance_predictor,
)

__all__ = [
    "ModelType",
    "PerformanceMetrics",
    "PerformancePredictor",
    "PredictionResult",
    "PredictionType",
    "QueryComplexityAnalyzer",
    "VectorDatabaseManager",
    "create_performance_predictor",
]
