"""AI/ML Performance Prediction and Optimization for Agent Zero.

This module provides advanced AI/ML capabilities for predicting and optimizing
ClickHouse performance using modern machine learning techniques:

- Query performance prediction
- Resource usage forecasting
- Anomaly detection for system metrics
- Automated optimization recommendations
- Vector database integration for RAG systems
- Real-time model inference
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

# Machine Learning imports
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Vector Database imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

# Deep Learning imports (optional)
try:
    import torch
    import torch.nn as nn

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of performance predictions."""

    QUERY_EXECUTION_TIME = "query_execution_time"
    RESOURCE_USAGE = "resource_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelType(Enum):
    """Available model types for predictions."""

    RANDOM_FOREST = "random_forest"
    ISOLATION_FOREST = "isolation_forest"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"


@dataclass
class PerformanceMetrics:
    """Performance metrics for model training and prediction."""

    timestamp: datetime
    query_time: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    active_connections: int
    query_complexity: float
    data_size: int
    result_rows: int

    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to feature vector for ML models."""
        return np.array(
            [
                self.cpu_usage,
                self.memory_usage,
                self.disk_io,
                self.network_io,
                self.active_connections,
                self.query_complexity,
                self.data_size,
                self.result_rows,
            ]
        )


@dataclass
class PredictionResult:
    """Result of a performance prediction."""

    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    timestamp: datetime
    features_used: list[str]
    model_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryComplexityAnalyzer:
    """Analyze SQL query complexity for ML features."""

    def __init__(self):
        self.complexity_keywords = {
            "join": 3.0,
            "union": 2.0,
            "group by": 2.5,
            "order by": 1.5,
            "having": 2.0,
            "window": 4.0,
            "subquery": 3.5,
            "with": 2.0,
            "cte": 2.0,
            "distinct": 1.5,
            "exists": 2.5,
            "in": 1.0,
        }

    def analyze(self, sql: str) -> float:
        """Calculate query complexity score."""
        if not sql:
            return 0.0

        sql_lower = sql.lower()
        complexity = 1.0  # Base complexity

        # Keyword-based complexity
        for keyword, weight in self.complexity_keywords.items():
            if keyword in sql_lower:
                complexity += weight

        # Length-based complexity
        complexity += len(sql) / 1000.0

        # Nested query detection
        paren_depth = 0
        max_depth = 0
        for char in sql:
            if char == "(":
                paren_depth += 1
                max_depth = max(max_depth, paren_depth)
            elif char == ")":
                paren_depth -= 1

        complexity += max_depth * 0.5

        return min(complexity, 20.0)  # Cap at 20


class PerformancePredictor:
    """AI/ML-based performance predictor for ClickHouse operations."""

    def __init__(self, model_type: ModelType = ModelType.RANDOM_FOREST):
        self.model_type = model_type
        self.models: dict[PredictionType, Any] = {}
        self.scalers: dict[PredictionType, StandardScaler] = {}
        self.training_data: list[PerformanceMetrics] = []
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.model_version = "v1.0.0"

        if not ML_AVAILABLE:
            logger.warning("ML libraries not available - predictions will use heuristics")

    def add_training_data(self, metrics: PerformanceMetrics) -> None:
        """Add training data for model improvement."""
        self.training_data.append(metrics)
        logger.debug(f"Added training data point: {len(self.training_data)} total samples")

    def train_model(self, prediction_type: PredictionType, min_samples: int = 100) -> bool:
        """Train ML model for specific prediction type."""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available - cannot train models")
            return False

        if len(self.training_data) < min_samples:
            logger.warning(f"Insufficient training data: {len(self.training_data)} < {min_samples}")
            return False

        try:
            # Prepare training data
            X = np.array([metrics.to_feature_vector() for metrics in self.training_data])

            # Target variable based on prediction type
            if prediction_type == PredictionType.QUERY_EXECUTION_TIME:
                y = np.array([metrics.query_time for metrics in self.training_data])
            elif prediction_type == PredictionType.RESOURCE_USAGE:
                y = np.array(
                    [metrics.cpu_usage + metrics.memory_usage for metrics in self.training_data]
                )
            else:
                logger.error(f"Unsupported prediction type for training: {prediction_type}")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model based on type
            if self.model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
            elif self.model_type == ModelType.ISOLATION_FOREST:
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_train_scaled)
            else:
                logger.error(f"Model type {self.model_type} not implemented")
                return False

            # Evaluate model
            if hasattr(model, "predict") and self.model_type != ModelType.ISOLATION_FOREST:
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"Model trained - MSE: {mse:.4f}, R²: {r2:.4f}")

            # Store model and scaler
            self.models[prediction_type] = model
            self.scalers[prediction_type] = scaler

            logger.info(
                f"Successfully trained {self.model_type.value} model for {prediction_type.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return False

    async def predict(
        self,
        prediction_type: PredictionType,
        current_metrics: PerformanceMetrics,
        query: str | None = None,
    ) -> PredictionResult:
        """Make performance prediction."""
        try:
            if ML_AVAILABLE and prediction_type in self.models:
                return await self._ml_predict(prediction_type, current_metrics, query)
            else:
                return await self._heuristic_predict(prediction_type, current_metrics, query)

        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            # Return fallback prediction
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                features_used=[],
                model_version=self.model_version,
                metadata={"error": str(e)},
            )

    async def _ml_predict(
        self, prediction_type: PredictionType, metrics: PerformanceMetrics, query: str | None
    ) -> PredictionResult:
        """Make ML-based prediction."""
        model = self.models[prediction_type]
        scaler = self.scalers[prediction_type]

        # Prepare feature vector
        feature_vector = metrics.to_feature_vector().reshape(1, -1)
        scaled_features = scaler.transform(feature_vector)

        # Make prediction
        if self.model_type == ModelType.ISOLATION_FOREST:
            # Anomaly detection
            anomaly_score = model.decision_function(scaled_features)[0]
            is_anomaly = model.predict(scaled_features)[0] == -1
            predicted_value = float(anomaly_score)
            confidence = abs(anomaly_score)
        else:
            # Regression prediction
            predicted_value = float(model.predict(scaled_features)[0])

            # Calculate confidence based on model type
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(scaled_features))
            elif hasattr(model, "estimators_"):
                # For ensemble methods, use prediction variance
                predictions = [tree.predict(scaled_features)[0] for tree in model.estimators_]
                confidence = 1.0 - (
                    np.std(predictions) / np.mean(predictions) if np.mean(predictions) != 0 else 0
                )
            else:
                confidence = 0.8  # Default confidence

        return PredictionResult(
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=datetime.now(),
            features_used=[
                "cpu_usage",
                "memory_usage",
                "disk_io",
                "network_io",
                "active_connections",
                "query_complexity",
                "data_size",
                "result_rows",
            ],
            model_version=self.model_version,
            metadata={"model_type": self.model_type.value, "query_provided": query is not None},
        )

    async def _heuristic_predict(
        self, prediction_type: PredictionType, metrics: PerformanceMetrics, query: str | None
    ) -> PredictionResult:
        """Make heuristic-based prediction when ML is not available."""

        if prediction_type == PredictionType.QUERY_EXECUTION_TIME:
            # Simple heuristic based on system load and query complexity
            base_time = 0.1  # 100ms base

            # Adjust for system load
            load_factor = (metrics.cpu_usage + metrics.memory_usage) / 200.0

            # Adjust for query complexity
            if query:
                complexity = self.complexity_analyzer.analyze(query)
                complexity_factor = complexity / 10.0
            else:
                complexity_factor = 1.0

            # Adjust for data size
            size_factor = min(metrics.data_size / 1000000.0, 5.0)  # Cap at 5x

            predicted_time = (
                base_time * (1 + load_factor) * (1 + complexity_factor) * (1 + size_factor)
            )
            confidence = 0.6

        elif prediction_type == PredictionType.RESOURCE_USAGE:
            # Predict based on current trends
            predicted_usage = (metrics.cpu_usage + metrics.memory_usage) * 1.1
            confidence = 0.5
            predicted_time = predicted_usage

        else:
            predicted_time = 1.0
            confidence = 0.3

        return PredictionResult(
            prediction_type=prediction_type,
            predicted_value=predicted_time,
            confidence=confidence,
            timestamp=datetime.now(),
            features_used=["heuristic_based"],
            model_version=self.model_version,
            metadata={"method": "heuristic"},
        )

    def get_optimization_recommendations(
        self, metrics: PerformanceMetrics, predictions: list[PredictionResult]
    ) -> list[str]:
        """Generate optimization recommendations based on predictions."""
        recommendations = []

        for prediction in predictions:
            if prediction.prediction_type == PredictionType.QUERY_EXECUTION_TIME:
                if prediction.predicted_value > 5.0:  # > 5 seconds
                    recommendations.extend(
                        [
                            "Consider adding appropriate indexes for the query",
                            "Review query plan and optimize JOIN operations",
                            "Consider query result caching",
                        ]
                    )

            elif prediction.prediction_type == PredictionType.RESOURCE_USAGE:
                if prediction.predicted_value > 80.0:  # > 80% usage
                    recommendations.extend(
                        [
                            "Scale up server resources (CPU/Memory)",
                            "Implement query rate limiting",
                            "Consider read replicas for load distribution",
                        ]
                    )

        # General recommendations based on metrics
        if metrics.active_connections > 100:
            recommendations.append("High connection count - consider connection pooling")

        if metrics.disk_io > 80.0:
            recommendations.append("High disk I/O - consider SSD upgrade or data archiving")

        return list(set(recommendations))  # Remove duplicates


class VectorDatabaseManager:
    """Manager for vector database operations and RAG systems."""

    def __init__(self):
        self.chroma_client = None
        self.embeddings_model = None
        self.collection = None

        if VECTOR_DB_AVAILABLE:
            self._initialize_vector_db()

    def _initialize_vector_db(self):
        """Initialize vector database and embeddings model."""
        try:
            self.chroma_client = chromadb.Client()
            self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Create collection for query patterns
            self.collection = self.chroma_client.create_collection(
                name="query_patterns",
                metadata={"description": "SQL query patterns and performance data"},
            )

            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            VECTOR_DB_AVAILABLE = False

    async def store_query_pattern(
        self, query: str, performance_data: PerformanceMetrics, pattern_id: str | None = None
    ) -> str:
        """Store query pattern with performance data."""
        if not VECTOR_DB_AVAILABLE or not self.collection:
            logger.warning("Vector database not available")
            return ""

        try:
            # Generate embeddings for the query
            query_embedding = self.embeddings_model.encode([query])[0].tolist()

            # Create unique ID
            if not pattern_id:
                pattern_id = f"query_{hash(query)}_{int(datetime.now().timestamp())}"

            # Store in vector database
            self.collection.add(
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[
                    {
                        "execution_time": performance_data.query_time,
                        "cpu_usage": performance_data.cpu_usage,
                        "memory_usage": performance_data.memory_usage,
                        "timestamp": performance_data.timestamp.isoformat(),
                        "complexity": (
                            self.complexity_analyzer.analyze(query)
                            if hasattr(self, "complexity_analyzer")
                            else 0
                        ),
                    }
                ],
                ids=[pattern_id],
            )

            logger.debug(f"Stored query pattern: {pattern_id}")
            return pattern_id

        except Exception as e:
            logger.error(f"Error storing query pattern: {e}")
            return ""

    async def find_similar_queries(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find similar queries using vector similarity."""
        if not VECTOR_DB_AVAILABLE or not self.collection:
            return []

        try:
            # Generate embeddings for the input query
            query_embedding = self.embeddings_model.encode([query])[0].tolist()

            # Search for similar queries
            results = self.collection.query(query_embeddings=[query_embedding], n_results=limit)

            similar_queries = []
            for i, doc in enumerate(results["documents"][0]):
                similar_queries.append(
                    {
                        "query": doc,
                        "similarity": 1
                        - results["distances"][0][i],  # Convert distance to similarity
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                    }
                )

            return similar_queries

        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []


# Factory function for creating performance predictor
def create_performance_predictor(
    model_type: ModelType = ModelType.RANDOM_FOREST,
) -> PerformancePredictor:
    """Create performance predictor with specified model type."""
    return PerformancePredictor(model_type=model_type)


# Example usage
async def demonstrate_ai_ml_features():
    """Demonstrate AI/ML features."""
    print("AI/ML Performance Prediction Demo")
    print(f"ML Available: {ML_AVAILABLE}")
    print(f"Vector DB Available: {VECTOR_DB_AVAILABLE}")

    # Create performance predictor
    predictor = create_performance_predictor(ModelType.RANDOM_FOREST)

    # Sample performance metrics
    metrics = PerformanceMetrics(
        timestamp=datetime.now(),
        query_time=1.5,
        cpu_usage=45.0,
        memory_usage=60.0,
        disk_io=30.0,
        network_io=15.0,
        active_connections=25,
        query_complexity=3.5,
        data_size=1000000,
        result_rows=1000,
    )

    # Make prediction
    prediction = await predictor.predict(
        PredictionType.QUERY_EXECUTION_TIME,
        metrics,
        "SELECT * FROM system.parts WHERE active = 1 ORDER BY modification_time DESC",
    )

    print(
        f"Prediction: {prediction.predicted_value:.2f}s (confidence: {prediction.confidence:.2f})"
    )

    # Get recommendations
    recommendations = predictor.get_optimization_recommendations(metrics, [prediction])
    print(f"Recommendations: {recommendations}")

    # Demonstrate vector database if available
    if VECTOR_DB_AVAILABLE:
        vector_mgr = VectorDatabaseManager()
        if vector_mgr.collection:
            await vector_mgr.store_query_pattern("SELECT count() FROM system.parts", metrics)

            similar = await vector_mgr.find_similar_queries("SELECT count(*) FROM system.tables")
            print(f"Similar queries found: {len(similar)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_ai_ml_features())
