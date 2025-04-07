"""Test configuration and fixtures."""

import os
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

# Set up mock environment variables for tests
# This allows tests to run even if the actual env vars aren't set
if "CLICKHOUSE_HOST" not in os.environ:
    os.environ["CLICKHOUSE_HOST"] = "mock_host"
if "CLICKHOUSE_USER" not in os.environ:
    os.environ["CLICKHOUSE_USER"] = "mock_user"
if "CLICKHOUSE_PASSWORD" not in os.environ:
    os.environ["CLICKHOUSE_PASSWORD"] = "mock_password"


@pytest.fixture(autouse=True)
def clear_prometheus_metrics():
    """Clear all Prometheus metrics before each test.

    This helps avoid 'Duplicated timeseries in CollectorRegistry' errors
    when running tests that import modules with Prometheus metrics.
    """
    # Get collectors from registry
    collectors = list(REGISTRY._collector_to_names.keys())

    # Remove each collector
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            # Collector might have been removed by another test
            pass

    yield

    # Clean up again after the test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


@pytest.fixture
def no_retry_settings():
    """Fixture to provide settings that disable query retries."""
    return {"disable_retries": True}


@pytest.fixture
def mock_clickhouse_client():
    """Fixture to provide a mock ClickHouse client."""
    from clickhouse_connect.driver.client import Client

    mock_client = MagicMock(spec=Client)
    return mock_client
