"""Test configuration and fixtures."""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set up mock environment variables for tests
# This allows tests to run even if the actual env vars aren't set
if "CLICKHOUSE_HOST" not in os.environ:
    os.environ["CLICKHOUSE_HOST"] = "mock_host"
if "CLICKHOUSE_USER" not in os.environ:
    os.environ["CLICKHOUSE_USER"] = "mock_user"
if "CLICKHOUSE_PASSWORD" not in os.environ:
    os.environ["CLICKHOUSE_PASSWORD"] = "mock_password"

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