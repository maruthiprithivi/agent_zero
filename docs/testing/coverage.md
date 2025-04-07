# Test Coverage for Standalone Server Features

This document summarizes test coverage for Agent Zero's standalone server features.

## Features Tested

### 1. Server Configuration (`server_config.py`)

- Default values, environment variables, and command-line overrides
- SSL/TLS and authentication configuration
- Password file handling

**Test File**: `tests/test_mock_server_config.py`

### 2. Query Metrics (`query_metrics.py`)

- SQL query type extraction and comment handling
- Tracking for successful and failed queries
- Different parameter passing styles

**Test File**: `tests/test_mock_query_metrics.py`

### 3. Monitoring Endpoints (`monitoring_endpoints.py`)

- Health check endpoint with and without authentication
- Metrics endpoint with and without authentication
- Error handling for ClickHouse connection failures

**Test File**: `tests/test_mock_monitoring_endpoints.py`

## Testing Approach

To avoid conflicts with Prometheus metrics, we use:

- **Isolated Mock Modules** - Avoiding global state and dependencies
- **Fixture-Based Testing** - Clean test state management
- **Dependency Injection** - For controlled behavior testing

## Run Coverage Reports

```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/test_mock_*.py --cov=agent_zero --cov-report=html
```

---

For more information, see the [testing documentation index](README.md) or [MCP endpoints testing guide](mcp-endpoints.md).
