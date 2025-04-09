# Agent Zero Testing Guide

This document covers testing Agent Zero and its standalone server features.

## Testing Resources

- [Test Coverage](coverage.md) - Coverage information for standalone server features
- [MCP Endpoints Testing](mcp-endpoints.md) - How to test MCP endpoints

## Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest

# Run standalone feature tests
python -m pytest tests/test_mock_*.py -v

# Run with coverage
python -m pytest tests/test_mock_*.py --cov=agent_zero
```

### Test Dependencies

```bash
pip install -r requirements-test.txt
```

## Standalone Server Testing Approach

We use mock modules to isolate components for testing:

- **Server Configuration**: Host, port, SSL/TLS, authentication settings
- **Monitoring Endpoints**: Health and metrics endpoints
- **Query Metrics**: Prometheus metrics tracking

When extending the server, maintain isolation between components to prevent Prometheus metric conflicts.

---

For more information, see the [main documentation](../README.md) or [standalone server documentation](../standalone-server.md).
