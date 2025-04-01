# Agent Zero: ClickHouse Monitoring MCP Server

Agent Zero is a Model Context Protocol (MCP) server for monitoring, analyzing, and managing ClickHouse databases. It enables AI assistants like Claude to perform sophisticated database operations, health checks, and troubleshooting on ClickHouse clusters. And more...

> **Note**: This project is currently in version 0.1.0 (early development).

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)](https://github.com/maruthiprithivi/agent_zero)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![Agent Zero](images/agent_zero.jpg)

## 🌟 Key Features

Agent Zero enables AI assistants to:

- **Query Performance Analysis**: Track slow queries, execution patterns, and bottlenecks
- **Resource Monitoring**: Monitor memory, CPU, and disk usage across the cluster
- **Table & Part Management**: Analyze table parts, merges, and storage efficiency
- **Error Investigation**: Identify and troubleshoot errors and exceptions
- **Health Checking**: Get comprehensive health status reports
- **Query Execution**: Run SELECT queries and analyze results safely

## 📋 Table of Contents

- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Module Breakdown](#-module-breakdown)
- [Environment Configuration](#-environment-configuration)
- [Development Guide](#-development-guide)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation & Setup

### Prerequisites

- Python 3.13 or higher
- Access to a ClickHouse database/cluster
- Claude AI assistant with MCP support

### Dependencies

Agent Zero relies on the following libraries:

- **mcp[cli]**: Core Model Context Protocol implementation (>=1.4.1)
- **clickhouse-connect**: ClickHouse client library (>=0.8.15)
- **python-dotenv**: Environment variable management (>=1.0.1)
- **uvicorn**: ASGI server for running the MCP service (>=0.34.0)
- **pydantic**: Data validation and settings management (>=2.10.6)
- **structlog**: Structured logging (>=25.2.0)
- **tenacity**: Retrying library (>=9.0.0)
- **aiohttp**: Asynchronous HTTP client/server (>=3.11.14)
- **prometheus-client**: Prometheus monitoring instrumentation (>=0.21.1)

### Using pip

```bash
pip install agent-zero

OR

uv pip install agent-zero
```

### Manual Installation

```bash
git clone https://github.com/maruthiprithivi/agent_zero.git
cd agent_zero
pip install -e .
```

### Environment Variables (This is not required while using Claude Desktop)

Agent Zero requires the following environment variables:

```bash
# Required
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password

# Optional (with defaults)
CLICKHOUSE_PORT=8443  # Default: 8443 if secure=true, 8123 if secure=false
CLICKHOUSE_SECURE=true  # Default: true
CLICKHOUSE_VERIFY=true  # Default: true
CLICKHOUSE_CONNECT_TIMEOUT=30  # Default: 30 seconds
CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300  # Default: 300 seconds
CLICKHOUSE_DATABASE=default  # Default: None
```

You can set these variables in your environment or use a `.env` file.

### Configuring Claude AI Assistant

#### Claude Desktop Configuration

1. Edit your Claude Desktop configuration file:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. Add the Agent Zero MCP server:

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "uv",
      "args": ["run", "--with", "agent_zero", "--python", "3.13", "agent_zero"],
      "env": {
        "CLICKHOUSE_HOST": "your-clickhouse-host",
        "CLICKHOUSE_PORT": "8443",
        "CLICKHOUSE_USER": "your-username",
        "CLICKHOUSE_PASSWORD": "your-password",
        "CLICKHOUSE_SECURE": "true",
        "CLICKHOUSE_VERIFY": "true",
        "CLICKHOUSE_CONNECT_TIMEOUT": "30",
        "CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "300"
      }
    }
  }
}
```

3. Restart Claude Desktop to apply the changes.

## 🔍 Usage Examples

### Basic Database Information

To get basic information about your ClickHouse databases and tables:

```
List all databases in my ClickHouse cluster
```

```
Show me all tables in the 'system' database
```

### Query Performance Analysis

To analyze query performance:

```
Show me the top 10 longest-running queries from the last 24 hours
```

```
Find queries that are consuming the most memory right now
```

```
Give me a breakdown of query types by hour for the past week
```

### Resource Usage Monitoring

To monitor resource usage:

```
Show memory usage trends across all hosts in my cluster for the past 3 days
```

```
What's the current CPU utilization across my ClickHouse cluster?
```

```
Give me a report on server sizing and resource allocation for all nodes
```

### Error Analysis

To investigate errors:

```
Show me recent errors in my ClickHouse cluster from the past 24 hours
```

```
Get the stack traces for LOGICAL_ERROR exceptions
```

```
Show error logs for query ID 'abc123'
```

### Health Check Reports

For comprehensive health checks:

```
Run a complete health check on my ClickHouse cluster
```

```
Are there any performance issues or bottlenecks in my ClickHouse setup?
```

```
Analyze my table parts and suggest optimization opportunities
```

## 📂 Project Structure

The project is organized as follows:

```
agent_zero/
├── __init__.py                # Package exports
├── main.py                    # Entry point for the MCP server
├── mcp_env.py                 # Environment configuration
├── mcp_server.py              # Main MCP server implementation
├── utils.py                   # Common utility functions
├── monitoring/                # Monitoring modules
│   ├── __init__.py            # Module exports
│   ├── error_analysis.py      # Error analysis tools
│   ├── insert_operations.py   # Insert operations monitoring
│   ├── parts_merges.py        # Parts and merges monitoring
│   ├── query_performance.py   # Query performance monitoring
│   ├── resource_usage.py      # Resource usage monitoring
│   ├── system_components.py   # System components monitoring
│   ├── table_statistics.py    # Table statistics tools
│   └── utility.py             # Utility functions
└── tests/                     # Test suite
    ├── __init__.py
    ├── conftest.py            # Test configuration
    ├── test_error_analysis.py # Tests for error analysis
    ├── test_query_performance.py # Tests for query performance
    ├── test_resource_usage.py # Tests for resource usage
    ├── test_tool.py           # Tests for basic tools
    └── utils.py               # Test utilities
```

## 🏗️ Architecture

Agent Zero follows a layered architecture:

1. **MCP Interface Layer** (`mcp_server.py`): Exposes functionality to Claude through the MCP protocol
2. **Monitoring Layer** (`monitoring/`): Specialized tools for different monitoring aspects
3. **Client Layer** (`mcp_env.py`, `utils.py`): Manages connection and interaction with ClickHouse
4. **Database Layer**: The ClickHouse database or cluster being monitored

Data flows as follows:

1. Claude sends a request to the MCP server
2. The MCP server routes the request to the appropriate tool
3. The tool uses the client layer to query ClickHouse
4. Results are processed and returned to Claude
5. Claude presents the information to the user

## 📊 Module Breakdown

### Core Modules

| Module          | Description                    | Key Features                                            |
| --------------- | ------------------------------ | ------------------------------------------------------- |
| `mcp_server.py` | Main MCP server implementation | Tool registration, request routing, client creation     |
| `mcp_env.py`    | Environment configuration      | Environment variable handling, configuration validation |
| `utils.py`      | Utility functions              | Retry mechanisms, logging, error formatting             |
| `main.py`       | Entry point                    | Server initialization and startup                       |

### Monitoring Modules

| Module                 | Description                 | Key Functions                                             |
| ---------------------- | --------------------------- | --------------------------------------------------------- |
| `query_performance.py` | Monitors query execution    | Current processes, duration stats, normalized query stats |
| `resource_usage.py`    | Tracks resource utilization | Memory usage, CPU usage, server sizing, uptime            |
| `parts_merges.py`      | Analyzes table parts        | Parts analysis, merge stats, partition statistics         |
| `error_analysis.py`    | Investigates errors         | Recent errors, stack traces, text log analysis            |
| `insert_operations.py` | Monitors inserts            | Async insert stats, written bytes distribution            |
| `system_components.py` | Monitors components         | Materialized views, blob storage, S3 queue stats          |
| `table_statistics.py`  | Analyzes tables             | Table stats, inactive parts analysis                      |
| `utility.py`           | Utility operations          | Drop tables scripts, monitoring view creation             |

## ⚙️ Environment Configuration

Agent Zero uses a typed configuration system for ClickHouse connection settings via the `ClickHouseConfig` class in `mcp_env.py`.

### Required Variables

- `CLICKHOUSE_HOST`: The hostname of the ClickHouse server
- `CLICKHOUSE_USER`: The username for authentication
- `CLICKHOUSE_PASSWORD`: The password for authentication

### Optional Variables

- `CLICKHOUSE_PORT`: The port number (default: 8443 if secure=True, 8123 if secure=False)
- `CLICKHOUSE_SECURE`: Enable HTTPS (default: true)
- `CLICKHOUSE_VERIFY`: Verify SSL certificates (default: true)
- `CLICKHOUSE_CONNECT_TIMEOUT`: Connection timeout in seconds (default: 30)
- `CLICKHOUSE_SEND_RECEIVE_TIMEOUT`: Send/receive timeout in seconds (default: 300)
- `CLICKHOUSE_DATABASE`: Default database to use (default: None)

### Configuration Usage

```python
from agent_zero.mcp_env import config

# Access configuration properties
host = config.host
port = config.port
secure = config.secure

# Get complete client configuration
client_config = config.get_client_config()
```

## 🛠️ Development Guide

### Setting Up Development Environment

1. Clone the repository:

```bash
git clone https://github.com/maruthiprithivi/agent_zero.git
cd agent_zero
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:

```bash
# With uv (recommended)
uv pip install -e .

# With pip
pip install -e .
```

Development dependencies include:

- **pytest**: Testing framework (>=8.3.5)

4. Set up environment variables for development:

```bash
# Create a .env file
cat > .env << EOF
CLICKHOUSE_HOST=localhost
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=password
CLICKHOUSE_SECURE=false
EOF
```

### Adding a New Monitoring Tool

1. Create or identify the appropriate module in the `monitoring/` directory.

2. Implement your monitoring function with proper error handling:

```python
# agent_zero/monitoring/your_module.py
import logging
from typing import Dict, List, Optional, Union, Any

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.utils import execute_query_with_retry, log_execution_time

logger = logging.getLogger("mcp-clickhouse")

@log_execution_time
def your_monitoring_function(
    client: Client,
    param1: str,
    param2: int = 10,
    settings: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Union[str, int, float]]]:
    """Your function description.

    Args:
        client: The ClickHouse client instance
        param1: Description of param1
        param2: Optional parameter (default: 10)
        settings: Optional query settings

    Returns:
        List of dictionaries with monitoring data
    """
    query = f"""
    SELECT
        column1,
        column2
    FROM your_table
    WHERE condition = '{param1}'
    LIMIT {param2}
    """

    logger.info(f"Retrieving data with param1={param1}, param2={param2}")

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error in your function: {str(e)}")
        # Optional fallback query if appropriate
        fallback_query = "SELECT 'fallback' AS result"
        logger.info("Using fallback query")
        return execute_query_with_retry(client, fallback_query, settings=settings)
```

3. **Export** your function in the module's `__init__.py`:

```python
# agent_zero/monitoring/__init__.py
from .your_module import your_monitoring_function

__all__ = [
    # ... existing exports
    "your_monitoring_function",
]
```

4. Add an MCP tool wrapper in `mcp_server.py`:

```python
# agent_zero/mcp_server.py
from agent_zero.monitoring import your_monitoring_function

@mcp.tool()
def monitor_your_feature(param1: str, param2: int = 10):
    """Description of your tool for Claude.

    Args:
        param1: Description of param1
        param2: Optional parameter (default: 10)

    Returns:
        Processed monitoring data
    """
    logger.info(f"Monitoring your feature with param1={param1}, param2={param2}")
    client = create_clickhouse_client()
    try:
        return your_monitoring_function(client, param1, param2)
    except Exception as e:
        logger.error(f"Error in your tool: {str(e)}")
        return f"Error monitoring your feature: {format_exception(e)}"
```

5. Write tests for your new functionality:

```python
# tests/test_your_module.py
from unittest.mock import MagicMock, patch
import unittest

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

from agent_zero.monitoring.your_module import your_monitoring_function
from tests.utils import create_mock_result

class TestYourModule(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=Client)
        self.mock_result = create_mock_result(
            column_names=["column1", "column2"],
            result_rows=[["value1", "value2"]]
        )
        self.mock_client.query.return_value = self.mock_result

    def test_your_monitoring_function(self):
        # Test basic functionality
        result = your_monitoring_function(self.mock_client, "test", 10)
        self.assertEqual(len(result), 1)
        self.mock_client.query.assert_called_once()

        # Test error handling
        self.mock_client.query.side_effect = ClickHouseError("Test error")
        self.mock_client.query.reset_mock()
        result = your_monitoring_function(self.mock_client, "test", 10)
        self.assertEqual(len(result), 1)  # Should return fallback result
```

### Code Style

This project follows these code style guidelines:

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://pep8.org/) guidelines for Python code
- Use type hints for all function parameters and return types
- Write comprehensive docstrings for all functions and classes
- Use meaningful variable and function names

## 🧪 Testing

### Running Tests

To run all tests:

```bash
python -m pytest
```

To run specific test files:

```bash
python -m pytest tests/test_query_performance.py
```

To run with coverage:

```bash
python -m pytest --cov=agent_zero
```

### Test Strategy

Tests are organized to match the module structure and include:

1. **Unit Tests**: Test individual functions in isolation with mocked dependencies
2. **Integration Tests**: Test interaction between components
3. **Mock Tests**: Use mock ClickHouse client to avoid external dependencies

### Test Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `mock_clickhouse_client`: A mocked ClickHouse client for testing
- `no_retry_settings`: Settings to disable query retries in tests

### Mock Utilities

The `tests/utils.py` file provides helpful utilities:

- `create_mock_result`: Creates mock query results for testing
- `assert_query_contains`: Compares queries while ignoring whitespace

## 🤝 Contributing

Contributions to Agent Zero are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Make your changes
4. Run tests: `python -m pytest`
5. Submit a pull request

Please follow the existing code style and add tests for any new functionality.

### Continuous Integration

Agent Zero uses GitHub Actions for continuous integration:

- **CI Workflow**: Automatically runs tests and linting on each push and pull request
- **Publish Workflow**: Handles publishing to PyPI when a new release is created

These workflows help maintain code quality and simplify the release process.

#### Testing GitHub Actions Locally

You can test GitHub Actions locally using [act](https://github.com/nektos/act):

1. Install act:

   ```bash
   # On macOS
   brew install act

   # On Linux
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   ```

2. Run the test script:

   ```bash
   # Run CI test job
   ./scripts/test-actions.sh test

   # Run CI lint job
   ./scripts/test-actions.sh lint

   # Run publish job
   ./scripts/test-actions.sh deploy
   ```

The script sets up the necessary configuration for act to run the workflows successfully.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔒 Security Considerations

- All queries are executed in read-only mode by default
- Ensure your ClickHouse user has appropriate permissions
- For production use, create a dedicated read-only user
- Always use HTTPS (secure=true) and SSL verification in production
- Store credentials securely and never hardcode them

## 📞 Support

If you encounter any issues or have questions, please file an issue on the [GitHub repository](https://github.com/maruthiprithivi/agent_zero/issues).
