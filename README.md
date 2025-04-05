# Agent Zero: ClickHouse Monitoring MCP Server

Agent Zero is a Model Context Protocol (MCP) server for monitoring, analyzing, and managing ClickHouse databases. It enables AI assistants like Claude to perform sophisticated database operations, health checks, and troubleshooting on ClickHouse clusters. And more...

> **Note**: This project is currently in version 0.1.0 (early development).

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)](https://github.com/maruthiprithivi/agent_zero)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![Agent Zero](https://media.githubusercontent.com/media/maruthiprithivi/agent_zero/refs/heads/fix-mcp-entrypoint/images/agent_zero.jpg)

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

First, create and activate a virtual environment:

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

Then install the package:

```bash
# Using pip
pip install ch-agent-zero

# OR using uv (recommended)
# First install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then install the package
uv pip install ch-agent-zero
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

1. Create and activate a virtual environment for Claude Desktop (recommended):

```bash
# Create a new virtual environment
python3 -m venv ~/claude-desktop-env

# Activate the virtual environment
# On macOS/Linux:
source ~/claude-desktop-env/bin/activate
# On Windows:
%USERPROFILE%\claude-desktop-env\Scripts\activate

# Install the package in the virtual environment
pip install ch-agent-zero
# OR using uv
uv pip install ch-agent-zero
```

2. Edit your Claude Desktop configuration file:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

3. Add the Agent Zero MCP server:

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "~/claude-desktop-env/bin/ch-agent-zero",
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

For users who prefer using uv, the following configuration can also be used:

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "~/claude-desktop-env/bin/uv",
      "args": [
        "run",
        "--with",
        "ch-agent-zero",
        "--python",
        "3.13",
        "ch-agent-zero"
      ],
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

> **Note**: Make sure to keep your virtual environment activated while using Claude Desktop with Agent Zero. If you close your terminal, you'll need to reactivate the virtual environment before using Claude Desktop again.

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

## 🔄 Recent Updates

### Test Framework Improvements

The testing framework has been significantly improved to make it more robust and reliable:

1. **Import Style Refactoring**: Changed import patterns in test files from direct imports to module imports:

   ```python
   # Old style (prone to patching issues)
   from agent_zero.mcp_server import list_tables, execute_query

   # New style (more reliable for patching)
   import agent_zero.mcp_server as mcp
   ```

2. **Enhanced Mock Data Handling**: Improved mock implementations to handle all expected parameters and provide more realistic test data:

   ```python
   # Added support for settings parameter and query-specific responses
   def mock_query_response(query, settings=None):
       if "system.tables" in query:
           # Return tables-specific data
       elif "SELECT * FROM specific_table" in query:
           # Return query-specific data
   ```

3. **Fixed Test Cases**:

   - Resolved issues in the `test_mcp_core.py` module to properly test database operations
   - Fixed patching issues that were causing tests to fail
   - Improved error handling test cases

4. **Comprehensive Test Suite**: All 53 test cases now pass successfully, providing thorough coverage of the codebase

### Performance Improvements

1. **Query Optimization**: Enhanced query patterns for better performance across all monitoring tools
2. **Caching Strategy**: Implemented more efficient result caching for commonly accessed metrics
3. **Resource Utilization**: Reduced memory and CPU usage for monitoring operations

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

### Testing Best Practices

When developing for Agent Zero, follow these testing best practices:

1. **Proper Mocking**:

   - Use module-level imports in tests (e.g., `import agent_zero.mcp_server as mcp`)
   - Patch at the appropriate level (e.g., `patch("agent_zero.mcp_server.function_name")`)
   - Create realistic mock data that matches the expected structure
   - Mock HTTP responses and database queries to avoid external dependencies

2. **Handling Query Parameters**:

   - When mocking query responses, make sure to handle all parameters including `settings`
   - Use conditional mocking to return different responses based on query content

3. **Error Handling Testing**:

   - Test both success paths and error paths
   - Verify that error messages are properly formatted and contain useful information
   - Test timeout scenarios for long-running operations

4. **Setup/Teardown**:
   - Use proper setup and teardown methods to initialize and clean up resources
   - Remember to stop patchers in teardown methods to avoid leaking mocks

Example mocking pattern for ClickHouse client:

```python
def mock_query_response(query, settings=None):
    """Mock query response based on query content."""
    if "system.tables" in query:
        result = MagicMock()
        result.column_names = ["name", "comment"]
        result.result_rows = [
            ["table1", "Test table 1"],
            ["table2", "Test table 2"],
        ]
    elif "system.columns" in query:
        result = MagicMock()
        result.column_names = ["table", "name", "comment"]
        result.result_rows = [
            ["table1", "id", "ID column"],
            ["table1", "name", "Name column"],
        ]
    else:
        result = MagicMock()
        result.column_names = ["name", "type"]
        result.result_rows = [
            ["id", "UInt32"],
            ["name", "String"],
        ]
    return result

# Use in setup
self.mock_client.query.side_effect = mock_query_response
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
import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError

import agent_zero.mcp_server as mcp
from agent_zero.monitoring.your_module import your_monitoring_function
from tests.utils import create_mock_result

class TestYourModule(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)

        # Create comprehensive mock for query responses
        def mock_query_response(query, settings=None):
            # Return different results based on query content
            if "your_table" in query:
                result = MagicMock()
                result.column_names = ["column1", "column2"]
                result.result_rows = [
                    ["value1", "value2"],
                    ["value3", "value4"],
                ]
            else:
                result = MagicMock()
                result.column_names = ["result"]
                result.result_rows = [["fallback"]]
            return result

        self.mock_client.query.side_effect = mock_query_response

        # Set up client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_your_monitoring_function(self):
        """Test basic functionality."""
        result = your_monitoring_function(self.mock_client, "test", 10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["column1"], "value1")
        self.assertEqual(result[0]["column2"], "value2")

        # Test with different parameters
        result = your_monitoring_function(self.mock_client, "different", 5)
        self.assertEqual(len(result), 2)

        # Test error handling
        self.mock_client.query.side_effect = ClickHouseError("Test error")
        result = your_monitoring_function(self.mock_client, "test", 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["result"], "fallback")

    def test_monitor_your_feature(self):
        """Test the MCP tool wrapper."""
        # Test successful execution
        with patch("agent_zero.monitoring.your_module.your_monitoring_function") as mock_func:
            mock_func.return_value = [{"column1": "value1", "column2": "value2"}]

            result = mcp.monitor_your_feature("test", 10)
            self.assertEqual(len(result), 1)
            mock_func.assert_called_once_with(self.mock_client, "test", 10)

        # Test error handling
        with patch("agent_zero.monitoring.your_module.your_monitoring_function") as mock_func:
            mock_func.side_effect = Exception("Test exception")

            result = mcp.monitor_your_feature("test", 10)
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error monitoring your feature", result)
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

### Test Coverage

Agent Zero has a comprehensive test suite covering different aspects of the system. Here's a breakdown of the test modules and what they cover:

#### Core Functionality Tests

- **`test_mcp_core.py`**: Tests for core MCP server functionality
  - Connection management with ClickHouse
  - Database and table listing operations
  - Query execution and timeout handling
  - Error handling for core operations

#### Monitoring Tool Tests

- **`test_error_analysis.py`**: Tests for error analysis capabilities
  - Error stack trace retrieval and analysis
  - Recent error logging and monitoring
  - Text log analysis for debugging
- **`test_insert_operations.py`**: Tests for insert operation monitoring
  - Asynchronous insert statistics tracking
  - Insert bytes distribution analysis
  - Recent insert query monitoring
- **`test_mcp_monitoring_tools.py`**: Tests for MCP monitoring tools
  - Cluster sizing and resource analysis
  - Process monitoring and management
  - Error analysis and stack trace examination
  - Memory and CPU usage monitoring
  - Query pattern and performance analysis
  - System uptime tracking
- **`test_parts_merges.py`**: Tests for parts and merge operations
  - Current merge operation monitoring
  - Merge statistics analysis
  - Part log event tracking
  - Partition statistics analysis
  - Parts analysis for optimization
- **`test_query_performance.py`**: Tests for query performance analysis
  - Current process monitoring
  - Normalized query statistics
  - Query duration metrics
  - Query type breakdown analysis
- **`test_resource_usage.py`**: Tests for resource utilization tracking
  - CPU usage monitoring
  - Memory usage analysis
  - Server sizing assessment
  - System uptime tracking
- **`test_system_components.py`**: Tests for system component monitoring
  - Blob storage statistics analysis
  - Materialized view performance tracking
  - S3 queue status monitoring
- **`test_table_statistics.py`**: Tests for table statistics tools
  - Table inactive parts monitoring
  - Comprehensive table statistics analysis
- **`test_utility_tools.py`**: Tests for utility operations
  - Table drop script generation
  - User-defined function listing

#### Integration Tests

- **`test_tool.py`**: Comprehensive tests for tool integration
  - Database and table listing functionality
  - SELECT query execution and error handling
  - Table and column comment handling
  - Integration of monitoring tools with the MCP interface

### Test Design Best Practices

Our test suite follows these best practices:

1. **Proper Mocking**: Uses mocks to isolate units of code and avoid external dependencies
2. **Comprehensive Coverage**: Tests both success and failure paths
3. **Modular Design**: Tests are organized to match the structure of the codebase
4. **Clear Naming**: Test names clearly indicate what functionality is being tested
5. **Robust Setup/Teardown**: Each test properly initializes and cleans up resources
6. **Parameterized Tests**: Where appropriate, tests use parameters to cover multiple scenarios
7. **Focused Testing**: Each test focuses on a specific piece of functionality

### Recent Improvements

The test suite has recently been improved to:

1. **Fix Import Style**: Updated import patterns to make patching more reliable
2. **Enhance Mock Data**: Improved mock data handling for more robust tests
3. **Handle Edge Cases**: Better handling of error conditions and edge cases
4. **Support Settings Parameters**: Updated mocks to handle settings parameters properly

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
