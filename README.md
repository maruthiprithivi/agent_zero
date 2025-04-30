# Agent Zero: ClickHouse Monitoring MCP Server

Agent Zero is a Model Context Protocol (MCP) server for monitoring, analyzing, and managing ClickHouse databases. It enables AI assistants like Claude to perform sophisticated database operations, health checks, and troubleshooting on ClickHouse clusters. And more...

> **Note**: This project is currently in version 0.0.1x (early development).
>
> **Important Update**: The project has been simplified by removing FastAPI and uvicorn dependencies. This change removes the following functionality:
>
> - HTTP API endpoints for monitoring (/health, /metrics)
> - Web-based dashboards and interfaces
> - HTTP-based authentication
>
> The core MCP functionality for ClickHouse monitoring remains intact and can be used through the native MCP interface or via Server-Sent Events (SSE) for remote connections.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.0.1x-brightgreen.svg)](https://github.com/maruthiprithivi/agent_zero)
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
- [Claude Desktop Configuration](#configuring-claude-ai-assistant)
- [Cursor IDE Configuration](#cursor-ide-configuration)
- [Standalone Server](#-standalone-server)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Module Breakdown](#-module-breakdown)
- [Environment Configuration](#-environment-configuration)
- [Logging and Tracing](#-logging-and-tracing)
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

Agent Zero relies on the following core libraries:

- **mcp[cli]**: Core Model Context Protocol implementation (>=1.4.1)
- **clickhouse-connect**: ClickHouse client library (>=0.8.15)
- **python-dotenv**: Environment variable management (>=1.0.1)
- **pydantic**: Data validation and settings management (>=2.10.6)
- **structlog**: Structured logging (>=25.2.0)
- **tenacity**: Retrying library (>=9.0.0)
- **aiohttp**: Asynchronous HTTP client (>=3.11.14)

Development dependencies:

- **pytest**: Testing framework (>=8.3.5)
- **black**: Code formatting (>=23.12.1)
- **ruff**: Fast Python linter

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

### Environment Variables

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

# Logging and Tracing (Optional)
CLICKHOUSE_ENABLE_QUERY_LOGGING=false  # Enable detailed query logging
CLICKHOUSE_LOG_QUERY_LATENCY=false    # Log query execution times
CLICKHOUSE_LOG_QUERY_ERRORS=true      # Log query errors
CLICKHOUSE_LOG_QUERY_WARNINGS=true    # Log query warnings
MCP_ENABLE_TRACING=false              # Enable MCP communication tracing

# MCP Server Configuration (Optional)
MCP_SERVER_HOST=127.0.0.1             # Host to bind to
MCP_SERVER_PORT=8505                  # Port to bind to
MCP_CURSOR_MODE=agent                 # Cursor IDE mode (agent, ask, edit)
MCP_CURSOR_TRANSPORT=sse              # Transport for Cursor IDE (sse, websocket)
```

You can set these variables in your environment or use a `.env` file.

### Configuring Claude AI Assistant

#### Claude Desktop Configuration

You can set up Agent Zero with Claude Desktop using either pip (traditional) or uv (recommended).

##### Method 1: Using uv (Recommended)

1. Install uv:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Find your uv installation path:

**On macOS/Linux:**

```bash
# This will show your uv installation path
which uv
# Example output: /Users/username/.cargo/bin/uv
```

**On Windows:**

```cmd
# Open Command Prompt or PowerShell and run:
where uv
# Example output: C:\Users\username\.cargo\bin\uv.exe
```

3. Configure Claude Desktop with uv:

Edit your Claude Desktop configuration file based on your OS:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add the following configuration (replace `<UV_PATH>` with the output from step 2):

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "<UV_PATH>",
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
        "CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "300",
        "MCP_ENABLE_TRACING": "false"
      }
    }
  }
}
```

### Cursor IDE Configuration

Agent Zero can be integrated with Cursor IDE using the same uv-based approach recommended for Claude Desktop.

#### Prerequisites

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Find your uv installation path:

```bash
which uv
# Example output: /Users/username/.cargo/bin/uv
```

#### Adding Agent Zero to Cursor

1. Open Cursor IDE settings
2. Navigate to the MCP Servers section
3. Add a new MCP server with the following configuration:

```json
{
  "name": "agent-zero",
  "command": "<UV_PATH>",
  "args": [
    "run",
    "--with",
    "ch-agent-zero",
    "--python",
    "3.13",
    "ch-agent-zero",
    "--cursor-mode",
    "agent"
  ],
  "env": {
    "CLICKHOUSE_HOST": "your-clickhouse-host",
    "CLICKHOUSE_PORT": "8443",
    "CLICKHOUSE_USER": "your-username",
    "CLICKHOUSE_PASSWORD": "your-password",
    "CLICKHOUSE_SECURE": "true",
    "CLICKHOUSE_VERIFY": "true",
    "CLICKHOUSE_CONNECT_TIMEOUT": "30",
    "CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "300",
    "MCP_ENABLE_TRACING": "false"
  }
}
```

Replace `<UV_PATH>` with the full path to your uv installation.

#### Cursor IDE Modes

Agent Zero supports different Cursor IDE modes by changing the `--cursor-mode` argument:

- **Agent Mode**: `--cursor-mode agent` - Provides full tool execution with all ClickHouse monitoring capabilities
- **Ask Mode**: `--cursor-mode ask` - Offers information retrieval tools for analyzing ClickHouse performance
- **Edit Mode**: `--cursor-mode edit` - Helps generate and modify ClickHouse queries

#### Cursor Transport Options

By default, Agent Zero uses the SSE (Server-Sent Events) transport for Cursor IDE. You can specify the transport with the `--cursor-transport` argument:

- **SSE**: `--cursor-transport sse` (default)
- **WebSocket**: `--cursor-transport websocket`

##### Method 2: Using pip with Virtual Environment

1. Create and activate a virtual environment for Claude Desktop:

**On macOS/Linux:**

```bash
# Create virtual environment
python3 -m venv ~/claude-desktop-env

# Activate virtual environment
source ~/claude-desktop-env/bin/activate
```

**On Windows:**

```cmd
# Create virtual environment
python -m venv %USERPROFILE%\claude-desktop-env

# Activate virtual environment
%USERPROFILE%\claude-desktop-env\Scripts\activate
```

2. Install Agent Zero in the virtual environment:

```bash
pip install ch-agent-zero
```

3. Find the ch-agent-zero installation path:

**On macOS/Linux:**

```bash
# This will show your ch-agent-zero installation path
which ch-agent-zero
# Example output: /Users/username/claude-desktop-env/bin/ch-agent-zero
```

**On Windows:**

```cmd
# This will show your ch-agent-zero installation path
where ch-agent-zero
# Example output: C:\Users\username\claude-desktop-env\Scripts\ch-agent-zero.exe
```

4. Configure Claude Desktop with pip:

Edit your Claude Desktop configuration file (same locations as mentioned above) and add:

**For macOS/Linux:**

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "<PATH_TO_YOUR_VENV>/bin/ch-agent-zero",
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

**For Windows:**

```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "<PATH_TO_YOUR_VENV>/Scripts/ch-agent-zero.exe",
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

#### Verification and Troubleshooting

After setting up either method:

1. Verify the installation:

```bash
# Test the installation
ch-agent-zero --version
```

2. Restart Claude Desktop to apply the changes

3. Test the connection by asking Claude to perform a simple ClickHouse operation:

```
Show me the list of databases in my ClickHouse cluster
```

If you encounter issues:

1. Enable tracing by adding `"MCP_ENABLE_TRACING": "true"` to your environment variables
2. Check console output in Claude Desktop for errors
3. Verify that the ClickHouse connection details are correct
4. Make sure the paths in your configuration match the actual installation paths

## 🧰 Command-Line Usage

Agent Zero can be used directly from the command line for testing or development purposes:

```bash
# Basic usage (runs with default host/port)
ch-agent-zero

# Specify host and port
ch-agent-zero --host 127.0.0.1 --port 8505

# Display version information
ch-agent-zero --version

# Show help
ch-agent-zero --help
```

These command-line options are primarily for debugging and development, as Agent Zero is designed to be used with Claude Desktop.

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

## 🖥️ Standalone Server

You can run Agent Zero as a standalone server using the following command:

```bash
ch-agent-zero [options]
```

### Server Options

| Option                 | Description                                         | Default   |
| ---------------------- | --------------------------------------------------- | --------- |
| `--host`               | Host to bind to                                     | 127.0.0.1 |
| `--port`               | Port to bind to                                     | 8505      |
| `--auth-username`      | Username for basic authentication                   | None      |
| `--auth-password`      | Password for basic authentication                   | None      |
| `--auth-password-file` | Path to file containing password for authentication | None      |

### Cursor IDE Integration Options

| Option               | Description                                       | Default |
| -------------------- | ------------------------------------------------- | ------- |
| `--cursor-mode`      | Cursor IDE mode (agent, ask, edit)                | None    |
| `--cursor-transport` | Transport to use with Cursor IDE (sse, websocket) | sse     |

### Examples

Standard server:

```bash
ch-agent-zero --host 0.0.0.0 --port 8080
```

With authentication:

```bash
ch-agent-zero --auth-username admin --auth-password-file /path/to/password.txt
```

For Cursor IDE in agent mode:

```bash
ch-agent-zero --cursor-mode agent
```

For Cursor IDE with WebSocket transport:

```bash
ch-agent-zero --cursor-mode agent --cursor-transport websocket
```

## 📂 Project Structure

The project is organized as follows:

```
agent_zero/
├── __init__.py                # Package exports
├── main.py                    # Entry point for the MCP server
├── mcp.py                     # MCP protocol definitions
├── mcp_env.py                # Environment configuration
├── mcp_server.py             # Main MCP server implementation
├── mcp_tracer.py             # MCP tracing functionality
├── database_logger.py         # Database logging utilities
├── server_config.py          # Server configuration handling
├── utils.py                  # Common utility functions
├── monitoring/               # Monitoring modules
│   ├── __init__.py           # Module exports
│   ├── error_analysis.py     # Error analysis tools
│   ├── insert_operations.py  # Insert operations monitoring
│   ├── parts_merges.py       # Parts and merges monitoring
│   ├── query_performance.py  # Query performance monitoring
│   ├── resource_usage.py     # Resource usage monitoring
│   ├── system_components.py  # System components monitoring
│   ├── table_statistics.py   # Table statistics tools
│   └── utility.py            # Utility functions
└── tests/                    # Test suite
    ├── __init__.py
    ├── conftest.py           # Test configuration
    ├── stubs/                # Stub implementations for tests
    ├── test_error_analysis.py # Tests for error analysis
    ├── test_query_performance.py # Tests for query performance
    ├── test_resource_usage.py # Tests for resource usage
    └── ...                   # Other test files
```

## 🏗️ Architecture

Agent Zero follows a simplified, streamlined architecture focused on the MCP protocol:

1. **MCP Interface Layer** (`mcp_server.py`):

   - Core of the application that exposes functionality to Claude through the MCP protocol
   - Provides a set of tools for interacting with and monitoring ClickHouse databases
   - Handles direct connections to ClickHouse via the FastMCP protocol
   - Smart transport selection based on configuration (stdio for default settings, SSE for custom host/port)

2. **Monitoring Layer** (`monitoring/`):

   - Specialized tools for different monitoring aspects
   - Modular design makes it easy to add new monitoring capabilities

3. **Client Layer** (`mcp_env.py`, `utils.py`):

   - Manages connection and interaction with ClickHouse
   - Handles environment variables and configuration

4. **Tracing Layer** (`mcp_tracer.py`):

   - Optional tracing of MCP communications
   - Detailed logging for debugging and diagnostics

5. **Database Layer**:
   - The ClickHouse database or cluster being monitored

Data flows directly from Claude to the ClickHouse database:

1. Claude sends a request to the MCP server
2. The MCP server routes the request to the appropriate tool
3. The tool uses the client layer to query ClickHouse directly
4. Results are processed and returned to Claude
5. Claude presents the information to the user

### Key Architectural Components

- **FastMCP Integration**: Direct integration with the FastMCP protocol for efficient communication with Claude
- **Threaded Query Execution**: Handles long-running queries in separate threads to prevent blocking
- **Connection Management**: Efficient connection handling with timeouts and retries
- **Typed Configuration**: Type-safe configuration management via environment variables
- **Comprehensive Toolset**: Wide range of monitoring tools exposed through a unified interface
- **Test Environment Detection**: Smart detection of test environments to support automated testing
- **Transport Protocol Selection**: Automatic selection between stdio and SSE transport based on configuration

## 📊 Module Breakdown

### Core Modules

| Module             | Description                    | Key Features                                            |
| ------------------ | ------------------------------ | ------------------------------------------------------- |
| `mcp_server.py`    | Main MCP server implementation | Tool registration, request routing, client creation     |
| `mcp_env.py`       | Environment configuration      | Environment variable handling, configuration validation |
| `utils.py`         | Utility functions              | Retry mechanisms, logging, error formatting             |
| `main.py`          | Entry point                    | Server initialization and startup                       |
| `server_config.py` | Server configuration handling  | Configuration for host, port, SSL, and auth             |

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

## 📝 Logging and Tracing

Agent Zero provides detailed logging and tracing options for debugging and monitoring.

### Database Query Logging

Tracks database queries, execution times, errors, and warnings.

| Environment Variable              | Description                   | Default |
| --------------------------------- | ----------------------------- | ------- |
| `CLICKHOUSE_ENABLE_QUERY_LOGGING` | Enable detailed query logging | `false` |
| `CLICKHOUSE_LOG_QUERY_LATENCY`    | Log query execution times     | `false` |
| `CLICKHOUSE_LOG_QUERY_ERRORS`     | Log query errors              | `true`  |
| `CLICKHOUSE_LOG_QUERY_WARNINGS`   | Log query warnings            | `true`  |

To enable:

```bash
# In shell
export CLICKHOUSE_ENABLE_QUERY_LOGGING=true
export CLICKHOUSE_LOG_QUERY_LATENCY=true

# Or in .env file
CLICKHOUSE_ENABLE_QUERY_LOGGING=true
CLICKHOUSE_LOG_QUERY_LATENCY=true
```

### MCP Server Tracing

Tracks MCP tool calls, request payloads, responses, and execution times.

| Environment Variable | Description        | Default |
| -------------------- | ------------------ | ------- |
| `MCP_ENABLE_TRACING` | Enable MCP tracing | `false` |

To enable:

```bash
# In shell
export MCP_ENABLE_TRACING=true

# Or in .env file
MCP_ENABLE_TRACING=true
```

### Test Environment Detection

Agent Zero includes built-in detection for test environments to support automated testing. This helps:

- Prevent infinite recursion issues when mocking the MCP server
- Properly handle parameter passing in test vs. production scenarios
- Support both real-world and testing scenarios with the same codebase

No additional configuration is needed for this functionality as it's automatically enabled.

### Performance Considerations

- Both logging systems can impact performance with high query volumes
- Enable only when needed for debugging or monitoring
- For production, consider enabling only error logging rather than full query logging
- Test environment detection adds minimal overhead to server startup

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
uv pip install -e ".[dev]"

# With pip
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

5. Set up environment variables for development:

```bash
# Create a .env file
cat > .env << EOF
CLICKHOUSE_HOST=localhost
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=password
CLICKHOUSE_SECURE=false
CH_AGENT_ZERO_DEBUG=1  # Enables development mode with default values
MCP_ENABLE_TRACING=true  # Enable tracing for development
EOF
```

### Code Style

This project follows these code style guidelines:

- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for fast Python linting
- Follow [PEP 8](https://pep8.org/) guidelines for Python code
- Use type hints for all function parameters and return types
- Write comprehensive docstrings for all functions and classes
- Use meaningful variable and function names

### Adding a New Monitoring Tool

1. Identify the appropriate module in the `monitoring/` directory or create a new one.

2. Implement your monitoring function with proper error handling:

```python
# agent_zero/monitoring/your_module.py
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError
from typing import Dict, List, Optional, Union, Any

from agent_zero.utils import execute_query_with_retry, log_execution_time
import logging

logger = logging.getLogger("mcp-clickhouse")

@log_execution_time
def your_monitoring_function(
    client: Client,
    param1: str,
    param2: int = 10,
    settings: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
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

    try:
        return execute_query_with_retry(client, query, settings=settings)
    except ClickHouseError as e:
        logger.error(f"Error in your function: {str(e)}")
        return []
```

3. Export your function in the module's `__init__.py`:

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
from agent_zero.mcp_tracer import trace_mcp_call

@mcp.tool()
@trace_mcp_call
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

5. Write tests for your new functionality.

### Working with FastMCP and Transport Protocols

When modifying the MCP server functionality, be aware of these important considerations:

1. **Avoid Recursive Function Calls**: Be careful when overriding or extending the `mcp.run()` method to prevent infinite recursion.

2. **Test Environment Detection**: The server includes logic to detect when it's running in a test environment vs. production:

   ```python
   # Example of how test environment detection works
   if mcp is not _original_mcp:
       # We're in a test environment with a mocked mcp
       return mcp.run(host=host, port=port, **ssl_args)
   else:
       # We're in a real production environment
       # Use appropriate transport
   ```

3. **Transport Protocol Selection**:

   - The server automatically uses the appropriate transport protocol based on configuration
   - Default configuration (127.0.0.1:8505) uses stdio transport
   - Custom host/port configuration uses SSE transport for network communication

4. **SSL Argument Handling**: When working with SSL configuration, ensure arguments are properly passed to the underlying FastMCP implementation.

## 🧪 Testing

### Testing Strategy

The tests are designed to validate the following aspects of the system:

1. **Core MCP Functionality**: Testing the MCP server implementation and tool registration
2. **ClickHouse Integration**: Testing the interaction with ClickHouse databases through the client
3. **Monitoring Tools**: Testing the various monitoring tools for accuracy and performance
4. **Configuration**: Testing the configuration handling and environment variable processing
5. **Error Handling**: Testing the system's behavior when errors occur
6. **Transport Protocols**: Testing different transport protocols (stdio and SSE)
7. **Test Environment Detection**: Ensuring the server correctly identifies and adapts to test contexts

### Test Isolation

The testing approach emphasizes test isolation to prevent tests from interfering with each other. This is achieved through:

- **Mock ClickHouse Client**: Using mock clients to avoid actual database connections
- **Environment Variable Isolation**: Resetting environment variables between tests
- **Independent Test Fixtures**: Creating fresh fixtures for each test
- **MCP Server Mocking**: Properly mocking the MCP server to avoid recursion issues

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

### Test Organization

Tests are organized to match the module structure and include:

1. **Unit Tests**: Test individual functions in isolation with mocked dependencies
2. **Integration Tests**: Test interaction between components
3. **Mock Tests**: Use mock ClickHouse client to avoid external dependencies
4. **Server Tests**: Validate server functionality with different configurations and transports

### Common Test Fixtures

Common test fixtures are defined in `conftest.py` and include:

- `no_retry_settings`: Fixture to disable query retries
- `mock_clickhouse_client`: Fixture to provide a mock ClickHouse client
- `mock_mcp_server`: Fixture to provide a properly mocked MCP server that avoids recursion issues

### Writing Tests for MCP Server Functionality

When writing tests that involve the MCP server:

1. **Use the mock_mcp_server fixture**: This fixture properly mocks the MCP server to avoid recursion issues.

2. **Test different transport configurations**:

   ```python
   def test_server_with_default_config():
       # Test with default configuration (stdio transport)

   def test_server_with_custom_host_port():
       # Test with custom host/port (SSE transport)
   ```

3. **Check SSL argument passing**: Ensure SSL arguments are correctly passed to the underlying FastMCP implementation.

4. **Test authentication configurations**: Verify that authentication configurations are properly handled.

## 🤝 Contributing

Contributions to Agent Zero are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Make your changes
4. Run tests: `python -m pytest`
5. Submit a pull request

Please follow the existing code style and add tests for any new functionality.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔒 Security Considerations

- All queries are executed in read-only mode by default
- Ensure your ClickHouse user has appropriate permissions
- For production use, create a dedicated read-only user
- Store credentials securely and never hardcode them
- Use the `settings` parameter to control query behavior safely
- Enable tracing for debugging but disable in production environments
- Set reasonable timeouts for connection and queries

## 📞 Support

If you encounter any issues or have questions, please file an issue on the [GitHub repository](https://github.com/maruthiprithivi/agent_zero/issues).
