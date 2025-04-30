# Agent Zero: ClickHouse Monitoring MCP Server

Agent Zero is a Model Context Protocol (MCP) server for monitoring, analyzing, and managing ClickHouse databases. It enables AI assistants like Claude to perform sophisticated database operations, health checks, and troubleshooting on ClickHouse clusters. And more...

> **Note**: This project is currently in version 0.0.1x (early development).
>
> **Important Update**: The project has been simplified by removing FastAPI and uvicorn dependencies. This change removes the following functionality:
>
> - HTTP API endpoints for monitoring (/health, /metrics)
> - Web-based dashboards and interfaces
> - Prometheus metrics collection and export via HTTP
> - Server-Sent Events (SSE) communication
> - HTTP-based authentication
>
> The core MCP functionality for ClickHouse monitoring remains intact and can be used through the native MCP interface.

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
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Module Breakdown](#-module-breakdown)
- [Environment Configuration](#-environment-configuration)
- [Development Guide](#-development-guide)
- [Testing](#-testing)
- [Documentation](#-documentation)
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
- **pydantic**: Data validation and settings management (>=2.10.6)
- **structlog**: Structured logging (>=25.2.0)
- **tenacity**: Retrying library (>=9.0.0)
- **aiohttp**: Asynchronous HTTP client/server (>=3.11.14)

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

You can set up Agent Zero with Claude Desktop using either pip (traditional) or uv (recommended). Choose the method that works best for you.

### Method 1: Using uv (Recommended)

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
        "CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "300"
      }
    }
  }
}
```

### Method 2: Using pip with Virtual Environment

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

> **Important Notes for Both Methods:**
>
> 1. Use the exact paths returned by the `which` or `where` commands
> 2. On Windows, convert backslashes (`\`) to forward slashes (`/`) in the configuration file
> 3. Make sure to replace `username` with your actual username in the paths
> 4. The virtual environment path can be customized, but make sure to use the correct path in the configuration

### Verification Steps

After setting up either method:

1. Verify the installation:

```bash
# Test the installation
ch-agent-zero --version
```

2. Restart Claude Desktop to apply the changes

3. Test the connection by asking Claude to perform a simple ClickHouse operation:

```bash
Show me the list of databases in my ClickHouse cluster
```

### Troubleshooting

If you encounter issues:

1. Verify that the paths in your configuration match the actual installation paths
2. Ensure the virtual environment (if using pip) is activated when testing
3. Check that all environment variables are correctly set
4. Make sure the ClickHouse connection details are correct

## 🚀 Deploying as a Standalone Server

You can deploy Agent Zero as a standalone MCP server, allowing multiple MCP clients (like Claude Desktop or other AI assistants) to connect to it. This is useful in scenarios where:

- You want to share a single Agent Zero instance across multiple users or devices
- You're deploying in an enterprise environment with centralized services
- You need to run the server on a different machine than your MCP clients

### Standalone Server Deployment

#### Step 1: Install Agent Zero

First, install Agent Zero on the server machine:

```bash
# Create a virtual environment
python3 -m venv /opt/agent-zero-env
source /opt/agent-zero-env/bin/activate

# Install Agent Zero
pip install ch-agent-zero
```

#### Step 2: Create a Configuration File

Create a configuration file for the server:

```bash
mkdir -p /etc/agent-zero
cat > /etc/agent-zero/config.env << EOF
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_SECURE=true
CLICKHOUSE_VERIFY=true
CLICKHOUSE_CONNECT_TIMEOUT=30
CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300
MCP_SERVER_HOST=0.0.0.0  # Listen on all interfaces
MCP_SERVER_PORT=8505     # MCP server port
EOF
```

#### Step 3: Create a Systemd Service (Linux)

For a production deployment on Linux, create a systemd service:

```bash
cat > /etc/systemd/system/agent-zero.service << EOF
[Unit]
Description=Agent Zero MCP Server
After=network.target

[Service]
ExecStart=/opt/agent-zero-env/bin/ch-agent-zero --host 0.0.0.0 --port 8505
WorkingDirectory=/opt/agent-zero
EnvironmentFile=/etc/agent-zero/config.env
User=agent-zero
Group=agent-zero
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agent-zero
sudo systemctl start agent-zero
```

#### Step 4: Verify the Service

Check that the service is running:

```bash
sudo systemctl status agent-zero

# Check the logs
sudo journalctl -u agent-zero -f
```

You can also test the MCP server directly:

```bash
curl http://localhost:8505/mcp/info
```

#### Docker Deployment (Alternative)

You can also deploy using Docker:

```bash
# Create a Dockerfile
cat > Dockerfile << EOF
FROM python:3.13-slim

WORKDIR /app

RUN pip install ch-agent-zero

ENV CLICKHOUSE_HOST=your-clickhouse-host
ENV CLICKHOUSE_PORT=8443
ENV CLICKHOUSE_USER=your-username
ENV CLICKHOUSE_PASSWORD=your-password
ENV CLICKHOUSE_SECURE=true
ENV CLICKHOUSE_VERIFY=true

EXPOSE 8505

CMD ["ch-agent-zero", "--host", "0.0.0.0", "--port", "8505"]
EOF

# Build and run the Docker image
docker build -t agent-zero .
docker run -d -p 8505:8505 --name agent-zero agent-zero
```

### Configuring MCP Clients to Use the Standalone Server

#### Claude Desktop Configuration

Edit your Claude Desktop configuration file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add the following configuration to connect to your standalone server:

```json
{
  "mcpServers": {
    "agent-zero": {
      "url": "http://your-server-ip:8505",
      "disableCommandExecution": true
    }
  }
}
```

Replace `your-server-ip` with the IP address or hostname of your standalone server.

#### Other MCP Clients

For other MCP clients, consult their documentation for how to configure external MCP servers. The key information you'll need to provide is:

- MCP Server URL: `http://your-server-ip:8505`
- Server name/identifier: `agent-zero` (or any name you prefer)

#### Security Considerations for Standalone Deployment

When deploying Agent Zero as a standalone server, consider these security measures:

1. **Use HTTPS**: For production deployments, configure the server with HTTPS:

   ```bash
   ch-agent-zero --host 0.0.0.0 --port 8505 --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
   ```

2. **Implement Authentication**: Add basic authentication to protect your MCP server:

   ```bash
   ch-agent-zero --host 0.0.0.0 --port 8505 --auth-username admin --auth-password-file /path/to/password_file
   ```

3. **Firewall Rules**: Restrict access to the MCP server port (8505) to only trusted clients.

4. **Reverse Proxy**: Consider placing the MCP server behind a reverse proxy like Nginx for additional security layers.

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
├── server_config.py           # Server configuration handling
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
    ├── stubs/                 # Stub implementations for tests
    ├── test_error_analysis.py # Tests for error analysis
    ├── test_query_performance.py # Tests for query performance
    ├── test_resource_usage.py # Tests for resource usage
    └── ...                    # Other test files
```

## 🏗️ Architecture

Agent Zero follows a simplified, streamlined architecture focused on the MCP protocol:

1. **MCP Interface Layer** (`mcp_server.py`):

   - Core of the application that exposes functionality to Claude through the MCP protocol
   - Provides a set of tools for interacting with and monitoring ClickHouse databases
   - Handles direct connections to ClickHouse via the FastMCP protocol without any HTTP intermediary

2. **Monitoring Layer** (`monitoring/`):

   - Specialized tools for different monitoring aspects
   - Modular design makes it easy to add new monitoring capabilities

3. **Client Layer** (`mcp_env.py`, `utils.py`):

   - Manages connection and interaction with ClickHouse
   - Handles environment variables and configuration

4. **Database Layer**:
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

### Server Configuration

Agent Zero has additional configuration options for the MCP server itself:

- `MCP_SERVER_HOST`: Host to bind to (default: 127.0.0.1)
- `MCP_SERVER_PORT`: Port to bind to (default: 8505)
- `MCP_SSL_CERTFILE`: SSL certificate file path (default: None)
- `MCP_SSL_KEYFILE`: SSL key file path (default: None)
- `MCP_AUTH_USERNAME`: Basic auth username (default: None)
- `MCP_AUTH_PASSWORD`: Basic auth password (default: None)
- `MCP_AUTH_PASSWORD_FILE`: Path to file containing basic auth password (default: None)

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

5. Write tests for your new functionality.

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
- Always use HTTPS (secure=true) and SSL verification in production
- Store credentials securely and never hardcode them

## 📞 Support

If you encounter any issues or have questions, please file an issue on the [GitHub repository](https://github.com/maruthiprithivi/agent_zero/issues).

## 📚 Documentation

All documentation is in the `/docs` directory:

- [Documentation Index](/docs/README.md)
- [Standalone Server](/docs/standalone-server.md)
- [Logging](/docs/logging.md)
- [Testing](/docs/testing/)

For reference: [Command-line Arguments](/docs/standalone-server.md#command-line-reference), [Environment Variables](/docs/standalone-server.md#environment-variables-reference)
