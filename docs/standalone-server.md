# Agent Zero Standalone Server Guide

This guide provides detailed information on deploying and configuring Agent Zero as a standalone server.

## Overview

Agent Zero can be deployed as a standalone Model Context Protocol (MCP) server, allowing multiple clients (such as Claude Desktop or other AI assistants) to connect to it.

## Key Features

- **Command-line Configuration**: Customize host, port, and other settings via command line arguments
- **SSL/TLS Support**: Secure your connections with SSL certificates
- **Basic Authentication**: Protect your server with username/password authentication
- **Monitoring Endpoints**: Monitor server health and metrics with built-in endpoints
- **Prometheus Integration**: Track performance metrics in your Prometheus installation

## Installation

```bash
# Create a virtual environment
python3 -m venv /opt/agent-zero-env
source /opt/agent-zero-env/bin/activate

# Install Agent Zero
pip install ch-agent-zero
```

## Starting the Server

### Basic Usage

```bash
ch-agent-zero
```

This starts the server on the default host (127.0.0.1) and port (8505).

### Custom Host and Port

```bash
ch-agent-zero --host 0.0.0.0 --port 8505
```

This starts the server listening on all interfaces (0.0.0.0) on port 8505.

### Environment Variables

You can also use environment variables to configure the server:

```bash
# Server configuration
export MCP_SERVER_HOST=0.0.0.0
export MCP_SERVER_PORT=8505

# ClickHouse connection
export CLICKHOUSE_HOST=your-clickhouse-host
export CLICKHOUSE_PORT=8443
export CLICKHOUSE_USER=your-username
export CLICKHOUSE_PASSWORD=your-password
export CLICKHOUSE_SECURE=true
export CLICKHOUSE_VERIFY=true

# Start the server
ch-agent-zero
```

## Security Features

### Enabling SSL/TLS

To enable HTTPS with SSL/TLS:

```bash
ch-agent-zero --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

You can also use environment variables:

```bash
export MCP_SSL_CERTFILE=/path/to/cert.pem
export MCP_SSL_KEYFILE=/path/to/key.pem
```

### Enabling Basic Authentication

To protect your server with basic authentication:

```bash
# Option 1: Direct password (less secure)
ch-agent-zero --auth-username admin --auth-password your-password

# Option 2: Password file (more secure)
echo "your-secure-password" > /path/to/password-file
chmod 600 /path/to/password-file
ch-agent-zero --auth-username admin --auth-password-file /path/to/password-file
```

You can also use environment variables:

```bash
export MCP_AUTH_USERNAME=admin
export MCP_AUTH_PASSWORD=your-password
# OR
export MCP_AUTH_PASSWORD_FILE=/path/to/password-file
```

## Monitoring

Agent Zero provides built-in monitoring endpoints:

### Health Check

```bash
curl http://localhost:8505/health
```

Response example:

```json
{
  "status": "healthy",
  "server": "agent-zero",
  "clickhouse_connected": true,
  "clickhouse_version": "23.8.1.2992",
  "timestamp": 1683043234.892453
}
```

If authentication is enabled:

```bash
curl -u admin:your-password http://localhost:8505/health
```

### Prometheus Metrics

```bash
curl http://localhost:8505/metrics
```

This returns metrics in the Prometheus text format, which can be scraped by a Prometheus server.

Example Prometheus configuration:

```yaml
scrape_configs:
  - job_name: agent-zero
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["your-server:8505"]
    basic_auth:
      username: admin
      password: your-password
```

## Available Metrics

Agent Zero exposes the following metrics:

- **http_requests_total**: Total number of HTTP requests by method, endpoint, and status
- **http_request_duration_seconds**: HTTP request latency by method and endpoint
- **clickhouse_connection_status**: ClickHouse connection status (1=up, 0=down)
- **mcp_tool_calls_total**: Total number of MCP tool calls by tool
- **clickhouse_queries_total**: Total number of ClickHouse queries by type
- **clickhouse_query_errors_total**: Total number of ClickHouse query errors by type
- **clickhouse_query_duration_seconds**: Duration of ClickHouse queries by type

## Systemd Service (Linux)

For production deployments on Linux, create a systemd service:

```bash
cat > /etc/systemd/system/agent-zero.service << EOF
[Unit]
Description=Agent Zero MCP Server
After=network.target

[Service]
ExecStart=/opt/agent-zero-env/bin/ch-agent-zero --host 0.0.0.0 --port 8505 --auth-username admin --auth-password-file /etc/agent-zero/password.txt --ssl-certfile /etc/agent-zero/cert.pem --ssl-keyfile /etc/agent-zero/key.pem
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

## Docker Deployment

You can also deploy Agent Zero in a Docker container:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install Agent Zero
RUN pip install ch-agent-zero

# Expose the default port
EXPOSE 8505

# Set environment variables (optional)
ENV MCP_SERVER_HOST=0.0.0.0

# Start the server
CMD ["ch-agent-zero"]
```

Build and run:

```bash
docker build -t agent-zero .
docker run -p 8505:8505 \
  -e CLICKHOUSE_HOST=your-host \
  -e CLICKHOUSE_USER=your-user \
  -e CLICKHOUSE_PASSWORD=your-password \
  agent-zero
```

## Configuring MCP Clients

### Claude Desktop

Edit your Claude Desktop configuration file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add the following configuration for HTTP:

```json
{
  "mcpServers": {
    "agent-zero": {
      "http": {
        "url": "http://localhost:8505"
      }
    }
  }
}
```

Or for HTTPS with authentication:

```json
{
  "mcpServers": {
    "agent-zero": {
      "http": {
        "url": "https://your-server:8505",
        "auth": {
          "username": "admin",
          "password": "your-password"
        },
        "tls": {
          "verify": true
        }
      }
    }
  }
}
```

## Command-Line Reference

### `ch-agent-zero` Command

```
Usage: ch-agent-zero [OPTIONS]

Options:
  --host TEXT                Host to bind to (default: 127.0.0.1)
  --port INTEGER             Port to bind to (default: 8505)
  --ssl-certfile TEXT        Path to SSL certificate file
  --ssl-keyfile TEXT         Path to SSL key file
  --auth-username TEXT       Username for basic authentication
  --auth-password TEXT       Password for basic authentication (not recommended, use --auth-password-file instead)
  --auth-password-file TEXT  Path to file containing password for authentication
  --help                     Show this message and exit.
```

## Environment Variables Reference

### Server Configuration

| Environment Variable     | Description                       | Default     |
| ------------------------ | --------------------------------- | ----------- |
| `MCP_SERVER_HOST`        | Host to bind to                   | `127.0.0.1` |
| `MCP_SERVER_PORT`        | Port to bind to                   | `8505`      |
| `MCP_SSL_CERTFILE`       | Path to SSL certificate file      | None        |
| `MCP_SSL_KEYFILE`        | Path to SSL key file              | None        |
| `MCP_AUTH_USERNAME`      | Username for basic authentication | None        |
| `MCP_AUTH_PASSWORD`      | Password for basic authentication | None        |
| `MCP_AUTH_PASSWORD_FILE` | Path to file containing password  | None        |

### ClickHouse Connection

| Environment Variable              | Description                     | Default                         |
| --------------------------------- | ------------------------------- | ------------------------------- |
| `CLICKHOUSE_HOST`                 | ClickHouse server hostname      | Required                        |
| `CLICKHOUSE_PORT`                 | ClickHouse server port          | `8443` if secure, `8123` if not |
| `CLICKHOUSE_USER`                 | ClickHouse username             | Required                        |
| `CLICKHOUSE_PASSWORD`             | ClickHouse password             | Required                        |
| `CLICKHOUSE_SECURE`               | Use HTTPS for ClickHouse        | `true`                          |
| `CLICKHOUSE_VERIFY`               | Verify SSL certificates         | `true`                          |
| `CLICKHOUSE_CONNECT_TIMEOUT`      | Connection timeout in seconds   | `30`                            |
| `CLICKHOUSE_SEND_RECEIVE_TIMEOUT` | Send/receive timeout in seconds | `300`                           |

## Troubleshooting

### Connection Issues

If clients cannot connect to your server:

1. Verify the server is running with `ps aux | grep ch-agent-zero`
2. Check that the port is open with `netstat -tuln | grep 8505`
3. Ensure your firewall allows connections on the specified port
4. If using SSL, verify your certificate is valid and not expired

### Authentication Problems

If you're having authentication issues:

1. Verify the username and password match what's configured on the server
2. If using a password file, check its permissions and ensure it contains the expected password
3. Try using a simpler password temporarily to rule out character escaping issues

### ClickHouse Connection Problems

If the server can't connect to ClickHouse:

1. Verify your ClickHouse credentials and connection settings
2. Check that ClickHouse is running and accessible from the server
3. Try connecting to ClickHouse directly with `clickhouse-client` to verify connectivity
4. Check the server logs for specific error messages

---

For more information, see the [main documentation](README.md) or [testing documentation](testing/README.md).
