# Agent Zero MCP Server - Deployment Guide (2025 Multi-IDE Edition)

This guide covers all deployment options for Agent Zero, from local development to enterprise deployments with multiple IDE integrations.

## 🚀 Quick Start

### Automated Installation

```bash
# Download and run the universal installer
curl -sSL https://raw.githubusercontent.com/maruthiprithivi/agent_zero/main/scripts/install.sh | bash

# Or clone and run locally
git clone https://github.com/maruthiprithivi/agent_zero.git
cd agent_zero
chmod +x scripts/install.sh
./scripts/install.sh
```

### Manual Installation

```bash
# Using uv (recommended)
uv tool install ch-agent-zero

# Using pip
pip install ch-agent-zero
```

## 📋 Deployment Modes

Agent Zero supports three deployment modes, each optimized for different use cases:

### 1. Local Mode (Default)
Perfect for development and local IDE integration.

```bash
# Basic local mode
ch-agent-zero

# With specific IDE optimization
ch-agent-zero --ide-type claude-code
```

**Features:**
- stdio transport for fast local communication
- Minimal resource usage
- Direct integration with local IDEs
- No network dependencies

### 2. Standalone Mode
HTTP/WebSocket server for remote access and team collaboration.

```bash
# Start standalone server
ch-agent-zero --deployment-mode standalone --host 0.0.0.0 --port 8505

# With SSL and authentication
ch-agent-zero --deployment-mode standalone \
  --ssl-enable --ssl-certfile cert.pem --ssl-keyfile key.pem \
  --auth-username admin --auth-password-file password.txt
```

**Features:**
- HTTP, WebSocket, and SSE transports
- Remote IDE access
- Basic authentication and SSL
- Health check and metrics endpoints
- Rate limiting

### 3. Enterprise Mode
Full-featured deployment with advanced security and monitoring.

```bash
# Enterprise deployment
ch-agent-zero --deployment-mode enterprise \
  --enable-metrics --enable-health-check \
  --rate-limit --oauth-enable \
  --oauth-client-id YOUR_CLIENT_ID \
  --oauth-client-secret YOUR_CLIENT_SECRET
```

**Features:**
- All standalone features
- OAuth 2.0 authentication
- Advanced metrics and monitoring
- OpenTelemetry integration
- Prometheus metrics export
- Advanced rate limiting and quotas

## 🖥️ IDE Integrations

### Claude Desktop

**Configuration Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Setup:**
```bash
# Generate configuration
ch-agent-zero generate-config --ide claude-desktop --output claude-desktop-config.json

# Or use the installer
./scripts/install.sh --ide claude-desktop
```

**Sample Configuration:**
```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "uv",
      "args": ["run", "--with", "ch-agent-zero", "ch-agent-zero"],
      "env": {
        "CLICKHOUSE_HOST": "your-host",
        "CLICKHOUSE_USER": "your-user",
        "CLICKHOUSE_PASSWORD": "your-password"
      }
    }
  }
}
```

### Claude Code

Claude Code supports both local and remote configurations with environment variable expansion.

**Local Configuration:**
```bash
# Create .claude.json in your project
ch-agent-zero generate-config --ide claude-code --output .claude.json

# Test the configuration
claude mcp
```

**Remote Configuration (Standalone Server):**
```json
{
  "mcpServers": {
    "agent-zero": {
      "transport": "sse",
      "url": "https://your-server.com:8505/mcp/sse",
      "headers": {
        "Authorization": "Bearer your-token"
      }
    }
  }
}
```

### Cursor IDE

Cursor supports multiple integration modes and transports.

**Agent Mode (Full Capabilities):**
```bash
ch-agent-zero --ide-type cursor --cursor-mode agent
```

**Ask Mode (Information Retrieval):**
```bash
ch-agent-zero --ide-type cursor --cursor-mode ask
```

**Edit Mode (Query Generation):**
```bash
ch-agent-zero --ide-type cursor --cursor-mode edit
```

**Configuration:**
```json
{
  "name": "agent-zero",
  "command": "ch-agent-zero",
  "args": ["--cursor-mode", "agent"],
  "env": {
    "CLICKHOUSE_HOST": "your-host",
    "MCP_CURSOR_MODE": "agent"
  }
}
```

### Windsurf IDE

Windsurf integration includes plugin support and SSE transport for remote connections.

**Local Setup:**
```bash
ch-agent-zero --ide-type windsurf --windsurf-plugins
```

**Remote Setup:**
```json
{
  "servers": {
    "agent-zero": {
      "type": "sse",
      "url": "https://your-server.com:8505/mcp/sse",
      "headers": {
        "Authorization": "Basic base64-credentials"
      }
    }
  }
}
```

### VS Code

Requires the MCP extension for VS Code.

```bash
# Install VS Code MCP extension first
code --install-extension mcp-extension

# Generate configuration
ch-agent-zero generate-config --ide vscode --output ~/.vscode/mcp_config.json
```

## 🐳 Docker Deployment

### Simple Docker Run

```bash
# Build the image
docker build -t agent-zero .

# Run with environment variables
docker run -d \
  --name agent-zero-mcp \
  -p 8505:8505 \
  -e CLICKHOUSE_HOST=your-host \
  -e CLICKHOUSE_USER=your-user \
  -e CLICKHOUSE_PASSWORD=your-password \
  agent-zero
```

### Docker Compose (Recommended)

```bash
# Create .env file with your settings
cp .env.example .env
# Edit .env with your ClickHouse details

# Start the stack
docker-compose up -d

# Include ClickHouse for testing
docker-compose --profile with-clickhouse up -d

# Include nginx proxy for SSL
docker-compose --profile with-proxy up -d
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-zero-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-zero-mcp
  template:
    metadata:
      labels:
        app: agent-zero-mcp
    spec:
      containers:
      - name: agent-zero
        image: agent-zero:latest
        ports:
        - containerPort: 8505
        env:
        - name: MCP_DEPLOYMENT_MODE
          value: "enterprise"
        - name: CLICKHOUSE_HOST
          valueFrom:
            secretKeyRef:
              name: clickhouse-secret
              key: host
        livenessProbe:
          httpGet:
            path: /health
            port: 8505
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8505
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-zero-service
spec:
  selector:
    app: agent-zero-mcp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8505
  type: LoadBalancer
```

## 🔧 Configuration

### Environment Variables

**ClickHouse Configuration:**
```bash
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_SECURE=true
CLICKHOUSE_VERIFY=true
CLICKHOUSE_CONNECT_TIMEOUT=30
CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300
CLICKHOUSE_DATABASE=default
```

**MCP Server Configuration:**
```bash
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8505
MCP_DEPLOYMENT_MODE=local|standalone|enterprise
MCP_TRANSPORT=stdio|sse|websocket|http
MCP_IDE_TYPE=claude_desktop|claude_code|cursor|windsurf|vscode
```

**Security Configuration:**
```bash
MCP_SSL_ENABLE=false
MCP_SSL_CERTFILE=/path/to/cert.pem
MCP_SSL_KEYFILE=/path/to/key.pem
MCP_AUTH_USERNAME=admin
MCP_AUTH_PASSWORD=secure-password
MCP_OAUTH_ENABLE=false
MCP_OAUTH_CLIENT_ID=your-client-id
MCP_OAUTH_CLIENT_SECRET=your-client-secret
```

**Feature Configuration:**
```bash
MCP_ENABLE_METRICS=false
MCP_ENABLE_HEALTH_CHECK=true
MCP_RATE_LIMIT_ENABLED=false
MCP_RATE_LIMIT_REQUESTS=100
MCP_TOOL_LIMIT=100
MCP_RESOURCE_LIMIT=50
MCP_ENABLE_STRUCTURED_OUTPUT=true
```

### Transport Selection

Agent Zero automatically selects the optimal transport based on your configuration:

- **stdio**: Local development with Claude Desktop, local Cursor/Windsurf
- **sse**: Remote connections, Claude Code, standalone mode
- **websocket**: Real-time bidirectional communication (Cursor advanced mode)
- **http**: Simple request/response patterns

### IDE-Specific Optimizations

**Claude Code:**
- Supports environment variable expansion in configs
- OAuth 2.0 authentication flow
- SSE transport for remote servers
- Project-level and user-level configurations

**Cursor:**
- Multiple modes: agent, ask, edit
- WebSocket and SSE transports
- Advanced tool selection
- Context-aware responses

**Windsurf:**
- Plugin integration support
- Team configuration management
- Enterprise whitelist support
- SSE transport for secure connections

## 📊 Monitoring and Metrics

### Health Checks

```bash
# Local health check
curl http://localhost:8505/health

# Example response
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "requests_total": 1500,
  "errors_total": 2,
  "active_connections": 5,
  "version": "0.0.1"
}
```

### Metrics Endpoint

```bash
# Prometheus-compatible metrics
curl http://localhost:8505/metrics

# Example metrics
{
  "tool_usage": {
    "list_databases": 45,
    "run_select_query": 120,
    "monitor_current_processes": 30
  },
  "average_response_times": {
    "list_databases": 0.05,
    "run_select_query": 1.2
  },
  "total_tools_available": 35
}
```

### OpenTelemetry Integration

```bash
# Enable OpenTelemetry tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
export MCP_ENABLE_TRACING=true
ch-agent-zero --deployment-mode enterprise
```

## 🔒 Security

### SSL/TLS Configuration

```bash
# Generate self-signed certificates for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start with SSL
ch-agent-zero --deployment-mode standalone \
  --ssl-enable --ssl-certfile cert.pem --ssl-keyfile key.pem
```

### Authentication

**Basic Authentication:**
```bash
# With username/password
ch-agent-zero --auth-username admin --auth-password mypassword

# With password file (recommended)
echo "secure-password" > password.txt
ch-agent-zero --auth-username admin --auth-password-file password.txt
```

**OAuth 2.0:**
```bash
ch-agent-zero --oauth-enable \
  --oauth-client-id your-client-id \
  --oauth-client-secret your-client-secret
```

### Rate Limiting

```bash
# Enable rate limiting (100 requests per minute)
ch-agent-zero --rate-limit --rate-limit-requests 100
```

## 🚀 Performance Tuning

### Server Optimization

```bash
# Increase tool and resource limits
ch-agent-zero --tool-limit 200 --resource-limit 100

# Optimize for high-throughput
ch-agent-zero --deployment-mode enterprise \
  --rate-limit-requests 1000 \
  --enable-metrics
```

### ClickHouse Optimization

```bash
# Increase timeouts for complex queries
export CLICKHOUSE_SEND_RECEIVE_TIMEOUT=600
export CLICKHOUSE_CONNECT_TIMEOUT=60
```

### Connection Pooling

Agent Zero automatically manages connection pooling with configurable limits:

```bash
# Configure connection pooling (via environment)
export CLICKHOUSE_MAX_CONNECTIONS=20
export CLICKHOUSE_CONNECTION_TIMEOUT=30
```

## 🔧 Troubleshooting

### Common Issues

**1. Connection Refused**
```bash
# Check if server is running
curl http://localhost:8505/health

# Check firewall settings
sudo ufw allow 8505
```

**2. Authentication Failures**
```bash
# Verify credentials
ch-agent-zero --show-config

# Test with curl
curl -u admin:password http://localhost:8505/health
```

**3. IDE Not Detecting Server**
```bash
# Verify configuration
ch-agent-zero generate-config --ide cursor

# Check IDE logs
tail -f ~/.cursor/logs/mcp.log
```

### Debug Mode

```bash
# Enable debug logging
export PYTHONLOGLEVEL=DEBUG
export MCP_ENABLE_TRACING=true
ch-agent-zero --show-config
```

### Configuration Validation

```bash
# Validate your configuration
ch-agent-zero --show-config

# Test specific IDE config
ch-agent-zero generate-config --ide claude-code --output - | jq .
```

## 📚 Examples

### Development Setup

```bash
# Local development with Claude Code
ch-agent-zero --ide-type claude-code

# Test with sample query
curl -X POST http://localhost:8505/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list", "id": 1}'
```

### Production Setup

```bash
# Enterprise deployment with full security
ch-agent-zero --deployment-mode enterprise \
  --host 0.0.0.0 --port 8505 \
  --ssl-enable --ssl-certfile /etc/ssl/cert.pem --ssl-keyfile /etc/ssl/key.pem \
  --oauth-enable --oauth-client-id $OAUTH_CLIENT_ID \
  --enable-metrics --enable-health-check \
  --rate-limit --rate-limit-requests 500 \
  --tool-limit 200
```

### Team Setup

```bash
# Standalone server for team access
docker-compose up -d

# Each team member configures their IDE
ch-agent-zero generate-config --ide cursor --deployment-mode standalone \
  --output cursor-team-config.json
```

For more examples and advanced configurations, see the [examples](./examples/) directory.

## 🆘 Support

- 📖 [Documentation](https://github.com/maruthiprithivi/agent_zero)
- 🐛 [Issues](https://github.com/maruthiprithivi/agent_zero/issues)
- 💬 [Discussions](https://github.com/maruthiprithivi/agent_zero/discussions)
- 📧 [Email](mailto:maruthiprithivi@gmail.com)