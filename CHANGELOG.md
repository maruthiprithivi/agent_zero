# Changelog

All notable changes to Agent Zero will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-IDE Support (2025 Edition)**: Complete integration support for Claude Desktop, Claude Code, Cursor, Windsurf, and VS Code
- **Multiple Deployment Modes**: Local, Standalone, and Enterprise deployment options
- **Standalone Server Mode**: Full HTTP/WebSocket server with async support using aiohttp
- **Enhanced Security**: SSL/TLS support, Basic Authentication, OAuth 2.0 integration
- **Advanced Monitoring**: Health checks, Prometheus metrics, OpenTelemetry tracing
- **Smart Transport Selection**: Automatic transport optimization based on IDE and deployment mode
- **Enhanced Configuration System**: Type-safe configuration with enums and comprehensive environment variable support
- **Configuration Generation**: Built-in command to generate IDE-specific configurations
- **Universal Installer**: Cross-platform installation script with IDE auto-configuration
- **Docker Support**: Complete containerization with docker-compose for enterprise deployments
- **Kubernetes Ready**: K8s deployment examples and configurations
- **Configuration Templates**: Pre-built configurations for all supported IDEs
- **Rate Limiting**: Configurable rate limiting for enterprise deployments
- **IDE-Specific Optimizations**: Tailored features and transport selection per IDE
- **Structured Output**: 2025-06-18 MCP specification compliance
- **Environment Variable Expansion**: Support for dynamic configuration in IDE configs

### Enhanced
- **Server Architecture**: Redesigned with support for multiple deployment modes and transports
- **Configuration Management**: Complete overhaul with type-safe configuration classes
- **Transport Layer**: Enhanced with SSE, WebSocket, and HTTP transport support
- **CLI Interface**: Expanded with comprehensive options for all deployment modes
- **Testing Framework**: Enhanced test isolation and mock server support
- **Documentation**: Complete rewrite with deployment guides and IDE-specific instructions

### Changed
- **Server Configuration**: Migrated from simple config to comprehensive ServerConfig class with enums
- **Entry Point**: Enhanced main.py with multi-mode support and configuration generation
- **MCP Server**: Updated with intelligent transport selection and deployment mode detection
- **Dependencies**: Added aiohttp, aiohttp-cors for standalone server support
- **Project Structure**: Added configs/, scripts/, and deployment files

### Technical Details

#### New Files Added
- `agent_zero/standalone_server.py` - Async HTTP/WebSocket server implementation
- `agent_zero/server_config.py` - Enhanced configuration system with enums
- `scripts/install.sh` - Universal cross-platform installer
- `configs/` - IDE-specific configuration templates
  - `claude-desktop.json` - Claude Desktop MCP configuration
  - `claude-code.json` - Claude Code configuration with env vars
  - `cursor.json` - Cursor IDE configuration
  - `windsurf.json` - Windsurf IDE configuration
  - `standalone-server.json` - Standalone server configuration
- `Dockerfile` - Multi-stage Docker build for enterprise deployment
- `docker-compose.yml` - Complete stack with ClickHouse and nginx
- `DEPLOYMENT.md` - Comprehensive deployment guide
- `CHANGELOG.md` - This changelog file

#### Enhanced Files
- `agent_zero/main.py` - Complete rewrite with multi-mode support and config generation
- `agent_zero/mcp_server.py` - Enhanced with deployment mode detection and transport selection
- `pyproject.toml` - Added new dependencies and optional dependency groups

#### New Configuration Options
```bash
# Deployment Configuration
MCP_DEPLOYMENT_MODE=local|standalone|enterprise
MCP_TRANSPORT=stdio|sse|websocket|http
MCP_IDE_TYPE=claude_desktop|claude_code|cursor|windsurf|vscode

# Security Configuration
MCP_SSL_ENABLE=false
MCP_AUTH_USERNAME=
MCP_OAUTH_ENABLE=false
MCP_OAUTH_CLIENT_ID=
MCP_OAUTH_CLIENT_SECRET=

# Feature Configuration
MCP_ENABLE_METRICS=false
MCP_ENABLE_HEALTH_CHECK=true
MCP_RATE_LIMIT_ENABLED=false
MCP_TOOL_LIMIT=100
MCP_RESOURCE_LIMIT=50

# IDE-Specific Configuration
MCP_CURSOR_MODE=agent|ask|edit
MCP_WINDSURF_PLUGINS_ENABLED=true
```

#### New CLI Commands
```bash
# Show current configuration
ch-agent-zero --show-config

# Generate IDE configurations
ch-agent-zero generate-config --ide cursor --output config.json

# Deployment modes
ch-agent-zero --deployment-mode standalone
ch-agent-zero --deployment-mode enterprise

# IDE optimizations
ch-agent-zero --ide-type claude-code
ch-agent-zero --ide-type cursor --cursor-mode agent
ch-agent-zero --ide-type windsurf --windsurf-plugins
```

#### Transport Selection Logic
- **stdio**: Local development (Claude Desktop, local IDE usage)
- **sse**: Remote connections (Claude Code, Windsurf, standalone mode)
- **websocket**: Real-time communication (Cursor advanced mode)
- **http**: Simple request/response (API access, health checks)

#### Supported IDE Integrations

1. **Claude Desktop**
   - stdio transport for optimal local performance
   - uv-based configuration with automatic Python handling
   - Environment variable isolation

2. **Claude Code**
   - Environment variable expansion in configurations
   - OAuth 2.0 authentication flow support
   - SSE transport for remote server connections
   - Project-level `.claude.json` configuration

3. **Cursor IDE**
   - Multiple modes: agent (full capabilities), ask (information retrieval), edit (query generation)
   - WebSocket transport for real-time bidirectional communication
   - SSE transport for stable remote connections
   - Context-aware tool selection and response formatting

4. **Windsurf IDE**
   - Plugin integration support with automatic discovery
   - Team configuration management
   - Enterprise whitelist support for approved servers
   - SSE transport for secure remote connections

5. **VS Code**
   - MCP extension compatibility
   - stdio transport for local development
   - SSE transport for remote server integration

#### Security Enhancements
- **SSL/TLS**: Full SSL support with configurable certificates
- **Basic Authentication**: Username/password authentication with secure password file support
- **OAuth 2.0**: Complete OAuth 2.0 flow implementation
- **Rate Limiting**: Configurable per-client rate limiting
- **CORS Support**: Cross-origin resource sharing for web clients

#### Monitoring & Observability
- **Health Checks**: Comprehensive health monitoring with uptime, request counts, and error rates
- **Metrics Collection**: Detailed metrics on tool usage, response times, and client connections
- **OpenTelemetry**: Distributed tracing support for enterprise deployments
- **Prometheus**: Metrics export in Prometheus format

#### Deployment Options
1. **Local Development**: Traditional stdio-based deployment for local IDE integration
2. **Standalone Server**: HTTP/WebSocket server for team collaboration and remote access
3. **Enterprise Deployment**: Full-featured deployment with security, monitoring, and scalability
4. **Docker**: Containerized deployment with docker-compose for production environments
5. **Kubernetes**: Cloud-native deployment with auto-scaling and load balancing

### Breaking Changes
- Configuration structure has changed - see migration guide in DEPLOYMENT.md
- Environment variable names have been standardized with `MCP_` prefix for server-specific settings
- CLI arguments have been reorganized into logical groups

### Migration Guide
For users upgrading from previous versions:

1. **Configuration Migration**:
   ```bash
   # Old way
   ch-agent-zero --cursor-mode agent

   # New way (recommended)
   ch-agent-zero --ide-type cursor --cursor-mode agent
   ```

2. **Environment Variables**:
   ```bash
   # Old variables still supported
   CLICKHOUSE_HOST=localhost

   # New MCP-specific variables
   MCP_IDE_TYPE=cursor
   MCP_DEPLOYMENT_MODE=local
   ```

3. **Docker Deployment**:
   ```bash
   # Use the new docker-compose setup
   docker-compose up -d
   ```

### Credits
- Built with FastMCP 2.0 for enhanced MCP protocol support
- Integrated with latest 2025 MCP specifications including June security updates
- Cross-platform compatibility with Windows, macOS, and Linux
- Comprehensive testing with mock-friendly architecture

---

## [0.0.1x] - 2024-12-XX (Previous Version)

### Added
- Initial ClickHouse monitoring MCP server implementation
- Basic tool set for database monitoring and analysis
- FastMCP integration for MCP protocol support
- Comprehensive test suite
- Docker support
- Basic documentation

### Features
- Query performance monitoring
- Resource usage tracking
- Error analysis tools
- Table and parts management
- Health checking capabilities
- Basic Cursor IDE integration

---

## Version History

- **v0.0.1x**: Initial release with core ClickHouse monitoring features
- **v0.1.0** (Planned): Multi-IDE support with enhanced deployment options
- **v0.2.0** (Planned): Advanced enterprise features and cloud integrations
