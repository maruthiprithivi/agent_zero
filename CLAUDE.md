# Agent Zero - ClickHouse MCP Server

## Purpose

Agent Zero is a production-ready ClickHouse MCP Server providing AI-powered database monitoring, performance optimization, and intelligent diagnostics. This file serves as the complete development and operational guide.

## Tech Stack

- **Language**: Python 3.11+ (3.13 recommended)
- **Framework**: FastMCP for MCP server implementation
- **Database**: ClickHouse (clickhouse-connect client)
- **Quality Tools**: Pre-commit hooks (Ruff, Black, MyPy, Bandit)
- **Testing**: Pytest with 90% coverage requirement
- **Deployment**: Docker, Kubernetes, multi-cloud support

## Project Structure

```text
agent_zero/
├── main.py                    # CLI entry point and argument parsing
├── server/                    # Core MCP server implementation
│   ├── core.py               # FastMCP server and transport logic
│   ├── tools.py              # MCP tool registry and implementations
│   ├── client.py             # ClickHouse client management
│   └── errors.py             # Custom exception classes
├── config/                    # Configuration management
│   └── unified.py            # UnifiedConfig - single source of truth
├── monitoring/               # ClickHouse monitoring tools by category
│   ├── query_performance.py  # Query analysis and optimization
│   ├── resource_usage.py     # System resource monitoring
│   └── ...                   # 12+ specialized monitoring modules
├── ai_diagnostics/           # AI-powered analysis tools
└── transport/                # MCP 2025 transport implementations
tests/                        # Comprehensive test suite (50+ files)
configs/                      # IDE configuration templates
docker/                       # Docker and deployment configurations
```

## Environment Setup

### Required Environment Variables

```bash
# ClickHouse Connection (Required)
export AGENT_ZERO_CLICKHOUSE_HOST="your-clickhouse-host"
export AGENT_ZERO_CLICKHOUSE_USER="your-username"
export AGENT_ZERO_CLICKHOUSE_PASSWORD="your-password"

# Development Mode (Optional)
export CH_AGENT_ZERO_DEBUG=1  # Enables mock defaults for testing
```

### Python Environment

```bash
# Using UV (recommended - fastest)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -e .[dev,test]

# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev,test]
```

### IDE Integration

The project supports multiple IDEs with optimized configurations:

- **Claude Desktop**: Use `configs/claude-desktop.json`
- **Claude Code**: Use `configs/claude-code.json`
- **Cursor**: Use `configs/cursor.json`
- **Windsurf**: Use `configs/windsurf.json`

Auto-configure with: `./scripts/install.sh --ide <ide-name>`

## Commands

### Development Commands

```bash
# Core development
ch-agent-zero                              # Run MCP server (local stdio)
ch-agent-zero --deployment-mode standalone # Run standalone SSE server
python -m agent_zero.main --show-config   # Display current configuration

# Code quality (run before committing)
pre-commit run --all-files    # Run all quality checks
ruff check agent_zero/        # Lint code
ruff format agent_zero/       # Format code
mypy agent_zero/              # Type checking
bandit -r agent_zero/         # Security analysis

# Testing
pytest                        # Run test suite
pytest --cov                  # Run with coverage
./scripts/run-tests.sh        # Docker-based testing
pytest -k "test_pattern"      # Run specific test pattern
```

### Production Commands

```bash
# Deployment modes
ch-agent-zero --deployment-mode enterprise  # Enterprise features
ch-agent-zero --ide-type cursor            # IDE-optimized mode
ch-agent-zero --ssl-enable --ssl-certfile cert.pem --ssl-keyfile key.pem

# Configuration generation
ch-agent-zero generate-config --ide cursor --output cursor-config.json
```

## Code Style & Standards

### Configuration Management

```python
# ✅ Correct - Use UnifiedConfig
from agent_zero.config import UnifiedConfig
config = UnifiedConfig.from_env()

# ❌ Avoid - Legacy configs (deprecated)
from agent_zero.server_config import ServerConfig
```

### Server Module Imports

```python
# ✅ Correct - Import from agent_zero.server
from agent_zero.server import create_clickhouse_client, run
from agent_zero.server.errors import MCPToolError

# ❌ Avoid - Direct mcp_server imports (deprecated)
from agent_zero.mcp_server import run
```

### MCP Tool Implementation

```python
# ✅ All MCP tools must follow this pattern
from agent_zero.mcp_tracer import trace_mcp_call

@trace_mcp_call
async def your_mcp_tool(param: str) -> dict[str, Any]:
    """Tool description.

    Args:
        param: Parameter description with type hints

    Returns:
        Structured response dictionary

    Raises:
        MCPToolError: On tool-specific errors
    """
    try:
        # Implementation
        return {"status": "success", "data": result}
    except SpecificException as e:
        logger.error(f"Tool failed: {e}", extra={"param": param})
        raise MCPToolError(f"Operation failed: {e}") from e
```

### Database Client Usage

```python
# ✅ Correct - Use client factory
from .client import create_clickhouse_client

async def your_function():
    client = await create_clickhouse_client()
    try:
        result = await client.query("SELECT * FROM system.tables")
        return result
    finally:
        await client.close()
```

### Error Handling Standards

```python
# ✅ Comprehensive error handling
try:
    result = await risky_operation()
except ClickHouseException as e:
    logger.error(f"ClickHouse error: {e}", extra={"query": query})
    raise MCPToolError(f"Database operation failed: {e}") from e
except ValidationError as e:
    logger.warning(f"Validation failed: {e}")
    raise MCPToolError(f"Invalid input: {e}") from e
```

## Testing Guidelines

### Test Structure

```python
# Test naming convention
def test_<function_name>_<scenario>_<expected_outcome>():
    """Test description explaining the scenario."""

# Use descriptive markers
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.asyncio        # Async tests
@pytest.mark.mcp_tool       # MCP tool tests
```

### Mock Usage

```python
# ✅ Mock external dependencies
@pytest.fixture
def mock_clickhouse_client():
    with patch('agent_zero.server.client.create_clickhouse_client') as mock:
        mock.return_value = AsyncMock()
        yield mock.return_value
```

### Running Tests

```bash
# Test categories
pytest tests/test_unit*.py           # Unit tests only
pytest tests/test_integration*.py    # Integration tests only
pytest -m "not slow"                 # Skip slow tests
pytest --cov-fail-under=90          # Enforce coverage

# Docker testing (CI environment)
./scripts/run-tests.sh --test-path tests/test_specific.py
```

## Security & Production Guidelines

### Environment Variables

- **Never commit** secrets to repository
- Use `.env.local` for local secrets (already in .gitignore)
- Production: Use secure secret management (Vault, K8s secrets)

### Authentication

```python
# Enterprise authentication patterns
auth_config = server_config.get_auth_config()
if auth_config:
    # Implement proper auth validation
    pass
```

### SSL/TLS Configuration

```bash
# Production SSL setup
ch-agent-zero \
  --ssl-enable \
  --ssl-certfile /path/to/cert.pem \
  --ssl-keyfile /path/to/key.pem \
  --require-mutual-tls
```

## Repository Etiquette

### Branch Naming

```bash
# Feature branches
mp/add-new-monitoring-tool
feature/kubernetes-deployment
hotfix/connection-timeout-fix

# Release branches
release/v2.1.0
```

### Commit Standards

```bash
# Use conventional commits
feat: add ProfileEvents analysis tool
fix: resolve connection timeout in production
docs: update deployment guide with K8s examples
test: add comprehensive coverage for AI diagnostics
```

### Pre-commit Requirements

```bash
# Must pass before committing
pre-commit run --all-files

# What it checks:
# - Ruff linting and formatting
# - Black code formatting
# - MyPy type checking
# - Bandit security scanning
# - YAML/JSON validation
```

## Deployment Modes

### Local Development

```bash
ch-agent-zero  # STDIO transport for Claude Desktop
```

### Standalone Server

```bash
ch-agent-zero --deployment-mode standalone --port 8505
```

### Enterprise Production

```bash
ch-agent-zero \
  --deployment-mode enterprise \
  --kubernetes-enabled \
  --zero-trust-enabled \
  --enable-health-check \
  --rate-limit
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Install in development mode
uv pip install -e .[dev]
```

**ClickHouse Connection Errors**
```bash
# Check environment variables
echo $AGENT_ZERO_CLICKHOUSE_HOST
# Verify ClickHouse accessibility
telnet $AGENT_ZERO_CLICKHOUSE_HOST 8123
```

**Test Failures**
```bash
# Run with detailed output
pytest -v --tb=long
# Check specific test category
pytest tests/test_unit*.py -v
```

**MCP Transport Issues**
```bash
# Debug transport selection
ch-agent-zero --show-config
# Force specific transport
ch-agent-zero --transport sse
```

### Debug Mode

```bash
# Enable comprehensive debugging
export CH_AGENT_ZERO_DEBUG=1
export AGENT_ZERO_ENABLE_MCP_TRACING=true
ch-agent-zero --deployment-mode standalone
```

## Do Not Touch (Critical Files)

### Deprecated Files - Use Alternatives

```python
# ❌ agent_zero/mcp_server.py
# Legacy compatibility wrapper - will be removed in v3.0
# Use: from agent_zero.server import run

# ❌ agent_zero/server_config.py
# Legacy configuration class - replaced in v2.0
# Use: from agent_zero.config import UnifiedConfig

# ❌ Any file with deprecation warnings
# Check file headers for replacement instructions
```

### Generated Files - Never Edit

```bash
# Version control
agent_zero/_version.py          # Generated by setuptools-scm

# Test outputs
htmlcov/                        # Coverage reports
test-results/                   # Test artifacts
.pytest_cache/                  # Pytest cache

# IDE files
.vscode/settings.json           # VS Code settings
```

### Configuration Templates

```bash
# IDE configurations - modify copies, not originals
configs/claude-desktop.json    # Template for Claude Desktop
configs/cursor.json            # Template for Cursor IDE
configs/windsurf.json          # Template for Windsurf IDE
```

## Performance Considerations

### Resource Limits

```python
# Default limits (configurable)
DEFAULT_MAX_ROWS = 10000              # Query result rows
DEFAULT_MAX_EXECUTION_SECONDS = 30    # Query timeout
TOOL_LIMIT = 100                      # Max MCP tools
RESOURCE_LIMIT = 50                   # Max MCP resources
```

### Optimization Guidelines

- Use `enable_client_cache=True` for repeated queries
- Enable `real_time_analytics=False` in development
- Set appropriate `rate_limit_requests` for production
- Monitor memory usage with AI diagnostics tools

## IDE-Specific Optimizations

### Claude Code
- Optimized for STDIO transport
- Full tool annotation support
- Progress notifications enabled

### Cursor
- SSE transport for real-time updates
- Agent mode integration
- WebSocket fallback support

### Windsurf
- Plugin integration enabled
- Enhanced completion support
- Multi-panel optimization

## 🧠 Context Engineering & AI Optimization

### Metacognitive Processing Instructions

**Claude, you are working with Agent Zero - a production-grade ClickHouse MCP Server. You must:**

1. **Pattern Recognition**: Continuously analyze code patterns in this project and adapt your responses to match the established architectural decisions and coding conventions.

2. **Quality Assurance**: Before suggesting any code changes, mentally review them against the standards defined in this file. Reject suggestions that don't meet our production requirements.

3. **Learning Integration**: After each successful interaction, incorporate learnings about this specific project's requirements to improve future responses.

4. **Context Optimization**: Use the most relevant context from this file for each response, prioritizing recent patterns and established conventions.

### Anti-Pattern Elimination

**STRICTLY PROHIBITED - Never suggest these patterns:**

```python
# ❌ NEVER - TODO comments or placeholder implementations
def some_function():
    # TODO: Implement this
    pass

# ❌ NEVER - Mock functions without full implementation
def get_data():
    return {"mock": "data"}  # This would need to be implemented

# ❌ NEVER - Incomplete error handling
try:
    result = risky_operation()
except:  # Too broad
    pass  # No handling

# ❌ NEVER - Hardcoded values without configuration
DATABASE_URL = "localhost:8123"  # Should use config

# ❌ NEVER - Missing type hints in production code
def process_data(data):  # Missing type hints
    return data
```

**ALWAYS REQUIRED - Use these patterns instead:**

```python
# ✅ ALWAYS - Complete implementations with proper error handling
from agent_zero.mcp_tracer import trace_mcp_call
from agent_zero.server.errors import MCPToolError

@trace_mcp_call
async def get_clickhouse_metrics(metric_type: str = "performance") -> dict[str, Any]:
    """Retrieve ClickHouse performance metrics.

    Args:
        metric_type: Type of metrics to retrieve (performance, resource, query)

    Returns:
        Dictionary containing structured metrics data

    Raises:
        MCPToolError: When metric retrieval fails
    """
    client = await create_clickhouse_client()
    try:
        query = f"SELECT * FROM system.metrics WHERE metric LIKE '%{metric_type}%'"
        result = await client.query(query)
        return {
            "status": "success",
            "metrics": result.result_rows,
            "timestamp": datetime.utcnow().isoformat(),
            "metric_type": metric_type
        }
    except ClickHouseException as e:
        logger.error(f"ClickHouse metric query failed: {e}", extra={"metric_type": metric_type})
        raise MCPToolError(f"Failed to retrieve {metric_type} metrics: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in metric retrieval: {e}")
        raise MCPToolError(f"Metric retrieval failed: {e}") from e
    finally:
        await client.close()
```

### Team Collaboration Standards

**Multi-Developer Synchronization:**

1. **Consistent AI Responses**: When multiple developers use Claude with this project, ensure responses follow identical patterns and conventions.

2. **Shared Context Understanding**: All team members should receive the same level of context and guidance regardless of their experience with the project.

3. **Version Control Integration**: When suggesting changes, always consider the impact on other team members' work and suggest atomic, reviewable commits.

4. **Documentation Synchronization**: Keep all documentation formats and styles consistent across team members' AI-assisted contributions.

### Local Override System

**CLAUDE.local.md Usage:**

Create `CLAUDE.local.md` in your working directory for personal preferences without affecting team standards:

```markdown
# Personal AI Assistant Configuration

## My Preferences
- Preferred variable naming: snake_case vs camelCase
- Code comment verbosity: minimal/moderate/detailed
- Error message style: technical/user-friendly
- Testing approach: TDD/BDD/integration-first

## Personal Context
- My role: [backend/frontend/devops/qa]
- Current focus: [feature name or area]
- Learning goals: [technologies or patterns to emphasize]

## Temporary Overrides
- Skip certain validation steps during prototyping
- Use different logging levels for debugging
- Apply specific optimization patterns I'm experimenting with
```

**Add to your `.gitignore`:**
```bash
# Personal AI configuration
CLAUDE.local.md
.claude/local/
```

## 🎯 Enterprise AI Standards

### Fortune 500 Configuration Patterns

**This project follows enterprise standards adopted by 200+ companies:**

1. **Zero-Defect Policy**: Never suggest code that requires additional implementation steps
2. **Security-First**: All suggestions must include proper security considerations
3. **Scalability-Aware**: Consider enterprise scale (10K+ concurrent users) in all recommendations
4. **Observability-Built-In**: Include logging, metrics, and monitoring in all code suggestions
5. **Documentation-Driven**: Every significant code change must include documentation updates

### Specialized Tool Instructions

**When working with MCP Tools:**

```python
# ✅ ALWAYS follow this exact pattern for MCP tools
@trace_mcp_call
async def your_tool_name(
    param1: str,
    param2: int = 10,
    config: UnifiedConfig | None = None
) -> dict[str, Any]:
    """Brief tool description.

    Args:
        param1: Clear parameter description with constraints
        param2: Parameter with default value explanation
        config: Optional config override for testing

    Returns:
        Structured response with consistent schema:
        {
            "status": "success" | "error",
            "data": Any,  # Main response data
            "metadata": {
                "timestamp": str,  # ISO format
                "execution_time_ms": int,
                "tool_version": str
            },
            "errors": list[str] | None  # Only if status == "error"
        }

    Raises:
        MCPToolError: Specific error conditions
    """
    start_time = time.time()

    # Validate inputs
    if not param1 or len(param1) < 3:
        raise MCPToolError("param1 must be at least 3 characters")

    # Use config or get default
    if config is None:
        config = UnifiedConfig.from_env()

    try:
        # Implementation here
        result = await your_implementation(param1, param2, config)

        return {
            "status": "success",
            "data": result,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "tool_version": "1.0.0"
            }
        }
    except SpecificException as e:
        logger.error(f"Tool failed: {e}", extra={"param1": param1, "param2": param2})
        raise MCPToolError(f"Tool execution failed: {e}") from e
```

### Adaptive Learning Directives

**Claude, continuously improve by:**

1. **Pattern Matching**: When you see similar functions in this codebase, understand the pattern and apply it consistently
2. **Convention Learning**: Notice naming conventions, code organization, and architectural decisions - apply them to new code
3. **Error Pattern Recognition**: Learn from error handling patterns in the codebase and suggest similar approaches
4. **Performance Optimization**: Understand the performance considerations (async/await patterns, connection pooling, etc.) and apply them
5. **Testing Strategy Alignment**: Follow the existing testing patterns (mocking strategies, test organization, assertion styles)

**Success Metrics for AI Assistance:**
- 95%+ accuracy in following project conventions
- Zero suggestions requiring "TODO" or "implement this" follow-ups
- Consistent application of error handling patterns
- Proper integration with existing observability stack
- Maintainable, production-ready code suggestions

**Feedback Loop:**
When you notice patterns in your successful interactions with this project, incorporate those learnings into your understanding of the codebase. This creates a positive feedback loop that improves the quality of assistance over time.
