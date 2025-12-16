# Agent Zero - ClickHouse MCP Server

🚀 **AI-Powered ClickHouse Monitoring & Analytics** - 66+ specialized tools for database performance optimization, monitoring, and intelligent diagnostics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Production Ready](https://img.shields.io/badge/Production-Ready-green.svg)]()

## What is Agent Zero?

Agent Zero is a comprehensive **Model Context Protocol (MCP) server** that bridges AI assistants with ClickHouse databases. It provides intelligent database operations, real-time performance monitoring, and AI-powered optimization recommendations.

### ✨ Key Features

- 🤖 **AI-Powered Analysis** - Machine learning bottleneck detection and predictive analytics
- 📊 **Comprehensive Monitoring** - 400+ ProfileEvents across 25+ categories
- ☁️ **Storage Analytics** - S3, Azure, and multi-cloud optimization
- 🔧 **Production Ready** - Enterprise security, high availability, Docker & Kubernetes support

## 🚀 Quick Start

### Installation

```bash
# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install ch-agent-zero

# Or install with pip
pip install ch-agent-zero
```

### Configuration

Set your ClickHouse connection details:

```bash
export AGENT_ZERO_CLICKHOUSE_HOST=your-clickhouse-host
export AGENT_ZERO_CLICKHOUSE_USER=your-username
export AGENT_ZERO_CLICKHOUSE_PASSWORD=your-password
```

### IDE Setup

**Claude Desktop** - Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "ch-agent-zero",
      "env": {
        "AGENT_ZERO_CLICKHOUSE_HOST": "your-host",
        "AGENT_ZERO_CLICKHOUSE_USER": "your-user",
        "AGENT_ZERO_CLICKHOUSE_PASSWORD": "your-password"
      }
    }
  }
}
```

**Auto-configure for Cursor/Windsurf**:
```bash
# Cursor IDE
bash <(curl -sSL https://raw.githubusercontent.com/maruthiprithivi/agent_zero/main/scripts/install.sh) --ide cursor

# Windsurf IDE
bash <(curl -sSL https://raw.githubusercontent.com/maruthiprithivi/agent_zero/main/scripts/install.sh) --ide windsurf
```

## 🎯 Usage Examples

### For Database Administrators
```bash
"Generate a comprehensive health report for my ClickHouse cluster with AI insights"
"Analyze performance bottlenecks and provide optimization recommendations"
"Show capacity planning recommendations based on usage trends"
```

### For Developers
```bash
"Optimize these slow queries and provide specific recommendations"
"Show ProfileEvents analysis for queries from my application"
"Compare performance before and after my deployment"
```

### For Data Engineers
```bash
"Monitor ETL pipeline performance and identify bottlenecks"
"Analyze storage compression efficiency across algorithms"
"Generate multi-cloud storage optimization strategies"
```

## 🏢 Production Deployment

### Docker
```bash
docker run -d \
  --name agent-zero \
  -p 8505:8505 \
  -e AGENT_ZERO_CLICKHOUSE_HOST=your-host \
  -e AGENT_ZERO_CLICKHOUSE_USER=your-user \
  -e AGENT_ZERO_CLICKHOUSE_PASSWORD=your-password \
  ghcr.io/maruthiprithivi/agent-zero:latest
```

### Kubernetes
```bash
helm repo add agent-zero https://charts.agent-zero.example.com
helm install agent-zero agent-zero/agent-zero \
  --set clickhouse.host=your-host \
  --set clickhouse.user=your-user \
  --set clickhouse.password=your-password
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [**User Guide**](docs/USER_GUIDE.md) | Complete usage guide with examples |
| [**Developer Guide**](docs/DEVELOPER_GUIDE.md) | Contributing and extending Agent Zero |
| [**Deployment Guide**](docs/DEPLOYMENT.md) | Production deployment guide |

### LLM Compatibility Testing

To verify compatibility with various LLM providers (OpenAI, Anthropic, Gemini, MoonshotAI), you can run the integration suite locally:

```bash
# Export your API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export MOONSHOT_API_KEY=...

# Run compatibility tests
pytest tests/llm_compatibility/
```

## 🛠️ Architecture

Agent Zero provides **66+ specialized MCP tools** across these categories:

- **Database Operations** (4 tools) - Database and table management
- **Query Performance** (8 tools) - Query analysis and optimization
- **Resource Monitoring** (5 tools) - System resource tracking
- **AI Diagnostics** (7 tools) - Machine learning insights
- **Storage & Cloud** (4 tools) - Cloud storage optimization
- **ProfileEvents** (8 tools) - Comprehensive event analysis
- And more...

## 🤝 Community & Support

- 📚 **[Documentation](docs/)** - Comprehensive guides and API reference
- 🐛 **[GitHub Issues](https://github.com/maruthiprithivi/agent_zero/issues)** - Bug reports and feature requests
- 💬 **[GitHub Discussions](https://github.com/maruthiprithivi/agent_zero/discussions)** - Community Q&A and support

## 📜 License

Released under the [MIT License](LICENSE). See [LICENSE](LICENSE) for details.

---

**Ready to supercharge your ClickHouse monitoring?**

[**View Examples →**](docs/USER_GUIDE.md) | [**Start Developing →**](docs/DEVELOPER_GUIDE.md) | [**Deploy →**](docs/DEPLOYMENT.md)
