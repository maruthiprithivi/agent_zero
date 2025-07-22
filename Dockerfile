# Agent Zero MCP Server - Multi-IDE Enterprise Dockerfile
FROM python:3.13-slim

# Metadata
LABEL maintainer="Maruthi Prithivi <maruthiprithivi@gmail.com>"
LABEL description="Agent Zero ClickHouse MCP Server - 2025 Multi-IDE Edition"
LABEL version="0.0.1"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MCP_DEPLOYMENT_MODE=enterprise
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8505

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN uv pip install --system -e .

# Copy application code
COPY agent_zero/ ./agent_zero/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash agentuser
RUN chown -R agentuser:agentuser /app
USER agentuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${MCP_SERVER_PORT}/health || exit 1

# Expose port
EXPOSE 8505

# Default command - can be overridden
CMD ["ch-agent-zero", "--deployment-mode", "enterprise", "--enable-health-check", "--enable-metrics"]