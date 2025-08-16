#!/bin/bash
# Agent Zero MCP Server - Universal Installation Script (2025 Multi-IDE Edition)
# Supports: Claude Desktop, Claude Code, Cursor, Windsurf, VS Code

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="ch-agent-zero"
GITHUB_REPO="maruthiprithivi/agent_zero"
DEFAULT_VERSION="latest"

# Platform detection
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PLATFORM="linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        PLATFORM="windows"
    else
        echo -e "${RED}Unsupported platform: $OSTYPE${NC}"
        exit 1
    fi
    echo -e "${BLUE}Detected platform: $PLATFORM${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
install_uv() {
    if command_exists uv; then
        echo -e "${GREEN}uv is already installed${NC}"
        return
    fi

    echo -e "${YELLOW}Installing uv...${NC}"
    if [[ "$PLATFORM" == "windows" ]]; then
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # Add to PATH for current session
    if [[ "$PLATFORM" == "windows" ]]; then
        export PATH="$USERPROFILE/.cargo/bin:$PATH"
    else
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
}

# Install Agent Zero
install_agent_zero() {
    echo -e "${YELLOW}Installing $PACKAGE_NAME...${NC}"

    if command_exists uv; then
        uv tool install "$PACKAGE_NAME"
    elif command_exists pip; then
        pip install "$PACKAGE_NAME"
    else
        echo -e "${RED}Neither uv nor pip found. Please install Python first.${NC}"
        exit 1
    fi

    echo -e "${GREEN}$PACKAGE_NAME installed successfully!${NC}"
}

# Get IDE configuration paths
get_config_paths() {
    case "$PLATFORM" in
        "macos")
            CLAUDE_DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
            CURSOR_CONFIG="$HOME/.cursor/mcp_config.json"
            WINDSURF_CONFIG="$HOME/.codeium/windsurf/mcp_config.json"
            VSCODE_CONFIG="$HOME/.vscode/mcp_config.json"
            ;;
        "linux")
            CLAUDE_DESKTOP_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
            CURSOR_CONFIG="$HOME/.cursor/mcp_config.json"
            WINDSURF_CONFIG="$HOME/.codeium/windsurf/mcp_config.json"
            VSCODE_CONFIG="$HOME/.vscode/mcp_config.json"
            ;;
        "windows")
            CLAUDE_DESKTOP_CONFIG="$APPDATA/Claude/claude_desktop_config.json"
            CURSOR_CONFIG="$APPDATA/Cursor/mcp_config.json"
            WINDSURF_CONFIG="$APPDATA/Windsurf/mcp_config.json"
            VSCODE_CONFIG="$APPDATA/Code/mcp_config.json"
            ;;
    esac
}

# Generate IDE configuration
generate_ide_config() {
    local ide="$1"
    local deployment_mode="${2:-local}"
    local output_file="$3"

    echo -e "${YELLOW}Generating configuration for $ide...${NC}"

    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"

    # Generate config using Agent Zero's built-in generator
    if command_exists ch-agent-zero; then
        ch-agent-zero generate-config --ide "$ide" --deployment-mode "$deployment_mode" --output "$output_file"
    elif command_exists uv; then
        uv run --with ch-agent-zero ch-agent-zero generate-config --ide "$ide" --deployment-mode "$deployment_mode" --output "$output_file"
    else
        echo -e "${RED}Agent Zero not found. Please install it first.${NC}"
        return 1
    fi

    echo -e "${GREEN}Configuration saved to: $output_file${NC}"
}

# Interactive IDE selection
select_ide() {
    echo -e "${BLUE}Select IDE(s) to configure:${NC}"
    echo "1) Claude Desktop"
    echo "2) Claude Code"
    echo "3) Cursor"
    echo "4) Windsurf"
    echo "5) VS Code"
    echo "6) All IDEs"
    echo "7) Skip IDE configuration"

    read -p "Enter your choice (1-7): " choice

    case $choice in
        1) configure_claude_desktop ;;
        2) configure_claude_code ;;
        3) configure_cursor ;;
        4) configure_windsurf ;;
        5) configure_vscode ;;
        6) configure_all_ides ;;
        7) echo -e "${YELLOW}Skipping IDE configuration${NC}" ;;
        *) echo -e "${RED}Invalid choice${NC}"; select_ide ;;
    esac
}

# Configure Claude Desktop
configure_claude_desktop() {
    echo -e "${BLUE}Configuring Claude Desktop...${NC}"
    generate_ide_config "claude-desktop" "local" "$CLAUDE_DESKTOP_CONFIG"

    echo -e "${GREEN}Claude Desktop configured!${NC}"
    echo -e "${YELLOW}Please restart Claude Desktop to apply changes.${NC}"
}

# Configure Claude Code
configure_claude_code() {
    echo -e "${BLUE}Configuring Claude Code...${NC}"

    # Claude Code uses project-level or user-level config
    local claude_code_config=".claude.json"
    if [[ -f ".claude.json" ]]; then
        echo -e "${YELLOW}Found existing .claude.json, backing up...${NC}"
        cp ".claude.json" ".claude.json.backup"
    fi

    generate_ide_config "claude-code" "local" "$claude_code_config"

    echo -e "${GREEN}Claude Code configured!${NC}"
    echo -e "${YELLOW}Configuration saved to .claude.json in current directory.${NC}"
    echo -e "${YELLOW}Run 'claude mcp' to verify the configuration.${NC}"
}

# Configure Cursor
configure_cursor() {
    echo -e "${BLUE}Configuring Cursor...${NC}"
    generate_ide_config "cursor" "local" "$CURSOR_CONFIG"

    echo -e "${GREEN}Cursor configured!${NC}"
    echo -e "${YELLOW}Please restart Cursor to apply changes.${NC}"
}

# Configure Windsurf
configure_windsurf() {
    echo -e "${BLUE}Configuring Windsurf...${NC}"
    generate_ide_config "windsurf" "local" "$WINDSURF_CONFIG"

    echo -e "${GREEN}Windsurf configured!${NC}"
    echo -e "${YELLOW}Please restart Windsurf to apply changes.${NC}"
}

# Configure VS Code
configure_vscode() {
    echo -e "${BLUE}Configuring VS Code...${NC}"
    generate_ide_config "vscode" "local" "$VSCODE_CONFIG"

    echo -e "${GREEN}VS Code configured!${NC}"
    echo -e "${YELLOW}Please install the MCP extension and restart VS Code.${NC}"
}

# Configure all IDEs
configure_all_ides() {
    echo -e "${BLUE}Configuring all supported IDEs...${NC}"

    configure_claude_desktop
    configure_claude_code
    configure_cursor
    configure_windsurf
    configure_vscode

    echo -e "${GREEN}All IDEs configured!${NC}"
}

# Setup environment variables
setup_environment() {
    echo -e "${BLUE}Setting up environment variables...${NC}"

    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        echo -e "${YELLOW}Creating .env file template...${NC}"
        cat > .env << 'EOF'
# ClickHouse Configuration (Required)
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password

# Optional ClickHouse Configuration
CLICKHOUSE_PORT=8443
CLICKHOUSE_SECURE=true
CLICKHOUSE_VERIFY=true
CLICKHOUSE_CONNECT_TIMEOUT=30
CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300
CLICKHOUSE_DATABASE=

# MCP Server Configuration (Optional)
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8505
MCP_DEPLOYMENT_MODE=local
MCP_TRANSPORT=stdio

# IDE Integration (Optional)
MCP_IDE_TYPE=
MCP_CURSOR_MODE=agent
MCP_WINDSURF_PLUGINS_ENABLED=true

# Features (Optional)
MCP_ENABLE_METRICS=false
MCP_ENABLE_HEALTH_CHECK=true
MCP_RATE_LIMIT_ENABLED=false
MCP_RATE_LIMIT_REQUESTS=100
MCP_TOOL_LIMIT=100
MCP_RESOURCE_LIMIT=50

# Security (Optional)
MCP_SSL_ENABLE=false
MCP_SSL_CERTFILE=
MCP_SSL_KEYFILE=
MCP_AUTH_USERNAME=
MCP_AUTH_PASSWORD=
MCP_OAUTH_ENABLE=false
MCP_OAUTH_CLIENT_ID=
MCP_OAUTH_CLIENT_SECRET=
EOF

        echo -e "${GREEN}.env file created!${NC}"
        echo -e "${YELLOW}Please edit .env file with your ClickHouse connection details.${NC}"
    else
        echo -e "${YELLOW}.env file already exists, skipping...${NC}"
    fi
}

# Verify installation
verify_installation() {
    echo -e "${BLUE}Verifying installation...${NC}"

    if command_exists ch-agent-zero; then
        echo -e "${GREEN}✓ Agent Zero CLI available${NC}"
        ch-agent-zero --version
    elif command_exists uv; then
        echo -e "${GREEN}✓ Agent Zero available via uv${NC}"
        uv run --with ch-agent-zero ch-agent-zero --version
    else
        echo -e "${RED}✗ Agent Zero not found${NC}"
        return 1
    fi

    # Test configuration generation
    echo -e "${BLUE}Testing configuration generation...${NC}"
    if ch-agent-zero generate-config --ide claude-desktop >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Configuration generation working${NC}"
    else
        echo -e "${YELLOW}⚠ Configuration generation may have issues${NC}"
    fi
}

# Show usage examples
show_usage_examples() {
    echo -e "${BLUE}Usage Examples:${NC}"
    echo
    echo -e "${YELLOW}Local development (stdio):${NC}"
    echo "  ch-agent-zero"
    echo
    echo -e "${YELLOW}Standalone server (HTTP/WebSocket):${NC}"
    echo "  ch-agent-zero --deployment-mode standalone --host 0.0.0.0 --port 8505"
    echo
    echo -e "${YELLOW}Claude Code integration:${NC}"
    echo "  ch-agent-zero --ide-type claude-code"
    echo
    echo -e "${YELLOW}Cursor IDE with specific mode:${NC}"
    echo "  ch-agent-zero --ide-type cursor --cursor-mode agent"
    echo
    echo -e "${YELLOW}Windsurf IDE with plugins:${NC}"
    echo "  ch-agent-zero --ide-type windsurf --windsurf-plugins"
    echo
    echo -e "${YELLOW}Generate IDE configuration:${NC}"
    echo "  ch-agent-zero generate-config --ide cursor --output cursor-config.json"
    echo
    echo -e "${YELLOW}Enterprise deployment with auth:${NC}"
    echo "  ch-agent-zero --deployment-mode enterprise --auth-username admin --auth-password-file /path/to/password"
    echo
}

# Main installation flow
main() {
    echo -e "${GREEN}Agent Zero MCP Server - Universal Installer (2025 Multi-IDE Edition)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo

    detect_platform
    get_config_paths

    # Install dependencies
    install_uv
    install_agent_zero

    # Setup environment
    setup_environment

    # Configure IDEs
    select_ide

    # Verify installation
    verify_installation

    # Show usage examples
    show_usage_examples

    echo
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Edit .env file with your ClickHouse connection details"
    echo "2. Test the installation: ch-agent-zero --show-config"
    echo "3. Start using Agent Zero in your preferred IDE"
    echo
    echo -e "${BLUE}For more information, visit: https://github.com/${GITHUB_REPO}${NC}"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Agent Zero MCP Server - Universal Installer"
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --ide IDE           Configure specific IDE (claude-desktop|claude-code|cursor|windsurf|vscode)"
        echo "  --deployment MODE   Set deployment mode (local|standalone|enterprise)"
        echo "  --skip-ide          Skip IDE configuration"
        echo
        exit 0
        ;;
    --ide)
        CONFIGURE_IDE="$2"
        shift 2
        ;;
    --deployment)
        DEPLOYMENT_MODE="$2"
        shift 2
        ;;
    --skip-ide)
        SKIP_IDE=true
        shift
        ;;
esac

# Run main installation
main "$@"
