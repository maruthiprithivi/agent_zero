#!/bin/bash
set -e

# Make script executable
# chmod +x scripts/test-actions.sh

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Go to the project root
cd "$PROJECT_ROOT"

# Create a .actrc file if it doesn't exist
if [ ! -f .actrc ]; then
  echo "Creating .actrc file"
  cat > .actrc << EOF
-P ubuntu-latest=catthehacker/ubuntu:act-latest
--secret-file=.secrets
EOF
fi

# Create a .secrets file if it doesn't exist
if [ ! -f .secrets ]; then
  echo "Creating .secrets file"
  cat > .secrets << EOF
PYPI_USERNAME=test
PYPI_PASSWORD=test
EOF
fi

# Make sure .secrets is in .gitignore
if ! grep -q ".secrets" .gitignore; then
  echo "Adding .secrets to .gitignore"
  echo ".secrets" >> .gitignore
fi

# Run act with the specified workflow
if [ $# -eq 0 ]; then
  # Default to running CI workflow if no arguments provided
  echo "Running CI workflow with act..."
  act --job test
else
  echo "Running $1 job from workflow..."
  act --job "$1"
fi
