#!/bin/bash

# Documentation Structure Maintenance Script
# Creates and maintains the documentation directory structure

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="${ROOT_DIR}/docs"
TESTING_DIR="${DOCS_DIR}/testing"

echo "🔄 Updating documentation structure..."

# Create necessary directories
mkdir -p "$TESTING_DIR"

echo "✅ Documentation structure updated"
echo "🎉 Done"
