#!/bin/bash
# test_build.sh - Local test for publish.yml

# Set the version for testing
VERSION="0.0.0"

# Clean any previous build artifacts
rm -rf dist build *.egg-info

# Create and enter venv
# uv venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Update version in files (similar to what the workflow does)
# Remove 'v' prefix if present
VERSION="${VERSION#v}"

# Update __init__.py
if [ -f agent_one/__init__.py ]; then
    sed -i 's/__version__ = "[^"]*"/__version__ = "'"${VERSION}"'"/' agent_one/__init__.py
else
    mkdir -p agent_one
    echo '"""Agent One package for ClickHouse database management."""' >agent_one/__init__.py
    echo '' >>agent_one/__init__.py
    echo '__version__ = "'"${VERSION}"'"' >>agent_one/__init__.py
fi

# Install dependencies
uv pip install -e .
uv pip install build pytest wheel pkginfo

# Run tests
uv run pytest

# Debug package structure
echo "Contents of project directory:"
ls -la
echo "Contents of agent_one directory:"
ls -la agent_one
echo "Checking for __init__.py:"
[ -f agent_one/__init__.py ] && echo "Present" || echo "Missing!"

# Build package with verbose output
uv pip install --upgrade build wheel setuptools >=61.0
uv run python -m build --sdist --wheel --verbose

# Inspect the built distributions
echo "Build complete - checking dist contents:"
ls -la dist/

# Inspect wheel metadata (similar to the workflow)
echo "Inspecting generated wheel(s):"
find dist -name "agent_one*.whl" -print0 | while IFS= read -r -d $'\0' wheel; do
    echo "--- Inspecting: $wheel ---"
    uv run python -m pkginfo "$wheel" | grep -E '^Name:|^Version:' || echo "ERROR: pkginfo failed or Name/Version missing for $wheel"

    # Optional deeper debug: Unpack and show METADATA
    echo "--- METADATA contents for $wheel ---"
    rm -rf temp_wheel_inspect
    uv run wheel unpack "$wheel" -d temp_wheel_inspect >>/dev/null
    METADATA_FILE=$(find temp_wheel_inspect -name METADATA)
    if [ -f "$METADATA_FILE" ]; then
        cat "$METADATA_FILE"
    else
        echo "ERROR: Could not find METADATA file within unpacked wheel $wheel"
    fi
    rm -rf temp_wheel_inspect
done

# Test installation of the built package
uv run python -m pip install dist/*.whl
echo "Package installed successfully!"
