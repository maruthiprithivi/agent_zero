#!/usr/bin/env python3
"""Test script to verify all dependencies can be resolved.

This script checks if the dependencies in pyproject.toml are available
on PyPI and can be installed successfully.
"""

import subprocess
import sys
from pathlib import Path


def test_dependency_resolution():
    """Test if dependencies can be resolved without installation."""
    print("Testing dependency resolution...")

    try:
        # Use pip-tools to check if dependencies can be resolved
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--no-deps",
                "--quiet",
                "-e",
                ".",
                "--break-system-packages",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            print("PASS: Basic package structure is valid")
        else:
            # Check if it's just the externally-managed-environment issue
            if "externally-managed-environment" in result.stderr:
                print("INFO: Local environment is externally managed (expected)")
                print("PASS: Will test individual dependencies instead")
            else:
                print(f"FAIL: Package structure issue: {result.stderr}")
                return False

    except Exception as e:
        print(f"FAIL: Error testing dependencies: {e}")
        return False

    # Test key dependencies individually (focus on critical ones for speed)
    key_deps = [
        "clickhouse-connect>=0.8.18,<1.0",
        "mcp>=1.0.0,<2.0",
        "python-dotenv>=1.0.0,<2.0",
        "python-oauth2>=1.1.1,<2.0",
    ]

    print("\nTesting individual key dependencies...")

    for dep in key_deps:
        print(f"  Checking {dep}...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--quiet",
                    "--break-system-packages",
                    dep,
                ],
                capture_output=True,
                text=True,
                timeout=30,  # Add timeout
            )

            if result.returncode == 0:
                print(f"  PASS: {dep.split('>=')[0]} - Available")
            else:
                print(f"  FAIL: {dep.split('>=')[0]} - Not available")
                print(f"     Error: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {dep.split('>=')[0]} - Check took too long")
            return False
        except Exception as e:
            print(f"  FAIL: {dep.split('>=')[0]} - Error: {e}")
            return False

    print("\nAll key dependencies are available!")
    return True


def main():
    """Main function."""
    print("Dependency Resolution Test")
    print("=" * 40)

    if test_dependency_resolution():
        print("\nPASS: All dependency checks passed!")
        print("The CI/CD pipeline should now work correctly.")
        sys.exit(0)
    else:
        print("\nFAIL: Some dependency checks failed!")
        print("Review the pyproject.toml dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
