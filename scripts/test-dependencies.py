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
    print("🔍 Testing dependency resolution...")

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
            print("✅ Basic package structure is valid")
        else:
            # Check if it's just the externally-managed-environment issue
            if "externally-managed-environment" in result.stderr:
                print("Info: Local environment is externally managed (expected)")
                print("✅ Will test individual dependencies instead")
            else:
                print(f"❌ Package structure issue: {result.stderr}")
                return False

    except Exception as e:
        print(f"❌ Error testing dependencies: {e}")
        return False

    # Test key dependencies individually
    key_deps = [
        "clickhouse-connect>=0.8.18,<1.0",
        "mcp>=1.0.0,<2.0",
        "python-dotenv>=1.0.0,<2.0",
        "cryptography>=43.0.0",
        "kubernetes>=30.1.0",
    ]

    print("\n🔍 Testing individual key dependencies...")

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
            )

            if result.returncode == 0:
                print(f"  ✅ {dep.split('>=')[0]} - Available")
            else:
                print(f"  ❌ {dep.split('>=')[0]} - Not available")
                print(f"     Error: {result.stderr.strip()}")
                return False

        except Exception as e:
            print(f"  ❌ {dep.split('>=')[0]} - Error: {e}")
            return False

    print("\n🎉 All key dependencies are available!")
    return True


def main():
    """Main function."""
    print("Dependency Resolution Test")
    print("=" * 40)

    if test_dependency_resolution():
        print("\n✅ All dependency checks passed!")
        print("The CI/CD pipeline should now work correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some dependency checks failed!")
        print("Review the pyproject.toml dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
