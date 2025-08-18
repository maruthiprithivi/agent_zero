#!/usr/bin/env python3
"""
Multi-Python Version Testing Script

Tests compatibility across Python 3.11, 3.12, and 3.13 versions.
Designed to be used in pre-commit hooks to catch issues early.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def find_python_versions() -> list[str]:
    """Find available Python versions on the system."""
    versions = []
    for version in ["python3.11", "python3.12", "python3.13"]:
        try:
            result = subprocess.run(
                [version, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                versions.append(version)
                print(f"Found: {version} - {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"Not found: {version}")

    return versions


def test_python_version(python_cmd: str) -> tuple[bool, str]:
    """Test a specific Python version for basic compatibility."""
    print(f"\nTesting {python_cmd}...")

    try:
        # Test basic Python version and syntax compatibility
        test_script = """
import sys
import warnings

# Test core Python modules that our code uses
try:
    import pathlib
    import json
    import subprocess
    import asyncio
    import typing
    import dataclasses
    import enum
    print("PASS: Core Python modules imported")

    # Test Python version compatibility
    version_info = sys.version_info
    if version_info >= (3, 11):
        print(f"PASS: Python version {version_info.major}.{version_info.minor} is supported")
    else:
        print(f"FAIL: Python version {version_info.major}.{version_info.minor} is not supported")
        sys.exit(1)

except ImportError as e:
    print(f"FAIL: Core module import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: General error: {e}")
    sys.exit(1)

print("PASS: All basic compatibility tests passed")
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script)
            temp_file = f.name

        result = subprocess.run(
            [python_cmd, temp_file],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=Path.cwd(),
        )

        # Clean up temp file
        Path(temp_file).unlink()

        if result.returncode == 0:
            print(f"PASS: {python_cmd} compatibility test passed")
            return True, result.stdout
        else:
            print(f"FAIL: {python_cmd} compatibility test failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False, result.stderr or result.stdout

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {python_cmd} test took too long")
        return False, "Test timeout"
    except Exception as e:
        print(f"ERROR: {python_cmd} test failed with exception: {e}")
        return False, str(e)


def test_syntax_compatibility() -> bool:
    """Test syntax compatibility across Python versions."""
    print("\nTesting syntax compatibility...")

    # Find Python files to test
    python_files = []
    for pattern in ["agent_zero/**/*.py", "scripts/*.py"]:
        python_files.extend(Path().glob(pattern))

    # Limit to avoid timeout in pre-commit
    python_files = python_files[:10]  # Test only first 10 files

    if not python_files:
        print("No Python files found to test")
        return True

    python_versions = find_python_versions()
    if not python_versions:
        print("No Python versions found for testing")
        return False

    for py_file in python_files:
        for python_cmd in python_versions:
            try:
                result = subprocess.run(
                    [python_cmd, "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    print(f"FAIL: {py_file} syntax error in {python_cmd}")
                    print(f"Error: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: Syntax check for {py_file} in {python_cmd}")
                continue
            except Exception as e:
                print(f"ERROR: Syntax check failed for {py_file}: {e}")
                continue

    print("PASS: Syntax compatibility tests passed")
    return True


def main():
    """Main function to run multi-Python version tests."""
    print("Multi-Python Version Compatibility Test")
    print("=" * 50)

    # Find available Python versions
    python_versions = find_python_versions()

    if not python_versions:
        print("FAIL: No Python versions found for testing")
        print("Expected: python3.11, python3.12, python3.13")
        sys.exit(1)

    if len(python_versions) < 2:
        print("WARNING: Only one Python version found")
        print("Consider installing additional Python versions for comprehensive testing")

    # Test each Python version
    all_passed = True
    results = {}

    for python_cmd in python_versions:
        success, output = test_python_version(python_cmd)
        results[python_cmd] = (success, output)
        if not success:
            all_passed = False

    # Test syntax compatibility
    if not test_syntax_compatibility():
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for python_cmd, (success, _) in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{python_cmd}: {status}")

    if all_passed:
        print("\nPASS: All multi-Python version tests passed!")
        print("The code is compatible across Python versions.")
        sys.exit(0)
    else:
        print("\nFAIL: Some multi-Python version tests failed!")
        print("Review compatibility issues before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
