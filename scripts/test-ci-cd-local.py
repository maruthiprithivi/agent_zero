#!/usr/bin/env python3
"""
Local CI/CD Testing Script

This script simulates the CI/CD pipeline locally to catch issues before
they reach GitHub Actions. It tests multiple Python versions, dependencies,
and clears caches as needed.
"""

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clear_caches():
    """Clear various caches that might cause stale dependency issues."""
    print("Clearing caches...")

    # Clear UV cache
    uv_cache_dirs = [
        Path.home() / ".cache" / "uv",
        Path.home() / ".local" / "share" / "uv",
        Path("/tmp") / "uv-cache",
    ]

    for cache_dir in uv_cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"CLEARED: {cache_dir}")
            except Exception as e:
                print(f"WARNING: Could not clear {cache_dir}: {e}")

    # Clear pip cache
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("CLEARED: pip cache")
        else:
            print(f"WARNING: Could not clear pip cache: {result.stderr}")
    except Exception as e:
        print(f"WARNING: Could not clear pip cache: {e}")

    # Clear Python cache
    pycache_dirs = list(Path().rglob("__pycache__"))
    for cache_dir in pycache_dirs:
        with contextlib.suppress(Exception):
            shutil.rmtree(cache_dir)

    if pycache_dirs:
        print(f"CLEARED: {len(pycache_dirs)} __pycache__ directories")


def test_dependency_resolution(python_cmd: str = "python3") -> bool:
    """Test dependency resolution with specified Python version."""
    print(f"\nTesting dependency resolution with {python_cmd}...")

    try:
        # Test using our existing script
        result = subprocess.run(
            [python_cmd, "scripts/test-dependencies.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print(f"PASS: Dependency resolution test with {python_cmd}")
            return True
        else:
            print(f"FAIL: Dependency resolution test with {python_cmd}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Dependency resolution test with {python_cmd}")
        return False
    except Exception as e:
        print(f"ERROR: Dependency resolution test failed: {e}")
        return False


def test_installation(python_cmd: str = "python3") -> bool:
    """Test installation in a temporary virtual environment."""
    print(f"\nTesting installation with {python_cmd}...")

    # Create temporary directory for testing
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Create virtual environment
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(temp_path / "test_env")],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                print(f"FAIL: Could not create virtual environment: {result.stderr}")
                return False

            # Activate virtual environment and install
            if os.name == "nt":  # Windows
                pip_cmd = str(temp_path / "test_env" / "Scripts" / "pip")
                python_env_cmd = str(temp_path / "test_env" / "Scripts" / "python")
            else:  # Unix/Linux/macOS
                pip_cmd = str(temp_path / "test_env" / "bin" / "pip")
                python_env_cmd = str(temp_path / "test_env" / "bin" / "python")

            # Install the package in editable mode
            result = subprocess.run(
                [pip_cmd, "install", "-e", ".[dev]"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                cwd=Path.cwd(),
            )

            if result.returncode == 0:
                print(f"PASS: Installation test with {python_cmd}")

                # Test basic import
                result = subprocess.run(
                    [python_env_cmd, "-c", "import agent_zero; print('Import successful')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    print(f"PASS: Import test with {python_cmd}")
                    return True
                else:
                    print(f"FAIL: Import test with {python_cmd}: {result.stderr}")
                    return False
            else:
                print(f"FAIL: Installation test with {python_cmd}")
                print(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: Installation test with {python_cmd}")
            return False
        except Exception as e:
            print(f"ERROR: Installation test failed: {e}")
            return False


def run_linting() -> bool:
    """Run linting checks locally."""
    print("\nRunning linting checks...")

    checks = [
        (["ruff", "check", "agent_zero"], "Ruff linting"),
        (["ruff", "format", "--check", "agent_zero"], "Ruff formatting"),
        (["black", "--check", "agent_zero"], "Black formatting"),
    ]

    all_passed = True

    for cmd, name in checks:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                all_passed = False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"SKIP: {name} - {e}")

    return all_passed


def main():
    """Main function to run local CI/CD simulation."""
    print("Local CI/CD Pipeline Simulation")
    print("=" * 50)

    # Parse command line arguments
    clear_cache = "--clear-cache" in sys.argv
    skip_install = "--skip-install" in sys.argv
    python_versions = []

    for arg in sys.argv[1:]:
        if arg.startswith("--python="):
            python_versions.append(arg.split("=", 1)[1])

    if not python_versions:
        python_versions = ["python3.11", "python3.12", "python3.13"]

    print(f"Testing Python versions: {python_versions}")
    print(f"Clear cache: {clear_cache}")
    print(f"Skip installation test: {skip_install}")

    # Clear caches if requested
    if clear_cache:
        clear_caches()

    # Test each Python version
    all_passed = True

    for python_cmd in python_versions:
        print(f"\n{'-' * 30}")
        print(f"Testing {python_cmd}")
        print(f"{'-' * 30}")

        # Check if Python version is available
        try:
            result = subprocess.run(
                [python_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                print(f"SKIP: {python_cmd} not available")
                continue
            print(f"Using: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"SKIP: {python_cmd} not available")
            continue

        # Test dependency resolution
        if not test_dependency_resolution(python_cmd):
            all_passed = False
            continue

        # Test installation (optional)
        if not skip_install and not test_installation(python_cmd):
            all_passed = False
            continue

    # Run multi-Python compatibility test
    print(f"\n{'-' * 30}")
    print("Multi-Python Compatibility Test")
    print(f"{'-' * 30}")

    try:
        result = subprocess.run(
            ["python3", "scripts/test-multi-python.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("PASS: Multi-Python compatibility test")
        else:
            print("FAIL: Multi-Python compatibility test")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            all_passed = False
    except Exception as e:
        print(f"ERROR: Multi-Python compatibility test failed: {e}")
        all_passed = False

    # Run linting checks
    if not run_linting():
        all_passed = False

    # Summary
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS")
    print(f"{'=' * 50}")

    if all_passed:
        print("PASS: All local CI/CD tests passed!")
        print("The code should work correctly in GitHub Actions.")
        sys.exit(0)
    else:
        print("FAIL: Some local CI/CD tests failed!")
        print("Fix issues before pushing to GitHub.")
        sys.exit(1)


if __name__ == "__main__":
    main()
