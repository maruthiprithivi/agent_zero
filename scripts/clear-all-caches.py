#!/usr/bin/env python3
"""
Clear All Development Caches

This script clears various caches that might cause issues during development
or CI/CD pipeline execution. Useful for troubleshooting dependency resolution
issues or stale cache problems.
"""

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path


def clear_uv_cache():
    """Clear UV package manager cache."""
    print("Clearing UV cache...")

    uv_cache_dirs = [
        Path.home() / ".cache" / "uv",
        Path.home() / ".local" / "share" / "uv",
        Path("/tmp") / "uv-cache",
        Path.cwd() / ".uv-cache",
    ]

    cleared = False
    for cache_dir in uv_cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"  CLEARED: {cache_dir}")
                cleared = True
            except Exception as e:
                print(f"  ERROR: Could not clear {cache_dir}: {e}")

    if not cleared:
        print("  INFO: No UV cache directories found")


def clear_pip_cache():
    """Clear pip cache."""
    print("Clearing pip cache...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  CLEARED: pip cache")
        else:
            print(f"  ERROR: Could not clear pip cache: {result.stderr}")
    except Exception as e:
        print(f"  ERROR: Could not clear pip cache: {e}")


def clear_python_cache():
    """Clear Python __pycache__ directories."""
    print("Clearing Python cache...")

    pycache_dirs = list(Path().rglob("__pycache__"))
    pyc_files = list(Path().rglob("*.pyc"))

    for cache_dir in pycache_dirs:
        with contextlib.suppress(Exception):
            shutil.rmtree(cache_dir)

    for pyc_file in pyc_files:
        with contextlib.suppress(Exception):
            pyc_file.unlink()

    total_cleared = len(pycache_dirs) + len(pyc_files)
    if total_cleared > 0:
        print(f"  CLEARED: {len(pycache_dirs)} __pycache__ dirs, {len(pyc_files)} .pyc files")
    else:
        print("  INFO: No Python cache files found")


def clear_build_artifacts():
    """Clear build artifacts."""
    print("Clearing build artifacts...")

    build_dirs = [
        Path() / "build",
        Path() / "dist",
        Path() / "*.egg-info",
    ]

    cleared = False
    for pattern in build_dirs:
        if "*" in str(pattern):
            # Handle glob patterns
            for path in Path().glob(pattern.name):
                if path.is_dir():
                    try:
                        shutil.rmtree(path)
                        print(f"  CLEARED: {path}")
                        cleared = True
                    except Exception as e:
                        print(f"  ERROR: Could not clear {path}: {e}")
        else:
            # Handle direct paths
            if pattern.exists():
                try:
                    if pattern.is_dir():
                        shutil.rmtree(pattern)
                    else:
                        pattern.unlink()
                    print(f"  CLEARED: {pattern}")
                    cleared = True
                except Exception as e:
                    print(f"  ERROR: Could not clear {pattern}: {e}")

    if not cleared:
        print("  INFO: No build artifacts found")


def clear_coverage_files():
    """Clear coverage files."""
    print("Clearing coverage files...")

    coverage_files = [
        Path() / ".coverage",
        Path() / "coverage.xml",
        Path() / "coverage.json",
        Path() / "htmlcov",
    ]

    cleared = False
    for path in coverage_files:
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"  CLEARED: {path}")
                cleared = True
            except Exception as e:
                print(f"  ERROR: Could not clear {path}: {e}")

    if not cleared:
        print("  INFO: No coverage files found")


def clear_test_artifacts():
    """Clear test artifacts."""
    print("Clearing test artifacts...")

    test_dirs = [
        Path() / ".pytest_cache",
        Path() / ".tox",
        Path() / "test-results",
    ]

    cleared = False
    for path in test_dirs:
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"  CLEARED: {path}")
                cleared = True
            except Exception as e:
                print(f"  ERROR: Could not clear {path}: {e}")

    if not cleared:
        print("  INFO: No test artifacts found")


def main():
    """Main function to clear all caches."""
    print("Cache Clearing Utility")
    print("=" * 30)

    # Parse command line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
Usage: python3 scripts/clear-all-caches.py [options]

Options:
  --uv-only      Clear only UV cache
  --pip-only     Clear only pip cache
  --python-only  Clear only Python cache
  --build-only   Clear only build artifacts
  --all          Clear all caches (default)
  --help, -h     Show this help message

Examples:
  python3 scripts/clear-all-caches.py
  python3 scripts/clear-all-caches.py --uv-only
  python3 scripts/clear-all-caches.py --all
"""
        )
        sys.exit(0)

    # Determine what to clear
    clear_specific = any(
        [
            "--uv-only" in sys.argv,
            "--pip-only" in sys.argv,
            "--python-only" in sys.argv,
            "--build-only" in sys.argv,
        ]
    )

    clear_all = "--all" in sys.argv or not clear_specific

    if clear_all or "--uv-only" in sys.argv:
        clear_uv_cache()

    if clear_all or "--pip-only" in sys.argv:
        clear_pip_cache()

    if clear_all or "--python-only" in sys.argv:
        clear_python_cache()

    if clear_all or "--build-only" in sys.argv:
        clear_build_artifacts()

    if clear_all:
        clear_coverage_files()
        clear_test_artifacts()

    print("\nCache clearing completed!")
    print("This should resolve most dependency resolution issues.")


if __name__ == "__main__":
    main()
