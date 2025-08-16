#!/usr/bin/env python3
"""
Version Management Utility for Agent Zero

This script provides utilities for managing versions across different branches
and deployment environments using setuptools-scm and PEP 440 standards.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: str, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd.split(), cwd=cwd or project_root, capture_output=True, text=True, check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def get_current_version() -> str:
    """Get the current version using setuptools-scm."""
    try:
        # Try to get version from setuptools-scm
        exit_code, stdout, stderr = run_command("python -m setuptools_scm")
        if exit_code == 0:
            return stdout.strip()

        # Fallback to importing from package
        from agent_zero import __version__

        return __version__
    except Exception as e:
        print(f"Error getting version: {e}")
        return "unknown"


def get_git_branch() -> str:
    """Get the current git branch."""
    exit_code, stdout, stderr = run_command("git branch --show-current")
    if exit_code == 0:
        return stdout.strip()
    return "unknown"


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    exit_code, stdout, stderr = run_command("git rev-parse --short HEAD")
    if exit_code == 0:
        return stdout.strip()
    return "unknown"


def is_clean_working_tree() -> bool:
    """Check if the working tree is clean."""
    exit_code, stdout, stderr = run_command("git status --porcelain")
    return exit_code == 0 and not stdout


def get_version_for_branch(branch: str, base_version: str = "0.2.0") -> str:
    """
    Generate appropriate version string based on branch name and PEP 440.

    Branch naming conventions:
    - main/master: stable releases (1.0.0)
    - develop: development releases (1.1.0.dev1)
    - release/*: release candidates (1.1.0rc1)
    - feature/*: alpha pre-releases (1.1.0a1.dev1)
    - hotfix/*: patch releases (1.0.1)
    """
    commit_hash = get_git_commit_hash()

    if branch in ["main", "master"]:
        # Stable release
        return base_version
    elif branch == "develop":
        # Development release
        return f"{base_version}.dev1+{commit_hash}"
    elif branch.startswith("release/"):
        # Release candidate
        rc_num = extract_number_from_branch(branch) or "1"
        return f"{base_version}rc{rc_num}"
    elif branch.startswith("feature/"):
        # Alpha pre-release with dev suffix
        alpha_num = extract_number_from_branch(branch) or "1"
        return f"{base_version}a{alpha_num}.dev1+{commit_hash}"
    elif branch.startswith("hotfix/"):
        # Patch release - increment patch version
        major, minor, patch = parse_version(base_version)
        return f"{major}.{minor}.{patch + 1}"
    else:
        # Unknown branch - development release
        return f"{base_version}.dev0+{commit_hash}.{branch.replace('/', '.')}"


def extract_number_from_branch(branch: str) -> str | None:
    """Extract version number from branch name."""
    # Look for patterns like release/1.0, feature/add-feature-v2, etc.
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", branch)
    if match:
        return match.group(1)
    return None


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into major, minor, patch."""
    # Remove any pre-release or local version parts
    clean_version = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if clean_version:
        return (
            int(clean_version.group(1)),
            int(clean_version.group(2)),
            int(clean_version.group(3)),
        )
    return (0, 2, 0)  # Default fallback


def validate_version(version: str) -> bool:
    """Validate version string against PEP 440."""
    # Simplified PEP 440 validation
    pep440_pattern = r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?(\+[a-z0-9]+(\.[a-z0-9]+)*)?$"
    return re.match(pep440_pattern, version.lower()) is not None


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type (major, minor, patch)."""
    major, minor, patch = parse_version(current)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def create_git_tag(version: str, message: str | None = None) -> bool:
    """Create a git tag for the version."""
    tag_name = f"v{version}"
    tag_message = message or f"Release version {version}"

    exit_code, stdout, stderr = run_command(f"git tag -a {tag_name} -m '{tag_message}'")
    if exit_code == 0:
        print(f"Created tag: {tag_name}")
        return True
    else:
        print(f"Failed to create tag: {stderr}")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Version management utility for Agent Zero")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Current version command
    current_parser = subparsers.add_parser("current", help="Show current version")

    # Version info command
    info_parser = subparsers.add_parser("info", help="Show version information")

    # Bump version command
    bump_parser = subparsers.add_parser("bump", help="Bump version")
    bump_parser.add_argument("type", choices=["major", "minor", "patch"], help="Version bump type")
    bump_parser.add_argument("--tag", action="store_true", help="Create git tag")

    # Branch version command
    branch_parser = subparsers.add_parser("branch", help="Get version for current branch")
    branch_parser.add_argument("--base", default="0.2.0", help="Base version")

    # Validate version command
    validate_parser = subparsers.add_parser("validate", help="Validate version string")
    validate_parser.add_argument("version", help="Version string to validate")

    # Tag command
    tag_parser = subparsers.add_parser("tag", help="Create git tag")
    tag_parser.add_argument("version", help="Version to tag")
    tag_parser.add_argument("-m", "--message", help="Tag message")

    args = parser.parse_args()

    if args.command == "current":
        print(get_current_version())

    elif args.command == "info":
        current_version = get_current_version()
        branch = get_git_branch()
        commit = get_git_commit_hash()
        clean = is_clean_working_tree()

        print(f"Current Version: {current_version}")
        print(f"Git Branch: {branch}")
        print(f"Commit Hash: {commit}")
        print(f"Working Tree: {'clean' if clean else 'dirty'}")
        print(f"Branch Version: {get_version_for_branch(branch)}")

    elif args.command == "bump":
        current_version = get_current_version()
        # Parse clean version without pre-release parts
        clean_current = re.match(r"^(\d+\.\d+\.\d+)", current_version)
        if clean_current:
            current_version = clean_current.group(1)

        new_version = bump_version(current_version, args.type)
        print(f"Bumped version: {current_version} -> {new_version}")

        if args.tag:
            create_git_tag(new_version)

    elif args.command == "branch":
        branch = get_git_branch()
        version = get_version_for_branch(branch, args.base)
        print(version)

    elif args.command == "validate":
        is_valid = validate_version(args.version)
        print(f"Version '{args.version}' is {'valid' if is_valid else 'invalid'}")
        sys.exit(0 if is_valid else 1)

    elif args.command == "tag":
        success = create_git_tag(args.version, args.message)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
