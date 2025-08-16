#!/usr/bin/env python3
"""
Cross-reference validation script for Agent Zero documentation.

This script validates internal links, cross-references, and documentation consistency
across all markdown files in the project.
"""

import argparse
import json
import re
import sys
from pathlib import Path


class DocumentationValidator:
    """Validates documentation cross-references and consistency."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.file_references: dict[str, set[str]] = {}
        self.external_links: set[str] = set()

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Starting documentation validation...")

        # Find all markdown files
        md_files = list(self.project_root.rglob("*.md"))
        print(f"📝 Found {len(md_files)} markdown files")

        # Validate each file
        for md_file in md_files:
            self.validate_file(md_file)

        # Cross-validation
        self.validate_cross_references()
        self.validate_consistency()

        # Report results
        self.report_results()

        return len(self.errors) == 0

    def validate_file(self, file_path: Path) -> None:
        """Validate a single markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            rel_path = file_path.relative_to(self.project_root)

            # Extract and validate links
            self.extract_links(content, rel_path)

            # Validate structure
            self.validate_structure(content, rel_path)

            # Validate code blocks
            self.validate_code_blocks(content, rel_path)

        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")

    def extract_links(self, content: str, file_path: Path) -> None:
        """Extract and categorize links from markdown content."""
        # Markdown link pattern: [text](url)
        link_pattern = r"\[([^\]]*)\]\(([^)]+)\)"
        links = re.findall(link_pattern, content)

        self.file_references[str(file_path)] = set()

        for text, url in links:
            if self.is_external_link(url):
                self.external_links.add(url)
            else:
                # Internal link
                self.file_references[str(file_path)].add(url)
                self.validate_internal_link(url, file_path)

    def is_external_link(self, url: str) -> bool:
        """Check if link is external."""
        return url.startswith(("http://", "https://", "mailto:", "ftp://"))

    def validate_internal_link(self, url: str, source_file: Path) -> None:
        """Validate internal link exists."""
        # Handle relative paths
        if url.startswith("#"):
            # Anchor link - validate in same file
            self.validate_anchor_link(url, source_file)
            return

        # Resolve relative path
        if url.startswith("./") or url.startswith("../") or not url.startswith("/"):
            target_path = (source_file.parent / url).resolve()
        else:
            target_path = (self.project_root / url.lstrip("/")).resolve()

        # Check if target exists
        if not target_path.exists():
            self.errors.append(f"Broken internal link in {source_file}: {url} -> {target_path}")

    def validate_anchor_link(self, anchor: str, file_path: Path) -> None:
        """Validate anchor link exists in the file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            anchor_id = anchor.lstrip("#").lower().replace(" ", "-")

            # Look for matching headers
            header_pattern = r"^#+\s+(.+)$"
            headers = re.findall(header_pattern, content, re.MULTILINE)

            header_ids = [h.lower().replace(" ", "-") for h in headers]

            if anchor_id not in header_ids:
                self.warnings.append(f"Anchor link may not exist in {file_path}: {anchor}")
        except Exception as e:
            self.errors.append(f"Error validating anchor in {file_path}: {e}")

    def validate_structure(self, content: str, file_path: Path) -> None:
        """Validate document structure."""
        lines = content.split("\n")

        # Check for title (H1)
        has_title = any(line.startswith("# ") for line in lines)
        if not has_title and file_path.name != "README.md":
            self.warnings.append(f"No H1 title found in {file_path}")

        # Check heading hierarchy
        self.validate_heading_hierarchy(lines, file_path)

        # Check for table of contents in long documents
        if len(lines) > 200:
            has_toc = any("table of contents" in line.lower() for line in lines)
            if not has_toc:
                self.warnings.append(f"Long document {file_path} should have table of contents")

    def validate_heading_hierarchy(self, lines: list[str], file_path: Path) -> None:
        """Validate heading hierarchy follows H1 > H2 > H3 pattern."""
        heading_levels = []

        for i, line in enumerate(lines, 1):
            if line.startswith("#"):
                level = len(line.split()[0])  # Count # characters
                heading_levels.append((level, i))

        # Check for proper hierarchy
        for i in range(1, len(heading_levels)):
            current_level, line_num = heading_levels[i]
            prev_level, _ = heading_levels[i - 1]

            if current_level > prev_level + 1:
                self.warnings.append(
                    f"Heading hierarchy jump in {file_path} line {line_num}: "
                    f"H{prev_level} to H{current_level}"
                )

    def validate_code_blocks(self, content: str, file_path: Path) -> None:
        """Validate code blocks are properly formatted."""
        # Check for unclosed code blocks
        code_block_pattern = r"```([a-zA-Z]*)"
        code_blocks = re.findall(code_block_pattern, content)

        if len(code_blocks) % 2 != 0:
            self.errors.append(f"Unclosed code block in {file_path}")

        # Check for language specification
        fenced_blocks = content.count("```")
        if fenced_blocks > 0:
            unspecified_blocks = content.count("```\n")
            if unspecified_blocks > fenced_blocks // 2:
                self.warnings.append(f"Code blocks in {file_path} should specify language")

    def validate_cross_references(self) -> None:
        """Validate cross-references between documents."""
        print("🔗 Validating cross-references...")

        # Check for orphaned files (no incoming links)
        all_files = set(self.file_references.keys())
        referenced_files = set()

        for file_refs in self.file_references.values():
            referenced_files.update(file_refs)

        # Note: This is a simplified check - in practice, you'd resolve paths
        orphaned = all_files - referenced_files
        for orphan in orphaned:
            if not any(x in orphan for x in ["README.md", "CLAUDE.md"]):
                self.warnings.append(f"Potentially orphaned file: {orphan}")

    def validate_consistency(self) -> None:
        """Validate consistency across documentation."""
        print("📊 Validating consistency...")

        # Check for consistent terminology
        self.validate_terminology()

        # Check for consistent code examples
        self.validate_code_consistency()

    def validate_terminology(self) -> None:
        """Check for consistent terminology usage."""
        # Define standard terms
        standard_terms = {
            "clickhouse": "ClickHouse",
            "mcp": "MCP",
            "api": "API",
            "ssl": "SSL",
            "tls": "TLS",
            "oauth": "OAuth",
        }

        md_files = list(self.project_root.rglob("*.md"))

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")

                for incorrect, correct in standard_terms.items():
                    # Look for incorrect usage (case-sensitive)
                    if incorrect in content and correct not in content:
                        self.warnings.append(
                            f"Inconsistent terminology in {md_file}: "
                            f"use '{correct}' instead of '{incorrect}'"
                        )
            except Exception:
                continue

    def validate_code_consistency(self) -> None:
        """Validate code examples follow consistent patterns."""
        # This would check for consistent:
        # - Import statements
        # - Error handling patterns
        # - Configuration usage
        # - Function signatures

        # Simplified implementation
        print("  ✓ Code consistency validation (placeholder)")

    def report_results(self) -> None:
        """Report validation results."""
        print("\n" + "=" * 60)
        print("📋 DOCUMENTATION VALIDATION RESULTS")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All documentation validation checks passed!")

        print("\n📈 SUMMARY:")
        print(f"  • Files validated: {len(self.file_references)}")
        print(f"  • External links found: {len(self.external_links)}")
        print(f"  • Errors: {len(self.errors)}")
        print(f"  • Warnings: {len(self.warnings)}")

    def export_results(self, output_file: Path) -> None:
        """Export validation results to JSON."""
        results = {
            "timestamp": "2025-01-16T00:00:00Z",  # Would use actual timestamp
            "summary": {
                "files_validated": len(self.file_references),
                "external_links": len(self.external_links),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "external_links": list(self.external_links),
        }

        output_file.write_text(json.dumps(results, indent=2))
        print(f"📄 Results exported to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Agent Zero documentation cross-references"
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )
    parser.add_argument("--output", type=Path, help="Export results to JSON file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    args = parser.parse_args()

    validator = DocumentationValidator(args.project_root)
    success = validator.validate_all()

    if args.output:
        validator.export_results(args.output)

    # Exit with error code if validation failed
    if not success or (args.strict and validator.warnings):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
