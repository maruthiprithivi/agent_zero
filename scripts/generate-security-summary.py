#!/usr/bin/env python3
"""
Security scan summary generator for Agent Zero.

This script aggregates security scan results from multiple tools and generates
a comprehensive summary report.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class SecuritySummaryGenerator:
    """Generates security summary from scan results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "UNKNOWN",
            "total_vulnerabilities": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "tools_run": [],
            "findings": [],
            "recommendations": [],
        }

    def generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive security summary."""
        print("Generating security summary...")

        # Process each scan type
        self.process_sast_results()
        self.process_dependency_results()
        self.process_container_results()
        self.process_secrets_results()

        # Determine overall status
        self.determine_overall_status()

        # Generate recommendations
        self.generate_recommendations()

        # Save summary
        self.save_summary()

        return self.summary

    def process_sast_results(self) -> None:
        """Process SAST scan results."""
        print("  Processing SAST results...")

        # Process Bandit results
        bandit_file = self.find_file("bandit-report.json")
        if bandit_file:
            self.process_bandit_results(bandit_file)
            self.summary["tools_run"].append("Bandit")

        # Process Semgrep results (would be in SARIF format)
        semgrep_files = list(self.results_dir.rglob("*semgrep*.sarif"))
        if semgrep_files:
            self.summary["tools_run"].append("Semgrep")

    def process_bandit_results(self, file_path: Path) -> None:
        """Process Bandit scan results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "results" in data:
                for finding in data["results"]:
                    severity = finding.get("issue_severity", "MEDIUM").upper()
                    confidence = finding.get("issue_confidence", "MEDIUM").upper()

                    self.summary["total_vulnerabilities"] += 1
                    self.increment_severity_count(severity)

                    self.summary["findings"].append(
                        {
                            "tool": "Bandit",
                            "type": "SAST",
                            "severity": severity,
                            "confidence": confidence,
                            "title": finding.get("test_name", "Unknown"),
                            "description": finding.get("issue_text", ""),
                            "file": finding.get("filename", ""),
                            "line": finding.get("line_number", 0),
                        }
                    )
        except Exception as e:
            print(f"  Warning: Error processing Bandit results: {e}")

    def process_dependency_results(self) -> None:
        """Process dependency scan results."""
        print("  Processing dependency scan results...")

        # Process Safety results
        safety_file = self.find_file("safety-report.json")
        if safety_file:
            self.process_safety_results(safety_file)
            self.summary["tools_run"].append("Safety")

        # Process pip-audit results
        pip_audit_file = self.find_file("pip-audit-report.json")
        if pip_audit_file:
            self.process_pip_audit_results(pip_audit_file)
            self.summary["tools_run"].append("pip-audit")

        # Process Snyk results
        snyk_file = self.find_file("snyk-report.json")
        if snyk_file:
            self.process_snyk_results(snyk_file)
            self.summary["tools_run"].append("Snyk")

    def process_safety_results(self, file_path: Path) -> None:
        """Process Safety scan results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                for vulnerability in data:
                    severity = self.map_cvss_to_severity(vulnerability.get("advisory", ""))

                    self.summary["total_vulnerabilities"] += 1
                    self.increment_severity_count(severity)

                    self.summary["findings"].append(
                        {
                            "tool": "Safety",
                            "type": "Dependency",
                            "severity": severity,
                            "title": f"Vulnerable package: {vulnerability.get('package_name', 'Unknown')}",
                            "description": vulnerability.get("advisory", ""),
                            "package": vulnerability.get("package_name", ""),
                            "installed_version": vulnerability.get("installed_version", ""),
                            "vulnerable_spec": vulnerability.get("vulnerable_spec", ""),
                        }
                    )
        except Exception as e:
            print(f"  Warning: Error processing Safety results: {e}")

    def process_pip_audit_results(self, file_path: Path) -> None:
        """Process pip-audit scan results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            vulnerabilities = data.get("vulnerabilities", [])
            for vulnerability in vulnerabilities:
                severity = "MEDIUM"  # pip-audit doesn't provide severity scores

                self.summary["total_vulnerabilities"] += 1
                self.increment_severity_count(severity)

                self.summary["findings"].append(
                    {
                        "tool": "pip-audit",
                        "type": "Dependency",
                        "severity": severity,
                        "title": f"Vulnerable package: {vulnerability.get('package', 'Unknown')}",
                        "description": vulnerability.get("description", ""),
                        "package": vulnerability.get("package", ""),
                        "installed_version": vulnerability.get("installed_version", ""),
                        "fix_versions": vulnerability.get("fix_versions", []),
                    }
                )
        except Exception as e:
            print(f"  Warning: Error processing pip-audit results: {e}")

    def process_snyk_results(self, file_path: Path) -> None:
        """Process Snyk scan results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            vulnerabilities = data.get("vulnerabilities", [])
            for vulnerability in vulnerabilities:
                severity = vulnerability.get("severity", "medium").upper()

                self.summary["total_vulnerabilities"] += 1
                self.increment_severity_count(severity)

                self.summary["findings"].append(
                    {
                        "tool": "Snyk",
                        "type": "Dependency",
                        "severity": severity,
                        "title": vulnerability.get("title", "Unknown vulnerability"),
                        "description": vulnerability.get("description", ""),
                        "package": vulnerability.get("packageName", ""),
                        "version": vulnerability.get("version", ""),
                        "cvss_score": vulnerability.get("cvssScore", 0),
                    }
                )
        except Exception as e:
            print(f"  Warning: Error processing Snyk results: {e}")

    def process_container_results(self) -> None:
        """Process container scan results."""
        print("  Processing container scan results...")

        # Look for Trivy results (SARIF format)
        trivy_files = list(self.results_dir.rglob("*trivy*.sarif"))
        if trivy_files:
            self.summary["tools_run"].append("Trivy")

        # Look for Docker Scout results
        scout_files = list(self.results_dir.rglob("*scout*.sarif"))
        if scout_files:
            self.summary["tools_run"].append("Docker Scout")

    def process_secrets_results(self) -> None:
        """Process secrets scan results."""
        print("  Processing secrets scan results...")

        # TruffleHog and GitLeaks results would be processed here
        # For now, just check if the tools ran

        # This is a placeholder - actual implementation would parse SARIF or JSON results
        if any(self.results_dir.rglob("*trufflehog*")):
            self.summary["tools_run"].append("TruffleHog")

        if any(self.results_dir.rglob("*gitleaks*")):
            self.summary["tools_run"].append("GitLeaks")

    def increment_severity_count(self, severity: str) -> None:
        """Increment the count for a given severity level."""
        severity = severity.upper()
        if severity == "CRITICAL":
            self.summary["critical_count"] += 1
        elif severity == "HIGH":
            self.summary["high_count"] += 1
        elif severity == "MEDIUM":
            self.summary["medium_count"] += 1
        elif severity == "LOW":
            self.summary["low_count"] += 1

    def map_cvss_to_severity(self, advisory: str) -> str:
        """Map CVSS score or advisory text to severity level."""
        advisory_lower = advisory.lower()
        if "critical" in advisory_lower:
            return "CRITICAL"
        elif "high" in advisory_lower:
            return "HIGH"
        elif "low" in advisory_lower:
            return "LOW"
        else:
            return "MEDIUM"

    def determine_overall_status(self) -> None:
        """Determine overall security status."""
        if self.summary["critical_count"] > 0:
            self.summary["overall_status"] = "CRITICAL"
        elif self.summary["high_count"] > 0:
            self.summary["overall_status"] = "HIGH"
        elif self.summary["medium_count"] > 0:
            self.summary["overall_status"] = "MEDIUM"
        elif self.summary["low_count"] > 0:
            self.summary["overall_status"] = "LOW"
        else:
            self.summary["overall_status"] = "CLEAN"

    def generate_recommendations(self) -> None:
        """Generate security recommendations."""
        recommendations = []

        if self.summary["critical_count"] > 0:
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: Critical vulnerabilities detected. "
                "Address these issues before deploying to production."
            )

        if self.summary["high_count"] > 0:
            recommendations.append(
                "HIGH PRIORITY: High severity vulnerabilities detected. "
                "Plan to address these issues in the current sprint."
            )

        # Dependency-specific recommendations
        dependency_findings = [f for f in self.summary["findings"] if f["type"] == "Dependency"]
        if dependency_findings:
            recommendations.append(
                "Dependency Updates: Review and update vulnerable dependencies. "
                "Consider using `pip install --upgrade` or `uv pip install --upgrade`."
            )

        # SAST-specific recommendations
        sast_findings = [f for f in self.summary["findings"] if f["type"] == "SAST"]
        if sast_findings:
            recommendations.append(
                "Code Review: Static analysis found potential security issues. "
                "Review flagged code and implement security best practices."
            )

        if not self.summary["findings"]:
            recommendations.append(
                "Good Security Posture: No significant vulnerabilities detected. "
                "Continue following security best practices."
            )

        self.summary["recommendations"] = recommendations

    def find_file(self, filename: str) -> Path | None:
        """Find a file in the results directory."""
        matches = list(self.results_dir.rglob(filename))
        return matches[0] if matches else None

    def save_summary(self) -> None:
        """Save summary to files."""
        # Save JSON summary
        json_file = Path("security-summary.json")
        with open(json_file, "w") as f:
            json.dump(self.summary, f, indent=2)

        # Save Markdown summary
        markdown_file = Path("security-summary.md")
        with open(markdown_file, "w") as f:
            f.write(self.generate_markdown_summary())

        print(f"  Summary saved to {json_file} and {markdown_file}")

    def generate_markdown_summary(self) -> str:
        """Generate markdown summary report."""
        markdown = f"""# Security Scan Summary

## Overall Status: {self.summary["overall_status"]}

**Scan completed:** {self.summary["timestamp"]}
**Total vulnerabilities found:** {self.summary["total_vulnerabilities"]}

### Vulnerability Breakdown

| Severity | Count |
|----------|-------|
| Critical | {self.summary["critical_count"]} |
| High | {self.summary["high_count"]} |
| Medium | {self.summary["medium_count"]} |
| Low | {self.summary["low_count"]} |

### Security Tools Used

{", ".join(self.summary["tools_run"]) if self.summary["tools_run"] else "No tools completed successfully"}

### Key Findings

"""

        if self.summary["findings"]:
            # Group findings by severity
            critical_findings = [f for f in self.summary["findings"] if f["severity"] == "CRITICAL"]
            high_findings = [f for f in self.summary["findings"] if f["severity"] == "HIGH"]

            if critical_findings:
                markdown += "#### Critical Issues\n\n"
                for finding in critical_findings[:5]:  # Limit to top 5
                    markdown += f"- **{finding['title']}** ({finding['tool']})\n"
                    markdown += f"  - {finding['description'][:100]}...\n\n"

            if high_findings:
                markdown += "#### High Severity Issues\n\n"
                for finding in high_findings[:5]:  # Limit to top 5
                    markdown += f"- **{finding['title']}** ({finding['tool']})\n"
                    markdown += f"  - {finding['description'][:100]}...\n\n"
        else:
            markdown += "No significant security issues found.\n\n"

        markdown += "### Recommendations\n\n"
        for rec in self.summary["recommendations"]:
            markdown += f"- {rec}\n"

        markdown += """
### Detailed Results

For complete details, download the security scan artifacts from the workflow run.

---
*This summary was automatically generated by Agent Zero security scanning pipeline.*
"""

        return markdown


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate security scan summary for Agent Zero")
    parser.add_argument("results_dir", type=Path, help="Directory containing security scan results")
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd(), help="Output directory for summary files"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Change to output directory
    os.chdir(args.output_dir)

    generator = SecuritySummaryGenerator(args.results_dir)
    summary = generator.generate_summary()

    print("\nSecurity Summary Generated:")
    print(f"   Overall Status: {summary['overall_status']}")
    print(f"   Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"   Tools Run: {len(summary['tools_run'])}")

    # Exit with appropriate code
    if summary["overall_status"] in ["CRITICAL", "HIGH"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
