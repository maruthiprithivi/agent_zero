#!/usr/bin/env python3
"""
Generate sitemap.xml for documentation site.

This script generates a sitemap for the documentation website
to improve SEO and search engine crawling.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin


def find_html_files(site_dir: Path) -> list[Path]:
    """Find all HTML files in the site directory."""
    html_files = []
    for html_file in site_dir.rglob("*.html"):
        # Skip files that shouldn't be in sitemap
        if html_file.name.startswith("404") or "search" in html_file.name.lower():
            continue
        html_files.append(html_file)
    return html_files


def generate_sitemap(site_dir: Path, base_url: str) -> str:
    """Generate sitemap XML content."""
    html_files = find_html_files(site_dir)

    # Start XML
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]

    # Current timestamp
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Process each HTML file
    for html_file in html_files:
        # Get relative path
        rel_path = html_file.relative_to(site_dir)

        # Convert to URL
        if rel_path.name == "index.html":
            if str(rel_path.parent) == ".":
                url_path = ""
            else:
                url_path = str(rel_path.parent) + "/"
        else:
            url_path = str(rel_path).replace("\\", "/")
            if url_path.endswith(".html"):
                url_path = url_path[:-5] + "/"

        full_url = urljoin(base_url.rstrip("/") + "/", url_path)

        # Determine priority and change frequency
        priority = "0.5"
        changefreq = "monthly"

        if url_path == "" or url_path == "/":
            priority = "1.0"
            changefreq = "weekly"
        elif "getting-started" in url_path or "quick" in url_path:
            priority = "0.8"
            changefreq = "weekly"
        elif "api" in url_path:
            priority = "0.7"
            changefreq = "monthly"

        # Add URL entry
        sitemap_lines.extend(
            [
                "  <url>",
                f"    <loc>{full_url}</loc>",
                f"    <lastmod>{timestamp}</lastmod>",
                f"    <changefreq>{changefreq}</changefreq>",
                f"    <priority>{priority}</priority>",
                "  </url>",
            ]
        )

    # Close XML
    sitemap_lines.append("</urlset>")

    return "\n".join(sitemap_lines)


def main():
    """Main function to generate sitemap."""
    if len(sys.argv) != 3:
        print("Usage: python generate_sitemap.py <site_directory> <base_url>")
        print("Example: python generate_sitemap.py site/ https://docs.agent-zero.com")
        sys.exit(1)

    site_dir = Path(sys.argv[1])
    base_url = sys.argv[2]

    if not site_dir.exists():
        print(f"Error: Site directory does not exist: {site_dir}")
        sys.exit(1)

    print(f"Generating sitemap for {site_dir} with base URL {base_url}")

    # Generate sitemap
    sitemap_content = generate_sitemap(site_dir, base_url)

    # Write sitemap file
    sitemap_path = site_dir / "sitemap.xml"
    with open(sitemap_path, "w", encoding="utf-8") as f:
        f.write(sitemap_content)

    # Count URLs
    url_count = sitemap_content.count("<url>")

    print(f"✅ Sitemap generated: {sitemap_path}")
    print(f"📄 URLs included: {url_count}")
    print(f"🔗 Base URL: {base_url}")


if __name__ == "__main__":
    main()
