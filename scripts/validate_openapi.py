#!/usr/bin/env python3
"""
Validate OpenAPI specification files.

This script validates OpenAPI specification files for correctness,
completeness, and adherence to OpenAPI standards.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. YAML validation will be limited.")
    yaml = None


def validate_openapi_spec(spec_path: Path) -> bool:
    """Validate OpenAPI specification file."""
    print(f"Validating OpenAPI specification: {spec_path}")

    try:
        # Load the specification
        if spec_path.suffix.lower() == ".yaml" or spec_path.suffix.lower() == ".yml":
            if yaml:
                with open(spec_path) as f:
                    spec = yaml.safe_load(f)
            else:
                print("⚠️  Cannot validate YAML files without PyYAML installed")
                return False
        else:
            with open(spec_path) as f:
                spec = json.load(f)

        # Basic validation checks
        errors = []
        warnings = []

        # Check required fields
        required_fields = ["openapi", "info", "paths"]
        for field in required_fields:
            if field not in spec:
                errors.append(f"Missing required field: {field}")

        # Validate OpenAPI version
        if "openapi" in spec:
            version = spec["openapi"]
            if not version.startswith("3."):
                warnings.append(f"OpenAPI version {version} is not 3.x")

        # Validate info section
        if "info" in spec:
            info = spec["info"]
            info_required = ["title", "version"]
            for field in info_required:
                if field not in info:
                    errors.append(f"Missing required info field: {field}")

        # Validate paths
        if "paths" in spec:
            paths = spec["paths"]
            if not paths:
                warnings.append("No paths defined in the specification")

            for path, path_item in paths.items():
                if not path.startswith("/"):
                    errors.append(f"Path should start with '/': {path}")

                # Validate HTTP methods
                valid_methods = [
                    "get",
                    "post",
                    "put",
                    "delete",
                    "options",
                    "head",
                    "patch",
                    "trace",
                ]
                for method in path_item:
                    if method not in valid_methods + ["parameters", "summary", "description"]:
                        warnings.append(f"Unknown method or field in path {path}: {method}")

        # Validate components if present
        if "components" in spec:
            components = spec["components"]
            if "schemas" in components:
                schemas = components["schemas"]
                for schema_name, schema_def in schemas.items():
                    if "type" not in schema_def and "$ref" not in schema_def:
                        warnings.append(f"Schema {schema_name} missing type or $ref")

        # Report results
        if errors:
            print("❌ Validation errors found:")
            for error in errors:
                print(f"  • {error}")

        if warnings:
            print("⚠️  Validation warnings:")
            for warning in warnings:
                print(f"  • {warning}")

        if not errors and not warnings:
            print("✅ OpenAPI specification is valid")
        elif not errors:
            print("✅ OpenAPI specification is valid (with warnings)")

        return len(errors) == 0

    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main function to validate OpenAPI specifications."""
    print("OpenAPI Specification Validator")
    print("=" * 40)

    # Look for OpenAPI specs
    spec_files = []

    # Check common locations
    common_paths = [
        "docs/api/openapi.yaml",
        "docs/api/openapi.yml",
        "docs/api/openapi.json",
        "docs/api/generated/openapi.yaml",
        "docs/api/generated/openapi.json",
    ]

    for path_str in common_paths:
        path = Path(path_str)
        if path.exists():
            spec_files.append(path)

    if not spec_files:
        print("❌ No OpenAPI specification files found")
        print("Looked in:")
        for path in common_paths:
            print(f"  • {path}")
        sys.exit(1)

    # Validate all found specifications
    all_valid = True
    for spec_file in spec_files:
        is_valid = validate_openapi_spec(spec_file)
        all_valid = all_valid and is_valid
        print()

    if all_valid:
        print("🎉 All OpenAPI specifications are valid!")
        sys.exit(0)
    else:
        print("❌ Some OpenAPI specifications have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
