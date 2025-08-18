#!/usr/bin/env python3
"""
Configuration validation script for Agent Zero.

This script validates environment configurations, checks required environment
variables, and ensures deployment readiness across all environments.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


class ConfigurationValidator:
    """Validates Agent Zero configuration files and environment setup."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.environments: dict[str, dict[str, Any]] = {}

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("Starting configuration validation...")

        # Load all environment configurations
        self.load_configurations()

        # Validate each configuration
        for env_name, config in self.environments.items():
            print(f"Validating {env_name} environment...")
            self.validate_configuration(env_name, config)

        # Cross-environment validation
        self.validate_cross_environment()

        # Report results
        self.report_results()

        return len(self.errors) == 0

    def load_configurations(self) -> None:
        """Load all environment configuration files."""
        env_dir = self.config_dir / "environments"

        if not env_dir.exists():
            self.errors.append(f"Environment configurations directory not found: {env_dir}")
            return

        for config_file in env_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config = json.load(f)

                env_name = config_file.stem
                self.environments[env_name] = config

            except json.JSONDecodeError as e:
                self.errors.append(f"Invalid JSON in {config_file}: {e}")
            except Exception as e:
                self.errors.append(f"Error loading {config_file}: {e}")

    def validate_configuration(self, env_name: str, config: dict[str, Any]) -> None:
        """Validate a single environment configuration."""
        print(f"  Validating {env_name} configuration structure...")

        # Validate required top-level keys
        required_keys = ["environment", "deployment_mode", "server", "clickhouse", "security"]
        for key in required_keys:
            if key not in config:
                self.errors.append(f"{env_name}: Missing required configuration key: {key}")

        # Validate configuration schema version
        validation_config = config.get("validation", {})
        schema_version = validation_config.get("config_schema_version")
        if not schema_version:
            self.warnings.append(f"{env_name}: No configuration schema version specified")
        elif schema_version != "2.1.0":
            self.warnings.append(
                f"{env_name}: Configuration schema version {schema_version} may be outdated"
            )

        # Validate server configuration
        self.validate_server_config(env_name, config.get("server", {}))

        # Validate ClickHouse configuration
        self.validate_clickhouse_config(env_name, config.get("clickhouse", {}))

        # Validate security configuration
        self.validate_security_config(env_name, config.get("security", {}))

        # Validate environment variables
        self.validate_environment_variables(env_name, validation_config)

        # Validate resource requirements
        self.validate_resource_requirements(env_name, validation_config)

        # Environment-specific validations
        if env_name == "production":
            self.validate_production_config(config)
        elif env_name == "staging":
            self.validate_staging_config(config)
        elif env_name == "development":
            self.validate_development_config(config)

    def validate_server_config(self, env_name: str, server_config: dict[str, Any]) -> None:
        """Validate server configuration."""
        # Validate host
        host = server_config.get("host", "127.0.0.1")
        if env_name == "production" and host in ["127.0.0.1", "localhost"]:
            self.errors.append(f"{env_name}: Production server should not bind to localhost")

        # Validate port
        port = server_config.get("port")
        if not isinstance(port, int) or port < 1024 or port > 65535:
            self.errors.append(f"{env_name}: Invalid server port: {port}")

        # Validate transport
        transport = server_config.get("transport")
        valid_transports = ["stdio", "sse", "websocket", "http"]
        if transport not in valid_transports:
            self.errors.append(f"{env_name}: Invalid transport: {transport}")

        # Validate log level
        log_level = server_config.get("log_level")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            self.errors.append(f"{env_name}: Invalid log level: {log_level}")

        # Production-specific server validation
        if env_name == "production":
            if not server_config.get("workers") or server_config.get("workers") < 2:
                self.warnings.append(f"{env_name}: Production should use multiple workers")

            if server_config.get("reload", False):
                self.errors.append(f"{env_name}: Auto-reload should be disabled in production")

    def validate_clickhouse_config(self, env_name: str, ch_config: dict[str, Any]) -> None:
        """Validate ClickHouse configuration."""
        # Validate required fields
        required_fields = ["host", "port", "user"]
        for field in required_fields:
            if not ch_config.get(field):
                self.errors.append(f"{env_name}: Missing ClickHouse {field}")

        # Validate port
        port = ch_config.get("port")
        if isinstance(port, str) and port.startswith("${"):
            # Environment variable - validate format
            if not re.match(r"^\$\{[A-Z_]+(?::-\d+)?\}$", port):
                self.errors.append(
                    f"{env_name}: Invalid environment variable format for ClickHouse port: {port}"
                )
        elif isinstance(port, int):
            if port not in [8123, 9000, 8443, 9440]:
                self.warnings.append(f"{env_name}: Unusual ClickHouse port: {port}")

        # Validate security settings
        if env_name in ["staging", "production"]:
            if not ch_config.get("secure"):
                self.errors.append(
                    f"{env_name}: ClickHouse secure connection required for {env_name}"
                )

            if not ch_config.get("verify"):
                self.warnings.append(
                    f"{env_name}: ClickHouse certificate verification should be enabled"
                )

        # Validate timeouts
        connect_timeout = ch_config.get("connect_timeout", 30)
        if connect_timeout < 5 or connect_timeout > 120:
            self.warnings.append(
                f"{env_name}: ClickHouse connect_timeout seems unusual: {connect_timeout}s"
            )

        # Validate connection pool
        pool_size = ch_config.get("pool_size", 5)
        max_overflow = ch_config.get("max_overflow", 10)

        if env_name == "production":
            if pool_size < 10:
                self.warnings.append(f"{env_name}: Consider larger connection pool for production")
            if max_overflow < pool_size:
                self.warnings.append(f"{env_name}: max_overflow should be >= pool_size")

    def validate_security_config(self, env_name: str, security_config: dict[str, Any]) -> None:
        """Validate security configuration."""
        # SSL validation
        if env_name in ["staging", "production"]:
            if not security_config.get("ssl_enable"):
                self.errors.append(f"{env_name}: SSL must be enabled for {env_name}")

            ssl_cert = security_config.get("ssl_certfile")
            ssl_key = security_config.get("ssl_keyfile")

            if ssl_cert and not ssl_cert.startswith("${"):
                # Not an environment variable - check file exists
                if not Path(ssl_cert).exists():
                    self.errors.append(f"{env_name}: SSL certificate file not found: {ssl_cert}")

            if ssl_key and not ssl_key.startswith("${"):
                if not Path(ssl_key).exists():
                    self.errors.append(f"{env_name}: SSL key file not found: {ssl_key}")

        # Authentication validation
        if env_name in ["staging", "production"]:
            if not security_config.get("auth_required"):
                self.errors.append(f"{env_name}: Authentication required for {env_name}")

        # CORS validation
        cors_origins = security_config.get("cors_origins", [])
        if env_name == "production" and "*" in cors_origins:
            self.errors.append(f"{env_name}: Wildcard CORS origins not allowed in production")

        # Rate limiting
        if env_name in ["staging", "production"]:
            if not security_config.get("rate_limit_enabled"):
                self.warnings.append(f"{env_name}: Rate limiting should be enabled for {env_name}")

        # Security headers
        if env_name in ["staging", "production"]:
            if not security_config.get("security_headers"):
                self.warnings.append(
                    f"{env_name}: Security headers should be enabled for {env_name}"
                )

    def validate_environment_variables(
        self, env_name: str, validation_config: dict[str, Any]
    ) -> None:
        """Validate required environment variables."""
        required_vars = validation_config.get("required_env_vars", [])
        optional_vars = validation_config.get("optional_env_vars", [])

        # Check if required variables are set (in current environment)
        missing_required = []
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        if missing_required:
            self.warnings.append(
                f"{env_name}: Required environment variables not set in current environment: "
                f"{', '.join(missing_required)} (Note: These may be set in deployment environment)"
            )

        # Validate environment variable naming
        all_vars = required_vars + optional_vars
        for var in all_vars:
            if not var.startswith("AGENT_ZERO_"):
                self.warnings.append(
                    f"{env_name}: Environment variable {var} should use AGENT_ZERO_ prefix"
                )

            if not re.match(r"^[A-Z_]+$", var):
                self.errors.append(f"{env_name}: Invalid environment variable name: {var}")

    def validate_resource_requirements(
        self, env_name: str, validation_config: dict[str, Any]
    ) -> None:
        """Validate resource requirements."""
        min_memory = validation_config.get("minimum_memory_mb")
        min_disk = validation_config.get("minimum_disk_space_gb")
        min_cpu = validation_config.get("minimum_cpu_cores")
        min_bandwidth = validation_config.get("required_network_bandwidth_mbps")

        # Environment-specific resource checks
        if env_name == "production":
            if not min_memory or min_memory < 4096:
                self.warnings.append(f"{env_name}: Production should require at least 4GB RAM")

            if not min_cpu or min_cpu < 2:
                self.warnings.append(f"{env_name}: Production should require at least 2 CPU cores")

            if not min_disk or min_disk < 50:
                self.warnings.append(
                    f"{env_name}: Production should require at least 50GB disk space"
                )

        elif env_name == "staging":
            if not min_memory or min_memory < 1024:
                self.warnings.append(f"{env_name}: Staging should require at least 1GB RAM")

    def validate_production_config(self, config: dict[str, Any]) -> None:
        """Validate production-specific requirements."""
        env_name = "production"

        # High availability checks
        monitoring = config.get("monitoring", {})
        if not monitoring.get("enable_opentelemetry"):
            self.warnings.append(
                f"{env_name}: OpenTelemetry should be enabled for production observability"
            )

        if not monitoring.get("enable_alerts"):
            self.errors.append(f"{env_name}: Alerting must be enabled in production")

        # Backup and disaster recovery
        backup = config.get("backup", {})
        if not backup.get("enable_config_backup"):
            self.errors.append(f"{env_name}: Configuration backup must be enabled in production")

        dr = config.get("disaster_recovery", {})
        if not dr.get("enable_dr"):
            self.warnings.append(f"{env_name}: Consider enabling disaster recovery for production")

        # Compliance
        compliance = config.get("compliance", {})
        if not compliance.get("enable_soc2_compliance"):
            self.warnings.append(f"{env_name}: SOC2 compliance should be enabled for production")

        # Performance
        performance = config.get("performance", {})
        if not performance.get("enable_auto_scaling"):
            self.warnings.append(f"{env_name}: Auto-scaling should be enabled for production")

        # Logging
        logging_config = config.get("logging", {})
        if logging_config.get("level") not in ["WARNING", "ERROR"]:
            self.warnings.append(f"{env_name}: Production log level should be WARNING or ERROR")

        if not logging_config.get("enable_centralized_logging"):
            self.warnings.append(
                f"{env_name}: Centralized logging should be enabled for production"
            )

    def validate_staging_config(self, config: dict[str, Any]) -> None:
        """Validate staging-specific requirements."""
        env_name = "staging"

        # Should mirror production security but allow some debugging
        features = config.get("features", {})
        if features.get("enable_debug_endpoints"):
            self.warnings.append(f"{env_name}: Debug endpoints in staging may pose security risk")

        # Monitoring should be enabled but less extensive than production
        monitoring = config.get("monitoring", {})
        if not monitoring.get("enable_health_checks"):
            self.errors.append(f"{env_name}: Health checks must be enabled in staging")

    def validate_development_config(self, config: dict[str, Any]) -> None:
        """Validate development-specific requirements."""
        env_name = "development"

        # Development should have debugging enabled
        features = config.get("features", {})
        if not features.get("enable_debug_endpoints"):
            self.warnings.append(f"{env_name}: Debug endpoints should be enabled for development")

        # Security can be relaxed in development
        security = config.get("security", {})
        if security.get("auth_required"):
            self.warnings.append(f"{env_name}: Authentication can be disabled for development")

    def validate_cross_environment(self) -> None:
        """Validate consistency across environments."""
        print("  Validating cross-environment consistency...")

        if not self.environments:
            return

        # Check that all environments have the same schema version
        schema_versions = set()
        for _env_name, config in self.environments.items():
            version = config.get("validation", {}).get("config_schema_version")
            if version:
                schema_versions.add(version)

        if len(schema_versions) > 1:
            self.warnings.append(
                f"Inconsistent schema versions across environments: {schema_versions}"
            )

        # Check for common configuration errors
        self.validate_environment_progression()

        # Validate port consistency
        self.validate_port_consistency()

    def validate_environment_progression(self) -> None:
        """Validate that environments follow proper progression (dev -> staging -> prod)."""
        envs = ["development", "staging", "production"]
        available_envs = [env for env in envs if env in self.environments]

        # Check security progression
        for i in range(len(available_envs) - 1):
            current_env = available_envs[i]
            next_env = available_envs[i + 1]

            current_security = self.environments[current_env].get("security", {})
            next_security = self.environments[next_env].get("security", {})

            # SSL should be more restrictive in higher environments
            if not current_security.get("ssl_enable") and next_security.get("ssl_enable"):
                continue  # This is expected
            elif current_security.get("ssl_enable") and not next_security.get("ssl_enable"):
                self.warnings.append(
                    f"Security regression: {next_env} has SSL disabled while {current_env} has it enabled"
                )

    def validate_port_consistency(self) -> None:
        """Validate that port configurations are consistent where appropriate."""
        server_ports = {}
        prometheus_ports = {}

        for env_name, config in self.environments.items():
            server_port = config.get("server", {}).get("port")
            if server_port:
                server_ports[env_name] = server_port

            prometheus_port = config.get("monitoring", {}).get("prometheus_port")
            if prometheus_port:
                prometheus_ports[env_name] = prometheus_port

        # Check for port conflicts within environment
        for env_name, _config in self.environments.items():
            server_port = server_ports.get(env_name)
            prometheus_port = prometheus_ports.get(env_name)

            if server_port and prometheus_port and server_port == prometheus_port:
                self.errors.append(
                    f"{env_name}: Server and Prometheus ports conflict: {server_port}"
                )

    def report_results(self) -> None:
        """Report validation results."""
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION RESULTS")
        print("=" * 60)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\nAll configuration validation checks passed!")

        print("\nSUMMARY:")
        print(f"  • Environments validated: {len(self.environments)}")
        print(f"  • Errors: {len(self.errors)}")
        print(f"  • Warnings: {len(self.warnings)}")

    def export_results(self, output_file: Path) -> None:
        """Export validation results to JSON."""
        results = {
            "timestamp": "2025-01-16T00:00:00Z",  # Would use actual timestamp
            "summary": {
                "environments_validated": len(self.environments),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "environments": list(self.environments.keys()),
        }

        output_file.write_text(json.dumps(results, indent=2))
        print(f"Results exported to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Agent Zero configuration files")
    parser.add_argument(
        "--config-dir", type=Path, default=Path.cwd() / "configs", help="Configuration directory"
    )
    parser.add_argument("--environment", help="Validate specific environment only")
    parser.add_argument("--output", type=Path, help="Export results to JSON file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    args = parser.parse_args()

    if not args.config_dir.exists():
        print(f"ERROR: Configuration directory not found: {args.config_dir}")
        sys.exit(1)

    validator = ConfigurationValidator(args.config_dir)

    # Load and validate
    if args.environment:
        # Validate specific environment
        env_file = args.config_dir / "environments" / f"{args.environment}.json"
        if not env_file.exists():
            print(f"ERROR: Environment configuration not found: {env_file}")
            sys.exit(1)

        try:
            with open(env_file) as f:
                config = json.load(f)
            validator.environments[args.environment] = config
            validator.validate_configuration(args.environment, config)
        except Exception as e:
            print(f"ERROR: Failed to load {env_file}: {e}")
            sys.exit(1)
    else:
        # Validate all environments
        success = validator.validate_all()

    if args.output:
        validator.export_results(args.output)

    # Exit with error code if validation failed
    if validator.errors or (args.strict and validator.warnings):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
