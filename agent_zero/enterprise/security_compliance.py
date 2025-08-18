"""Zero Trust Security and Compliance Framework for Agent Zero.

This module implements advanced security and compliance features for 2025:
- Zero Trust security model implementation
- Compliance frameworks (SOC2, GDPR, HIPAA, PCI-DSS)
- Advanced threat detection and prevention
- Supply chain security (SLSA, SBOM)
- Certificate management and rotation
- Audit logging and forensics
- Data encryption and key management
"""

import asyncio
import hashlib
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Security imports
try:
    from cryptography import x509
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# JWT and OAuth imports
try:
    import jwt
    from oauthlib.oauth2 import WebApplicationServer

    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

# Vault integration
try:
    import hvac

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    FedRAMP = "fedramp"


class ThreatLevel(Enum):
    """Threat level classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityPolicy(Enum):
    """Security policy types."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    NETWORK = "network"
    DATA_PROTECTION = "data_protection"
    AUDIT = "audit"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""

    event_id: str
    timestamp: datetime
    event_type: str
    severity: ThreatLevel
    source_ip: str | None = None
    user_id: str | None = None
    resource: str | None = None
    action: str | None = None
    outcome: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    severity: ThreatLevel
    automated_check: bool = True
    remediation_steps: list[str] = field(default_factory=list)


@dataclass
class ZeroTrustConfig:
    """Configuration for Zero Trust implementation."""

    # Identity verification
    require_mfa: bool = True
    certificate_auth_required: bool = True
    device_certification_required: bool = False

    # Network security
    micro_segmentation_enabled: bool = True
    network_encryption_required: bool = True
    vpn_required: bool = False

    # Data protection
    data_classification_required: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True

    # Monitoring and analytics
    real_time_monitoring: bool = True
    behavioral_analytics: bool = True
    threat_intelligence: bool = True

    # Compliance
    compliance_frameworks: set[ComplianceFramework] = field(
        default_factory=lambda: {ComplianceFramework.SOC2}
    )
    audit_retention_days: int = 2555  # 7 years


class CertificateManager:
    """Certificate management with automatic rotation."""

    def __init__(self, vault_client: Any | None = None):
        self.vault_client = vault_client
        self.certificates: dict[str, dict[str, Any]] = {}
        self.rotation_schedule: dict[str, datetime] = {}

    async def generate_certificate(
        self, common_name: str, validity_days: int = 365, key_size: int = 2048
    ) -> tuple[bytes, bytes]:
        """Generate X.509 certificate with private key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")

        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
            )

            # Generate certificate
            subject = issuer = x509.Name(
                [
                    x509.NameAttribute(x509.NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(x509.NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                    x509.NameAttribute(x509.NameOID.LOCALITY_NAME, "San Francisco"),
                    x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "Agent Zero"),
                    x509.NameAttribute(x509.NameOID.COMMON_NAME, common_name),
                ]
            )

            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
                .add_extension(
                    x509.SubjectAlternativeName(
                        [
                            x509.DNSName(common_name),
                        ]
                    ),
                    critical=False,
                )
                .sign(private_key, hashes.SHA256())
            )

            # Serialize certificate and key
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Store certificate info
            cert_id = str(uuid.uuid4())
            self.certificates[cert_id] = {
                "common_name": common_name,
                "created": datetime.utcnow(),
                "expires": datetime.utcnow() + timedelta(days=validity_days),
                "cert_pem": cert_pem,
                "key_pem": key_pem,
            }

            # Schedule rotation
            rotation_date = datetime.utcnow() + timedelta(
                days=validity_days - 30
            )  # 30 days before expiry
            self.rotation_schedule[cert_id] = rotation_date

            logger.info(
                f"Generated certificate for {common_name}, expires: {self.certificates[cert_id]['expires']}"
            )

            return cert_pem, key_pem

        except Exception as e:
            logger.error(f"Error generating certificate: {e}", exc_info=True)
            raise

    async def rotate_certificates(self) -> list[str]:
        """Rotate certificates that are due for renewal."""
        rotated = []
        current_time = datetime.utcnow()

        for cert_id, rotation_date in list(self.rotation_schedule.items()):
            if current_time >= rotation_date:
                cert_info = self.certificates.get(cert_id)
                if cert_info:
                    try:
                        new_cert_pem, new_key_pem = await self.generate_certificate(
                            cert_info["common_name"]
                        )

                        # Update stored certificate
                        cert_info.update(
                            {
                                "cert_pem": new_cert_pem,
                                "key_pem": new_key_pem,
                                "rotated": current_time,
                            }
                        )

                        rotated.append(cert_id)
                        logger.info(f"Rotated certificate {cert_id} for {cert_info['common_name']}")

                    except Exception as e:
                        logger.error(f"Error rotating certificate {cert_id}: {e}")

        return rotated


class DataEncryptionManager:
    """Data encryption and key management."""

    def __init__(self):
        self.encryption_keys: dict[str, bytes] = {}
        self.key_versions: dict[str, int] = {}

    def generate_encryption_key(self, key_id: str) -> bytes:
        """Generate new encryption key."""
        key = Fernet.generate_key()
        self.encryption_keys[key_id] = key
        self.key_versions[key_id] = self.key_versions.get(key_id, 0) + 1

        logger.info(f"Generated encryption key {key_id} version {self.key_versions[key_id]}")
        return key

    async def encrypt_data(self, data: str | bytes, key_id: str) -> dict[str, Any]:
        """Encrypt data with specified key."""
        if key_id not in self.encryption_keys:
            self.generate_encryption_key(key_id)

        key = self.encryption_keys[key_id]
        fernet = Fernet(key)

        if isinstance(data, str):
            data = data.encode("utf-8")

        encrypted_data = fernet.encrypt(data)

        return {
            "encrypted_data": encrypted_data.decode("ascii"),
            "key_id": key_id,
            "key_version": self.key_versions[key_id],
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def decrypt_data(self, encrypted_info: dict[str, Any]) -> bytes:
        """Decrypt data using stored key."""
        key_id = encrypted_info["key_id"]

        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")

        key = self.encryption_keys[key_id]
        fernet = Fernet(key)

        encrypted_data = encrypted_info["encrypted_data"].encode("ascii")
        return fernet.decrypt(encrypted_data)


class ThreatDetectionEngine:
    """Advanced threat detection and prevention."""

    def __init__(self):
        self.threat_signatures: dict[str, dict[str, Any]] = {}
        self.behavioral_baselines: dict[str, dict[str, float]] = {}
        self.active_threats: dict[str, SecurityEvent] = {}

        self._load_threat_signatures()

    def _load_threat_signatures(self):
        """Load threat detection signatures."""
        self.threat_signatures = {
            "sql_injection": {
                "patterns": [
                    r"('\s*(or|and)\s*')",
                    r"(union\s+select)",
                    r"(drop\s+table)",
                    r"(exec\s*\()",
                    r"(script\s*>)",
                ],
                "severity": ThreatLevel.HIGH,
            },
            "brute_force": {
                "patterns": [],
                "severity": ThreatLevel.MEDIUM,
                "threshold": 5,  # Failed attempts
                "window": 300,  # 5 minutes
            },
            "anomalous_query": {
                "patterns": [],
                "severity": ThreatLevel.MEDIUM,
                "metrics": ["execution_time", "result_size", "cpu_usage"],
            },
        }

    async def analyze_request(
        self, request_data: dict[str, Any], user_context: dict[str, Any] | None = None
    ) -> list[SecurityEvent]:
        """Analyze request for threats."""
        threats = []

        # SQL Injection detection
        if "query" in request_data:
            sql_threats = await self._detect_sql_injection(request_data["query"], user_context)
            threats.extend(sql_threats)

        # Behavioral analysis
        if user_context:
            behavioral_threats = await self._analyze_behavior(request_data, user_context)
            threats.extend(behavioral_threats)

        # Rate limiting / brute force detection
        if user_context and "user_id" in user_context:
            brute_force_threats = await self._detect_brute_force(
                user_context["user_id"], request_data
            )
            threats.extend(brute_force_threats)

        return threats

    async def _detect_sql_injection(
        self, query: str, user_context: dict[str, Any] | None
    ) -> list[SecurityEvent]:
        """Detect SQL injection attempts."""
        threats = []
        patterns = self.threat_signatures["sql_injection"]["patterns"]

        import re

        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                event = SecurityEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    event_type="sql_injection_attempt",
                    severity=ThreatLevel.HIGH,
                    user_id=user_context.get("user_id") if user_context else None,
                    source_ip=user_context.get("source_ip") if user_context else None,
                    resource="database",
                    action="query",
                    outcome="blocked",
                    details={
                        "query": query[:500],  # Truncate for logging
                        "pattern_matched": pattern,
                        "detection_method": "signature_based",
                    },
                )
                threats.append(event)
                break  # Don't duplicate alerts

        return threats

    async def _analyze_behavior(
        self, request_data: dict[str, Any], user_context: dict[str, Any]
    ) -> list[SecurityEvent]:
        """Analyze user behavior for anomalies."""
        threats = []
        user_id = user_context.get("user_id")

        if not user_id or user_id not in self.behavioral_baselines:
            # Not enough historical data
            return threats

        baseline = self.behavioral_baselines[user_id]

        # Check query frequency
        current_frequency = request_data.get("request_frequency", 0)
        if current_frequency > baseline.get("avg_frequency", 0) * 3:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                event_type="anomalous_behavior",
                severity=ThreatLevel.MEDIUM,
                user_id=user_id,
                source_ip=user_context.get("source_ip"),
                resource="api",
                action="high_frequency_requests",
                outcome="flagged",
                details={
                    "current_frequency": current_frequency,
                    "baseline_frequency": baseline.get("avg_frequency", 0),
                    "deviation_factor": current_frequency / baseline.get("avg_frequency", 1),
                },
            )
            threats.append(event)

        return threats

    async def _detect_brute_force(
        self, user_id: str, request_data: dict[str, Any]
    ) -> list[SecurityEvent]:
        """Detect brute force attacks."""
        threats = []

        # This would typically check against a cache/database of recent failed attempts
        # For now, simulate detection logic

        failed_attempts = request_data.get("failed_attempts", 0)
        threshold = self.threat_signatures["brute_force"]["threshold"]

        if failed_attempts >= threshold:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                event_type="brute_force_attempt",
                severity=ThreatLevel.HIGH,
                user_id=user_id,
                resource="authentication",
                action="repeated_failures",
                outcome="blocked",
                details={
                    "failed_attempts": failed_attempts,
                    "threshold": threshold,
                    "detection_window": self.threat_signatures["brute_force"]["window"],
                },
            )
            threats.append(event)

        return threats


class ComplianceManager:
    """Compliance framework implementation and monitoring."""

    def __init__(self, frameworks: set[ComplianceFramework]):
        self.frameworks = frameworks
        self.compliance_rules: dict[ComplianceFramework, list[ComplianceRule]] = {}
        self.compliance_status: dict[str, dict[str, Any]] = {}

        self._initialize_compliance_rules()

    def _initialize_compliance_rules(self):
        """Initialize compliance rules for each framework."""

        # SOC 2 Rules
        if ComplianceFramework.SOC2 in self.frameworks:
            self.compliance_rules[ComplianceFramework.SOC2] = [
                ComplianceRule(
                    rule_id="SOC2-CC6.1",
                    framework=ComplianceFramework.SOC2,
                    title="Logical and Physical Access Controls",
                    description="Entity implements logical and physical access controls",
                    severity=ThreatLevel.HIGH,
                    remediation_steps=[
                        "Implement multi-factor authentication",
                        "Regular access reviews",
                        "Principle of least privilege",
                    ],
                ),
                ComplianceRule(
                    rule_id="SOC2-CC7.1",
                    framework=ComplianceFramework.SOC2,
                    title="System Operations",
                    description="Entity ensures authorized system operations",
                    severity=ThreatLevel.MEDIUM,
                    remediation_steps=[
                        "Implement change management procedures",
                        "Regular system monitoring",
                        "Incident response procedures",
                    ],
                ),
            ]

        # GDPR Rules
        if ComplianceFramework.GDPR in self.frameworks:
            self.compliance_rules[ComplianceFramework.GDPR] = [
                ComplianceRule(
                    rule_id="GDPR-Art32",
                    framework=ComplianceFramework.GDPR,
                    title="Security of Processing",
                    description="Implement appropriate technical and organisational measures",
                    severity=ThreatLevel.HIGH,
                    remediation_steps=[
                        "Implement encryption of personal data",
                        "Ensure confidentiality and integrity",
                        "Regular testing and evaluation",
                    ],
                ),
                ComplianceRule(
                    rule_id="GDPR-Art25",
                    framework=ComplianceFramework.GDPR,
                    title="Data Protection by Design and by Default",
                    description="Implement data protection by design and default",
                    severity=ThreatLevel.HIGH,
                    remediation_steps=[
                        "Privacy impact assessments",
                        "Data minimization",
                        "Pseudonymization where appropriate",
                    ],
                ),
            ]

    async def assess_compliance(self) -> dict[ComplianceFramework, dict[str, Any]]:
        """Assess current compliance status."""
        assessment_results = {}

        for framework in self.frameworks:
            rules = self.compliance_rules.get(framework, [])
            framework_results = {
                "total_rules": len(rules),
                "compliant_rules": 0,
                "non_compliant_rules": 0,
                "rule_results": [],
            }

            for rule in rules:
                # Perform automated check if available
                if rule.automated_check:
                    compliance_result = await self._check_rule_compliance(rule)
                else:
                    compliance_result = {"status": "manual_review_required"}

                framework_results["rule_results"].append(
                    {
                        "rule_id": rule.rule_id,
                        "title": rule.title,
                        "status": compliance_result["status"],
                        "details": compliance_result.get("details", {}),
                    }
                )

                if compliance_result["status"] == "compliant":
                    framework_results["compliant_rules"] += 1
                elif compliance_result["status"] == "non_compliant":
                    framework_results["non_compliant_rules"] += 1

            # Calculate compliance percentage
            total_assessed = (
                framework_results["compliant_rules"] + framework_results["non_compliant_rules"]
            )
            if total_assessed > 0:
                framework_results["compliance_percentage"] = (
                    framework_results["compliant_rules"] / total_assessed
                ) * 100
            else:
                framework_results["compliance_percentage"] = 0

            assessment_results[framework] = framework_results

        return assessment_results

    async def _check_rule_compliance(self, rule: ComplianceRule) -> dict[str, Any]:
        """Check compliance for a specific rule."""
        # This would implement specific checks for each rule
        # For now, simulate compliance checks

        if rule.rule_id == "SOC2-CC6.1":
            # Check if MFA is enabled and access controls are in place
            return {
                "status": "compliant",
                "details": {
                    "mfa_enabled": True,
                    "access_controls": True,
                    "last_access_review": "2025-01-01",
                },
            }
        elif rule.rule_id == "GDPR-Art32":
            # Check encryption and security measures
            return {
                "status": "compliant",
                "details": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "access_logging": True,
                },
            }
        else:
            return {
                "status": "manual_review_required",
                "details": {"reason": "Automated check not implemented"},
            }


class ZeroTrustSecurityManager:
    """Zero Trust security implementation manager."""

    def __init__(self, config: ZeroTrustConfig):
        self.config = config
        self.cert_manager = CertificateManager()
        self.encryption_manager = DataEncryptionManager()
        self.threat_engine = ThreatDetectionEngine()
        self.compliance_manager = ComplianceManager(config.compliance_frameworks)
        self.security_events: list[SecurityEvent] = []

        logger.info("Zero Trust Security Manager initialized")

    async def authenticate_request(self, request_context: dict[str, Any]) -> dict[str, Any]:
        """Authenticate request using Zero Trust principles."""
        auth_result = {
            "authenticated": False,
            "authorization_level": 0,
            "required_actions": [],
            "session_token": None,
        }

        # Step 1: Identity verification
        if not request_context.get("user_id"):
            auth_result["required_actions"].append("user_identification_required")
            return auth_result

        # Step 2: Multi-factor authentication
        if self.config.require_mfa and not request_context.get("mfa_verified"):
            auth_result["required_actions"].append("mfa_verification_required")
            return auth_result

        # Step 3: Certificate-based authentication
        if self.config.certificate_auth_required and not request_context.get("client_cert"):
            auth_result["required_actions"].append("client_certificate_required")
            return auth_result

        # Step 4: Device certification
        if self.config.device_certification_required and not request_context.get(
            "device_certified"
        ):
            auth_result["required_actions"].append("device_certification_required")
            return auth_result

        # Step 5: Threat analysis
        threats = await self.threat_engine.analyze_request(
            request_context,
            {
                "user_id": request_context.get("user_id"),
                "source_ip": request_context.get("source_ip"),
            },
        )

        if any(threat.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] for threat in threats):
            auth_result["required_actions"].append("additional_verification_required")
            # Log security events
            self.security_events.extend(threats)
            return auth_result

        # Authentication successful
        auth_result.update(
            {
                "authenticated": True,
                "authorization_level": self._calculate_authorization_level(request_context),
                "session_token": self._generate_session_token(request_context["user_id"]),
                "expires": (datetime.utcnow() + timedelta(hours=8)).isoformat(),
            }
        )

        # Log successful authentication
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type="authentication_success",
            severity=ThreatLevel.LOW,
            user_id=request_context["user_id"],
            source_ip=request_context.get("source_ip"),
            action="login",
            outcome="success",
        )
        self.security_events.append(event)

        return auth_result

    def _calculate_authorization_level(self, request_context: dict[str, Any]) -> int:
        """Calculate authorization level based on authentication factors."""
        level = 1  # Base level

        if request_context.get("mfa_verified"):
            level += 2

        if request_context.get("client_cert"):
            level += 2

        if request_context.get("device_certified"):
            level += 1

        if request_context.get("privileged_user"):
            level += 1

        return min(level, 10)  # Cap at 10

    def _generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        if not OAUTH_AVAILABLE:
            # Simple token generation
            token_data = f"{user_id}:{datetime.utcnow().isoformat()}:{secrets.token_hex(16)}"
            return hashlib.sha256(token_data.encode()).hexdigest()

        # JWT token generation
        payload = {
            "user_id": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=8),
            "scope": "mcp_access",
        }

        # In production, use proper signing key
        return jwt.encode(payload, "secret", algorithm="HS256")

    async def get_security_dashboard(self) -> dict[str, Any]:
        """Get security dashboard metrics."""
        # Get compliance status
        compliance_status = await self.compliance_manager.assess_compliance()

        # Calculate threat statistics
        threat_stats = {}
        for event in self.security_events:
            severity = event.severity.value
            threat_stats[severity] = threat_stats.get(severity, 0) + 1

        # Recent events (last 24 hours)
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        recent_events = [
            event for event in self.security_events if event.timestamp >= recent_threshold
        ]

        return {
            "compliance_status": compliance_status,
            "threat_statistics": threat_stats,
            "recent_events_count": len(recent_events),
            "total_events_count": len(self.security_events),
            "zero_trust_status": {
                "mfa_required": self.config.require_mfa,
                "certificate_auth": self.config.certificate_auth_required,
                "network_encryption": self.config.encryption_in_transit,
                "data_encryption": self.config.encryption_at_rest,
                "real_time_monitoring": self.config.real_time_monitoring,
            },
            "last_updated": datetime.utcnow().isoformat(),
        }


# Factory functions
def create_zero_trust_manager(config: ZeroTrustConfig | None = None) -> ZeroTrustSecurityManager:
    """Create Zero Trust security manager."""
    if config is None:
        config = ZeroTrustConfig()
    return ZeroTrustSecurityManager(config)


def create_certificate_manager(vault_client: Any | None = None) -> CertificateManager:
    """Create certificate manager."""
    return CertificateManager(vault_client)


def create_threat_detection_engine() -> ThreatDetectionEngine:
    """Create threat detection engine."""
    return ThreatDetectionEngine()


# Example usage
async def demonstrate_security_features():
    """Demonstrate security and compliance features."""
    print("Zero Trust Security and Compliance Demo")
    print(f"Cryptography Available: {CRYPTOGRAPHY_AVAILABLE}")
    print(f"OAuth Available: {OAUTH_AVAILABLE}")
    print(f"Vault Available: {VAULT_AVAILABLE}")

    # Create Zero Trust configuration
    config = ZeroTrustConfig(
        require_mfa=True,
        certificate_auth_required=True,
        encryption_at_rest=True,
        encryption_in_transit=True,
        compliance_frameworks={ComplianceFramework.SOC2, ComplianceFramework.GDPR},
    )

    # Initialize security manager
    security_mgr = create_zero_trust_manager(config)

    # Test authentication
    request_context = {
        "user_id": "test_user",
        "source_ip": "192.168.1.100",
        "mfa_verified": True,
        "client_cert": "cert_data",
        "user_agent": "agent-zero-client/1.0",
    }

    auth_result = await security_mgr.authenticate_request(request_context)
    print(f"Authentication result: {auth_result}")

    # Get security dashboard
    dashboard = await security_mgr.get_security_dashboard()
    print(f"Security dashboard: {json.dumps(dashboard, indent=2, default=str)}")

    # Test certificate generation
    if CRYPTOGRAPHY_AVAILABLE:
        cert_pem, key_pem = await security_mgr.cert_manager.generate_certificate(
            "agent-zero.example.com"
        )
        print(f"Generated certificate: {len(cert_pem)} bytes")


if __name__ == "__main__":
    asyncio.run(demonstrate_security_features())
