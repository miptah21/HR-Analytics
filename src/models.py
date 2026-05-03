import hashlib
import secrets
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from datetime import datetime, timezone, timedelta
from src.database import Base


# ── Password Hashing (stdlib only — no bcrypt dependency) ─────────────
_PWD_SALT_LENGTH = 32
_PWD_ITERATIONS = 600_000  # OWASP 2024 recommended minimum for SHA-256


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256 with random salt.

    Returns: 'salt$hash' string for storage.
    """
    salt = secrets.token_hex(_PWD_SALT_LENGTH)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PWD_ITERATIONS
    )
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored 'salt$hash' string."""
    try:
        salt, expected_hash = stored_hash.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PWD_ITERATIONS
    )
    return secrets.compare_digest(dk.hex(), expected_hash)


def generate_session_token() -> str:
    """Generate a cryptographically secure session token."""
    return f"ses-{secrets.token_urlsafe(48)}"


# ── PII Masking Utility ───────────────────────────────────────────────
def mask_employee_id(employee_id: str) -> str:
    """Hash employee ID for privacy-preserving audit trail storage.

    Uses SHA-256 with a static salt to create a deterministic but
    irreversible pseudonymized identifier. This allows audit trail
    queries by hashed ID while preventing PII exposure if the
    database is compromised.

    The original employee_id is still accepted at the API boundary
    for usability — the hashing happens at the persistence layer.
    """
    salt = "hr-attrition-v1"  # In production, use env var
    return hashlib.sha256(f"{salt}:{employee_id}".encode()).hexdigest()[:16]


class PredictionLog(Base):
    """
    Audit trail for high-risk HR predictive models as required by EU AI Act.
    Stores input features, prediction probabilities, and metadata.

    G-10: employee_id is stored as a SHA-256 pseudonymized hash to protect
    PII while maintaining queryability. The employee_id_hash column enables
    lookups without exposing raw employee identifiers.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Context — G-10: employee_id stored as hash for PII protection
    employee_id = Column(String, index=True, nullable=True) # Optional, can be anonymized
    employee_id_hash = Column(String(16), index=True, nullable=True)  # SHA-256 pseudonym
    requester_id = Column(String, index=True, nullable=True) # Who requested the score
    requester_role = Column(String, nullable=True) # RBAC role of the requester
    
    # Inputs & Outputs
    input_features = Column(JSON, nullable=False) # Store the raw input payload
    engineered_features = Column(JSON, nullable=True) # Post-engineering feature vector
    attrition_probability = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    
    # Explainability
    top_risk_drivers = Column(JSON, nullable=True) # Store SHAP top factors
    
    # Gemini Context (optional)
    generated_strategy = Column(String, nullable=True)
    
    # Model Provenance (EU AI Act Art. 12 — full traceability)
    model_version = Column(String, nullable=True)
    population_stats_hash = Column(String, nullable=True)


class HumanOverride(Base):
    """
    Records human override decisions for EU AI Act Art. 14 compliance.
    When an HR professional disagrees with the AI assessment, their
    override decision and rationale are persisted here.
    """
    __tablename__ = "human_overrides"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    prediction_log_id = Column(Integer, nullable=True, index=True)
    employee_id = Column(String, index=True, nullable=False)
    original_risk_tier = Column(String, nullable=False)
    override_risk_tier = Column(String, nullable=False)
    override_reason = Column(Text, nullable=False)
    overridden_by = Column(String, nullable=True) # Who made the override


class ScoringExclusion(Base):
    """
    GDPR Article 21 — Right to Object.

    Records employees who have exercised their right to be excluded
    from AI-based attrition scoring. The predict endpoint checks this
    table before processing any prediction request.

    When an employee is excluded:
    - No new predictions can be made for their employee ID
    - Existing audit logs are retained (legal obligation under Art. 6(1)(c))
    - The exclusion is itself logged for compliance
    """
    __tablename__ = "scoring_exclusions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    employee_id = Column(String, index=True, unique=True, nullable=False)
    employee_id_hash = Column(String(16), index=True, nullable=True)
    reason = Column(Text, nullable=False)  # Why exclusion was requested
    requested_by = Column(String, nullable=False)  # DPO, employee, or HR
    approved_by = Column(String, nullable=True)  # Who approved the exclusion
    is_active = Column(Boolean, default=True, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by = Column(String, nullable=True)


# ── Causal ML: Treatment Registry ───────────────────────────────────────
class InterventionLog(Base):
    """
    Treatment Assignment Registry for Causal Machine Learning.
    Logs every applied intervention (treatment) for an employee.
    This provides the longitudinal "Treatment" data for future Uplift model retraining.
    """
    __tablename__ = "intervention_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    prediction_log_id = Column(Integer, nullable=True, index=True) # Link to original score
    employee_id = Column(String, index=True, nullable=False)
    
    # Intervention Details
    intervention_type = Column(String, nullable=False) # e.g., "Salary Hike >= 15%", "Overtime Removed"
    intervention_details = Column(Text, nullable=True) # Additional notes
    applied_by = Column(String, nullable=True) # Who approved the intervention


# ── User Management ───────────────────────────────────────────────────
class User(Base):
    """
    System user for authentication.

    Replaces the old API-key-based auth with proper username/password login.
    Passwords are stored as PBKDF2-SHA256 hashes (OWASP 2024 compliant).
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(500), nullable=False)
    display_name = Column(String(200), nullable=True)
    role = Column(String(50), nullable=False, default="analyst")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    def set_password(self, password: str) -> None:
        self.password_hash = hash_password(password)

    def check_password(self, password: str) -> bool:
        return verify_password(password, self.password_hash)


class UserSession(Base):
    """
    Session tokens for authenticated users.

    Each login creates a session with an 8-hour expiry.
    The session token is sent via X-API-Key header for backward compatibility.
    """
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    @property
    def is_expired(self) -> bool:
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > expires


# Default admin credentials (changed on first login in production)
DEFAULT_ADMIN_USER = {
    "username": "admin",
    "password": "admin123",
    "display_name": "System Administrator",
    "role": "admin",
}

