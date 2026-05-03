"""
HR Attrition Intelligence API — v1
Production-grade FastAPI service with authentication, rate limiting,
input validation, and unified feature engineering.
"""
import os
from dotenv import load_dotenv

# Load .env file explicitly so API keys are available
load_dotenv()
import asyncio
import hashlib
import hmac
import json
import re
import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
except ImportError:
    class RateLimitExceeded(Exception):
        """Fallback exception used when slowapi is unavailable in test envs."""

    def get_remote_address(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    async def _rate_limit_exceeded_handler(
        request: Request,
        exc: RateLimitExceeded,
    ) -> Response:
        return Response("Rate limit exceeded", status_code=429)

    class Limiter:
        """No-op limiter fallback for lightweight imports/tests."""

        def __init__(self, key_func: Any):
            self.key_func = key_func

        def limit(self, _limit_value: str):
            def decorator(func: Any) -> Any:
                return func

            return decorator

from src.features import engineer_features_single, load_population_stats

# Database imports
from sqlalchemy.orm import Session
from src.database import get_db, engine, Base, get_db_info, SessionLocal
from src.models import PredictionLog, HumanOverride, ScoringExclusion, mask_employee_id

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr-attrition-api")

# ── Rate Limiter ───────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Paths & Config ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_attrition.json"

# Security config from environment
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")
IS_DEV_MODE = os.environ.get("ENV", "development") == "development"

# ── RBAC Configuration ────────────────────────────────────────────────
# Roles: admin, hr_partner, analyst, auditor
# Keys are configured via RBAC_KEYS env var as JSON, e.g.:
#   RBAC_KEYS={"key1":"admin","key2":"hr_partner","key3":"analyst"}
# In dev mode without RBAC_KEYS, all requests get admin role.
RBAC_KEYS: dict[str, str] = {}
try:
    _rbac_raw = os.environ.get("RBAC_KEYS", "")
    if _rbac_raw:
        RBAC_KEYS = json.loads(_rbac_raw)
except json.JSONDecodeError:
    logger.warning("RBAC_KEYS env var is not valid JSON — falling back to single-key mode.")

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"predict", "override", "dashboard", "audit", "export", "system"},
    "hr_partner": {"predict", "override", "dashboard", "audit", "export"},
    "analyst": {"dashboard", "audit", "export"},
    "auditor": {"audit"},
}

# ── Brute-Force Protection ─────────────────────────────────────────────
# Track failed login attempts per IP. Lock out after MAX_FAILED_ATTEMPTS
# within LOCKOUT_WINDOW_SECONDS.
import time as _time

MAX_FAILED_ATTEMPTS = 10
LOCKOUT_WINDOW_SECONDS = 900  # 15 minutes

# In-memory store: {ip: [(timestamp, key_prefix), ...]}
_failed_attempts: dict[str, list[tuple[float, str]]] = {}


def _record_failed_attempt(ip: str, key_prefix: str) -> None:
    """Record a failed authentication attempt for an IP."""
    now = _time.time()
    if ip not in _failed_attempts:
        _failed_attempts[ip] = []
    _failed_attempts[ip].append((now, key_prefix))
    # Prune old entries
    cutoff = now - LOCKOUT_WINDOW_SECONDS
    _failed_attempts[ip] = [(t, k) for t, k in _failed_attempts[ip] if t > cutoff]
    logger.warning(
        f"Failed auth attempt from {ip} (key: {key_prefix}...) — "
        f"{len(_failed_attempts[ip])}/{MAX_FAILED_ATTEMPTS} in window"
    )


def _is_locked_out(ip: str) -> bool:
    """Check if an IP is currently locked out due to too many failures."""
    if ip not in _failed_attempts:
        return False
    now = _time.time()
    cutoff = now - LOCKOUT_WINDOW_SECONDS
    recent = [t for t, _ in _failed_attempts[ip] if t > cutoff]
    return len(recent) >= MAX_FAILED_ATTEMPTS


def _clear_failed_attempts(ip: str) -> None:
    """Clear failed attempts for an IP after successful auth."""
    _failed_attempts.pop(ip, None)


# ── Enums for strict validation ────────────────────────────────────────
class JobRoleEnum(str, Enum):
    SALES_EXEC = "Sales Executive"
    RESEARCH_SCIENTIST = "Research Scientist"
    LAB_TECH = "Laboratory Technician"
    MFG_DIRECTOR = "Manufacturing Director"
    HC_REP = "Healthcare Representative"
    MANAGER = "Manager"
    SALES_REP = "Sales Representative"
    RESEARCH_DIR = "Research Director"
    HUMAN_RESOURCES = "Human Resources"


class BusinessTravelEnum(str, Enum):
    NON_TRAVEL = "Non-Travel"
    TRAVEL_RARELY = "Travel_Rarely"
    TRAVEL_FREQUENTLY = "Travel_Frequently"


class OvertimeEnum(str, Enum):
    YES = "Yes"
    NO = "No"


# ── Application State ─────────────────────────────────────────────────
class AppState:
    """Container for loaded model artifacts to avoid global mutable state."""

    model: Any = None                       # Raw model (for SHAP)
    calibrated_model: Any = None             # Calibrated wrapper (for predictions)
    explainer: Any = None
    uplift_model: Any = None                 # Causal ML T-Learner
    population_stats: dict[str, Any] | None = None
    model_version: str = "unknown"           # Model provenance hash
    stats_hash: str = "unknown"              # Population stats hash

    # G-2: Shadow model for A/B testing (runs in parallel, results logged but not served)
    shadow_model: Any = None
    shadow_calibrated: Any = None
    shadow_version: str | None = None


app_state = AppState()


# ── Lifespan (modern FastAPI startup/shutdown) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, cleanup on shutdown."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_attrition_model.py first."
        )

    import joblib
    import shap
    import xgboost as xgb
    from src.causal_uplift import TLearnerUplift

    app_state.model = xgb.XGBClassifier()
    app_state.model.load_model(str(MODEL_DIR / "xgb_attrition.json" if 'MODEL_DIR' in locals() else MODEL_PATH))
    app_state.explainer = shap.TreeExplainer(app_state.model)
    app_state.population_stats = load_population_stats()
    
    # Compute model provenance hashes for audit trail
    app_state.model_version = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    stats_path = BASE_DIR / "models" / "population_stats.json"
    if stats_path.exists():
        app_state.stats_hash = hashlib.sha256(stats_path.read_bytes()).hexdigest()[:12]
    
    # Load calibrated model if available (falls back to raw model)
    calibrated_path = BASE_DIR / "models" / "xgb_calibrated.joblib"
    if calibrated_path.exists():
        app_state.calibrated_model = joblib.load(str(calibrated_path))
        logger.info("Calibrated model loaded (Sigmoid / Platt Scaling).")
    else:
        app_state.calibrated_model = app_state.model
        logger.warning("Calibrated model not found — using raw XGBoost probabilities.")
        
    # Load Causal Uplift Model
    uplift_path = BASE_DIR / "models" / "uplift_tlearner.joblib"
    if uplift_path.exists():
        app_state.uplift_model = TLearnerUplift.load(uplift_path)
        logger.info("Causal Uplift Model loaded (T-Learner).")
    else:
        logger.warning("Uplift model not found. Causal inference disabled.")
    
    # Initialize Database Schema
    Base.metadata.create_all(bind=engine)

    # Seed default admin user if no users exist
    from src.models import User, DEFAULT_ADMIN_USER
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username=DEFAULT_ADMIN_USER["username"],
                display_name=DEFAULT_ADMIN_USER["display_name"],
                role=DEFAULT_ADMIN_USER["role"],
            )
            admin.set_password(DEFAULT_ADMIN_USER["password"])
            db.add(admin)
            db.commit()
            logger.info("Default admin user created (username: admin, password: admin123). Change in production!")
    finally:
        db.close()

    logger.info(
        "Model loaded: version=%s, stats=%s",
        app_state.model_version, app_state.stats_hash,
    )

    yield  # Application runs

    # Cleanup
    app_state.model = None
    app_state.calibrated_model = None
    app_state.explainer = None
    app_state.uplift_model = None
    app_state.population_stats = None
    logger.info("Resources cleaned up.")


from fastapi.staticfiles import StaticFiles


# ── Security Headers Middleware ────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Attrition Intelligence API",
    description=(
        "Real-time predictive scoring and explainability API for HR Systems. "
        "Provides SHAP-based risk drivers and LLM-powered retention strategies."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for fairness audit and drift reports
outputs_dir = BASE_DIR / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


# ── Authentication & RBAC ──────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthContext:
    """Authentication context with role information."""
    def __init__(self, api_key: str, role: str):
        self.api_key = api_key
        self.role = role
        self.permissions = ROLE_PERMISSIONS.get(role, set())

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> AuthContext:
    """Validate API key or session token and resolve RBAC role.

    Authentication priority:
    1. Session token (ses-* prefix) → resolve via database
    2. RBAC_KEYS env var → static key-to-role mapping
    3. API_SECRET_KEY → legacy single-key mode
    4. Dev mode bypass → admin role (no keys configured)

    Brute-force protection: locks out IPs after 10 failed attempts in 15 minutes.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Check brute-force lockout
    if _is_locked_out(client_ip):
        logger.error(f"Locked out IP attempted auth: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Try again later.",
        )

    # ── Session Token Mode (username/password login) ──────────────
    if api_key and api_key.startswith("ses-"):
        from src.models import UserSession, User
        db = SessionLocal()
        try:
            session = db.query(UserSession).filter(
                UserSession.token == api_key,
                UserSession.is_active == True,
            ).first()
            if not session or session.is_expired:
                _record_failed_attempt(client_ip, api_key[:10])
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired or invalid. Please login again.",
                )
            user = db.query(User).filter(User.id == session.user_id).first()
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is disabled.",
                )
            _clear_failed_attempts(client_ip)
            return AuthContext(api_key=api_key, role=user.role)
        finally:
            db.close()

    # ── Dev Mode Bypass ───────────────────────────────────────────
    if IS_DEV_MODE and not API_SECRET_KEY and not RBAC_KEYS:
        logger.warning("Running in DEV MODE with no API key — authentication bypassed (admin role).")
        return AuthContext(api_key="dev-mode", role="admin")

    # ── RBAC mode: look up key → role ─────────────────────────────
    if RBAC_KEYS:
        if not api_key:
            _record_failed_attempt(client_ip, "(empty)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key.",
            )
        role = RBAC_KEYS.get(api_key)
        if role is None:
            _record_failed_attempt(client_ip, api_key[:6] if api_key else "(none)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        _clear_failed_attempts(client_ip)
        return AuthContext(api_key=api_key, role=role)

    # ── Legacy single-key mode ────────────────────────────────────
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API_SECRET_KEY not set.",
        )

    if not api_key or not hmac.compare_digest(api_key, API_SECRET_KEY):
        _record_failed_attempt(client_ip, (api_key or "")[:6])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    _clear_failed_attempts(client_ip)
    return AuthContext(api_key=api_key, role="admin")


def require_permission(auth: AuthContext, permission: str) -> None:
    """Check that the authenticated user has the required permission."""
    if not auth.has_permission(permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{auth.role}' does not have '{permission}' permission.",
        )


# ── Pydantic Schemas ───────────────────────────────────────────────────
class EmployeeData(BaseModel):
    """Input schema for employee attrition prediction with strict validation."""

    EmployeeID: str = Field(..., min_length=1, max_length=50)
    Age: int = Field(..., ge=18, le=70, description="Employee age (18-70)")
    JobRole: JobRoleEnum
    JobLevel: int = Field(..., ge=1, le=5)
    MonthlyIncome: float = Field(..., gt=0, le=100000)
    PercentSalaryHike: float = Field(..., ge=0, le=50)
    OverTime: OvertimeEnum
    DistanceFromHome: int = Field(..., ge=0, le=100)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=50)
    YearsInCurrentRole: int = Field(..., ge=0, le=50)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=50)
    YearsWithCurrManager: int = Field(..., ge=0, le=50)
    TotalWorkingYears: int = Field(..., ge=0, le=50)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    JobInvolvement: int = Field(..., ge=1, le=4)
    BusinessTravel: BusinessTravelEnum

    @field_validator("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager")
    @classmethod
    def validate_years_not_exceed_company(cls, v: int, info) -> int:
        """Years in role/promo/manager cannot exceed total company tenure."""
        company_years = info.data.get("YearsAtCompany")
        if company_years is not None and v > company_years:
            raise ValueError(
                f"Cannot exceed YearsAtCompany ({company_years})"
            )
        return v


class PredictionResponse(BaseModel):
    """Output schema for attrition prediction."""

    EmployeeID: str
    Risk_Probability: float
    Risk_Tier: str
    Expected_Financial_Loss: float
    Top_Risk_Drivers: dict[str, float]
    Retention_Strategy: str = ""
    Recommended_Action: str = ""
    Causal_Uplift_Score: float = 0.0
    Uplift_Recommendation: str = ""
    Explainability_Disclaimer: str = (
        "SHAP values reflect statistical correlations in historical data, "
        "not causal relationships. Features may serve as proxies for "
        "protected characteristics. Recommended actions are model scenarios, "
        "not proven interventions. Always apply human judgment before "
        "making employment decisions based on these outputs."
    )
    Causal_Warning: str = (
        "This system identifies statistical patterns associated with attrition. "
        "It does NOT prove that changing any factor will reduce turnover. "
        "Use these insights as starting points for human investigation, not as prescriptions."
    )
    Model_Version: str = ""


class DashboardSummary(BaseModel):
    """Aggregate risk summary for the overview dashboard."""

    total_employees_scored: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    total_value_at_risk: float
    average_risk_probability: float
    top_systemic_drivers: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    stats_loaded: bool
    version: str
    llm_providers: int = 0
    llm_chain: list[str] = []


# ── Helper: Build Feature DataFrame ───────────────────────────────────
def _build_feature_df(employee: EmployeeData) -> pd.DataFrame:
    """Convert employee data to model-ready feature DataFrame.

    Uses the unified feature engineering module with population stats
    computed during training to ensure zero train-serving skew.
    """
    record = employee.model_dump()
    # Convert enums back to strings
    record["JobRole"] = record["JobRole"].value if hasattr(record["JobRole"], "value") else record["JobRole"]
    record["OverTime"] = record["OverTime"].value if hasattr(record["OverTime"], "value") else record["OverTime"]
    record["BusinessTravel"] = record["BusinessTravel"].value if hasattr(record["BusinessTravel"], "value") else record["BusinessTravel"]

    features = engineer_features_single(record, app_state.population_stats)
    df = pd.DataFrame([features])

    # Align columns with XGBoost expectation
    if hasattr(app_state.model, "get_booster"):
        expected_cols = app_state.model.get_booster().feature_names
    else:
        expected_cols = getattr(app_state.model, "feature_names_in_", None)
    if expected_cols is None:
        raise RuntimeError("Loaded model does not expose feature names.")

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]


# ── Helper: LLM Retention Strategy (async, multi-provider fallback) ────
async def _generate_retention_strategy(
    job_role: str,
    top_drivers: dict[str, float],
) -> str:
    """Generate LLM-powered retention strategy via the multi-provider fallback chain.

    Uses the src.llm module which chains through:
        Gemini (primary) → Gemini (backup) → Groq → Groq_2 → Groq_3

    Prompt injection is mitigated via input sanitization.
    """
    from src.llm import generate_text

    # Sanitize inputs to prevent prompt injection
    safe_role = re.sub(r'[^a-zA-Z0-9 \-]', '', str(job_role))[:50]
    safe_drivers = {
        re.sub(r'[^a-zA-Z0-9_]', '', str(k))[:30]: round(float(v), 4)
        for k, v in list(top_drivers.items())[:5]
    }

    prompt = f"""You are an HR Business Partner. A '{safe_role}' is at flight risk.
Top risk factors from our model: {safe_drivers}.

Provide a concise, highly specific retention strategy for their manager.
Format exactly as 2 short bullet points (max 15 words per bullet). Do NOT use markdown bolding (**). Do NOT add intro/outro text.
CRITICAL: Your bullet points MUST directly address the specific factors listed above (e.g., if StockOptionLevel is a factor, mention compensation/equity; if WorkLifeBalance, mention scheduling). Do NOT give generic advice."""

    return await generate_text(
        prompt=prompt,
        fallback_message="AI strategy unavailable. Schedule a 1:1 check-in with the employee.",
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    summary="Predict attrition risk for a single employee",
    tags=["Prediction"],
)
@limiter.limit("60/minute")
async def predict_attrition(
    request: Request,
    employee: EmployeeData,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Score an individual employee for attrition risk.

    Returns probability, risk tier, expected financial loss, top SHAP
    drivers, an LLM-generated retention strategy (for Medium/High risk),
    and a counterfactual recommended action.

    Every prediction is persisted to the audit database for EU AI Act
    Art. 12 record-keeping compliance.

    **RBAC:** Requires 'predict' permission (roles: admin, hr_partner).
    """
    require_permission(auth, "predict")
    try:
        # 0. GDPR Art. 21 — Check scoring exclusion
        exclusion = db.query(ScoringExclusion).filter(
            ScoringExclusion.employee_id == str(employee.EmployeeID),
            ScoringExclusion.is_active == True,
        ).first()
        if exclusion:
            raise HTTPException(
                status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
                detail=(
                    f"Employee {employee.EmployeeID} has exercised their right to object "
                    f"to AI-based scoring (GDPR Art. 21). Scoring is blocked. "
                    f"Contact the DPO to review or revoke this exclusion."
                ),
            )

        # 1. Feature Engineering (unified with training)
        X_infer = _build_feature_df(employee)

        # 2. Prediction (calibrated model for reliable probabilities)
        proba = float(app_state.calibrated_model.predict_proba(X_infer)[0][1])

        # 3. Decision Logic & Cost Model
        if proba >= 0.60:
            tier = "High"
        elif proba >= 0.30:
            tier = "Medium"
        else:
            tier = "Low"

        annual_salary = employee.MonthlyIncome * 12
        replacement_cost = annual_salary * 1.5
        expected_loss = round(proba * replacement_cost, 2)

        # 4. SHAP Local Interpretability
        shap_vals = app_state.explainer.shap_values(X_infer)[0]

        feature_impacts = list(zip(X_infer.columns, shap_vals))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        top_drivers = {
            feat: round(float(val), 4)
            for feat, val in feature_impacts
            if val > 0
        }
        top_3_drivers = dict(list(top_drivers.items())[:3])

        # 5. LLM Retention Copilot (async, non-blocking)
        retention_strategy = "Continue regular check-ins."
        if tier in ["High", "Medium"]:
            retention_strategy = await _generate_retention_strategy(
                employee.JobRole.value, top_3_drivers
            )

        # 6. Counterfactual Engine
        recommended_action = ""
        if tier in ["High", "Medium"]:
            recommended_action = _compute_counterfactual(employee, proba)

        # 6b. Causal Uplift (T-Learner)
        causal_uplift_score = 0.0
        uplift_recommendation = ""
        if app_state.uplift_model is not None and tier in ["High", "Medium"]:
            try:
                # Add proxy treatment feature to match training data
                X_uplift = X_infer.copy()
                X_uplift["Treatment_HighHike"] = (employee.PercentSalaryHike >= 15)
                
                should_intervene, uplift_magnitude = app_state.uplift_model.recommend_intervention(X_uplift)
                # Ensure the CATE is negative meaning reduction in attrition
                causal_uplift_score = round(-uplift_magnitude * 100, 2)
                if should_intervene:
                    uplift_recommendation = f"High ROI: Giving a salary hike >15% is estimated to reduce attrition probability by {abs(causal_uplift_score)}%."
                else:
                    uplift_recommendation = "Low ROI: A salary hike is NOT expected to significantly reduce this employee's flight risk."
            except Exception as e:
                logger.warning(f"Uplift modeling failed for {employee.EmployeeID}: {e}")

        # 7. Persist to Audit Database (EU AI Act Art. 12)
        # G-10: PII masking — store hashed employee ID alongside raw ID
        _eid = str(employee.EmployeeID)
        audit_entry = PredictionLog(
            employee_id=_eid,
            employee_id_hash=mask_employee_id(_eid),
            requester_id=auth.api_key[:8] if auth.api_key != "dev-mode" else "dev",
            requester_role=auth.role,
            input_features=employee.model_dump(mode="json"),
            engineered_features={col: float(X_infer[col].iloc[0]) for col in X_infer.columns},
            attrition_probability=round(proba, 4),
            risk_tier=tier,
            top_risk_drivers=top_3_drivers,
            generated_strategy=retention_strategy,
            model_version=app_state.model_version,
            population_stats_hash=app_state.stats_hash,
        )
        db.add(audit_entry)
        db.commit()

        logger.info(
            "Prediction: employee=%s risk=%.4f tier=%s loss=%.2f [LOGGED id=%s model=%s]",
            employee.EmployeeID, proba, tier, expected_loss,
            audit_entry.id, app_state.model_version,
        )

        # G-2: Shadow model comparison (non-blocking, logged only)
        if app_state.shadow_calibrated is not None:
            try:
                shadow_proba = float(app_state.shadow_calibrated.predict_proba(X_infer)[0][1])
                shadow_tier = "High" if shadow_proba >= 0.60 else ("Medium" if shadow_proba >= 0.30 else "Low")
                logger.info(
                    "SHADOW: employee=%s primary=%.4f(%s) shadow=%.4f(%s) delta=%.4f model=%s",
                    employee.EmployeeID, proba, tier, shadow_proba, shadow_tier,
                    abs(proba - shadow_proba), app_state.shadow_version,
                )
            except Exception as e:
                logger.warning("Shadow model prediction failed: %s", e)

        return PredictionResponse(
            EmployeeID=employee.EmployeeID,
            Risk_Probability=round(proba, 4),
            Risk_Tier=tier,
            Expected_Financial_Loss=expected_loss,
            Top_Risk_Drivers=top_3_drivers,
            Retention_Strategy=retention_strategy,
            Recommended_Action=recommended_action,
            Causal_Uplift_Score=causal_uplift_score,
            Uplift_Recommendation=uplift_recommendation,
            Model_Version=app_state.model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed for %s: %s", employee.EmployeeID, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please contact the HR Analytics team.",
        )


def _compute_counterfactual(employee: EmployeeData, base_proba: float) -> str:
    """Compute minimal actionable intervention that would lower risk the most.

    Uses DiCE (Diverse Counterfactual Explanations) for model-agnostic
    counterfactual search when available (GAP-19 upgrade). Falls back to
    the manual lever-based approach if DiCE is not installed.

    Only actionable features are varied — immutable attributes like Age,
    TotalWorkingYears, and tenure are locked.
    """
    try:
        return _dice_counterfactual(employee, base_proba)
    except Exception:
        return _manual_counterfactual(employee, base_proba)


def _dice_counterfactual(employee: EmployeeData, base_proba: float) -> str:
    """DiCE-powered counterfactual search with actionability constraints."""
    import dice_ml

    # Build the feature dataframe for this employee
    X_infer = _build_feature_df(employee)

    # Define which features HR can actually change (actionable levers)
    features_to_vary = [
        col for col in X_infer.columns
        if any(k in col.lower() for k in [
            "over_time", "percent_salary_hike", "work_life_balance",
            "job_involvement", "monthly_income",
        ])
    ]

    if not features_to_vary:
        raise ValueError("No actionable features found")

    # DiCE needs continuous feature names
    continuous_features = [c for c in X_infer.columns if X_infer[c].dtype != "object"]

    d = dice_ml.Data(
        dataframe=pd.concat([X_infer, pd.DataFrame({"outcome": [1]})], axis=1),
        continuous_features=continuous_features,
        outcome_name="outcome",
    )
    m = dice_ml.Model(model=app_state.model, backend="sklearn", model_type="classifier")
    exp = dice_ml.Dice(d, m, method="random")

    dice_exp = exp.generate_counterfactuals(
        X_infer,
        total_CFs=3,
        desired_class="opposite",
        features_to_vary=features_to_vary,
    )

    # Extract the best counterfactual
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    if cf_df is None or cf_df.empty:
        raise ValueError("DiCE produced no counterfactuals")

    # Find the feature that changed the most meaningfully
    changes = []
    for col in features_to_vary:
        if col in cf_df.columns and col in X_infer.columns:
            orig = float(X_infer[col].iloc[0])
            new = float(cf_df[col].iloc[0])
            if abs(new - orig) > 0.01:
                direction = "increase" if new > orig else "decrease"
                readable = col.replace("_", " ").title()
                changes.append(f"{direction} {readable} ({orig:.1f} → {new:.1f})")

    if changes:
        return f"DiCE suggests: {'; '.join(changes[:2])}"
    raise ValueError("No meaningful changes found")


def _manual_counterfactual(employee: EmployeeData, base_proba: float) -> str:
    """Fallback: manually test 3 actionable levers and pick the best."""
    interventions: dict[str, EmployeeData] = {}

    if employee.OverTime == OvertimeEnum.YES:
        interventions["Remove Overtime"] = employee.model_copy(
            update={"OverTime": OvertimeEnum.NO}
        )

    new_hike = min(employee.PercentSalaryHike + 5, 25)
    if new_hike != employee.PercentSalaryHike:
        interventions[f"Increase Salary Hike to {new_hike}%"] = employee.model_copy(
            update={"PercentSalaryHike": new_hike}
        )

    new_wlb = min(employee.WorkLifeBalance + 1, 4)
    if new_wlb != employee.WorkLifeBalance:
        interventions["Improve Work-Life Balance (+1 level)"] = employee.model_copy(
            update={"WorkLifeBalance": new_wlb}
        )

    if not interventions:
        return "No simple automated lever found. Deep HR intervention needed."

    drops: dict[str, float] = {}
    for label, cf_employee in interventions.items():
        cf_df = _build_feature_df(cf_employee)
        cf_proba = float(app_state.model.predict_proba(cf_df)[0][1])
        drops[label] = base_proba - cf_proba

    best_action = max(drops, key=drops.get)
    best_drop = drops[best_action]

    if best_drop > 0.05:
        return f"{best_action} (Lowers risk by {(best_drop * 100):.1f}%)"

    return "No simple automated lever found. Deep HR intervention needed."


@app.get(
    "/v1/dashboard/summary",
    response_model=DashboardSummary,
    summary="Get aggregate risk summary from latest scoring run",
    tags=["Dashboard"],
)
async def get_dashboard_summary(
    auth: AuthContext = Depends(verify_api_key),
) -> DashboardSummary:
    """Return aggregate risk statistics from the latest batch scoring run.

    Reads from `analytics.predictions` generated by the training pipeline in PostgreSQL.

    **RBAC:** Requires 'dashboard' permission (roles: admin, hr_partner, analyst).
    """
    require_permission(auth, "dashboard")
    try:
        if engine.url.drivername.startswith("postgres"):
            df = pd.read_sql_table("predictions", engine, schema="analytics")
        else:
            df = pd.read_sql_table("predictions", engine)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No risk scores found in database. Run the training pipeline first. Error: {e}",
        )

    # SHAP bar chart data (from global SHAP, approximated via stored scores)
    top_drivers = []
    shap_path = BASE_DIR / "outputs" / "risk_summary.csv"
    if shap_path.exists():
        summary = pd.read_csv(shap_path)
        top_drivers = summary.to_dict(orient="records")

    tier_counts = df["Risk_Tier"].value_counts()

    return DashboardSummary(
        total_employees_scored=len(df),
        high_risk_count=int(tier_counts.get("High", 0)),
        medium_risk_count=int(tier_counts.get("Medium", 0)),
        low_risk_count=int(tier_counts.get("Low", 0)),
        total_value_at_risk=round(df["Expected_Loss"].sum(), 2),
        average_risk_probability=round(df["Predicted_Probability"].mean(), 4),
        top_systemic_drivers=top_drivers,
    )


@app.get(
    "/v1/dashboard/employees",
    summary="Get detailed employee risk scores for Analytics Dashboard",
    tags=["Dashboard"],
)
async def get_dashboard_employees(
    auth: AuthContext = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return full employee risk table for the analytics dashboard.

    **RBAC:** Requires 'dashboard' permission (roles: admin, hr_partner, analyst).
    """
    require_permission(auth, "dashboard")
    try:
        if engine.url.drivername.startswith("postgres"):
            df = pd.read_sql_table("predictions", engine, schema="analytics")
        else:
            df = pd.read_sql_table("predictions", engine)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No risk scores found in database. Run the training pipeline first. Error: {e}",
        )
        
    # Merge with original dataset to get satisfaction/performance metrics for Employee Profiles
    dataset_path = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
    if dataset_path.exists():
        orig_df = pd.read_csv(dataset_path)
        cols_to_keep = ["EmployeeNumber", "PerformanceRating", "JobSatisfaction", "EnvironmentSatisfaction", 
                        "RelationshipSatisfaction", "WorkLifeBalance", "YearsAtCompany", "PercentSalaryHike"]
        if all(col in orig_df.columns for col in cols_to_keep):
            df = df.merge(orig_df[cols_to_keep], on="EmployeeNumber", how="left")

    df = df.fillna("")
    return df.to_dict(orient="records")


@app.get(
    "/v1/dashboard/employees/{employee_id}/narrative",
    summary="Get SHAP and LLM narrative for a specific employee",
    tags=["Dashboard"],
)
async def get_employee_narrative(
    request: Request,
    employee_id: int,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Dynamically generate SHAP explanation and LLM retention strategies for an employee."""
    require_permission(auth, "dashboard")
    
    dataset_path = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    orig_df = pd.read_csv(dataset_path)
    emp_row = orig_df[orig_df["EmployeeNumber"] == employee_id]
    if emp_row.empty:
        raise HTTPException(status_code=404, detail="Employee not found")
        
    emp_dict = emp_row.iloc[0].to_dict()
    # Map CSV column names to Pydantic model names if necessary
    if "EmployeeNumber" in emp_dict and "EmployeeID" not in emp_dict:
        emp_dict["EmployeeID"] = str(emp_dict["EmployeeNumber"])
        
    from src.api import EmployeeData
    valid_fields = EmployeeData.model_fields.keys()
    filtered_dict = {k: v for k, v in emp_dict.items() if k in valid_fields}
    
    try:
        emp_data = EmployeeData(**filtered_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse employee data: {e}")
    prediction = await predict_attrition(request, emp_data, auth, db=db)
    
    return {
        "AI_Contextual_Narrative": prediction.Retention_Strategy,
        "Recommended_Interventions": prediction.Recommended_Action,
        "Top_Risk_Drivers": list(prediction.Top_Risk_Drivers.keys())
    }

@app.get(
    "/v1/dashboard/employees/{employee_id}/profile",
    summary="Get raw employee profile data",
    tags=["Dashboard"],
)
async def get_employee_profile(
    employee_id: int,
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return the raw feature vector for an employee."""
    require_permission(auth, "dashboard")
    
    dataset_path = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    orig_df = pd.read_csv(dataset_path)
    emp_row = orig_df[orig_df["EmployeeNumber"] == employee_id]
    if emp_row.empty:
        raise HTTPException(status_code=404, detail="Employee not found")
        
    emp_dict = emp_row.iloc[0].to_dict()
    if "EmployeeNumber" in emp_dict and "EmployeeID" not in emp_dict:
        emp_dict["EmployeeID"] = str(emp_dict["EmployeeNumber"])
        
    from src.api import EmployeeData
    valid_fields = EmployeeData.model_fields.keys()
    filtered_dict = {k: v for k, v in emp_dict.items() if k in valid_fields}
    
    return filtered_dict


# ── Dashboard: Drift Status ──────────────────────────────────────────
@app.get(
    "/v1/dashboard/drift",
    summary="Get model drift status (data drift + SHAP attribution drift)",
    tags=["Dashboard"],
)
async def get_drift_status(
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return the latest drift monitoring reports.

    Combines:
    - Evidently AI data drift report (input distribution shifts)
    - SHAP attribution drift report (model reasoning shifts)

    **RBAC:** Requires 'dashboard' permission.
    """
    require_permission(auth, "dashboard")
    result: dict[str, Any] = {"data_drift": None, "shap_drift": None}

    # Evidently data drift
    evidently_path = BASE_DIR / "outputs" / "evidently_drift_report.json"
    if evidently_path.exists():
        try:
            result["data_drift"] = json.loads(evidently_path.read_text())
        except Exception:
            result["data_drift"] = {"status": "report_exists", "path": str(evidently_path)}

    # SHAP attribution drift
    shap_drift_path = BASE_DIR / "outputs" / "shap_drift_report.json"
    if shap_drift_path.exists():
        try:
            result["shap_drift"] = json.loads(shap_drift_path.read_text())
        except Exception:
            result["shap_drift"] = {"status": "report_exists", "path": str(shap_drift_path)}

    if not result["data_drift"] and not result["shap_drift"]:
        raise HTTPException(
            status_code=404,
            detail="No drift reports found. Run the training pipeline first.",
        )

    return result


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Return API health status — no authentication required."""
    from src.llm import get_provider_status
    llm_providers = get_provider_status()
    return HealthResponse(
        status="operational",
        model_loaded=app_state.model is not None,
        stats_loaded=app_state.population_stats is not None,
        version="1.1.0",
        llm_providers=len(llm_providers),
        llm_chain=[p["name"] for p in llm_providers],
    )


# ── Audit Trail (EU AI Act Art. 12) ──────────────────────────────────
class AuditLogEntry(BaseModel):
    """Schema for a single audit log entry."""
    id: int
    timestamp: str
    employee_id: str | None
    requester_id: str | None
    attrition_probability: float
    risk_tier: str
    top_risk_drivers: dict[str, float] | None
    generated_strategy: str | None


class AuditLogResponse(BaseModel):
    """Paginated response for audit log queries."""
    total_count: int
    page: int
    page_size: int
    logs: list[AuditLogEntry]


@app.get(
    "/v1/audit-logs",
    response_model=AuditLogResponse,
    summary="Query prediction audit trail",
    tags=["Compliance"],
)
async def get_audit_logs(
    page: int = 1,
    page_size: int = 20,
    risk_tier: str | None = None,
    employee_id: str | None = None,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> AuditLogResponse:
    """Query the prediction audit trail with pagination and filters.

    Provides full traceability of every prediction made by the system,
    as required by EU AI Act Article 12 (Record-Keeping).

    **RBAC:** Requires 'audit' permission (roles: admin, hr_partner, analyst, auditor).
    """
    require_permission(auth, "audit")
    from sqlalchemy import desc

    query = db.query(PredictionLog)

    if risk_tier:
        query = query.filter(PredictionLog.risk_tier == risk_tier)
    if employee_id:
        query = query.filter(PredictionLog.employee_id == employee_id)

    total_count = query.count()

    logs = (
        query
        .order_by(desc(PredictionLog.timestamp))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return AuditLogResponse(
        total_count=total_count,
        page=page,
        page_size=page_size,
        logs=[
            AuditLogEntry(
                id=log.id,
                timestamp=log.timestamp.isoformat() if log.timestamp else "",
                employee_id=log.employee_id,
                requester_id=log.requester_id,
                attrition_probability=log.attrition_probability,
                risk_tier=log.risk_tier,
                top_risk_drivers=log.top_risk_drivers,
                generated_strategy=log.generated_strategy,
            )
            for log in logs
        ],
    )


# ── Human Override (EU AI Act Art. 14) ────────────────────────────────
class OverrideRequest(BaseModel):
    """Schema for human override of AI prediction."""
    employee_id: str = Field(..., min_length=1)
    prediction_log_id: int | None = None
    original_risk_tier: str
    override_risk_tier: str = Field(..., description="Human-assessed risk tier")
    override_reason: str = Field(..., min_length=10, max_length=1000)


class InterventionLogRequest(BaseModel):
    employee_id: str
    prediction_log_id: int | None = None
    intervention_type: str
    intervention_details: str | None = None


class OverrideResponse(BaseModel):
    id: int
    message: str


@app.post(
    "/v1/override",
    response_model=OverrideResponse,
    summary="Record human override of AI prediction",
    tags=["Compliance"],
)
@limiter.limit("30/minute")
async def record_override(
    request: Request,
    override: OverrideRequest,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> OverrideResponse:
    """Record when a human overrides the AI risk assessment.

    Required by EU AI Act Article 14 for effective human oversight.

    **RBAC:** Requires 'override' permission (roles: admin, hr_partner).
    """
    require_permission(auth, "override")
    entry = HumanOverride(
        prediction_log_id=override.prediction_log_id,
        employee_id=override.employee_id,
        original_risk_tier=override.original_risk_tier,
        override_risk_tier=override.override_risk_tier,
        override_reason=override.override_reason,
        overridden_by=f"{auth.role}:{auth.api_key[:8]}" if auth.api_key != "dev-mode" else "dev-user",
    )
    db.add(entry)
    db.commit()
    logger.info(
        "Human override recorded: employee=%s %s→%s by=%s",
        override.employee_id, override.original_risk_tier,
        override.override_risk_tier, entry.overridden_by,
    )
    return OverrideResponse(id=entry.id, message="Override recorded successfully.")


@app.post(
    "/v1/intervention",
    response_model=OverrideResponse,
    summary="Record an applied HR intervention",
    tags=["Causal Inference"],
)
@limiter.limit("30/minute")
async def log_intervention(
    request: Request,
    intervention: InterventionLogRequest,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> OverrideResponse:
    """Record an applied HR intervention (Treatment Assignment Registry).
    
    This tracks longitudinal causal data for future Uplift Model retraining.
    
    **RBAC:** Requires 'override' permission (roles: admin, hr_partner).
    """
    require_permission(auth, "override")
    entry = InterventionLog(
        prediction_log_id=intervention.prediction_log_id,
        employee_id=intervention.employee_id,
        intervention_type=intervention.intervention_type,
        intervention_details=intervention.intervention_details,
        applied_by=f"{auth.role}:{auth.api_key[:8]}" if auth.api_key != "dev-mode" else "dev-user",
    )
    db.add(entry)
    db.commit()
    logger.info(
        "Intervention recorded: employee=%s type=%s by=%s",
        intervention.employee_id, intervention.intervention_type, entry.applied_by,
    )
    return OverrideResponse(id=entry.id, message="Intervention recorded successfully.")


# ── Legacy endpoint redirect ──────────────────────────────────────────
@app.post("/predict", include_in_schema=False, deprecated=True)
async def predict_legacy(
    request: Request,
    employee: EmployeeData,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Legacy endpoint — redirects to /v1/predict."""
    logger.warning("Deprecated /predict endpoint called — migrate to /v1/predict")
    response = await predict_attrition(request, employee, auth, db)
    return response


# ── Dashboard: Trend Analysis ──────────────────────────────────────
@app.get(
    "/v1/dashboard/trends",
    summary="Get risk trend data over time from audit logs",
    tags=["Dashboard"],
)
async def get_dashboard_trends(
    days: int = 30,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return time-series trend data from prediction audit logs.

    Groups predictions by date and risk tier, enabling temporal
    trend analysis in the dashboard.

    **RBAC:** Requires 'dashboard' permission.
    """
    require_permission(auth, "dashboard")
    from sqlalchemy import func, cast, Date
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(
            cast(PredictionLog.timestamp, Date).label("date"),
            PredictionLog.risk_tier,
            func.count().label("count"),
            func.avg(PredictionLog.attrition_probability).label("avg_probability"),
        )
        .filter(PredictionLog.timestamp >= cutoff)
        .group_by(cast(PredictionLog.timestamp, Date), PredictionLog.risk_tier)
        .order_by(cast(PredictionLog.timestamp, Date))
        .all()
    )

    trends = []
    for row in rows:
        trends.append({
            "date": row.date.isoformat() if row.date else "",
            "risk_tier": row.risk_tier,
            "count": row.count,
            "avg_probability": round(float(row.avg_probability), 4) if row.avg_probability else 0,
        })

    return {"period_days": days, "data": trends}


# ── Dashboard: Cohort Comparison ───────────────────────────────────
@app.get(
    "/v1/dashboard/cohorts",
    summary="Compare risk metrics across departments or groups",
    tags=["Dashboard"],
)
async def get_dashboard_cohorts(
    group_by: str = "Department",
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return cohort-level risk comparison for dashboard analytics.

    Aggregates risk scores by department, job level, or other grouping
    columns to enable side-by-side comparison.

    **RBAC:** Requires 'dashboard' permission.
    """
    require_permission(auth, "dashboard")
    try:
        if engine.url.drivername.startswith("postgres"):
            df = pd.read_sql_table("predictions", engine, schema="analytics")
        else:
            df = pd.read_sql_table("predictions", engine)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No risk scores found. Run the training pipeline first. Error: {e}",
        )

    # Validate group_by column exists
    allowed_groups = ["Department", "JobRole", "JobLevel", "Risk_Tier"]
    if group_by not in df.columns:
        # Fall back to Risk_Tier which always exists
        group_by = "Risk_Tier"

    cohorts = []
    for name, group in df.groupby(group_by):
        cohorts.append({
            "cohort": str(name),
            "count": len(group),
            "avg_probability": round(float(group["Predicted_Probability"].mean()), 4)
            if "Predicted_Probability" in group.columns else 0,
            "high_risk_count": int((group["Risk_Tier"] == "High").sum())
            if "Risk_Tier" in group.columns else 0,
            "total_value_at_risk": round(float(group["Expected_Loss"].sum()), 2)
            if "Expected_Loss" in group.columns else 0,
        })

    cohorts.sort(key=lambda x: x["avg_probability"], reverse=True)
    return {"group_by": group_by, "cohorts": cohorts}


# ── Reports: Data Export ──────────────────────────────────────────
@app.get(
    "/v1/reports/export",
    summary="Export risk scores as CSV or JSON",
    tags=["Reports"],
)
async def export_risk_scores(
    format: str = "json",
    auth: AuthContext = Depends(verify_api_key),
) -> Any:
    """Export the latest risk scores for offline analysis.

    Supports JSON and CSV formats.

    **RBAC:** Requires 'export' permission (roles: admin, hr_partner, analyst).
    """
    require_permission(auth, "export")
    try:
        if engine.url.drivername.startswith("postgres"):
            df = pd.read_sql_table("predictions", engine, schema="analytics")
        else:
            df = pd.read_sql_table("predictions", engine)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No risk scores found. Error: {e}",
        )

    if format.lower() == "csv":
        from starlette.responses import StreamingResponse
        import io
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=risk_scores_export.csv"},
        )

    return {
        "export_date": pd.Timestamp.now().isoformat(),
        "total_records": len(df),
        "data": df.fillna("").to_dict(orient="records"),
    }


# ── System: RBAC Info ────────────────────────────────────────────
@app.get(
    "/v1/auth/whoami",
    summary="Return current user role and permissions",
    tags=["System"],
)
async def whoami(
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return the role and permissions of the authenticated API key."""
    return {
        "role": auth.role,
        "permissions": sorted(auth.permissions),
        "api_key_prefix": auth.api_key[:8] + "..." if len(auth.api_key) > 8 else auth.api_key,
    }


# ── GDPR Art. 21 — Scoring Exclusion Management ──────────────────────
class ExclusionRequest(BaseModel):
    """Schema for requesting employee exclusion from AI scoring."""
    employee_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=10, max_length=1000)
    requested_by: str = Field(..., min_length=1, description="DPO, employee name, or HR contact")


class ExclusionResponse(BaseModel):
    id: int
    employee_id: str
    message: str


@app.post(
    "/v1/gdpr/exclude",
    response_model=ExclusionResponse,
    summary="Exclude employee from AI scoring (GDPR Art. 21)",
    tags=["Compliance"],
)
@limiter.limit("10/minute")
async def exclude_employee(
    request: Request,
    exclusion: ExclusionRequest,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> ExclusionResponse:
    """Record an employee's right to object to AI-based scoring.

    Once excluded, the /v1/predict endpoint will reject scoring
    requests for this employee with HTTP 451 (Unavailable For Legal Reasons).

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")

    # Check if already excluded
    existing = db.query(ScoringExclusion).filter(
        ScoringExclusion.employee_id == exclusion.employee_id,
        ScoringExclusion.is_active == True,
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Employee {exclusion.employee_id} is already excluded (id={existing.id}).",
        )

    entry = ScoringExclusion(
        employee_id=exclusion.employee_id,
        employee_id_hash=mask_employee_id(exclusion.employee_id),
        reason=exclusion.reason,
        requested_by=exclusion.requested_by,
        approved_by=f"{auth.role}:{auth.api_key[:8]}" if auth.api_key != "dev-mode" else "dev-user",
    )
    db.add(entry)
    db.commit()
    logger.info("GDPR Art.21 exclusion recorded: employee=%s by=%s", exclusion.employee_id, entry.approved_by)
    return ExclusionResponse(
        id=entry.id,
        employee_id=exclusion.employee_id,
        message="Employee excluded from AI scoring. Existing audit logs retained per Art. 6(1)(c).",
    )


@app.delete(
    "/v1/gdpr/exclude/{employee_id}",
    summary="Revoke scoring exclusion for an employee",
    tags=["Compliance"],
)
@limiter.limit("10/minute")
async def revoke_exclusion(
    request: Request,
    employee_id: str,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Revoke an active scoring exclusion, re-enabling AI scoring for the employee.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    from datetime import datetime, timezone

    exclusion = db.query(ScoringExclusion).filter(
        ScoringExclusion.employee_id == employee_id,
        ScoringExclusion.is_active == True,
    ).first()
    if not exclusion:
        raise HTTPException(status_code=404, detail=f"No active exclusion found for {employee_id}.")

    exclusion.is_active = False
    exclusion.revoked_at = datetime.now(timezone.utc)
    exclusion.revoked_by = f"{auth.role}:{auth.api_key[:8]}" if auth.api_key != "dev-mode" else "dev-user"
    db.commit()
    logger.info("GDPR exclusion revoked: employee=%s by=%s", employee_id, exclusion.revoked_by)
    return {"message": f"Exclusion revoked for {employee_id}. AI scoring re-enabled."}


@app.get(
    "/v1/gdpr/exclusions",
    summary="List all scoring exclusions",
    tags=["Compliance"],
)
async def list_exclusions(
    active_only: bool = True,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List all employees excluded from AI scoring.

    **RBAC:** Requires 'audit' permission.
    """
    require_permission(auth, "audit")
    query = db.query(ScoringExclusion)
    if active_only:
        query = query.filter(ScoringExclusion.is_active == True)
    exclusions = query.order_by(ScoringExclusion.timestamp.desc()).all()
    return {
        "total": len(exclusions),
        "active_only": active_only,
        "exclusions": [
            {
                "id": e.id,
                "employee_id": e.employee_id,
                "reason": e.reason,
                "requested_by": e.requested_by,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "is_active": e.is_active,
            }
            for e in exclusions
        ],
    }


# ── Model Rollback (G-6) ────────────────────────────────────────────
@app.post(
    "/v1/system/model/rollback",
    summary="Rollback to a previous model version",
    tags=["System"],
)
@limiter.limit("5/minute")
async def rollback_model(
    request: Request,
    model_filename: str = "xgb_attrition_previous.json",
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Hot-swap the serving model to a previous version.

    The system keeps the previous model alongside the current one.
    This endpoint loads the specified model file and replaces the
    in-memory model without requiring a restart.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    import xgboost as xgb
    import shap
    import joblib

    target_path = BASE_DIR / "models" / model_filename
    if not target_path.exists():
        available = [f.name for f in (BASE_DIR / "models").glob("xgb_*.json")]
        raise HTTPException(
            status_code=404,
            detail=f"Model file '{model_filename}' not found. Available: {available}",
        )

    old_version = app_state.model_version

    # Load new model
    new_model = xgb.XGBClassifier()
    new_model.load_model(str(target_path))
    new_version = hashlib.sha256(target_path.read_bytes()).hexdigest()[:12]

    # Load calibrated version if available
    calibrated_name = model_filename.replace(".json", "").replace("xgb_attrition", "xgb_calibrated") + ".joblib"
    calibrated_path = BASE_DIR / "models" / calibrated_name
    if calibrated_path.exists():
        new_calibrated = joblib.load(str(calibrated_path))
    else:
        new_calibrated = new_model

    # Hot-swap
    app_state.model = new_model
    app_state.calibrated_model = new_calibrated
    app_state.explainer = shap.TreeExplainer(new_model)
    app_state.model_version = new_version

    logger.warning(
        "MODEL ROLLBACK: %s → %s (file=%s, by=%s)",
        old_version, new_version, model_filename, auth.role,
    )

    return {
        "message": "Model rollback successful.",
        "previous_version": old_version,
        "current_version": new_version,
        "model_file": model_filename,
        "calibrated": calibrated_path.exists(),
    }


@app.get(
    "/v1/system/models",
    summary="List available model versions",
    tags=["System"],
)
async def list_models(
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all model files available for serving or rollback.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    models_dir = BASE_DIR / "models"
    model_files = []
    for f in sorted(models_dir.glob("xgb_*.json")):
        version_hash = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        is_current = version_hash == app_state.model_version
        model_files.append({
            "filename": f.name,
            "version_hash": version_hash,
            "size_bytes": f.stat().st_size,
            "is_current": is_current,
        })
    return {
        "current_version": app_state.model_version,
        "shadow_version": app_state.shadow_version,
        "available_models": model_files,
        "db_info": get_db_info(),
    }


# ── G-2: Shadow Model A/B Testing ────────────────────────────────────
@app.post(
    "/v1/system/shadow/load",
    summary="Load a shadow model for A/B comparison",
    tags=["System"],
)
@limiter.limit("5/minute")
async def load_shadow_model(
    request: Request,
    model_filename: str = "xgb_attrition_previous.json",
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Load a shadow model that runs in parallel with the primary model.

    Shadow predictions are logged but never served to the client.
    This enables safe offline comparison before promoting a new model.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    import xgboost as xgb
    import joblib

    target_path = BASE_DIR / "models" / model_filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found.")

    shadow = xgb.XGBClassifier()
    shadow.load_model(str(target_path))
    shadow_version = hashlib.sha256(target_path.read_bytes()).hexdigest()[:12]

    calibrated_name = model_filename.replace(".json", "").replace("xgb_attrition", "xgb_calibrated") + ".joblib"
    calibrated_path = BASE_DIR / "models" / calibrated_name
    if calibrated_path.exists():
        app_state.shadow_calibrated = joblib.load(str(calibrated_path))
    else:
        app_state.shadow_calibrated = shadow

    app_state.shadow_model = shadow
    app_state.shadow_version = shadow_version

    logger.info("Shadow model loaded: %s (version=%s)", model_filename, shadow_version)
    return {
        "message": "Shadow model loaded for A/B comparison.",
        "shadow_version": shadow_version,
        "primary_version": app_state.model_version,
        "calibrated": calibrated_path.exists(),
    }


@app.delete(
    "/v1/system/shadow",
    summary="Remove shadow model",
    tags=["System"],
)
async def remove_shadow_model(
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove the shadow model, stopping A/B comparison logging.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    if app_state.shadow_model is None:
        raise HTTPException(status_code=404, detail="No shadow model is currently loaded.")

    old_version = app_state.shadow_version
    app_state.shadow_model = None
    app_state.shadow_calibrated = None
    app_state.shadow_version = None
    logger.info("Shadow model removed (was version=%s)", old_version)
    return {"message": f"Shadow model removed (was {old_version})."}


# ── G-1: Feature Store Info ───────────────────────────────────────────
@app.get(
    "/v1/system/feature-store",
    summary="Feature store metadata and population statistics",
    tags=["System"],
)
async def feature_store_info(
    auth: AuthContext = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return metadata about the feature engineering configuration.

    Exposes the population statistics that serve as the centralized
    feature store for ensuring zero train-serving skew.

    **RBAC:** Requires 'system' permission (roles: admin).
    """
    require_permission(auth, "system")
    stats = app_state.population_stats or {}
    stats_path = BASE_DIR / "models" / "population_stats.json"

    return {
        "feature_store_type": "embedded",
        "description": "Unified feature engineering via src/features.py with serialized population statistics",
        "stats_hash": app_state.stats_hash,
        "stats_file": str(stats_path),
        "stats_exists": stats_path.exists(),
        "population_stats": stats,
        "features_documented": [
            "engagement_index", "burnout_risk", "compa_ratio",
            "promotion_stagnation", "income_growth_gap", "loyalty_index",
            "career_velocity", "manager_stability", "travel_burden",
        ],
        "train_serving_skew_prevention": "features.py is the Single Source of Truth for both training and inference",
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


# ── Authentication: Login/Logout ──────────────────────────────────────
class LoginRequest(BaseModel):
    """Login credentials."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


@app.post(
    "/v1/auth/login",
    summary="Login with username and password",
    tags=["Auth"],
)
@limiter.limit("5/minute")
async def login(
    request: Request,
    credentials: LoginRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Authenticate with username/password and receive a session token.

    The session token should be sent via `X-API-Key` header for subsequent requests.
    Sessions expire after 8 hours.
    """
    from src.models import User as UserModel, UserSession, generate_session_token
    client_ip = request.client.host if request.client else "unknown"

    # Check brute-force lockout
    if _is_locked_out(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Try again later.",
        )

    user = db.query(UserModel).filter(
        UserModel.username == credentials.username,
        UserModel.is_active == True,
    ).first()

    if not user or not user.check_password(credentials.password):
        _record_failed_attempt(client_ip, credentials.username[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    _clear_failed_attempts(client_ip)

    # Create session token
    from datetime import datetime, timezone, timedelta
    token = generate_session_token()
    session = UserSession(
        token=token,
        user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
    )
    db.add(session)

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    logger.info(f"User '{user.username}' logged in from {client_ip}")

    return {
        "token": token,
        "role": user.role,
        "permissions": sorted(ROLE_PERMISSIONS.get(user.role, set())),
        "display_name": user.display_name or user.username,
        "expires_in_hours": 8,
    }


@app.post(
    "/v1/auth/logout",
    summary="Logout and invalidate session",
    tags=["Auth"],
)
async def logout_endpoint(
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Invalidate the current session token."""
    from src.models import UserSession
    if auth.api_key.startswith("ses-"):
        session = db.query(UserSession).filter(
            UserSession.token == auth.api_key,
        ).first()
        if session:
            session.is_active = False
            db.commit()
    return {"message": "Logged out successfully."}


@app.get(
    "/v1/auth/whoami",
    summary="Get current user identity and permissions",
    tags=["Auth"],
)
async def whoami_endpoint(
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the currently authenticated user's role and permissions."""
    display_name = ""
    # Try to fetch display_name from user if token-based
    if auth.api_key.startswith("ses-"):
        from src.models import UserSession, User
        session = db.query(UserSession).filter(UserSession.token == auth.api_key).first()
        if session:
            user = db.query(User).filter(User.id == session.user_id).first()
            if user:
                display_name = user.display_name or user.username
    
    return {
        "role": auth.role,
        "permissions": sorted(list(auth.permissions)),
        "api_key_prefix": auth.api_key[:6] if auth.api_key else "",
        "display_name": display_name,
    }



# ── Admin: User Management ────────────────────────────────────────────
class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=200)
    display_name: str | None = Field(None, max_length=200)
    role: str = Field("analyst")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid = {"admin", "hr_partner", "analyst", "auditor"}
        if v not in valid:
            raise ValueError(f"Role must be one of: {', '.join(sorted(valid))}")
        return v


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=6, max_length=200)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid = {"admin", "hr_partner", "analyst", "auditor"}
        if v not in valid:
            raise ValueError(f"Role must be one of: {', '.join(sorted(valid))}")
        return v


def _user_to_dict(user) -> dict:
    """Convert User ORM object to API response dict."""
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


@app.get(
    "/v1/admin/users",
    summary="List all users",
    tags=["Admin"],
)
async def list_users(
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List all system users. Requires admin role."""
    require_permission(auth, "system")
    from src.models import User as UserModel
    users = db.query(UserModel).order_by(UserModel.id).all()
    return {
        "total": len(users),
        "users": [_user_to_dict(u) for u in users],
    }


@app.post(
    "/v1/admin/users",
    summary="Create a new user",
    tags=["Admin"],
    status_code=201,
)
async def create_user(
    user_data: CreateUserRequest,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create a new user account. Requires admin role."""
    require_permission(auth, "system")
    from src.models import User as UserModel

    existing = db.query(UserModel).filter(UserModel.username == user_data.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{user_data.username}' already exists.",
        )

    user = UserModel(
        username=user_data.username,
        display_name=user_data.display_name or user_data.username,
        role=user_data.role,
    )
    user.set_password(user_data.password)
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"User created: {user.username} (role: {user.role}) by {auth.role}")
    return {"message": "User created.", "user": _user_to_dict(user)}


@app.put(
    "/v1/admin/users/{user_id}",
    summary="Update a user",
    tags=["Admin"],
)
async def update_user(
    user_id: int,
    update: UpdateUserRequest,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update user details. Requires admin role."""
    require_permission(auth, "system")
    from src.models import User as UserModel

    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Check if we are demoting or disabling the only active admin
    if user.role == "admin":
        is_demoting = update.role is not None and update.role != "admin"
        is_disabling = update.is_active is False
        if is_demoting or is_disabling:
            active_admins = db.query(UserModel).filter(
                UserModel.role == "admin", 
                UserModel.is_active == True, 
                UserModel.id != user_id
            ).count()
            if active_admins == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot disable or change role of the only active admin.",
                )

    if update.display_name is not None:
        user.display_name = update.display_name
    if update.role is not None:
        user.role = update.role
    if update.is_active is not None:
        user.is_active = update.is_active
    if update.password is not None:
        user.set_password(update.password)

    db.commit()
    db.refresh(user)

    logger.info(f"User updated: {user.username} by {auth.role}")
    return {"message": "User updated.", "user": _user_to_dict(user)}


@app.delete(
    "/v1/admin/users/{user_id}",
    summary="Delete a user",
    tags=["Admin"],
)
async def delete_user(
    user_id: int,
    auth: AuthContext = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a user account. Requires admin role. Cannot delete yourself."""
    require_permission(auth, "system")
    from src.models import User as UserModel, UserSession

    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Prevent self-deletion
    if auth.api_key.startswith("ses-"):
        current_session = db.query(UserSession).filter(
            UserSession.token == auth.api_key,
        ).first()
        if current_session and current_session.user_id == user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account.",
            )

    # Prevent deleting the only active admin
    if user.role == "admin":
        active_admins = db.query(UserModel).filter(
            UserModel.role == "admin", 
            UserModel.is_active == True, 
            UserModel.id != user_id
        ).count()
        if active_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the only active admin.",
            )

    # Invalidate all sessions for this user
    db.query(UserSession).filter(UserSession.user_id == user_id).delete()
    db.delete(user)
    db.commit()

    logger.info(f"User deleted: {user.username} by {auth.role}")
    return {"message": f"User '{user.username}' deleted."}
