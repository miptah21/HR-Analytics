"""
Tests for Role-Based Access Control (RBAC) in the HR Analytics API.

Validates that:
- RBAC role resolution works correctly from API keys
- Permission enforcement blocks unauthorized access
- Dev mode defaults to admin
- Each role has the correct permission set
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_model_loading():
    """Mock model artifacts so tests don't require actual model files."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.65, 0.35]])
    mock_model.feature_names_in_ = [f"feat_{i}" for i in range(53)]

    mock_calibrated = MagicMock()
    mock_calibrated.predict_proba.return_value = np.array([[0.65, 0.35]])

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [np.random.randn(53)]

    mock_stats = {
        "role_avg_income": {
            "Sales Executive": 6924.28,
            "Research Scientist": 3239.97,
            "Laboratory Technician": 3237.17,
            "Manufacturing Director": 7295.14,
            "Healthcare Representative": 7528.76,
            "Manager": 17181.68,
            "Sales Representative": 2626.0,
            "Research Director": 16033.55,
            "Human Resources": 4235.75,
        },
        "level_hike_median": {"1": 14.0, "2": 14.0, "3": 14.0, "4": 14.0, "5": 14.0},
    }

    with patch("src.api.app_state") as mock_state:
        mock_state.model = mock_model
        mock_state.calibrated_model = mock_calibrated
        mock_state.explainer = mock_explainer
        mock_state.population_stats = mock_stats
        mock_state.model_version = "test123"
        mock_state.stats_hash = "hash456"
        yield mock_state


VALID_EMPLOYEE = {
    "EmployeeID": "RBAC-001",
    "Age": 35,
    "JobRole": "Sales Executive",
    "JobLevel": 3,
    "MonthlyIncome": 8000,
    "PercentSalaryHike": 14,
    "OverTime": "No",
    "DistanceFromHome": 5,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 7,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 3,
    "TotalWorkingYears": 12,
    "JobSatisfaction": 3,
    "EnvironmentSatisfaction": 3,
    "RelationshipSatisfaction": 3,
    "JobInvolvement": 3,
    "BusinessTravel": "Travel_Rarely",
}


# ── AuthContext Unit Tests ────────────────────────────────────────────

class TestAuthContext:
    """Test the AuthContext class and RBAC resolution."""

    def test_admin_has_all_permissions(self):
        from src.api import AuthContext
        ctx = AuthContext(api_key="test-key", role="admin")
        assert ctx.has_permission("predict")
        assert ctx.has_permission("override")
        assert ctx.has_permission("dashboard")
        assert ctx.has_permission("audit")
        assert ctx.has_permission("export")
        assert ctx.has_permission("system")

    def test_hr_partner_permissions(self):
        from src.api import AuthContext
        ctx = AuthContext(api_key="test-key", role="hr_partner")
        assert ctx.has_permission("predict")
        assert ctx.has_permission("override")
        assert ctx.has_permission("dashboard")
        assert ctx.has_permission("audit")
        assert ctx.has_permission("export")
        assert not ctx.has_permission("system")

    def test_analyst_permissions(self):
        from src.api import AuthContext
        ctx = AuthContext(api_key="test-key", role="analyst")
        assert not ctx.has_permission("predict")
        assert not ctx.has_permission("override")
        assert ctx.has_permission("dashboard")
        assert ctx.has_permission("audit")
        assert ctx.has_permission("export")

    def test_auditor_permissions(self):
        from src.api import AuthContext
        ctx = AuthContext(api_key="test-key", role="auditor")
        assert not ctx.has_permission("predict")
        assert not ctx.has_permission("override")
        assert not ctx.has_permission("dashboard")
        assert ctx.has_permission("audit")
        assert not ctx.has_permission("export")

    def test_unknown_role_has_no_permissions(self):
        from src.api import AuthContext
        ctx = AuthContext(api_key="test-key", role="unknown")
        assert not ctx.has_permission("predict")
        assert not ctx.has_permission("dashboard")
        assert not ctx.has_permission("audit")


# ── RBAC Integration Tests ───────────────────────────────────────────

class TestRBACIntegration:
    """Test RBAC enforcement at the API level."""

    def test_dev_mode_bypasses_auth(self, mock_model_loading):
        """Dev mode (no keys set) should bypass auth with admin role."""
        with patch("src.api.IS_DEV_MODE", True), \
             patch("src.api.API_SECRET_KEY", ""), \
             patch("src.api.RBAC_KEYS", {}):
            from src.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/v1/auth/whoami")
            assert response.status_code == 200
            data = response.json()
            assert data["role"] == "admin"

    def test_rbac_keys_resolve_role(self, mock_model_loading):
        """RBAC_KEYS should resolve key → role correctly."""
        rbac = {"sk-test-analyst": "analyst"}
        with patch("src.api.IS_DEV_MODE", False), \
             patch("src.api.API_SECRET_KEY", ""), \
             patch("src.api.RBAC_KEYS", rbac):
            from src.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/v1/auth/whoami",
                headers={"X-API-Key": "sk-test-analyst"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["role"] == "analyst"

    def test_rbac_invalid_key_rejected(self, mock_model_loading):
        """Invalid key in RBAC mode returns 401."""
        rbac = {"sk-valid": "admin"}
        with patch("src.api.IS_DEV_MODE", False), \
             patch("src.api.API_SECRET_KEY", ""), \
             patch("src.api.RBAC_KEYS", rbac):
            from src.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/v1/auth/whoami",
                headers={"X-API-Key": "sk-invalid"},
            )
            assert response.status_code == 401

    def test_auditor_cannot_predict(self, mock_model_loading):
        """Auditor role should be blocked from /v1/predict."""
        rbac = {"sk-auditor": "auditor"}
        with patch("src.api.IS_DEV_MODE", False), \
             patch("src.api.API_SECRET_KEY", ""), \
             patch("src.api.RBAC_KEYS", rbac):
            from src.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/v1/predict",
                json=VALID_EMPLOYEE,
                headers={"X-API-Key": "sk-auditor"},
            )
            assert response.status_code == 403
            assert "predict" in response.json()["detail"]


# ── require_permission Unit Test ─────────────────────────────────────

class TestRequirePermission:
    """Test the permission enforcement helper."""

    def test_allowed_permission_passes(self):
        from src.api import AuthContext, require_permission
        ctx = AuthContext(api_key="test", role="admin")
        # Should not raise
        require_permission(ctx, "predict")

    def test_denied_permission_raises(self):
        from src.api import AuthContext, require_permission
        from fastapi import HTTPException
        ctx = AuthContext(api_key="test", role="auditor")
        with pytest.raises(HTTPException) as exc_info:
            require_permission(ctx, "predict")
        assert exc_info.value.status_code == 403
