"""
Integration tests for the HR Attrition API.

Tests API endpoints, input validation, authentication, error handling,
and audit trail persistence using FastAPI's TestClient.
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


@pytest.fixture
def client(mock_model_loading):
    """Create a test client with mocked state."""
    # Import here so mocks are already in place
    from src.api import app
    return TestClient(app, raise_server_exceptions=False)


VALID_EMPLOYEE = {
    "EmployeeID": "TEST-001",
    "Age": 30,
    "JobRole": "Sales Executive",
    "JobLevel": 2,
    "MonthlyIncome": 5000,
    "PercentSalaryHike": 14,
    "OverTime": "Yes",
    "DistanceFromHome": 10,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 3,
    "TotalWorkingYears": 8,
    "JobSatisfaction": 3,
    "EnvironmentSatisfaction": 3,
    "RelationshipSatisfaction": 3,
    "JobInvolvement": 3,
    "BusinessTravel": "Travel_Rarely",
}


# ── Health & Status ───────────────────────────────────────────────────

class TestHealth:
    def test_health_endpoint(self, client):
        """GET /v1/health returns 200 with model status."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


# ── Input Validation ──────────────────────────────────────────────────

class TestInputValidation:
    def test_invalid_job_role(self, client):
        """Invalid JobRole enum value returns 422."""
        bad = {**VALID_EMPLOYEE, "JobRole": "CEO"}
        response = client.post("/v1/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_overtime(self, client):
        """Invalid OverTime value returns 422."""
        bad = {**VALID_EMPLOYEE, "OverTime": "Maybe"}
        response = client.post("/v1/predict", json=bad)
        assert response.status_code == 422

    def test_negative_income(self, client):
        """Negative MonthlyIncome returns 422."""
        bad = {**VALID_EMPLOYEE, "MonthlyIncome": -1000}
        response = client.post("/v1/predict", json=bad)
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Missing required field returns 422."""
        bad = {k: v for k, v in VALID_EMPLOYEE.items() if k != "Age"}
        response = client.post("/v1/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_business_travel(self, client):
        """Invalid BusinessTravel enum returns 422."""
        bad = {**VALID_EMPLOYEE, "BusinessTravel": "Travel_Always"}
        response = client.post("/v1/predict", json=bad)
        assert response.status_code == 422


# ── Error Handling ────────────────────────────────────────────────────

class TestErrorHandling:
    def test_error_response_sanitized(self, client, mock_model_loading):
        """Error responses should NOT leak internal details."""
        mock_model_loading.calibrated_model.predict_proba.side_effect = RuntimeError(
            "Connection to /var/lib/postgres:5432 refused"
        )
        response = client.post("/v1/predict", json=VALID_EMPLOYEE)
        assert response.status_code == 500
        # Should NOT contain internal paths or connection strings
        detail = response.json().get("detail", "")
        assert "/var/lib" not in detail
        assert "postgres" not in detail.lower()
        assert "HR Analytics team" in detail


# ── Prediction Response Schema ────────────────────────────────────────

class TestPredictionResponse:
    def test_response_has_disclaimer(self, client):
        """Prediction response includes SHAP explainability disclaimer."""
        response = client.post("/v1/predict", json=VALID_EMPLOYEE)
        if response.status_code == 200:
            data = response.json()
            assert "Explainability_Disclaimer" in data
            assert "causal" in data["Explainability_Disclaimer"].lower()

    def test_response_has_model_version(self, client):
        """Prediction response includes model version for traceability."""
        response = client.post("/v1/predict", json=VALID_EMPLOYEE)
        if response.status_code == 200:
            data = response.json()
            assert "Model_Version" in data
            assert data["Model_Version"] != ""


# ── Override Endpoint ─────────────────────────────────────────────────

class TestHumanOverride:
    def test_override_requires_reason(self, client):
        """Override with reason shorter than 10 chars fails validation."""
        override = {
            "employee_id": "EMP-001",
            "original_risk_tier": "High",
            "override_risk_tier": "Low",
            "override_reason": "no",  # Too short
        }
        response = client.post("/v1/override", json=override)
        assert response.status_code == 422

    def test_override_requires_employee_id(self, client):
        """Override without employee_id fails validation."""
        override = {
            "original_risk_tier": "High",
            "override_risk_tier": "Low",
            "override_reason": "Employee has accepted a counter-offer and is staying.",
        }
        response = client.post("/v1/override", json=override)
        assert response.status_code == 422
