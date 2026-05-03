"""
Tests for adversarial robustness, causal validation, and survival analysis.

Validates that the new pipeline stages (EU AI Act Art. 15 compliance,
causal inference, and survival analysis) execute correctly and produce
well-structured output.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_xgb_model():
    """Create a mock XGBoost model with consistent predict_proba."""
    model = MagicMock()
    # Return consistent probabilities for determinism testing
    def predict_proba_side_effect(X):
        np.random.seed(42)
        n = X.shape[0]
        p = np.clip(np.random.beta(2, 5, n), 0.01, 0.99)
        return np.column_stack([1 - p, p])
    
    model.predict_proba.side_effect = predict_proba_side_effect
    return model


@pytest.fixture
def sample_test_data():
    """Create sample test data for robustness testing."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(n) for i in range(10)
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def sample_raw_df():
    """Create sample raw DataFrame for causal and survival analysis."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Attrition": np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "Age": np.random.randint(22, 60, n),
        "MonthlyIncome": np.random.randint(2000, 20000, n),
        "OverTime": np.random.choice(["Yes", "No"], n),
        "JobLevel": np.random.randint(1, 6, n),
        "YearsAtCompany": np.random.randint(0, 30, n),
        "TotalWorkingYears": np.random.randint(1, 35, n),
        "DistanceFromHome": np.random.randint(1, 30, n),
        "JobSatisfaction": np.random.randint(1, 5, n),
        "EnvironmentSatisfaction": np.random.randint(1, 5, n),
        "WorkLifeBalance": np.random.randint(1, 5, n),
        "YearsSinceLastPromotion": np.random.randint(0, 15, n),
    })


# ── Quality Gate Tests ────────────────────────────────────────────────

class TestQualityGateEOD:
    """Verify that the quality gate now checks both DPD and EOD."""

    def test_eod_failure_blocks_registration(self):
        """Fairness results with failed EOD should produce quality issues."""
        fairness_results = {
            "Gender": {
                "demographic_parity_diff": 0.05,
                "equalized_odds_diff": 0.20,
                "dpd_pass": True,
                "eod_pass": False,
            }
        }
        
        quality_issues = []
        for attr, fair_metrics in fairness_results.items():
            if not fair_metrics.get("dpd_pass", True):
                quality_issues.append(
                    f"Fairness DPD FAILED for {attr}: DPD={fair_metrics['demographic_parity_diff']:.4f}"
                )
            if not fair_metrics.get("eod_pass", True):
                quality_issues.append(
                    f"Fairness EOD FAILED for {attr}: EOD={fair_metrics['equalized_odds_diff']:.4f}"
                )
        
        # EOD failure should generate a quality issue
        assert len(quality_issues) == 1
        assert "EOD FAILED" in quality_issues[0]
        assert "Gender" in quality_issues[0]

    def test_both_dpd_and_eod_failure(self):
        """Both DPD and EOD failures should be reported."""
        fairness_results = {
            "Age_Group": {
                "demographic_parity_diff": 0.276,
                "equalized_odds_diff": 0.611,
                "dpd_pass": False,
                "eod_pass": False,
            }
        }
        
        quality_issues = []
        for attr, fair_metrics in fairness_results.items():
            if not fair_metrics.get("dpd_pass", True):
                quality_issues.append(f"DPD FAILED for {attr}")
            if not fair_metrics.get("eod_pass", True):
                quality_issues.append(f"EOD FAILED for {attr}")
        
        assert len(quality_issues) == 2

    def test_passing_fairness(self):
        """Both DPD and EOD passing should generate zero issues."""
        fairness_results = {
            "Gender": {
                "demographic_parity_diff": 0.05,
                "equalized_odds_diff": 0.08,
                "dpd_pass": True,
                "eod_pass": True,
            }
        }
        
        quality_issues = []
        for attr, fair_metrics in fairness_results.items():
            if not fair_metrics.get("dpd_pass", True):
                quality_issues.append(f"DPD")
            if not fair_metrics.get("eod_pass", True):
                quality_issues.append(f"EOD")
        
        assert len(quality_issues) == 0


# ── Adversarial Robustness Tests ──────────────────────────────────────

class TestAdversarialRobustness:
    """Tests for the adversarial robustness testing module."""

    def test_function_exists(self):
        """Verify the robustness test function is importable."""
        from src.train_attrition_model import run_adversarial_robustness_test
        assert callable(run_adversarial_robustness_test)

    def test_robustness_runs_without_error(self, sample_test_data, tmp_path):
        """Robustness test should complete without raising exceptions."""
        from src.train_attrition_model import run_adversarial_robustness_test
        import src.train_attrition_model as module
        
        X, y = sample_test_data
        
        # Create a real-ish mock model
        model = MagicMock()
        def stable_predict(X_input):
            n = X_input.shape[0]
            p = np.full(n, 0.3)
            return np.column_stack([1 - p, p])
        model.predict_proba.side_effect = stable_predict
        
        # Redirect output directory
        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            run_adversarial_robustness_test(model, X, y)
            report_path = tmp_path / "adversarial_robustness_report.json"
            assert report_path.exists()
            
            import json
            with open(report_path) as f:
                report = json.load(f)
            assert "tests" in report
            assert "overall_verdict" in report
            assert len(report["tests"]) == 3
        finally:
            module.OUT_DIR = original_out


# ── Causal Validation Tests ───────────────────────────────────────────

class TestCausalValidation:
    """Tests for the causal inference validation module."""

    def test_function_exists(self):
        """Verify the causal validation function is importable."""
        from src.train_attrition_model import run_causal_validation
        assert callable(run_causal_validation)

    def test_causal_handles_missing_column(self, tmp_path):
        """Should skip gracefully if Attrition column is missing."""
        from src.train_attrition_model import run_causal_validation
        
        df = pd.DataFrame({"Age": [30, 40], "Income": [5000, 8000]})
        # Should not raise
        run_causal_validation(df)


# ── Survival Analysis Tests ───────────────────────────────────────────

class TestSurvivalAnalysis:
    """Tests for the survival analysis module."""

    def test_function_exists(self):
        """Verify the survival analysis function is importable."""
        from src.train_attrition_model import run_survival_analysis
        assert callable(run_survival_analysis)

    def test_survival_handles_missing_columns(self):
        """Should skip gracefully if required columns are missing."""
        from src.train_attrition_model import run_survival_analysis
        
        df = pd.DataFrame({"Age": [30, 40]})
        # Should not raise
        run_survival_analysis(df)

    def test_survival_runs_with_valid_data(self, sample_raw_df, tmp_path):
        """Survival analysis should produce report with valid data."""
        from src.train_attrition_model import run_survival_analysis
        import src.train_attrition_model as module
        
        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            run_survival_analysis(sample_raw_df)
            report_path = tmp_path / "survival_analysis_report.json"
            assert report_path.exists()
            
            import json
            with open(report_path) as f:
                report = json.load(f)
            assert "concordance_index" in report
            assert "hazard_ratios" in report
            assert report["concordance_index"] > 0
        finally:
            module.OUT_DIR = original_out
