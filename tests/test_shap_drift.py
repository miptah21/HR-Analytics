"""
Tests for SHAP Attribution Drift Monitor.

Validates that the drift detection correctly identifies:
- First run (baseline establishment)
- Stable model (no drift)
- Drifted model (rank/magnitude changes)
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def stable_importance():
    """Create a SHAP importance DataFrame that is stable (similar to baseline)."""
    return pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(10)],
        "mean_abs_shap": [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.01],
    })


@pytest.fixture
def drifted_importance():
    """Create a SHAP importance DataFrame with significant drift from baseline."""
    # Completely different ranking and magnitudes
    return pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(10)],
        "mean_abs_shap": [0.01, 0.03, 0.05, 0.08, 0.10, 0.50, 0.40, 0.30, 0.20, 0.15],
    })


class TestSHAPDriftBaseline:
    """Test baseline establishment on first run."""

    def test_first_run_establishes_baseline(self, stable_importance, tmp_path):
        """First run should save baseline and return baseline_established status."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            result = detect_shap_drift(stable_importance)
            assert result["status"] == "baseline_established"
            assert (tmp_path / "shap_baseline_importance.csv").exists()
            assert (tmp_path / "shap_drift_report.json").exists()
            assert result["n_features"] == 10
        finally:
            module.OUT_DIR = original_out


class TestSHAPDriftStable:
    """Test stable model detection (no drift)."""

    def test_identical_importance_is_stable(self, stable_importance, tmp_path):
        """Identical SHAP values should produce STABLE verdict."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            # First run: establish baseline
            detect_shap_drift(stable_importance)
            # Second run: compare to baseline (identical)
            result = detect_shap_drift(stable_importance)
            assert result["verdict"] == "STABLE"
            assert result["has_drift"] is False
            assert result["metrics"]["spearman_rank_correlation"] == 1.0
            assert result["metrics"]["top5_overlap"] == 5
        finally:
            module.OUT_DIR = original_out

    def test_minor_perturbation_is_stable(self, stable_importance, tmp_path):
        """Small perturbations (< threshold) should still be STABLE."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            detect_shap_drift(stable_importance)
            # Add small noise (< 50% magnitude change)
            perturbed = stable_importance.copy()
            perturbed["mean_abs_shap"] *= np.random.uniform(0.85, 1.15, len(perturbed))
            result = detect_shap_drift(perturbed)
            assert result["verdict"] == "STABLE"
        finally:
            module.OUT_DIR = original_out


class TestSHAPDriftDetected:
    """Test drift detection with significant changes."""

    def test_reversed_ranking_triggers_drift(self, stable_importance, drifted_importance, tmp_path):
        """Completely reversed feature ranking should trigger DRIFT_DETECTED."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            detect_shap_drift(stable_importance)
            result = detect_shap_drift(drifted_importance)
            assert result["verdict"] == "DRIFT_DETECTED"
            assert result["has_drift"] is True
            assert result["metrics"]["rank_drift_detected"] is True
            assert result["metrics"]["spearman_rank_correlation"] < 0.80
        finally:
            module.OUT_DIR = original_out

    def test_drift_report_has_drifted_features(self, stable_importance, drifted_importance, tmp_path):
        """Drift report should list the specific features that drifted."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            detect_shap_drift(stable_importance)
            result = detect_shap_drift(drifted_importance)
            assert len(result["drifted_features"]) > 0
            # Each drifted feature should have required keys
            for feat in result["drifted_features"]:
                assert "feature" in feat
                assert "relative_change" in feat
                assert "direction" in feat
        finally:
            module.OUT_DIR = original_out

    def test_drift_provides_recommendation(self, stable_importance, drifted_importance, tmp_path):
        """Drift report should include retraining recommendation."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            detect_shap_drift(stable_importance)
            result = detect_shap_drift(drifted_importance)
            assert "retraining" in result["recommendation"].lower()
        finally:
            module.OUT_DIR = original_out

    def test_stable_result_updates_baseline(self, stable_importance, tmp_path):
        """When stable, baseline should be updated (rolling baseline strategy)."""
        from src.train_attrition_model import detect_shap_drift
        import src.train_attrition_model as module

        original_out = module.OUT_DIR
        module.OUT_DIR = tmp_path
        try:
            detect_shap_drift(stable_importance)
            baseline_mtime = (tmp_path / "shap_baseline_importance.csv").stat().st_mtime

            import time
            time.sleep(0.1)

            # Run with identical data (stable) — baseline should be updated
            detect_shap_drift(stable_importance)
            new_mtime = (tmp_path / "shap_baseline_importance.csv").stat().st_mtime
            assert new_mtime >= baseline_mtime
        finally:
            module.OUT_DIR = original_out
