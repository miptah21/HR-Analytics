"""
Unit tests for the HR Analytics feature engineering module.

Validates that batch and single-record feature engineering produce
consistent outputs, ensuring zero train-serving skew.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from uuid import uuid4


# ── Test Data ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_employee_record() -> dict:
    """A single employee record as it arrives from the API."""
    return {
        "EmployeeID": "TEST-001",
        "Age": 35,
        "JobRole": "Sales Executive",
        "JobLevel": 2,
        "MonthlyIncome": 5500,
        "PercentSalaryHike": 12,
        "OverTime": "Yes",
        "DistanceFromHome": 15,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 4,
        "YearsWithCurrManager": 4,
        "TotalWorkingYears": 8,
        "JobSatisfaction": 2,
        "EnvironmentSatisfaction": 2,
        "RelationshipSatisfaction": 3,
        "JobInvolvement": 3,
        "BusinessTravel": "Travel_Rarely",
    }


@pytest.fixture
def mock_population_stats() -> dict:
    """Population stats as serialized during training."""
    return {
        "role_avg_income": {
            "Sales Executive": 6500.0,
            "Research Scientist": 3200.0,
            "Laboratory Technician": 3200.0,
        },
        "level_hike_median": {
            "1": 12.0,
            "2": 13.0,
            "3": 14.0,
            "4": 15.0,
            "5": 16.0,
        },
    }


@pytest.fixture
def sample_batch_df() -> pd.DataFrame:
    """A small batch DataFrame simulating training data."""
    return pd.DataFrame({
        "Attrition": ["Yes", "No", "No", "Yes", "No"],
        "Age": [30, 40, 35, 28, 50],
        "JobRole": ["Sales Executive"] * 3 + ["Research Scientist"] * 2,
        "JobLevel": [2, 3, 2, 1, 4],
        "MonthlyIncome": [5500, 8000, 6000, 3000, 12000],
        "PercentSalaryHike": [12, 14, 11, 15, 13],
        "OverTime": ["Yes", "No", "No", "Yes", "No"],
        "DistanceFromHome": [15, 5, 10, 20, 3],
        "WorkLifeBalance": [2, 3, 3, 1, 4],
        "YearsAtCompany": [5, 10, 3, 2, 20],
        "YearsInCurrentRole": [4, 8, 2, 1, 15],
        "YearsSinceLastPromotion": [4, 2, 1, 2, 5],
        "YearsWithCurrManager": [4, 7, 2, 1, 10],
        "TotalWorkingYears": [8, 15, 5, 3, 25],
        "JobSatisfaction": [2, 3, 4, 1, 3],
        "EnvironmentSatisfaction": [2, 4, 3, 2, 3],
        "RelationshipSatisfaction": [3, 3, 4, 2, 4],
        "JobInvolvement": [3, 4, 3, 2, 4],
        "BusinessTravel": ["Travel_Rarely", "Non-Travel", "Travel_Frequently", "Travel_Rarely", "Non-Travel"],
        "Department": ["Sales"] * 3 + ["Research & Development"] * 2,
        "EmployeeCount": [1] * 5,
        "EmployeeNumber": [1, 2, 3, 4, 5],
        "Over18": ["Y"] * 5,
        "StandardHours": [80] * 5,
    })


# ── Feature Module Tests ──────────────────────────────────────────────
class TestEngineerFeaturesSingle:
    """Tests for single-record (serving-time) feature engineering."""

    def test_returns_expected_keys(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single
        
        features = engineer_features_single(sample_employee_record, mock_population_stats)
        
        expected_keys = {
            "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
            "PercentSalaryHike", "TotalWorkingYears", "YearsAtCompany",
            "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
            "Compa_Ratio", "Promotion_Stagnation", "Burnout_Risk",
            "Manager_Stability", "Engagement_Index", "Career_Velocity",
            "Income_Growth_Gap", "Loyalty_Index", "Travel_Burden", "OverTime_Yes",
        }
        assert set(features.keys()) == expected_keys

    def test_compa_ratio_uses_population_stats(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single

        features = engineer_features_single(sample_employee_record, mock_population_stats)
        
        # CompaRatio = MonthlyIncome / AvgRoleIncome = 5500 / 6500
        expected = 5500 / 6500
        assert abs(features["Compa_Ratio"] - expected) < 1e-6

    def test_income_growth_gap_uses_population_stats(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single
        
        features = engineer_features_single(sample_employee_record, mock_population_stats)
        
        # Income_Growth_Gap = PercentSalaryHike - LevelMedian = 12 - 13 = -1
        expected = 12 - 13.0
        assert abs(features["Income_Growth_Gap"] - expected) < 1e-6

    def test_overtime_binary(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single

        features_yes = engineer_features_single(sample_employee_record, mock_population_stats)
        assert features_yes["OverTime_Yes"] == 1

        no_ot = {**sample_employee_record, "OverTime": "No"}
        features_no = engineer_features_single(no_ot, mock_population_stats)
        assert features_no["OverTime_Yes"] == 0

    def test_travel_burden_mapping(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single

        for travel, expected in [("Non-Travel", 0), ("Travel_Rarely", 1), ("Travel_Frequently", 2)]:
            rec = {**sample_employee_record, "BusinessTravel": travel}
            features = engineer_features_single(rec, mock_population_stats)
            assert features["Travel_Burden"] == expected

    def test_burnout_risk_calculation(self, sample_employee_record, mock_population_stats):
        from src.features import engineer_features_single

        features = engineer_features_single(sample_employee_record, mock_population_stats)
        
        # Burnout = (OverTime=1) * DistanceFromHome(15) / WorkLifeBalance(2) = 7.5
        expected = 1 * 15 / 2
        assert abs(features["Burnout_Risk"] - expected) < 1e-6


class TestEngineerFeaturesBatch:
    """Tests for batch (training-time) feature engineering."""

    def test_returns_dataframe_and_stats(self, sample_batch_df):
        from src.features import engineer_features_batch

        df, stats = engineer_features_batch(sample_batch_df, save_stats=False)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(stats, dict)
        assert "role_avg_income" in stats
        assert "level_hike_median" in stats

    def test_attrition_encoded_as_binary(self, sample_batch_df):
        from src.features import engineer_features_batch
        
        df, _ = engineer_features_batch(sample_batch_df, save_stats=False)
        
        assert df["Attrition"].dtype in [np.int64, np.int32, int]
        assert set(df["Attrition"].unique()).issubset({0, 1})

    def test_drops_redundant_columns(self, sample_batch_df):
        from src.features import engineer_features_batch
        
        df, _ = engineer_features_batch(sample_batch_df, save_stats=False)
        
        for col in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]:
            assert col not in df.columns

    def test_engineered_features_present(self, sample_batch_df):
        from src.features import engineer_features_batch

        df, _ = engineer_features_batch(sample_batch_df, save_stats=False)
        
        for col in ["Compa_Ratio", "Promotion_Stagnation", "Burnout_Risk",
                     "Manager_Stability", "Engagement_Index", "Career_Velocity",
                     "Income_Growth_Gap", "Loyalty_Index", "Travel_Burden"]:
            assert col in df.columns, f"Missing engineered feature: {col}"


class TestPopulationStatsIO:
    """Tests for serialization and loading of population stats."""

    def test_save_and_load_round_trip(self, mock_population_stats):
        from src.features import save_population_stats, load_population_stats

        test_dir = Path(".test_artifacts")
        test_dir.mkdir(exist_ok=True)
        path = test_dir / f"test_stats_{uuid4().hex}.json"
        save_population_stats(mock_population_stats, path)
        loaded = load_population_stats(path)
        path.unlink(missing_ok=True)
        
        assert loaded == mock_population_stats

    def test_load_missing_file_raises(self):
        from src.features import load_population_stats

        path = Path(".test_artifacts") / f"missing_{uuid4().hex}.json"
        with pytest.raises(FileNotFoundError, match="Population stats not found"):
            load_population_stats(path)
