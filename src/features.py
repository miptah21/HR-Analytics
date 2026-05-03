"""
Unified Feature Engineering Module — Single Source of Truth.

This module is imported by BOTH the training pipeline and the serving API
to guarantee feature parity and eliminate train-serving skew.

Population statistics (role averages, level medians) are serialized at
training time and loaded at serving time.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
STATS_PATH = BASE_DIR / "models" / "population_stats.json"

# ── Ordinal mappings (shared) ──────────────────────────────────────────
TRAVEL_MAP: dict[str, int] = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2,
}

# Columns that are constant or redundant and should always be dropped
DROP_COLS: list[str] = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
    "Avg_Role_Income",
    "Median_Level_Hike",
]


# ── Population Statistics ──────────────────────────────────────────────
def compute_population_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute and return population-level statistics needed for features.

    These stats are derived from the *training* population and must be
    persisted so the serving layer uses the exact same reference values.
    """
    role_income: dict[str, float] = (
        df.groupby("JobRole")["MonthlyIncome"]
        .mean()
        .round(2)
        .to_dict()
    )
    level_hike_median: dict[str, float] = (
        df.groupby("JobLevel")["PercentSalaryHike"]
        .median()
        .round(2)
        .to_dict()
    )
    # Store keys as strings for JSON compatibility
    level_hike_median_str = {str(k): v for k, v in level_hike_median.items()}

    return {
        "role_avg_income": role_income,
        "level_hike_median": level_hike_median_str,
    }


def save_population_stats(stats: dict[str, Any], path: Path | None = None) -> None:
    """Persist population statistics to JSON alongside the model."""
    target = path or STATS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(stats, f, indent=2)


def load_population_stats(path: Path | None = None) -> dict[str, Any]:
    """Load serialized population statistics for serving."""
    target = path or STATS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Population stats not found at {target}. "
            "Run the training pipeline first to generate them."
        )
    with open(target) as f:
        return json.load(f)


# ── Feature Engineering (Batch — Training) ─────────────────────────────
def engineer_features_batch(
    df: pd.DataFrame,
    save_stats: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full feature engineering for batch/training context.

    Computes population statistics from the dataset, creates all derived
    features, and optionally saves the stats for serving-time use.

    Returns:
        Tuple of (engineered DataFrame, population stats dict).
    """
    df = df.copy()

    # Encode target (handle both object and StringDtype from pandas 3.x)
    if not pd.api.types.is_numeric_dtype(df["Attrition"]):
        df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Compute population stats from THIS dataset
    stats = compute_population_stats(df)

    # Population-derived features
    df["Avg_Role_Income"] = df.groupby("JobRole")["MonthlyIncome"].transform("mean")
    df["Compa_Ratio"] = df["MonthlyIncome"] / df["Avg_Role_Income"]

    df["Median_Level_Hike"] = df.groupby("JobLevel")["PercentSalaryHike"].transform("median")
    df["Income_Growth_Gap"] = df["PercentSalaryHike"] - df["Median_Level_Hike"]

    # Deterministic features (no population dependency)
    df = _add_deterministic_features(df)

    # Drop constant/redundant
    df.drop(
        columns=[c for c in DROP_COLS if c in df.columns],
        inplace=True,
    )

    if save_stats:
        save_population_stats(stats)

    return df, stats


# ── Feature Engineering (Single Record — Serving) ──────────────────────
def engineer_features_single(
    record: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Feature engineering for a single employee record at serving time.

    Uses pre-computed population statistics to ensure parity with training.

    Args:
        record: Raw employee data (flat dict from Pydantic model).
        stats: Population statistics loaded from `population_stats.json`.

    Returns:
        Dict of engineered feature values (numeric only).
    """
    role_incomes: dict[str, float] = stats["role_avg_income"]
    level_hike_medians: dict[str, float] = stats["level_hike_median"]

    monthly_income: float = record["MonthlyIncome"]
    job_role: str = record["JobRole"]
    job_level: int = record["JobLevel"]
    overtime: str = record["OverTime"]
    distance: int = record["DistanceFromHome"]
    wlb: int = record["WorkLifeBalance"]
    years_company: int = record["YearsAtCompany"]
    years_role: int = record["YearsInCurrentRole"]
    years_promo: int = record["YearsSinceLastPromotion"]
    years_mgr: int = record["YearsWithCurrManager"]
    total_years: int = record["TotalWorkingYears"]
    pct_hike: float = record["PercentSalaryHike"]
    job_sat: int = record["JobSatisfaction"]
    env_sat: int = record["EnvironmentSatisfaction"]
    rel_sat: int = record["RelationshipSatisfaction"]
    job_inv: int = record["JobInvolvement"]
    travel: str = record["BusinessTravel"]

    # Compa Ratio — using training-time population average
    avg_income = role_incomes.get(job_role, monthly_income)
    compa_ratio = monthly_income / avg_income if avg_income else 1.0

    # Income Growth Gap — using training-time population median
    level_median = level_hike_medians.get(str(job_level), pct_hike)
    income_growth_gap = pct_hike - level_median

    # Deterministic features
    promotion_stagnation = years_promo / (years_company + 1)
    burnout_risk = (1 if overtime == "Yes" else 0) * distance / max(wlb, 1)
    manager_stability = years_mgr / (years_company + 1)
    engagement_index = (job_sat + env_sat + rel_sat + job_inv) / 4
    career_velocity = job_level / (total_years + 1)
    loyalty_index = years_company / (total_years + 1)
    travel_burden = TRAVEL_MAP.get(travel, 1)

    return {
        "Age": record["Age"],
        "DistanceFromHome": distance,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": record.get("NumCompaniesWorked", 1),
        "PercentSalaryHike": pct_hike,
        "TotalWorkingYears": total_years,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_mgr,
        "Compa_Ratio": compa_ratio,
        "Promotion_Stagnation": promotion_stagnation,
        "Burnout_Risk": burnout_risk,
        "Manager_Stability": manager_stability,
        "Engagement_Index": engagement_index,
        "Career_Velocity": career_velocity,
        "Income_Growth_Gap": income_growth_gap,
        "Loyalty_Index": loyalty_index,
        "Travel_Burden": travel_burden,
        "OverTime_Yes": 1 if overtime == "Yes" else 0,
    }


# ── Shared Deterministic Features ──────────────────────────────────────
def _add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that depend only on the individual record, not population."""
    df["Promotion_Stagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    df["Burnout_Risk"] = (
        (df["OverTime"] == "Yes").astype(int)
        * df["DistanceFromHome"]
        / df["WorkLifeBalance"]
    )

    df["Manager_Stability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

    df["Engagement_Index"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["JobInvolvement"]
    ) / 4

    df["Career_Velocity"] = df["JobLevel"] / (df["TotalWorkingYears"] + 1)
    df["Loyalty_Index"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["Travel_Burden"] = df["BusinessTravel"].map(TRAVEL_MAP)

    return df
