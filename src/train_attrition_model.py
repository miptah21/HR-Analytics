"""
AI-Powered Employee Attrition Intelligence System
Production Pipeline: Feature Engineering → XGBoost → SHAP → Fairness → Cost Model → Risk Scoring

This pipeline uses the shared feature engineering module (src/features.py)
to ensure parity with the serving API.
"""
import os
from dotenv import load_dotenv

# Ensure environment variables (like DATABASE_URL) are loaded from .env
load_dotenv()
import json
import hashlib
import warnings
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.features import engineer_features_batch
from src.causal_uplift import TLearnerUplift

# Suppress only non-critical warnings, keep DeprecationWarnings visible
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


# ── 1. PREPROCESSING ──────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features/target and encode categoricals."""
    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X, y


# ── 2. OPTUNA HYPERPARAMETER TUNING ───────────────────────────────────
def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> dict[str, Any]:
    """Bayesian hyperparameter search with Optuna using stratified 5-fold CV.
    
    GAP-17 improvements:
    - Added L1/L2 regularization (reg_alpha, reg_lambda) to prevent overfitting
    - Uses threshold tuning per fold to maximize F2 directly
    - Increased CV folds from 3→5 for more reliable estimation on small data
    """
    scale_w = float((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_w,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        threshold = trial.suggest_float("threshold", 0.2, 0.5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores: list[float] = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            probas = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            preds = (probas >= threshold).astype(int)
            scores.append(fbeta_score(y_train.iloc[val_idx], preds, beta=2))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  Best F2-Score (CV): {study.best_value:.4f}")
    best = study.best_params
    # Extract threshold separately from model params
    best["_optimal_threshold"] = best.pop("threshold", 0.5)
    return best


# ── 3. MODEL TRAINING ─────────────────────────────────────────────────
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
) -> xgb.XGBClassifier:
    """Train final XGBoost with optimized hyperparameters."""
    scale_w = float((y_train == 0).sum() / (y_train == 1).sum())
    # Filter out non-XGBoost keys (e.g., _optimal_threshold from Optuna)
    xgb_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
    params = {
        **xgb_params,
        "scale_pos_weight": scale_w,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ── 3b. PROBABILITY CALIBRATION ───────────────────────────────────────
def calibrate_model(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """Calibrate a fitted XGBoost model on held-out calibration data.

    This addresses the known issue of gradient boosting models producing
    overconfident probability estimates, which is critical for:
    - Accurate financial loss calculations (probability * replacement cost)
    - Reliable risk tier assignments (thresholds at 0.30 and 0.60)
    - EU AI Act transparency (Art. 13) — stakeholders deserve accurate confidence scores

    Sigmoid calibration is intentionally used here because the calibration
    split is small; isotonic calibration can overfit when calibration samples
    are limited.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="sigmoid",
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            model,
            method="sigmoid",
            cv="prefit",
        )
    calibrated.fit(X_train, y_train)
    return calibrated


# ── 4. EVALUATION ─────────────────────────────────────────────────────
def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Full evaluation with classification report, F2, AUC, calibration, and confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print(f"  F2-Score: {f2:.4f}  |  ROC-AUC: {auc:.4f}  |  Brier Score: {brier:.4f}")

    # Calibration curve
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fraction_pos, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_predicted, fraction_pos, "s-", label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Stay", "Leave"],
        yticklabels=["Stay", "Leave"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {"f2": f2, "auc": auc, "brier": brier, "report": report}


# ── 4b. SUBPOPULATION PERFORMANCE (EU AI Act Art. 15) ────────────────
def evaluate_subpopulations(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute performance metrics across demographic subpopulations.

    Required by EU AI Act Article 15 to ensure accuracy does not
    vary significantly across protected groups. Logs per-group
    AUC and F2 scores for Gender and Age bins when available.
    """
    subpop_results: dict[str, Any] = {}
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Gender subgroup (one-hot encoded)
    if "Gender_Male" in X_test.columns:
        for gender_val, label in [(1, "Male"), (0, "Female")]:
            mask = X_test["Gender_Male"] == gender_val
            if mask.sum() >= 10 and y_test[mask].nunique() == 2:
                group_auc = roc_auc_score(y_test[mask], y_proba[mask])
                group_f2 = fbeta_score(y_test[mask], y_pred[mask], beta=2)
                subpop_results[f"Gender_{label}"] = {
                    "count": int(mask.sum()),
                    "auc": round(float(group_auc), 4),
                    "f2": round(float(group_f2), 4),
                }

    # Age subgroups
    if "Age" in X_test.columns:
        age_bins = pd.cut(
            X_test["Age"], bins=[17, 30, 40, 50, 70],
            labels=["18-30", "31-40", "41-50", "51+"],
        )
        for label in ["18-30", "31-40", "41-50", "51+"]:
            mask = age_bins == label
            if mask.sum() >= 10 and y_test[mask].nunique() == 2:
                group_auc = roc_auc_score(y_test[mask], y_proba[mask])
                group_f2 = fbeta_score(y_test[mask], y_pred[mask], beta=2)
                subpop_results[f"Age_{label}"] = {
                    "count": int(mask.sum()),
                    "auc": round(float(group_auc), 4),
                    "f2": round(float(group_f2), 4),
                }

    if subpop_results:
        print("\n  -- Subpopulation Performance (EU AI Act Art. 15) --")
        print("  " + "-" * 55)
        for group, group_metrics in subpop_results.items():
            print(f"  [{group}] n={group_metrics['count']}  AUC={group_metrics['auc']:.4f}  F2={group_metrics['f2']:.4f}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "subpopulation_metrics.json", "w") as f:
            json.dump(subpop_results, f, indent=2)
        print(f"\n  Subpopulation metrics saved to {OUT_DIR / 'subpopulation_metrics.json'}")

    return subpop_results


# ── 5. COMPREHENSIVE FAIRNESS AUDIT ──────────────────────────────────
def run_fairness_audit(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Expanded algorithmic fairness audit across multiple dimensions.

    Checks Demographic Parity Difference and Equalized Odds Difference
    across Gender, Age (binned), and Marital Status if columns are present.
    """
    results: dict[str, Any] = {}

    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )
    except ImportError:
        print("\n  [INFO] fairlearn not installed. Skipping fairness audit.")
        return results

    y_pred = model.predict(X_test)

    # Define protected attribute columns to check
    protected_attrs: dict[str, str | None] = {
        "Gender": None,
        "Age_Group": None,
        "MaritalStatus": None,
    }

    # Find Gender column (one-hot encoded)
    if "Gender_Male" in X_test.columns:
        protected_attrs["Gender"] = "Gender_Male"

    # Bin Age into groups for intersectional analysis
    if "Age" in X_test.columns:
        age_bins = pd.cut(
            X_test["Age"], bins=[17, 30, 40, 50, 70], labels=["18-30", "31-40", "41-50", "51+"]
        )
        protected_attrs["Age_Group"] = "Age_binned"
        X_test = X_test.copy()
        X_test["Age_binned"] = age_bins

    # Find MaritalStatus columns
    for col in X_test.columns:
        if col.startswith("MaritalStatus_"):
            protected_attrs["MaritalStatus"] = col
            break

    print("\n  -- Algorithmic Fairness Audit --")
    print("  " + "-" * 55)

    for attr_name, col_name in protected_attrs.items():
        if col_name is None or col_name not in X_test.columns:
            continue

        sensitive = X_test[col_name]
        dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
        eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)

        status_dpd = "[WARNING]" if abs(dpd) > 0.1 else "[PASS]"
        status_eod = "[WARNING]" if abs(eod) > 0.1 else "[PASS]"

        print(f"\n  [{attr_name}] (column: {col_name})")
        print(f"    Demographic Parity Diff: {dpd:.4f}  {status_dpd}")
        print(f"    Equalized Odds Diff:     {eod:.4f}  {status_eod}")

        results[attr_name] = {
            "demographic_parity_diff": round(float(dpd), 4),
            "equalized_odds_diff": round(float(eod), 4),
            "dpd_pass": bool(abs(dpd) <= 0.1),
            "eod_pass": bool(abs(eod) <= 0.1),
        }

    if not results:
        print("  No protected attribute columns found in test set.")
    else:
        # Save audit results
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "fairness_audit.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Fairness audit saved to {OUT_DIR / 'fairness_audit.json'}")

    return results


# ── 6. SHAP EXPLAINABILITY ────────────────────────────────────────────
def explain_model(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[shap.TreeExplainer, np.ndarray]:
    """Generate global and local SHAP explanations."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("Global SHAP: Systemic Attrition Drivers")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150)
    plt.close()

    # Bar importance
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_bar.png", dpi=150)
    plt.close()

    # Save global SHAP importances as CSV for dashboard
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(OUT_DIR / "shap_global_importance.csv", index=False)

    return explainer, shap_values


# ── 6b. SHAP ATTRIBUTION DRIFT MONITOR ────────────────────────────────
def detect_shap_drift(
    current_shap_importance: pd.DataFrame,
    rank_correlation_threshold: float = 0.80,
    magnitude_change_threshold: float = 0.50,
) -> dict:
    """Detect SHAP attribution drift by comparing current vs baseline importance.

    SHAP drift is a **leading indicator** of concept drift — when the model
    starts relying on different features (or the same features with very
    different weights), it signals that the relationship between features
    and the target has shifted. This is important because:

    1. Data drift (Evidently) catches *input distribution* changes
    2. SHAP drift catches *model reasoning* changes
    3. Together they provide complete observability

    Checks:
    - Rank correlation (Spearman) between feature importance orderings
    - Top-5 feature set stability (are the same features still dominant?)
    - Per-feature magnitude change (normalized SHAP value shifts)

    Args:
        current_shap_importance: DataFrame with 'feature' and 'mean_abs_shap' columns
        rank_correlation_threshold: Minimum Spearman correlation to pass (default 0.80)
        magnitude_change_threshold: Max relative magnitude change per feature (default 0.50 = 50%)

    Returns:
        Dict with drift status, metrics, and individual feature drift details.
    """
    import json
    from scipy.stats import spearmanr

    baseline_path = OUT_DIR / "shap_baseline_importance.csv"
    report_path = OUT_DIR / "shap_drift_report.json"

    # If no baseline exists, establish one
    if not baseline_path.exists():
        current_shap_importance.to_csv(baseline_path, index=False)
        report = {
            "status": "baseline_established",
            "message": "No prior baseline found. Current SHAP importance saved as baseline.",
            "n_features": len(current_shap_importance),
            "top_5_features": current_shap_importance["feature"].head(5).tolist(),
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("  [INFO] SHAP baseline established (first run).")
        return report

    # Load baseline
    baseline = pd.read_csv(baseline_path)

    # Merge on feature name
    merged = baseline.merge(
        current_shap_importance,
        on="feature",
        how="outer",
        suffixes=("_baseline", "_current"),
    ).fillna(0)

    # 1. Rank Correlation (Spearman)
    spearman_corr, spearman_p = spearmanr(
        merged["mean_abs_shap_baseline"].rank(),
        merged["mean_abs_shap_current"].rank(),
    )
    rank_drift = spearman_corr < rank_correlation_threshold

    # 2. Top-5 Stability
    baseline_top5 = set(baseline.nlargest(5, "mean_abs_shap")["feature"].tolist())
    current_top5 = set(current_shap_importance.nlargest(5, "mean_abs_shap")["feature"].tolist())
    top5_overlap = len(baseline_top5 & current_top5)
    top5_stable = top5_overlap >= 3  # At least 3/5 must be the same

    # 3. Per-feature Magnitude Change
    feature_drifts = []
    max_baseline = merged["mean_abs_shap_baseline"].max()
    if max_baseline > 0:
        for _, row in merged.iterrows():
            base_val = row["mean_abs_shap_baseline"]
            curr_val = row["mean_abs_shap_current"]
            if base_val > 0:
                relative_change = abs(curr_val - base_val) / base_val
            else:
                relative_change = 1.0 if curr_val > 0 else 0.0

            if relative_change > magnitude_change_threshold:
                feature_drifts.append({
                    "feature": row["feature"],
                    "baseline_importance": round(float(base_val), 6),
                    "current_importance": round(float(curr_val), 6),
                    "relative_change": round(float(relative_change), 4),
                    "direction": "increased" if curr_val > base_val else "decreased",
                })

    # Overall verdict
    has_drift = rank_drift or not top5_stable or len(feature_drifts) > 3
    verdict = "DRIFT_DETECTED" if has_drift else "STABLE"

    report = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "verdict": verdict,
        "has_drift": bool(has_drift),
        "metrics": {
            "spearman_rank_correlation": round(float(spearman_corr), 4),
            "spearman_p_value": round(float(spearman_p), 6),
            "rank_drift_detected": bool(rank_drift),
            "threshold": float(rank_correlation_threshold),
            "top5_overlap": int(top5_overlap),
            "top5_baseline": sorted(baseline_top5),
            "top5_current": sorted(current_top5),
            "top5_stable": bool(top5_stable),
            "features_with_magnitude_drift": len(feature_drifts),
        },
        "drifted_features": sorted(feature_drifts, key=lambda x: x["relative_change"], reverse=True),
        "recommendation": (
            "Model retraining recommended — SHAP attributions have significantly shifted."
            if has_drift else
            "Model explanations remain consistent with baseline."
        ),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    status_tag = "[WARN]" if has_drift else "[PASS]"
    print(f"  {status_tag} SHAP Drift: rho={spearman_corr:.3f}, Top-5 overlap={top5_overlap}/5, "
          f"Drifted features={len(feature_drifts)}")

    # Update baseline if stable (rolling baseline strategy)
    if not has_drift:
        current_shap_importance.to_csv(baseline_path, index=False)

    return report


# ── 7. RISK SCORING & COST MODEL ──────────────────────────────────────
def build_risk_framework(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_original: pd.DataFrame,
    X_columns: pd.Index,
) -> pd.DataFrame:
    """Assign risk tiers and calculate financial impact."""
    y_proba = model.predict_proba(X_test)[:, 1]

    risk_df = pd.DataFrame(
        {
            "Predicted_Probability": y_proba,
            "Actual": y_test.values,
        },
        index=X_test.index,
    )

    # Risk tiers
    risk_df["Risk_Tier"] = pd.cut(
        risk_df["Predicted_Probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    # Attach income for cost model
    if "MonthlyIncome" in df_original.columns:
        risk_df["MonthlyIncome"] = df_original.loc[X_test.index, "MonthlyIncome"].values
        risk_df["Annual_Salary"] = risk_df["MonthlyIncome"] * 12
        risk_df["Replacement_Cost"] = risk_df["Annual_Salary"] * 1.5
        risk_df["Expected_Loss"] = (
            risk_df["Predicted_Probability"] * risk_df["Replacement_Cost"]
        )

    # Attach categorical features for Dashboard
    for col in ["EmployeeNumber", "Department", "JobRole"]:
        if col in df_original.columns:
            risk_df[col] = df_original.loc[X_test.index, col].values

    # Save to CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(OUT_DIR / "risk_scores.csv", index=False)

    # Save to Database (for API Dashboard)
    try:
        from src.database import engine as db_engine
        # SQLite doesn't support schemas in the same way, so we just use the table name.
        # But if it's Postgres, we use the analytics schema.
        is_postgres = db_engine.url.drivername.startswith("postgres")
        
        if is_postgres:
            from sqlalchemy import text
            with db_engine.begin() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics;"))
            risk_df.to_sql("predictions", db_engine, schema="analytics", if_exists="replace", index=False)
        else:
            # SQLite fallback
            risk_df.to_sql("predictions", db_engine, if_exists="replace", index=False)
            
        print("  Saved risk predictions to database (predictions table)")
    except Exception as e:
        print(f"  Warning: Could not save to database. Dashboard data unavailable. Error: {e}")

    # Summary stats
    summary = (
        risk_df.groupby("Risk_Tier", observed=True)
        .agg(
            Count=("Predicted_Probability", "count"),
            Avg_Risk=("Predicted_Probability", "mean"),
            Total_Expected_Loss=("Expected_Loss", "sum"),
        )
        .round(2)
    )
    print("\n-- Risk Tier Summary --")
    print(summary)
    print(f"\n  Total Value at Risk: ${risk_df['Expected_Loss'].sum():,.0f}")

    summary.to_csv(OUT_DIR / "risk_summary.csv")
    return risk_df


# ── 8. EDA CHARTS ─────────────────────────────────────────────────────
def generate_eda(df: pd.DataFrame) -> None:
    """Generate key exploratory charts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Attrition rate by OverTime
    ot = df.groupby("OverTime")["Attrition"].mean() * 100
    axes[0].bar(ot.index.astype(str), ot.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Attrition Rate by OverTime")
    axes[0].set_ylabel("Attrition %")

    # Attrition by Department
    dept = df.groupby("Department")["Attrition"].mean() * 100
    axes[1].barh(dept.index, dept.values, color="#3498db")
    axes[1].set_title("Attrition Rate by Department")
    axes[1].set_xlabel("Attrition %")

    # Income distribution by attrition
    for label, group in df.groupby("Attrition"):
        tag = "Left" if label == 1 else "Stayed"
        axes[2].hist(group["MonthlyIncome"], bins=30, alpha=0.6, label=tag)
    axes[2].set_title("Income Distribution by Attrition")
    axes[2].legend()
    axes[2].set_xlabel("Monthly Income")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)
    print("  EDA charts saved.")


# ── 9. MODEL OBSERVABILITY (Evidently AI) ─────────────────────────────
def generate_drift_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Generate HTML report for Data Drift and Data Quality using Evidently."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_train, current_data=X_test)
        report.save_html(str(OUT_DIR / "evidently_drift_report.html"))
        report.save_json(str(OUT_DIR / "evidently_drift_report.json"))
        print("  [PASS] Evidently AI Data Drift & Quality report saved.")
    except ImportError:
        print("  [INFO] evidently not installed. Skipping drift report.")


# ── 7. ADVERSARIAL ROBUSTNESS (EU AI Act Art. 15) ─────────────────────
def run_adversarial_robustness_test(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Test model resilience against adversarial perturbations.

    EU AI Act Article 15 requires high-risk AI systems to be resilient
    against errors, faults, and adversarial attacks. This tests:
    - Feature perturbation robustness (noise injection)
    - Boundary stability (small shifts shouldn't flip predictions)
    - Overall model consistency under adversarial conditions

    Results are persisted to outputs/adversarial_robustness_report.json
    for inclusion in the AI System Card (Annex IV documentation).
    """
    try:
        from scipy import stats as scipy_stats
        import json

        np.random.seed(42)
        X_test_arr = X_test.values.astype(np.float64)
        base_proba = model.predict_proba(X_test_arr)[:, 1]
        base_preds = (base_proba >= 0.5).astype(int)

        results = {"test_date": pd.Timestamp.now().isoformat(), "tests": []}

        # ── Test 1: Gaussian Noise Injection ──────────────────────────
        noise_levels = [0.01, 0.05, 0.10, 0.20]
        noise_results = []
        for eps in noise_levels:
            noise = np.random.normal(0, eps, X_test_arr.shape)
            X_perturbed = X_test_arr + noise * np.abs(X_test_arr).mean(axis=0)
            perturbed_proba = model.predict_proba(X_perturbed)[:, 1]
            perturbed_preds = (perturbed_proba >= 0.5).astype(int)
            flip_rate = (base_preds != perturbed_preds).mean()
            mae_shift = np.abs(base_proba - perturbed_proba).mean()
            noise_results.append({
                "epsilon": eps,
                "prediction_flip_rate": round(float(flip_rate), 4),
                "mean_probability_shift": round(float(mae_shift), 4),
            })
            print(f"    eps={eps:.2f}: flip_rate={flip_rate:.2%}, prob_shift={mae_shift:.4f}")

        results["tests"].append({
            "name": "Gaussian Noise Injection",
            "description": "Measures prediction stability under random Gaussian noise at different magnitudes.",
            "verdict": "PASS" if noise_results[-1]["prediction_flip_rate"] < 0.30 else "WARN",
            "details": noise_results,
        })

        # ── Test 2: Feature Boundary Attack ───────────────────────────
        n_boundary_attacks = min(50, len(X_test_arr))
        boundary_flips = 0
        boundary_attempts = 0
        for i in range(n_boundary_attacks):
            sample = X_test_arr[i:i+1].copy()
            orig_pred = model.predict_proba(sample)[:, 1][0]
            for feat_idx in range(X_test_arr.shape[1]):
                perturbed = sample.copy()
                feat_std = np.std(X_test_arr[:, feat_idx])
                if feat_std == 0:
                    continue
                # Try small perturbation (1% of feature std)
                perturbed[0, feat_idx] += 0.01 * feat_std
                new_pred = model.predict_proba(perturbed)[:, 1][0]
                boundary_attempts += 1
                if (orig_pred >= 0.5) != (new_pred >= 0.5):
                    boundary_flips += 1

        boundary_flip_rate = boundary_flips / max(boundary_attempts, 1)
        results["tests"].append({
            "name": "Feature Boundary Stability",
            "description": "Tests if a 1% perturbation of individual features flips predictions near decision boundaries.",
            "verdict": "PASS" if boundary_flip_rate < 0.05 else "WARN",
            "attempts": boundary_attempts,
            "flips": boundary_flips,
            "flip_rate": round(float(boundary_flip_rate), 4),
        })
        print(f"    Boundary: {boundary_flips}/{boundary_attempts} flips ({boundary_flip_rate:.2%})")

        # ── Test 3: Prediction Consistency (identical inputs) ─────────
        consistency_checks = 100
        sample_indices = np.random.choice(len(X_test_arr), min(consistency_checks, len(X_test_arr)), replace=False)
        inconsistencies = 0
        for idx in sample_indices:
            p1 = model.predict_proba(X_test_arr[idx:idx+1])[:, 1][0]
            p2 = model.predict_proba(X_test_arr[idx:idx+1])[:, 1][0]
            if abs(p1 - p2) > 1e-10:
                inconsistencies += 1

        results["tests"].append({
            "name": "Deterministic Consistency",
            "description": "Verifies identical inputs produce identical outputs (non-stochastic inference).",
            "verdict": "PASS" if inconsistencies == 0 else "FAIL",
            "checks": consistency_checks,
            "inconsistencies": inconsistencies,
        })
        print(f"    Determinism: {inconsistencies}/{consistency_checks} inconsistencies")

        # ── Overall Verdict ───────────────────────────────────────────
        overall = "PASS" if all(t["verdict"] == "PASS" for t in results["tests"]) else "CONDITIONAL"
        results["overall_verdict"] = overall
        results["eu_ai_act_article"] = "Article 15 — Accuracy, Robustness and Cybersecurity"
        results["note"] = (
            "This test suite covers noise robustness and consistency. "
            "Full Article 15 compliance also requires data poisoning and "
            "model evasion tests using IBM ART (adversarial-robustness-toolbox)."
        )

        report_path = OUT_DIR / "adversarial_robustness_report.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [{'PASS' if overall == 'PASS' else 'WARN'}] Report saved -> {report_path}")

    except Exception as e:
        print(f"  [WARN] Adversarial robustness test error: {e}")
        print("  Continuing pipeline — robustness testing is advisory at PoC stage.")


# ── 8. CAUSAL INFERENCE VALIDATION (DoWhy) ────────────────────────────
def run_causal_validation(raw_df: pd.DataFrame) -> None:
    """Validate that SHAP-identified drivers have causal support.

    Addresses the critical gap: SHAP measures feature attribution
    (correlation-based), NOT causal effect. Without causal validation,
    interventions based on SHAP can be ineffective or harmful.

    Uses DoWhy to estimate Average Treatment Effect (ATE) for key
    binary/ordinal treatments and validate via refutation tests.

    Results are persisted to outputs/causal_validation_report.json.
    """
    try:
        import dowhy
        from dowhy import CausalModel
        import json
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        df = raw_df.copy()
        if "Attrition" not in df.columns:
            print("  [SKIP] No 'Attrition' column for causal analysis.")
            return

        # Encode target (handle both pandas 2 'object' and pandas 3 'str' dtypes)
        if pd.api.types.is_string_dtype(df["Attrition"]) or df["Attrition"].dtype == object:
            df["Attrition_Binary"] = (df["Attrition"].astype(str) == "Yes").astype(int)
        else:
            df["Attrition_Binary"] = df["Attrition"].astype(int)

        # Encode OverTime
        if "OverTime" in df.columns and (pd.api.types.is_string_dtype(df["OverTime"]) or df["OverTime"].dtype == object):
            df["OverTime_Num"] = (df["OverTime"].astype(str) == "Yes").astype(int)

        # Define treatment-outcome pairs to test (based on top SHAP drivers)
        treatments = []
        if "OverTime_Num" in df.columns:
            treatments.append({
                "name": "OverTime",
                "treatment": "OverTime_Num",
                "common_causes": ["Age", "MonthlyIncome", "JobLevel",
                                  "YearsAtCompany", "TotalWorkingYears",
                                  "DistanceFromHome", "JobSatisfaction"],
            })

        causal_results = {"analysis_date": pd.Timestamp.now().isoformat(), "treatments": []}

        for t in treatments:
            try:
                # Filter to available columns only
                available_causes = [c for c in t["common_causes"] if c in df.columns]
                graph_nodes = [t["treatment"], "Attrition_Binary"] + available_causes

                # Build causal graph (simple DAG: confounders → treatment → outcome)
                gml_edges = ""
                for cause in available_causes:
                    gml_edges += f'edge [source "{cause}" target "{t["treatment"]}"]\n'
                    gml_edges += f'edge [source "{cause}" target "Attrition_Binary"]\n'
                gml_edges += f'edge [source "{t["treatment"]}" target "Attrition_Binary"]\n'

                gml_graph = f"""graph [directed 1
{chr(10).join([f'node [id "{n}" label "{n}"]' for n in graph_nodes])}
{gml_edges}]"""

                model = CausalModel(
                    data=df[graph_nodes].dropna(),
                    treatment=t["treatment"],
                    outcome="Attrition_Binary",
                    graph=gml_graph,
                )

                identified = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified,
                    method_name="backdoor.linear_regression",
                )
                ate = float(estimate.value)

                # Refutation: Random common cause (placebo test)
                refutation = model.refute_estimate(
                    identified, estimate,
                    method_name="random_common_cause",
                    num_simulations=20,
                )
                refutation_pval = float(refutation.refutation_result.get("p_value", 0.0)) if hasattr(refutation, "refutation_result") and isinstance(refutation.refutation_result, dict) else None

                result = {
                    "treatment": t["name"],
                    "average_treatment_effect": round(ate, 4),
                    "interpretation": (
                        f"{'Increasing' if ate > 0 else 'Decreasing'} {t['name']} is estimated to "
                        f"{'increase' if ate > 0 else 'decrease'} attrition probability by "
                        f"{abs(ate)*100:.1f} percentage points"
                    ),
                    "causal_support": "SUPPORTED" if abs(ate) > 0.01 else "WEAK",
                    "refutation_method": "random_common_cause",
                    "refutation_note": str(refutation),
                }
                causal_results["treatments"].append(result)
                print(f"    {t['name']}: ATE = {ate:+.4f} → {'SUPPORTED' if abs(ate) > 0.01 else 'WEAK'}")

            except Exception as inner_e:
                causal_results["treatments"].append({
                    "treatment": t["name"],
                    "error": str(inner_e),
                    "causal_support": "INCONCLUSIVE",
                })
                print(f"    {t['name']}: Error — {inner_e}")

        causal_results["methodology_note"] = (
            "Causal effects estimated via backdoor adjustment with linear regression. "
            "Results assume the specified DAG is correct (no unmeasured confounders). "
            "On synthetic data, these estimates validate methodology only, not real-world causal claims."
        )

        report_path = OUT_DIR / "causal_validation_report.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(causal_results, f, indent=2)
        print(f"  Report saved → {report_path}")

    except ImportError:
        print("  [INFO] dowhy not installed. Skipping causal validation.")
        print("  Install with: pip install dowhy")
    except Exception as e:
        print(f"  [WARN] Causal validation error: {e}")


# ── 9. SURVIVAL ANALYSIS ─────────────────────────────────────────────
def run_survival_analysis(raw_df: pd.DataFrame) -> None:
    """Run time-to-event survival analysis alongside binary classification.

    Binary classification answers "will they leave?" but NOT "when?"
    Survival analysis (Cox PH model) provides:
    - Hazard ratios for each feature (time-aware effect sizes)
    - Median survival times per risk cohort
    - Time-dependent risk curves

    This is architecturally important because:
    - HR interventions have different urgency levels
    - Budget allocation depends on time horizon
    - EU AI Act favors multi-method validation

    Uses YearsAtCompany as the duration proxy and Attrition as event indicator.
    Results are persisted to outputs/survival_analysis_report.json.
    """
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        import json

        df = raw_df.copy()

        # Setup survival data
        if "Attrition" not in df.columns or "YearsAtCompany" not in df.columns:
            print("  [SKIP] Missing required columns for survival analysis.")
            return

        # Encode event indicator (handle both pandas 2 'object' and pandas 3 'str' dtypes)
        if pd.api.types.is_string_dtype(df["Attrition"]) or df["Attrition"].dtype == object:
            df["Event"] = (df["Attrition"].astype(str) == "Yes").astype(int)
        else:
            df["Event"] = df["Attrition"].astype(int)

        # Duration must be positive
        df["Duration"] = pd.to_numeric(df["YearsAtCompany"], errors="coerce").clip(lower=0.5)

        # Need minimum events for meaningful analysis
        if df["Event"].sum() < 5:
            print("  [SKIP] Fewer than 5 events — insufficient for survival analysis.")
            return

        # Select features for Cox PH model
        cox_features = [
            "Age", "MonthlyIncome", "DistanceFromHome",
            "JobSatisfaction", "EnvironmentSatisfaction",
            "WorkLifeBalance", "YearsSinceLastPromotion",
            "TotalWorkingYears", "JobLevel",
        ]

        # Encode OverTime if present
        if "OverTime" in df.columns:
            if pd.api.types.is_string_dtype(df["OverTime"]) or df["OverTime"].dtype == object:
                df["OverTime_Flag"] = (df["OverTime"].astype(str) == "Yes").astype(int)
            else:
                df["OverTime_Flag"] = df["OverTime"]
            cox_features.append("OverTime_Flag")

        available = [f for f in cox_features if f in df.columns]
        surv_df = df[available + ["Duration", "Event"]].dropna()

        # Ensure all columns are numeric (handles pandas 3 str dtype)
        for col in surv_df.columns:
            surv_df[col] = pd.to_numeric(surv_df[col], errors="coerce")
        surv_df = surv_df.dropna()

        # Fit Cox Proportional Hazards model
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(surv_df, duration_col="Duration", event_col="Event")

        # Extract hazard ratios
        summary = cph.summary
        hazard_ratios = {}
        for feature in summary.index:
            hr = float(summary.loc[feature, "exp(coef)"])
            p_val = float(summary.loc[feature, "p"])
            ci_lower = float(summary.loc[feature, "exp(coef) lower 95%"])
            ci_upper = float(summary.loc[feature, "exp(coef) upper 95%"])
            hazard_ratios[feature] = {
                "hazard_ratio": round(hr, 4),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
                "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
                "interpretation": (
                    f"{'Increases' if hr > 1 else 'Decreases'} attrition hazard by "
                    f"{abs(hr - 1) * 100:.1f}% per unit increase"
                    f" (p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'})"
                ),
            }

        # Kaplan-Meier by risk tier (using median income as proxy)
        income_col = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
        median_income = income_col.median()
        dur = pd.to_numeric(df["Duration"], errors="coerce").astype(float)
        evt = pd.to_numeric(df["Event"], errors="coerce").astype(float)
        km_results = {}
        for label, mask in [("Below Median Income", income_col < median_income),
                            ("Above Median Income", income_col >= median_income)]:
            kmf = KaplanMeierFitter()
            sub_dur = dur.loc[mask].dropna()
            sub_evt = evt.loc[mask].dropna()
            common_idx = sub_dur.index.intersection(sub_evt.index)
            if len(common_idx) < 5:
                continue
            kmf.fit(sub_dur.loc[common_idx], event_observed=sub_evt.loc[common_idx], label=label)
            km_results[label] = {
                "median_survival_years": round(float(kmf.median_survival_time_), 2)
                if not np.isinf(kmf.median_survival_time_) else "Not reached",
                "n_subjects": int(mask.sum()),
                "n_events": int(sub_evt.loc[common_idx].sum()),
            }

        # Build report
        report = {
            "analysis_date": pd.Timestamp.now().isoformat(),
            "model": "Cox Proportional Hazards (penalizer=0.1)",
            "concordance_index": round(float(cph.concordance_index_), 4),
            "n_observations": len(surv_df),
            "n_events": int(surv_df["Event"].sum()),
            "hazard_ratios": hazard_ratios,
            "kaplan_meier_cohorts": km_results,
            "methodology_note": (
                "YearsAtCompany is used as duration proxy. On synthetic data, "
                "this approximates time-at-risk but is not a true observation period. "
                "Real HRIS data with hire dates and termination dates enables proper "
                "left-truncated, right-censored survival analysis."
            ),
        }

        # Print top significant hazard ratios
        sig_features = [(f, v) for f, v in hazard_ratios.items() if v["significant"]]
        sig_features.sort(key=lambda x: abs(x[1]["hazard_ratio"] - 1), reverse=True)
        for feat, vals in sig_features[:5]:
            hr = vals["hazard_ratio"]
            print(f"    {feat}: HR={hr:.3f} ({'UP risk' if hr > 1 else 'DOWN risk'}), p={vals['p_value']:.4f}")
        print(f"  Concordance Index: {report['concordance_index']:.4f}")

        report_path = OUT_DIR / "survival_analysis_report.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved -> {report_path}")

        # Save survival plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Cox coefficients
            cph.plot(ax=axes[0])
            axes[0].set_title("Cox PH — Feature Hazard Ratios", fontsize=12)
            axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

            # Plot 2: Kaplan-Meier curves
            for label, mask in [("Below Median Income", df["MonthlyIncome"] < median_income),
                                ("Above Median Income", df["MonthlyIncome"] >= median_income)]:
                kmf = KaplanMeierFitter()
                subset = df.loc[mask]
                kmf.fit(subset["Duration"], event_observed=subset["Event"], label=label)
                kmf.plot_survival_function(ax=axes[1])
            axes[1].set_title("Kaplan-Meier Survival by Income Cohort", fontsize=12)
            axes[1].set_xlabel("Years at Company")
            axes[1].set_ylabel("Survival Probability")

            plt.tight_layout()
            plt.savefig(OUT_DIR / "survival_analysis.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Plot saved → {OUT_DIR / 'survival_analysis.png'}")
        except Exception as plot_err:
            print(f"  [INFO] Survival plot skipped: {plot_err}")

    except ImportError:
        print("  [INFO] lifelines not installed. Skipping survival analysis.")
        print("  Install with: pip install lifelines")
    except Exception as e:
        print(f"  [WARN] Survival analysis error: {e}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  HR Attrition Intelligence System — Production Pipeline")
    print("=" * 60)

    # Step 1: Load & Engineer
    print("\n[1/9] Loading data from Postgres fct_attrition_features...")
    from src.database import engine
    
    try:
        # Try reading the data cleaned and transformed by dbt (from Postgres)
        raw_df = pd.read_sql("SELECT * FROM public.fct_attrition_features", engine)
        print("  Successfully loaded data from database (fct_attrition_features).")
        
        # Map snake_case back to PascalCase (or whatever the original dataset used) 
        # to maintain compatibility with the REST API and SHAP visualizations
        column_mapping = {
            "employee_number": "EmployeeNumber", "age": "Age", "attrition": "Attrition",
            "distance_from_home": "DistanceFromHome", "monthly_income": "MonthlyIncome",
            "num_companies_worked": "NumCompaniesWorked", "percent_salary_hike": "PercentSalaryHike",
            "total_working_years": "TotalWorkingYears", "years_at_company": "YearsAtCompany",
            "years_in_current_role": "YearsInCurrentRole", "years_since_last_promotion": "YearsSinceLastPromotion",
            "years_with_curr_manager": "YearsWithCurrManager", "over_time": "OverTime",
            "over_time_yes": "OverTime_Yes", "department": "Department", "education": "Education",
            "education_field": "EducationField", "environment_satisfaction": "EnvironmentSatisfaction",
            "gender": "Gender", "job_involvement": "JobInvolvement", "job_level": "JobLevel",
            "job_role": "JobRole", "job_satisfaction": "JobSatisfaction", "marital_status": "MaritalStatus",
            "performance_rating": "PerformanceRating", "relationship_satisfaction": "RelationshipSatisfaction",
            "stock_option_level": "StockOptionLevel", "training_times_last_year": "TrainingTimesLastYear",
            "work_life_balance": "WorkLifeBalance", "compa_ratio": "Compa_Ratio",
            "income_growth_gap": "Income_Growth_Gap", "promotion_stagnation": "Promotion_Stagnation",
            "burnout_risk": "Burnout_Risk", "manager_stability": "Manager_Stability",
            "engagement_index": "Engagement_Index", "career_velocity": "Career_Velocity",
            "loyalty_index": "Loyalty_Index", "travel_burden": "Travel_Burden"
        }
        df = raw_df.rename(columns=column_mapping)
    except Exception as e:
        print(f"  [Fallback] Database table missing. Loading from raw CSV: {DATA_PATH.name}...")
        raw_df = pd.read_csv(DATA_PATH)
        # Convert Attrition to int
        raw_df['Attrition'] = (raw_df['Attrition'] == 'Yes').astype(int)
        # Apply the identical feature engineering used by the API to ensure parity
        df, _ = engineer_features_batch(raw_df)

    # For API inference, the API needs population stats!
    from src.features import save_population_stats, compute_population_stats
    stats = compute_population_stats(df)
    save_population_stats(stats)
    print(f"  Dataset: {df.shape[0]} employees, {df.shape[1]} features")
    print(f"  Attrition rate: {df['Attrition'].mean():.1%}")
    print(f"  Population stats saved to {BASE_DIR / 'models' / 'population_stats.json'}")

    # Step 1b: Data Version Tracking (GAP-03)
    print("\n[1b/10] Recording data version...")
    from src.data_version import DataVersionTracker
    version_tracker = DataVersionTracker()
    data_version = version_tracker.record_version(
        df, source=str(DATA_PATH), metadata={"pipeline": "train_attrition_model"}
    )

    # Step 2: EDA
    print("\n[2/9] Generating EDA charts...")
    generate_eda(df)

    # Drop identifiers and columns that were replaced by features (e.g. OverTime)
    drop_cols = ["EmployeeNumber", "OverTime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Step 3: Preprocess
    print("\n[3/9] Preprocessing...")
    X, y = preprocess(df)
    # Three-way split: train (60%) / calibration (20%) / test (20%)
    # This prevents data leakage in the calibration step (GAP-16 fix)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    print(f"  Train: {X_train.shape[0]} | Calibration: {X_cal.shape[0]} | Test: {X_test.shape[0]}")

    # Step 4: Hyperparameter Tuning
    print("\n[4/9] Optimizing hyperparameters (Optuna, 30 trials)...")
    best_params = optimize_hyperparams(X_train, y_train, n_trials=30)
    print(f"  Best params: {best_params}")

    # Step 5: Train
    print("\n[5/10] Training final model...")
    model = train_model(X_train, y_train, best_params)

    # Step 6: Evaluate raw model
    print("\n[6/10] Evaluating raw model...")
    metrics = evaluate(model, X_test, y_test)

    # Step 6b: Probability Calibration (using held-out calibration set)
    print("\n[6b/10] Applying Sigmoid (Platt Scaling) calibration (on held-out cal set)...")
    calibrated_model = calibrate_model(model, X_cal, y_cal)

    # Compare Brier scores
    raw_proba = model.predict_proba(X_test)[:, 1]
    cal_proba = calibrated_model.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test, raw_proba)
    cal_brier = brier_score_loss(y_test, cal_proba)
    cal_auc = roc_auc_score(y_test, cal_proba)
    improvement_pct = ((raw_brier - cal_brier) / raw_brier) * 100

    print(f"  Raw Brier:        {raw_brier:.4f}")
    print(f"  Calibrated Brier: {cal_brier:.4f}")
    print(f"  Calibrated AUC:   {cal_auc:.4f}")
    print(f"  Improvement:      {improvement_pct:+.1f}%")

    # Updated calibration curve (before vs after)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frac_raw, mean_raw = calibration_curve(y_test, raw_proba, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y_test, cal_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_raw, frac_raw, "s--", color="#ef4444", label="XGBoost (Raw)", alpha=0.7)
    ax.plot(mean_cal, frac_cal, "o-", color="#22c55e", label="XGBoost (Calibrated)", linewidth=2)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration: Before vs After")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # Step 6c: Uplift Modeling (Causal ML)
    print("\n[6c/10] Training T-Learner Uplift Model (Causal Inference)...")
    # Proxy Treatment: We consider a salary hike of >= 15% as a retention intervention (Treatment=1)
    X_train_uplift = X_train.copy()
    X_train_uplift["Treatment_HighHike"] = (X_train_uplift["PercentSalaryHike"] >= 15).astype(int)
    
    uplift_model = TLearnerUplift(treatment_col="Treatment_HighHike")
    uplift_model.fit(X_train_uplift, y_train)

    # Extract threshold early so it's available for fairness stage
    optimal_threshold = best_params.get("_optimal_threshold", 0.5)

    # ── Step 7d: FAIRNESS MITIGATION STAGE ─────────────────────────────
    print("\n[7d/18] Running Fairness Mitigation Stage...")
    from src.fairness_mitigation import (
        reconstruct_sensitive_features,
        smote_enn_oversample,
        smote_oversample_minority,
        compute_fairness_sample_weights,
        compute_multi_attribute_weights,
        compute_subgroup_thresholds,
        compute_dpd_constrained_thresholds,
        apply_subgroup_thresholds,
        evaluate_fairness_metrics,
        bootstrap_fairness_ci,
        fit_equalized_odds_postprocessor,
        safe_threshold_predict,
        adaptive_eod_threshold,
        adaptive_dpd_threshold,
        FAIRNESS_THRESHOLDS,
    )

    # 7d-A: Reconstruct sensitive features for all splits
    sf_train = reconstruct_sensitive_features(X_train)
    sf_cal = reconstruct_sensitive_features(X_cal)
    sf_test = reconstruct_sensitive_features(X_test)

    # Evaluate baseline fairness BEFORE mitigation
    print("\n  -- Baseline Fairness (before mitigation) --")
    baseline_proba = model.predict_proba(X_test)[:, 1]
    baseline_pred = model.predict(X_test)
    for attr_name in ["Age_Group", "Gender", "MaritalStatus"]:
        if attr_name in sf_test:
            fair_m = evaluate_fairness_metrics(
                y_test, baseline_pred, sf_test[attr_name], baseline_proba
            )
            status = "PASS" if fair_m.get("eod_pass", False) else "FAIL"
            print(f"  [{attr_name}] EOD={fair_m.get('eod','N/A')}, "
                  f"DPD={fair_m.get('dpd','N/A')}, "
                  f"F2={fair_m.get('global_f2','N/A')} [{status}]")
            # Show subgroup detail
            for sg, sm in fair_m.get("subgroups", {}).items():
                print(f"    {sg}: F2={sm['f2']}, n={sm['n']}")

    # 7d-B: SMOTE-ENN hybrid augmentation for Age minority groups
    print("\n  [7d-B] SMOTE-ENN Augmentation for Age 51+ minority...")
    if "Age_Group" in sf_train:
        X_train_aug, y_train_aug, sf_train_age_aug = smote_enn_oversample(
            X_train, y_train, sf_train["Age_Group"],
            min_positive_per_group=35,
            k_neighbors=3,
        )
    else:
        X_train_aug, y_train_aug = X_train, y_train
        sf_train_age_aug = sf_train.get("Age_Group", np.array([]))

    # 7d-B2: SMOTE augmentation for MaritalStatus "Other" minority
    print("\n  [7d-B2] SMOTE Augmentation for MaritalStatus 'Other'...")
    sf_train_aug_full = reconstruct_sensitive_features(X_train_aug)
    if "MaritalStatus" in sf_train_aug_full:
        X_train_aug, y_train_aug, sf_train_ms_aug = smote_oversample_minority(
            X_train_aug, y_train_aug, sf_train_aug_full["MaritalStatus"],
            min_positive_per_group=25,
            k_neighbors=3,
        )
    else:
        sf_train_ms_aug = np.array([])

    # 7d-C: Multi-attribute sample weights (Age + MaritalStatus)
    print("\n  [7d-C] Cost-Sensitive Retraining with Multi-Attribute Weights...")
    sf_retrain = reconstruct_sensitive_features(X_train_aug)
    weight_attrs = {}
    if "Age_Group" in sf_retrain:
        weight_attrs["Age_Group"] = sf_retrain["Age_Group"]
    if "MaritalStatus" in sf_retrain:
        weight_attrs["MaritalStatus"] = sf_retrain["MaritalStatus"]

    if weight_attrs:
        sample_weights = compute_multi_attribute_weights(
            X_train_aug, y_train_aug, weight_attrs, boost_factor=1.5,
        )
        print(f"    Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        print(f"    Attributes weighted: {list(weight_attrs.keys())}")

        # Retrain XGBoost with multi-attribute sample weights
        xgb_params_fair = {k: v for k, v in best_params.items() if not k.startswith("_")}
        scale_w = float((y_train_aug == 0).sum() / max((y_train_aug == 1).sum(), 1))
        fair_model = xgb.XGBClassifier(
            **xgb_params_fair,
            scale_pos_weight=scale_w,
            eval_metric="logloss",
            random_state=42,
        )
        fair_model.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)
        print("    Fair model retrained with SMOTE + multi-attribute weights.")

        # Recalibrate the fair model
        try:
            from sklearn.frozen import FrozenEstimator
            fair_calibrated = CalibratedClassifierCV(
                FrozenEstimator(fair_model), method="sigmoid"
            )
        except ImportError:
            fair_calibrated = CalibratedClassifierCV(
                fair_model, method="sigmoid", cv="prefit"
            )
        fair_calibrated.fit(X_cal, y_cal)
    else:
        fair_model = model
        fair_calibrated = calibrated_model

    # 7d-D: Post-processing with ThresholdOptimizer (Hardt et al. 2016)
    # Try Fairlearn ThresholdOptimizer for principled Equalized Odds enforcement
    print("\n  [7d-D] Fitting ThresholdOptimizer (Equalized Odds)...")
    fair_proba_cal = fair_calibrated.predict_proba(X_cal)[:, 1]
    group_thresholds_all: dict[str, dict[str, float]] = {}
    age_postprocessor = None

    if "Age_Group" in sf_cal:
        age_postprocessor, to_ok = fit_equalized_odds_postprocessor(
            fair_calibrated, X_cal, y_cal, sf_cal["Age_Group"],
        )
        if to_ok:
            print("    ThresholdOptimizer fitted for Age_Group.")
        else:
            print("    ThresholdOptimizer failed; using manual subgroup thresholds.")
            gt = compute_subgroup_thresholds(
                np.asarray(y_cal), fair_proba_cal, sf_cal["Age_Group"],
                default_threshold=optimal_threshold,
                min_threshold=0.08,
                max_threshold=0.50,
            )
            group_thresholds_all["Age_Group"] = gt
            print(f"    Age_Group manual thresholds: {gt}")
    print(f"    Gender: global threshold {optimal_threshold:.4f}")

    # 7d-D2: ThresholdOptimizer for MaritalStatus (with DPD fallback)
    ms_postprocessor = None
    if "MaritalStatus" in sf_cal:
        print("\n  [7d-D2] Fitting ThresholdOptimizer for MaritalStatus...")
        ms_postprocessor, ms_to_ok = fit_equalized_odds_postprocessor(
            fair_calibrated, X_cal, y_cal, sf_cal["MaritalStatus"],
        )
        if ms_to_ok:
            print("    ThresholdOptimizer fitted for MaritalStatus.")
        else:
            print("    ThresholdOptimizer failed; using DPD-constrained grid search.")
            ms_postprocessor = None
            ms_thresholds = compute_dpd_constrained_thresholds(
                np.asarray(y_cal), fair_proba_cal, sf_cal["MaritalStatus"],
                default_threshold=optimal_threshold,
                min_threshold=0.05,
                max_threshold=0.55,
                n_steps=30,
                min_global_f2=0.35,
                dpd_target=0.12,
            )
            group_thresholds_all["MaritalStatus"] = ms_thresholds
            print(f"    MaritalStatus DPD-constrained thresholds: {ms_thresholds}")

    # 7d-E: Evaluate AFTER mitigation with adaptive quality gates
    print("\n  -- Fairness AFTER Mitigation (with bootstrap CIs) --")
    fair_proba_test = fair_calibrated.predict_proba(X_test)[:, 1]
    fair_auc = roc_auc_score(y_test, fair_proba_test)

    fairness_pass_all = True
    fairness_results_v2: dict[str, Any] = {}

    for attr_name in ["Age_Group", "Gender", "MaritalStatus"]:
        if attr_name not in sf_test:
            continue

        # Predict with ThresholdOptimizer or manual thresholds
        if attr_name == "Age_Group" and age_postprocessor is not None:
            fair_pred = safe_threshold_predict(
                age_postprocessor, X_test, sf_test[attr_name],
            )
            # Safety: if TO predicts all zeros (degenerate), fall back
            if fair_pred is not None and fair_pred.sum() == 0:
                print(f"    [WARN] ThresholdOptimizer predicted all zeros for {attr_name}. "
                      f"Falling back to subgroup thresholds.")
                fair_pred = None
            if fair_pred is None:
                # Compute manual thresholds on the fly if not already done
                if attr_name not in group_thresholds_all:
                    gt = compute_subgroup_thresholds(
                        np.asarray(y_cal), fair_proba_cal, sf_cal[attr_name],
                        default_threshold=optimal_threshold,
                        min_threshold=0.08, max_threshold=0.50,
                    )
                    group_thresholds_all[attr_name] = gt
                fair_pred = apply_subgroup_thresholds(
                    fair_proba_test, sf_test[attr_name],
                    group_thresholds_all[attr_name],
                    default_threshold=optimal_threshold,
                )
        elif attr_name == "Age_Group" and attr_name in group_thresholds_all:
            fair_pred = apply_subgroup_thresholds(
                fair_proba_test, sf_test[attr_name],
                group_thresholds_all[attr_name],
                default_threshold=optimal_threshold,
            )
        elif attr_name == "MaritalStatus" and ms_postprocessor is not None:
            # MaritalStatus ThresholdOptimizer
            fair_pred = safe_threshold_predict(
                ms_postprocessor, X_test, sf_test[attr_name],
            )
            if fair_pred is not None and fair_pred.sum() == 0:
                print(f"    [WARN] ThresholdOptimizer predicted all zeros for {attr_name}. "
                      f"Falling back to subgroup thresholds.")
                fair_pred = None
            if fair_pred is None:
                if attr_name not in group_thresholds_all:
                    gt = compute_subgroup_thresholds(
                        np.asarray(y_cal), fair_proba_cal, sf_cal[attr_name],
                        default_threshold=optimal_threshold,
                        min_threshold=0.08, max_threshold=0.50,
                    )
                    group_thresholds_all[attr_name] = gt
                fair_pred = apply_subgroup_thresholds(
                    fair_proba_test, sf_test[attr_name],
                    group_thresholds_all[attr_name],
                    default_threshold=optimal_threshold,
                )
        elif attr_name in group_thresholds_all:
            # Any attribute with computed subgroup thresholds (DPD fallback)
            fair_pred = apply_subgroup_thresholds(
                fair_proba_test, sf_test[attr_name],
                group_thresholds_all[attr_name],
                default_threshold=optimal_threshold,
            )
        else:
            # Gender: global threshold
            fair_pred = (fair_proba_test >= optimal_threshold).astype(int)

        after_metrics = evaluate_fairness_metrics(
            y_test, fair_pred, sf_test[attr_name], fair_proba_test
        )
        before_metrics = evaluate_fairness_metrics(
            y_test, baseline_pred, sf_test[attr_name], baseline_proba
        )

        # Compute adaptive thresholds based on subgroup sizes
        sf_test_arr = np.asarray(sf_test[attr_name])
        y_test_arr = np.asarray(y_test)
        min_n_pos = min(
            (y_test_arr[sf_test_arr == g] == 1).sum()
            for g in np.unique(sf_test_arr)
        )
        min_n = min(
            (sf_test_arr == g).sum()
            for g in np.unique(sf_test_arr)
        )
        adaptive_eod_t = adaptive_eod_threshold(min_n_pos)
        adaptive_dpd_t = adaptive_dpd_threshold(min_n)

        eod_before = before_metrics.get("eod", 0)
        eod_after = after_metrics.get("eod", 0)
        dpd_after = after_metrics.get("dpd", 0)
        f2_after = after_metrics.get("global_f2", 0)
        # Use ADAPTIVE thresholds instead of fixed
        status_eod = "PASS" if eod_after <= adaptive_eod_t else "FAIL"
        status_dpd = "PASS" if dpd_after <= adaptive_dpd_t else "FAIL"
        direction = "IMPROVED" if eod_after < eod_before else "REGRESSED"

        # Bootstrap CI for uncertainty quantification
        boot_ci = bootstrap_fairness_ci(
            y_test_arr, np.asarray(fair_pred), sf_test_arr, n_bootstrap=1000,
        )
        ci_tag = "INCONCLUSIVE" if boot_ci.get("statistically_inconclusive") else "CONCLUSIVE"

        print(f"  [{attr_name}] (min_n_pos={min_n_pos}, adaptive EOD<={adaptive_eod_t})")
        print(f"    EOD: {eod_before:.4f} -> {eod_after:.4f} ({direction}) [{status_eod}]")
        print(f"    EOD 95% CI: [{boot_ci['eod_ci_lower']:.4f}, {boot_ci['eod_ci_upper']:.4f}] "
              f"width={boot_ci['ci_width']:.4f} [{ci_tag}]")
        print(f"    DPD: {before_metrics.get('dpd',0):.4f} -> {dpd_after:.4f} [{status_dpd}]")
        print(f"    F2:  {before_metrics.get('global_f2',0)} -> {f2_after}")
        for sg, sm in after_metrics.get("subgroups", {}).items():
            bf2 = before_metrics.get("subgroups", {}).get(sg, {}).get("f2", "N/A")
            print(f"      {sg}: F2 {bf2} -> {sm['f2']} (n={sm['n']})")

        if status_eod == "FAIL" or status_dpd == "FAIL":
            fairness_pass_all = False

        fairness_results_v2[attr_name] = {
            "eod_before": round(float(eod_before), 4),
            "eod_after": round(float(eod_after), 4),
            "dpd_after": round(float(dpd_after), 4),
            "f2_after": round(float(f2_after), 4),
            "eod_pass": status_eod == "PASS",
            "dpd_pass": status_dpd == "PASS",
            "adaptive_eod_threshold": adaptive_eod_t,
            "adaptive_dpd_threshold": adaptive_dpd_t,
            "bootstrap_ci": boot_ci,
            "min_subgroup_n_positive": int(min_n_pos),
            "thresholds": group_thresholds_all.get(attr_name, {}),
        }

    # Save fairness remediation report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remediation_report = {
        "strategy": "smote_enn_reweighing_threshold_optimizer_bootstrap_ci",
        "thresholds": group_thresholds_all,
        "quality_gate": {
            **FAIRNESS_THRESHOLDS,
            "note": "Adaptive thresholds scaled to subgroup sample size",
        },
        "results": fairness_results_v2,
        "fair_auc": round(float(fair_auc), 4),
        "all_passed": fairness_pass_all,
        "eu_ai_act_compliance": {
            "article_10": "Data augmentation (SMOTE-ENN) + reweighing for representative training",
            "article_15": "Adversarial robustness + bootstrap uncertainty quantification",
            "note": "EU AI Act does not prescribe numeric fairness thresholds. "
                    "Compliance documented through mitigation process and statistical rigor.",
        },
    }
    with open(OUT_DIR / "fairness_mitigation_report.json", "w") as f:
        json.dump(remediation_report, f, indent=2, default=str)

    # Use the fair model + subgroup thresholds as the primary model
    if fairness_pass_all or fair_auc >= FAIRNESS_THRESHOLDS["min_global_auc"]:
        print(f"\n  Using FAIR model (AUC={fair_auc:.4f}). Subgroup thresholds active.")
        model = fair_model
        calibrated_model = fair_calibrated
        # Save thresholds for inference
        with open(MODEL_DIR / "subgroup_thresholds.json", "w") as f:
            json.dump(group_thresholds_all, f, indent=2)
    else:
        print(f"\n  [WARN] Fair model AUC too low ({fair_auc:.4f}). Keeping original model.")

    # ── End Fairness Mitigation Stage ──────────────────────────────────

    # Step 7: Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgb_attrition.json"))
    joblib.dump(calibrated_model, str(MODEL_DIR / "xgb_calibrated.joblib"))
    uplift_model.save(str(MODEL_DIR / "uplift_tlearner.joblib"))
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Compute model version hash for provenance
    model_hash = hashlib.sha256(
        (MODEL_DIR / "xgb_attrition.json").read_bytes()
    ).hexdigest()[:12]

    # Apply optimized threshold for F2 evaluation (GAP-17)
    cal_preds_tuned = (cal_proba >= optimal_threshold).astype(int)
    f2_tuned = fbeta_score(y_test, cal_preds_tuned, beta=2)
    print(f"\n  F2 @ default 0.5 threshold:   {metrics['f2']:.4f}")
    print(f"  F2 @ tuned {optimal_threshold:.2f} threshold:  {f2_tuned:.4f}")

    # Also compute F2 with subgroup thresholds / ThresholdOptimizer for the fair model
    if "Age_Group" in sf_test and age_postprocessor is not None:
        fair_proba_final = calibrated_model.predict_proba(X_test)[:, 1]
        fair_pred_final = safe_threshold_predict(
            age_postprocessor, X_test, sf_test["Age_Group"],
        )
        # Guard against degenerate all-zeros solution
        if fair_pred_final is not None and fair_pred_final.sum() == 0:
            fair_pred_final = None
        if fair_pred_final is None:
            if "Age_Group" in group_thresholds_all:
                fair_pred_final = apply_subgroup_thresholds(
                    fair_proba_final, sf_test["Age_Group"],
                    group_thresholds_all["Age_Group"],
                    default_threshold=optimal_threshold,
                )
            else:
                fair_pred_final = (fair_proba_final >= optimal_threshold).astype(int)
        f2_fair = fbeta_score(y_test, fair_pred_final, beta=2)
        print(f"  F2 @ fair thresholds:         {f2_fair:.4f}")
    elif "Age_Group" in sf_test and "Age_Group" in group_thresholds_all:
        fair_proba_final = calibrated_model.predict_proba(X_test)[:, 1]
        fair_pred_final = apply_subgroup_thresholds(
            fair_proba_final, sf_test["Age_Group"],
            group_thresholds_all["Age_Group"],
            default_threshold=optimal_threshold,
        )
        f2_fair = fbeta_score(y_test, fair_pred_final, beta=2)
        print(f"  F2 @ subgroup thresholds:    {f2_fair:.4f}")
    else:
        f2_fair = f2_tuned

    # Step 7b: Subpopulation performance analysis (EU AI Act Art. 15)
    subpop_metrics = evaluate_subpopulations(model, X_test, y_test)

    # Step 7c: Fairness Audit (must run before quality gate so registration
    # decisions use metrics from this training run, not stale output files).
    print("\n[7c/10] Running Algorithmic Fairness Audit...")
    fairness_results = run_fairness_audit(model, X_test, y_test)

    # Save training metadata for audit trail
    training_meta = {
        "model_file": "xgb_attrition.json",
        "calibrated_model_file": "xgb_calibrated.joblib",
        "model_version": model_hash,
        "data_version": data_version["data_hash_short"],
        "optimal_threshold": optimal_threshold,
        "dataset": str(DATA_PATH.name),
        "dataset_rows": int(df.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
        "calibration_size": int(X_cal.shape[0]),
        "test_size": int(X_test.shape[0]),
        "metrics": {
            "f2_score": round(metrics["f2"], 4),
            "f2_score_tuned": round(f2_tuned, 4),
            "roc_auc": round(metrics["auc"], 4),
            "brier_score_raw": round(raw_brier, 4),
            "brier_score_calibrated": round(cal_brier, 4),
            "calibration_improvement_pct": round(improvement_pct, 1),
            "calibrated_auc": round(cal_auc, 4),
        },
        "best_params": best_params,
    }
    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(training_meta, f, indent=2)
    print(f"  Model + calibrated model + metadata saved to {MODEL_DIR}")

    # ── MLflow Experiment Tracking & Model Registry (GAP-01/04) ──────
    if MLFLOW_AVAILABLE:
        mlflow_db = str(BASE_DIR / "mlflow_registry.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("hr-attrition-intelligence")

        # ── Quality Gate: Validate before registration ──────────────
        MIN_AUC_THRESHOLD = 0.70
        MIN_F2_THRESHOLD = 0.40

        quality_passed = True
        quality_issues: list[str] = []

        if cal_auc < MIN_AUC_THRESHOLD:
            quality_issues.append(f"AUC {cal_auc:.4f} below threshold {MIN_AUC_THRESHOLD}")
            quality_passed = False
        if f2_fair < MIN_F2_THRESHOLD:
            quality_issues.append(f"F2 {f2_fair:.4f} below threshold {MIN_F2_THRESHOLD}")
            quality_passed = False

        # Use fairness_results_v2 (from step 7d with subgroup thresholds)
        # instead of fairness_results (from step 7c with default threshold)
        for attr, fair_v2 in fairness_results_v2.items():
            if not fair_v2.get("dpd_pass", True):
                quality_issues.append(
                    f"Fairness DPD FAILED for {attr}: DPD={fair_v2['dpd_after']:.4f}"
                )
                quality_passed = False
            if not fair_v2.get("eod_pass", True):
                quality_issues.append(
                    f"Fairness EOD FAILED for {attr}: EOD={fair_v2['eod_after']:.4f}"
                )
                quality_passed = False

        if quality_issues:
            print("\n  [!] Quality Gate Issues:")
            for issue in quality_issues:
                print(f"    - {issue}")

        lifecycle_stage = "Staging" if quality_passed else "Rejected"
        print(f"\n  Model lifecycle stage: {lifecycle_stage}")

        with mlflow.start_run(run_name=f"train-{model_hash}") as run:
            # Log parameters
            xgb_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
            mlflow.log_params(xgb_params)
            mlflow.log_param("optimal_threshold", optimal_threshold)
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("cal_size", X_cal.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            mlflow.log_param("lifecycle_stage", lifecycle_stage)

            # Log metrics
            mlflow.log_metric("f2_score", metrics["f2"])
            mlflow.log_metric("f2_score_tuned", f2_tuned)
            mlflow.log_metric("roc_auc", metrics["auc"])
            mlflow.log_metric("brier_raw", raw_brier)
            mlflow.log_metric("brier_calibrated", cal_brier)
            mlflow.log_metric("calibrated_auc", cal_auc)
            mlflow.log_metric("quality_gate_passed", int(quality_passed))

            # Log model artifacts
            mlflow.log_artifact(str(MODEL_DIR / "xgb_attrition.json"))
            mlflow.log_artifact(str(MODEL_DIR / "xgb_calibrated.joblib"))
            mlflow.log_artifact(str(MODEL_DIR / "best_params.json"))
            mlflow.log_artifact(str(MODEL_DIR / "training_metadata.json"))
            mlflow.log_artifact(str(MODEL_DIR / "population_stats.json"))
            fairness_path = OUT_DIR / "fairness_audit.json"
            if fairness_path.exists():
                mlflow.log_artifact(str(fairness_path))
            subpop_path = OUT_DIR / "subpopulation_metrics.json"
            if subpop_path.exists():
                mlflow.log_artifact(str(subpop_path))

            # Register model only if quality gate passes
            if quality_passed:
                mlflow.xgboost.log_model(
                    model, artifact_path="model",
                    registered_model_name="hr-attrition-xgboost",
                )
                print(f"\n  MLflow: Run {run.info.run_id} logged")
                print(f"  MLflow: Model registered as 'hr-attrition-xgboost' (Stage: {lifecycle_stage})")
            else:
                mlflow.xgboost.log_model(model, artifact_path="model")
                print(f"\n  MLflow: Run {run.info.run_id} logged (NOT registered — quality gate failed)")
                print(f"  Issues: {'; '.join(quality_issues)}")
    else:
        print("\n  MLflow not installed — skipping experiment tracking.")
        print("  Install with: pip install mlflow-skinny")

    # Step 8: SHAP (uses raw model for tree-based explainability)
    print("\n[7/13] Generating SHAP explanations...")
    explain_model(model, X_train, X_test)

    # Step 8b: SHAP Attribution Drift Monitor
    print("\n[8/13] Checking SHAP Attribution Drift...")
    shap_csv = OUT_DIR / "shap_global_importance.csv"
    if shap_csv.exists():
        current_importance = pd.read_csv(shap_csv)
        detect_shap_drift(current_importance)
    else:
        print("  [SKIP] No SHAP importance CSV found — skipping drift check.")

    # Step 9: Risk Framework & Cost Model (uses calibrated model for accurate probabilities)
    print("\n[9/13] Building Risk Framework & Cost Model...")
    build_risk_framework(calibrated_model, X_test, y_test, raw_df, X.columns)

    # Step 10: Data Drift
    print("\n[10/13] Generating Data Drift Report (Evidently AI)...")
    generate_drift_report(X_train, X_test)

    # Step 11: Adversarial Robustness Testing (EU AI Act Art. 15)
    print("\n[11/13] Running Adversarial Robustness Tests (EU AI Act Art. 15)...")
    run_adversarial_robustness_test(model, X_test, y_test)

    # Step 12: Causal Inference Validation
    print("\n[12/13] Running Causal Inference Validation (DoWhy)...")
    run_causal_validation(raw_df)

    # Step 13: Survival Analysis
    print("\n[13/13] Running Survival Analysis...")
    run_survival_analysis(raw_df)

    print("\n" + "=" * 60)
    print("  Pipeline complete. All outputs saved to ./outputs/")
    print(f"  Model version: {model_hash}")
    print(f"  F2 (tuned threshold): {f2_tuned:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
