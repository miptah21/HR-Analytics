"""
Fairness Remediation Runner v2 - Per-Attribute Optimal Strategy.

Strategy based on empirical results:
  - Age_Group:      Full 3-layer (most critical, needs all layers)
  - Gender:         Post-processing ONLY (in-processing over-corrected)
  - MaritalStatus:  Full 3-layer (already passing with this approach)
  - Combined:       Multi-attribute joint mitigation for final model
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features import engineer_features_batch
from src.fairness_mitigation import (
    FairnessMitigator,
    compute_fairness_sample_weights,
    reconstruct_sensitive_features,
    evaluate_fairness_metrics,
    train_fair_model,
    optimize_group_thresholds,
    _safe_threshold_predict,
    FAIRNESS_THRESHOLDS,
)

DATA_PATH = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


def load_and_prepare():
    """Load data, engineer features, split."""
    raw_df = pd.read_csv(DATA_PATH)
    df, _ = engineer_features_batch(raw_df, save_stats=False)
    df = df.drop(columns=[c for c in ["EmployeeNumber", "OverTime"] if c in df.columns])

    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )
    return X_train, y_train, X_cal, y_cal, X_test, y_test, raw_df


def load_params():
    """Load best hyperparameters."""
    params_path = MODEL_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.06}


def run_per_attribute_mitigation(X_train, y_train, X_cal, y_cal, X_test, y_test, best_params):
    """Run optimal strategy per attribute."""
    results = {}

    # --- Age Group: Full 3-layer ---
    print("\n" + "=" * 60)
    print("  [1/3] AGE GROUP - Full 3-Layer Mitigation")
    print("=" * 60)
    age_mitigator = FairnessMitigator(
        sensitive_feature_name="Age_Group",
        apply_reweighing=True,
        apply_inprocessing=True,
        apply_postprocessing=True,
    )
    age_result = age_mitigator.run(
        X_train, y_train, X_cal, y_cal, X_test, y_test,
        base_params=best_params,
    )
    results["Age_Group"] = {
        "strategy": "full_3_layer",
        "before": age_result.metrics_before,
        "after": age_result.metrics_after,
        "gate": age_result.quality_gate_passed,
        "issues": age_result.issues,
    }

    # --- Gender: Post-processing ONLY ---
    print("\n" + "=" * 60)
    print("  [2/3] GENDER - Post-Processing Only (ThresholdOptimizer)")
    print("=" * 60)
    gender_mitigator = FairnessMitigator(
        sensitive_feature_name="Gender",
        apply_reweighing=False,
        apply_inprocessing=False,
        apply_postprocessing=True,
    )
    gender_result = gender_mitigator.run(
        X_train, y_train, X_cal, y_cal, X_test, y_test,
        base_params=best_params,
    )
    results["Gender"] = {
        "strategy": "postprocessing_only",
        "before": gender_result.metrics_before,
        "after": gender_result.metrics_after,
        "gate": gender_result.quality_gate_passed,
        "issues": gender_result.issues,
    }

    # --- MaritalStatus: Full 3-layer ---
    print("\n" + "=" * 60)
    print("  [3/3] MARITAL STATUS - Full 3-Layer Mitigation")
    print("=" * 60)
    ms_mitigator = FairnessMitigator(
        sensitive_feature_name="MaritalStatus",
        apply_reweighing=True,
        apply_inprocessing=True,
        apply_postprocessing=True,
    )
    ms_result = ms_mitigator.run(
        X_train, y_train, X_cal, y_cal, X_test, y_test,
        base_params=best_params,
    )
    results["MaritalStatus"] = {
        "strategy": "full_3_layer",
        "before": ms_result.metrics_before,
        "after": ms_result.metrics_after,
        "gate": ms_result.quality_gate_passed,
        "issues": ms_result.issues,
    }

    return results


def run_combined_mitigation(X_train, y_train, X_cal, y_cal, X_test, y_test, best_params):
    """Run combined multi-attribute mitigation using Age+MaritalStatus joint feature."""
    print("\n" + "=" * 60)
    print("  COMBINED MULTI-ATTRIBUTE MITIGATION")
    print("  (Age_Group + MaritalStatus joint constraint)")
    print("=" * 60)

    # Reconstruct all sensitive features
    sf_train = reconstruct_sensitive_features(X_train)
    sf_cal = reconstruct_sensitive_features(X_cal)
    sf_test = reconstruct_sensitive_features(X_test)

    # Create combined sensitive feature: "Age_Group|MaritalStatus"
    combined_train = np.array([
        f"{a}|{m}" for a, m in zip(sf_train["Age_Group"], sf_train["MaritalStatus"])
    ])
    combined_cal = np.array([
        f"{a}|{m}" for a, m in zip(sf_cal["Age_Group"], sf_cal["MaritalStatus"])
    ])
    combined_test = np.array([
        f"{a}|{m}" for a, m in zip(sf_test["Age_Group"], sf_test["MaritalStatus"])
    ])

    print(f"\n  Combined groups: {len(np.unique(combined_train))}")
    for g in sorted(np.unique(combined_train)):
        n = (combined_train == g).sum()
        print(f"    {g}: n={n}")

    # Train baseline
    xgb_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
    scale_w = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    baseline = xgb.XGBClassifier(
        **xgb_params, scale_pos_weight=scale_w, eval_metric="logloss", random_state=42,
    )
    baseline.fit(X_train, y_train)

    # Evaluate baseline on each attribute separately
    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]

    print("\n  --- Baseline (before combined mitigation) ---")
    for attr_name in ["Age_Group", "Gender", "MaritalStatus"]:
        sf = sf_test[attr_name]
        metrics = evaluate_fairness_metrics(y_test, baseline_pred, sf, baseline_proba)
        print(f"    {attr_name}: EOD={metrics['eod']}, DPD={metrics['dpd']}, F2={metrics['global_f2']}")

    # Step 1: Reweigh on combined feature
    print("\n  [Layer 1] Computing combined sample weights...")
    combined_weights = compute_fairness_sample_weights(combined_train, y_train)
    print(f"    Weight range: [{combined_weights.min():.3f}, {combined_weights.max():.3f}]")

    # Step 2: Train with combined fairness constraint
    print("  [Layer 2] Training ExponentiatedGradient on combined features...")
    try:
        fair_model = train_fair_model(
            X_train, y_train,
            sensitive_features=combined_train,
            base_params=best_params,
            sample_weights=combined_weights,
            max_iter=80,
        )
        print("    Combined constrained model trained successfully.")
    except Exception as e:
        print(f"    [WARN] Combined in-processing failed: {e}")
        print("    Falling back to weighted baseline...")
        baseline.fit(X_train, y_train, sample_weight=combined_weights)
        fair_model = baseline

    # Step 3: Per-attribute ThresholdOptimizer (post-processing)
    print("  [Layer 3] Optimizing per-attribute thresholds...")
    postprocessors = {}
    for attr_name in ["Age_Group", "Gender", "MaritalStatus"]:
        try:
            pp = optimize_group_thresholds(
                fair_model, X_cal, y_cal, sf_cal[attr_name]
            )
            postprocessors[attr_name] = pp
            print(f"    {attr_name}: ThresholdOptimizer fitted")
        except Exception as e:
            print(f"    {attr_name}: ThresholdOptimizer failed - {e}")

    # Evaluate after mitigation on each attribute
    print("\n  --- After Combined Mitigation ---")
    combined_results = {}
    for attr_name in ["Age_Group", "Gender", "MaritalStatus"]:
        sf = sf_test[attr_name]
        if attr_name in postprocessors:
            fair_pred = _safe_threshold_predict(
                postprocessors[attr_name], fair_model, X_test, sf
            )
        else:
            fair_pred = fair_model.predict(X_test)

        try:
            fair_proba = fair_model.predict_proba(X_test)[:, 1]
        except Exception:
            fair_proba = None

        after_metrics = evaluate_fairness_metrics(y_test, fair_pred, sf, fair_proba)
        before_metrics = evaluate_fairness_metrics(y_test, baseline_pred, sf_test[attr_name], baseline_proba)

        eod_before = before_metrics['eod']
        eod_after = after_metrics['eod']
        dpd_before = before_metrics['dpd']
        dpd_after = after_metrics['dpd']
        f2_after = after_metrics['global_f2']

        status = "IMPROVED" if eod_after < eod_before else "REGRESSED"
        gate_eod = "PASS" if eod_after <= FAIRNESS_THRESHOLDS["eod_max"] else "FAIL"
        gate_dpd = "PASS" if dpd_after <= FAIRNESS_THRESHOLDS["dpd_max"] else "FAIL"

        print(f"    {attr_name}:")
        print(f"      EOD: {eod_before:.4f} -> {eod_after:.4f} ({status}) [{gate_eod}]")
        print(f"      DPD: {dpd_before:.4f} -> {dpd_after:.4f} [{gate_dpd}]")
        print(f"      F2:  {before_metrics['global_f2']} -> {f2_after}")

        # Subgroup detail
        for g, m in after_metrics.get("subgroups", {}).items():
            b_f2 = before_metrics.get("subgroups", {}).get(g, {}).get("f2", "N/A")
            print(f"        {g}: F2 {b_f2} -> {m['f2']} (n={m['n']})")

        combined_results[attr_name] = {
            "before": before_metrics,
            "after": after_metrics,
            "eod_pass": gate_eod == "PASS",
            "dpd_pass": gate_dpd == "PASS",
        }

    # Save final consolidated report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_report = {
        "remediation_date": pd.Timestamp.now().isoformat(),
        "strategy": "combined_multi_attribute",
        "mitigation_layers": [
            "combined_reweighing (Age+MaritalStatus)",
            "exponentiated_gradient (combined constraint)",
            "per_attribute_threshold_optimizer"
        ],
        "quality_gate_thresholds": FAIRNESS_THRESHOLDS,
        "results": {k: {"before": v["before"], "after": v["after"]} for k, v in combined_results.items()},
    }
    with open(OUT_DIR / "fairness_remediation_final.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    # Overall verdict
    all_pass = all(
        v["eod_pass"] and v["dpd_pass"] for v in combined_results.values()
    )
    print(f"\n  Overall Quality Gate: {'PASSED' if all_pass else 'FAILED'}")
    return combined_results, fair_model, postprocessors


def main():
    print("=" * 60)
    print("  FAIRNESS REMEDIATION v2")
    print("  Per-Attribute Optimal Strategy + Combined Mitigation")
    print("=" * 60)

    print("\n[1/4] Loading data and features...")
    X_train, y_train, X_cal, y_cal, X_test, y_test, raw_df = load_and_prepare()
    print(f"  Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

    print("\n[2/4] Loading hyperparameters...")
    best_params = load_params()

    print("\n[3/4] Per-attribute mitigation...")
    per_attr = run_per_attribute_mitigation(
        X_train, y_train, X_cal, y_cal, X_test, y_test, best_params
    )

    print("\n[4/4] Combined multi-attribute mitigation...")
    combined, fair_model, postprocessors = run_combined_mitigation(
        X_train, y_train, X_cal, y_cal, X_test, y_test, best_params
    )

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print("\n  Per-Attribute Strategy Results:")
    for attr, res in per_attr.items():
        strategy = res["strategy"]
        eod_b = res["before"].get("eod", "N/A")
        eod_a = res["after"].get("eod", "N/A")
        gate = "PASS" if res["gate"] else "FAIL"
        print(f"    {attr} ({strategy}): EOD {eod_b} -> {eod_a} [{gate}]")

    print("\n  Combined Strategy Results:")
    for attr, res in combined.items():
        eod_b = res["before"].get("eod", "N/A")
        eod_a = res["after"].get("eod", "N/A")
        eod_ok = "PASS" if res["eod_pass"] else "FAIL"
        dpd_ok = "PASS" if res["dpd_pass"] else "FAIL"
        print(f"    {attr}: EOD {eod_b} -> {eod_a} [EOD:{eod_ok}] [DPD:{dpd_ok}]")

    print(f"\n  Reports saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
