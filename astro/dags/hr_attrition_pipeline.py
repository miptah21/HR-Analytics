"""
HR Attrition Batch Scoring DAG
Orchestrates nightly re-scoring of employee attrition risk.

Uses Python callables instead of subprocess to maintain Airflow's
task isolation, logging, and XCom capabilities.
"""
from airflow.decorators import dag, task
from pendulum import datetime


@dag(
    dag_id="hr_attrition_batch_scoring",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hr_analytics", "mlops"],
    description="Nightly batch scoring of employee attrition risk",
    doc_md="""
    ### HR Attrition Batch Scoring Pipeline

    This DAG runs the full ML pipeline nightly:
    1. Extract live HRIS data
    2. Run feature engineering + model training/scoring
    3. Generate fairness audit
    4. Upsert risk scores to the dashboard database

    **Owner:** HR Analytics Team
    """,
)
def hr_attrition_pipeline():

    @task
    def extract_hris_data() -> bool:
        """Extract live HRIS data from source system using dlt.

        This queries the simulated HRIS API (local CSV) and lands
        the data directly into the `raw_hris` schema in PostgreSQL.
        """
        print("Running dlt pipeline to extract HRIS data...")
        import sys
        from pathlib import Path
        
        # Add include directory to sys.path to ensure module is found
        include_path = str(Path(__file__).parent.parent / "include")
        if include_path not in sys.path:
            sys.path.append(include_path)
            
        from hris_dlt_pipeline import run_pipeline
        load_info = run_pipeline()
        print(f"Data extraction complete: {load_info}")
        return True

    @task
    def run_attrition_model(data_ready: bool) -> str:
        """Execute the ML pipeline using Python imports.

        Calls the main() function directly instead of shelling out
        via subprocess, preserving Airflow's logging and error handling.
        """
        if not data_ready:
            raise ValueError("Data extraction did not succeed.")

        from src.train_attrition_model import main
        main()

        return "outputs/risk_scores.csv"

    @task
    def validate_fairness(scores_path: str) -> bool:
        """Validate that the fairness audit passed acceptable thresholds.

        Acts as a quality gate — if bias exceeds thresholds, the pipeline
        stops before publishing results to the dashboard.
        """
        import json
        from pathlib import Path

        audit_path = Path("outputs/fairness_audit.json")
        if not audit_path.exists():
            print("No fairness audit found — proceeding with caution.")
            return True

        with open(audit_path) as f:
            results = json.load(f)

        for attr, metrics in results.items():
            if not metrics.get("dpd_pass", True):
                raise ValueError(
                    f"Fairness gate FAILED for {attr}: "
                    f"DPD={metrics['demographic_parity_diff']:.4f} exceeds 0.1 threshold."
                )

        print("Fairness validation passed for all protected attributes.")
        return True

    @task
    def update_dashboard_database(
        scores_path: str,
        fairness_ok: bool,
    ) -> None:
        """Upsert new risk scores to the SQL backend driving Power BI.

        Only proceeds if the fairness validation gate passed.
        """
        if not fairness_ok:
            raise ValueError("Cannot publish scores — fairness validation failed.")

        print(f"Upserting {scores_path} to Power BI staging tables...")
        import pandas as pd
        from sqlalchemy import create_engine, text
        import os
        
        # Project root relative to this DAG file (astro/dags/...)
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        abs_scores_path = project_root / scores_path
        
        if not abs_scores_path.exists():
            raise FileNotFoundError(f"Scores file not found at {abs_scores_path}")
            
        df = pd.read_csv(abs_scores_path)
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL environment variable is required.")
        engine = create_engine(db_url)
        
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics;"))
            
        # Write to analytics.predictions, replacing the table for simplicity in this PoC
        df.to_sql("predictions", con=engine, schema="analytics", if_exists="replace", index=False)
        print(f"Successfully upserted {len(df)} records into analytics.predictions")

    from airflow.models.baseoperator import chain
    from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig, RenderConfig
    from pathlib import Path

    _project_config = ProjectConfig(dbt_project_path=str(Path(__file__).parent / "dbt"))
    _profile_config = ProfileConfig(
        profile_name="hr_analytics", 
        target_name="dev", 
        profiles_yml_filepath=str(Path(__file__).parent / "dbt" / "profiles.yml")
    )

    dbt_transform = DbtTaskGroup(
        group_id="dbt_transform",
        project_config=_project_config,
        profile_config=_profile_config,
        execution_config=ExecutionConfig(),
        render_config=RenderConfig(),
    )

    @task
    def validate_data() -> bool:
        """Run Great Expectations data quality checks on the dbt output."""
        print("Running data quality validation...")
        import sys
        from pathlib import Path
        
        include_path = str(Path(__file__).parent.parent / "include")
        if include_path not in sys.path:
            sys.path.append(include_path)
            
        from ge_validation import run_ge_validation
        return run_ge_validation()

    @task
    def check_drift_and_alert(scores_path: str) -> bool:
        """Parse Evidently drift report and determine if retraining is warranted.

        G-12: This now returns a boolean to trigger automated retraining
        when significant drift is detected, closing the observability loop.
        """
        import json
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        drift_path = project_root / "outputs" / "evidently_drift_report.json"
        shap_drift_path = project_root / "outputs" / "shap_drift_report.json"

        needs_retrain = False

        # Check Evidently data drift
        if drift_path.exists():
            with open(drift_path) as f:
                report = json.load(f)

            for metric in report.get("metrics", []):
                result = metric.get("result", {})
                if "dataset_drift" in result:
                    dataset_drift = result["dataset_drift"]
                    drift_share = result.get("drift_share", 0)
                    print(f"  Dataset drift detected: {dataset_drift}")
                    print(f"  Drifted features share: {drift_share:.1%}")
                    if dataset_drift:
                        needs_retrain = True
                    break
        else:
            print("No Evidently drift report found — skipping data drift check.")

        # Check SHAP attribution drift
        if shap_drift_path.exists():
            with open(shap_drift_path) as f:
                shap_report = json.load(f)
            has_shap_drift = shap_report.get("has_drift", False)
            verdict = shap_report.get("verdict", "UNKNOWN")
            print(f"  SHAP drift verdict: {verdict}")
            if has_shap_drift:
                needs_retrain = True
        else:
            print("No SHAP drift report found — skipping attribution drift check.")

        if needs_retrain:
            print("  ⚠ ALERT: Significant drift detected! Auto-retrain will be triggered.")
        else:
            print("  ✓ No significant drift. Current model remains valid.")

        return needs_retrain

    @task
    def auto_retrain_on_drift(needs_retrain: bool, data_ready: bool) -> str | None:
        """G-12: Conditionally trigger model retraining when drift is detected.

        Archives the current model before retraining to enable rollback (G-6).
        Only runs if drift detection flagged significant drift.
        """
        if not needs_retrain:
            print("  ✓ No retraining needed — drift within acceptable bounds.")
            return None

        if not data_ready:
            raise ValueError("Cannot retrain — data validation did not succeed.")

        import shutil
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "xgb_attrition.json"
        calibrated_path = project_root / "models" / "xgb_calibrated.joblib"
        archive_model = project_root / "models" / "xgb_attrition_previous.json"
        archive_calibrated = project_root / "models" / "xgb_calibrated_previous.joblib"

        # G-6: Archive current model for rollback
        if model_path.exists():
            shutil.copy2(model_path, archive_model)
            print(f"  Archived current model to {archive_model.name}")
        if calibrated_path.exists():
            shutil.copy2(calibrated_path, archive_calibrated)
            print(f"  Archived calibrated model to {archive_calibrated.name}")

        # Trigger full retraining
        print("  🔄 Starting drift-triggered retraining with fresh HPO...")
        from src.train_attrition_model import main
        main()

        print("  ✅ Drift-triggered retraining complete.")
        return "outputs/risk_scores.csv"

    extracted = extract_hris_data()
    validated_dq = validate_data()
    
    # Wire data validation into model training (GAP-15 fix)
    model_run = run_attrition_model(validated_dq)
    
    chain(extracted, dbt_transform, validated_dq, model_run)
    
    fairness_ok = validate_fairness(model_run)
    drift_detected = check_drift_and_alert(model_run)
    update_dashboard_database(model_run, fairness_ok)

    # G-12: Auto-retrain if drift detected
    auto_retrain_on_drift(drift_detected, validated_dq)


pipeline = hr_attrition_pipeline()

