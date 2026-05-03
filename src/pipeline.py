"""
Modular ML Pipeline Orchestrator (G-5)

Decomposes the monolithic train_attrition_model.py into named, trackable
pipeline steps. Each step is a self-contained unit that can be:
  - Run independently for debugging
  - Skipped via configuration
  - Monitored with timing and status metadata
  - Extended with new steps without modifying the orchestrator

The original functions in train_attrition_model.py remain unchanged —
this module wraps them into a structured pipeline contract.

Usage:
    from src.pipeline import Pipeline, PipelineConfig
    pipeline = Pipeline(config=PipelineConfig(skip_steps={"survival_analysis"}))
    result = pipeline.run()
"""
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("hr-pipeline")


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""
    name: str
    status: StepStatus
    duration_seconds: float = 0.0
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    data_path: str | None = None
    n_trials: int = 30
    skip_steps: set[str] = field(default_factory=set)
    fail_fast: bool = True  # Stop on first failure
    verbose: bool = True


@dataclass
class PipelineContext:
    """Shared state passed between pipeline steps.

    Each step reads from and writes to this context, enabling
    loose coupling between steps while maintaining data flow.
    """
    # Data artifacts
    raw_df: Any = None
    X: Any = None
    y: Any = None
    X_train: Any = None
    X_cal: Any = None
    X_test: Any = None
    y_train: Any = None
    y_cal: Any = None
    y_test: Any = None

    # Model artifacts
    best_params: dict[str, Any] = field(default_factory=dict)
    optimal_threshold: float = 0.5
    model: Any = None
    calibrated_model: Any = None

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    quality_passed: bool = False
    quality_issues: list[str] = field(default_factory=list)

    # Hashes
    model_hash: str = ""
    data_hash: str = ""


class PipelineStep:
    """A named, executable pipeline step."""

    def __init__(
        self,
        name: str,
        display_name: str,
        func: Callable[[PipelineContext], None],
        step_number: int,
        total_steps: int,
    ):
        self.name = name
        self.display_name = display_name
        self.func = func
        self.step_number = step_number
        self.total_steps = total_steps

    def execute(self, ctx: PipelineContext) -> StepResult:
        """Run this step and return the result."""
        print(f"\n[{self.step_number}/{self.total_steps}] {self.display_name}...")
        start = time.time()
        try:
            self.func(ctx)
            duration = time.time() - start
            print(f"  ✓ Completed in {duration:.1f}s")
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                duration_seconds=round(duration, 2),
            )
        except Exception as e:
            duration = time.time() - start
            logger.error("Step '%s' failed: %s", self.name, e, exc_info=True)
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                duration_seconds=round(duration, 2),
                error=str(e),
            )


@dataclass
class PipelineResult:
    """Overall pipeline execution result."""
    total_duration_seconds: float
    steps: list[StepResult]
    success: bool

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "total_steps": len(self.steps),
            "passed": sum(1 for s in self.steps if s.status == StepStatus.SUCCESS),
            "failed": sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            "skipped": sum(1 for s in self.steps if s.status == StepStatus.SKIPPED),
            "success": self.success,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration": s.duration_seconds,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


class Pipeline:
    """Modular ML pipeline orchestrator.

    Wraps the existing functions from train_attrition_model.py into
    a structured, configurable pipeline with monitoring and error handling.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.context = PipelineContext()
        self._steps: list[PipelineStep] = []

    def _register_steps(self) -> None:
        """Register all pipeline steps from train_attrition_model.py."""
        from src.train_attrition_model import (
            preprocess,
            optimize_hyperparams,
            train_model,
            calibrate_model,
            evaluate,
            evaluate_subpopulations,
            run_fairness_audit,
            explain_model,
            detect_shap_drift,
            build_risk_framework,
            generate_eda,
            generate_drift_report,
            run_adversarial_robustness_test,
            run_causal_validation,
            run_survival_analysis,
            BASE_DIR, DATA_PATH, OUT_DIR, MODEL_DIR,
        )
        from src.features import engineer_features_batch
        from src.data_version import fingerprint_csv
        import pandas as pd
        import hashlib
        import json
        import joblib

        # Store references for step closures
        ctx = self.context
        config = self.config

        def step_load_data(ctx: PipelineContext) -> None:
            data_path = config.data_path or str(DATA_PATH)

            # Try database first, fall back to CSV
            db_url = None
            import os
            db_url = os.getenv("DATABASE_URL")
            if db_url and "sqlite" not in db_url:
                try:
                    from sqlalchemy import create_engine
                    engine = create_engine(db_url)
                    query = "SELECT * FROM marts.fct_attrition_features"
                    ctx.raw_df = pd.read_sql(query, engine)
                    print(f"  Loaded {len(ctx.raw_df)} rows from database mart")
                    return
                except Exception as e:
                    print(f"  DB load failed ({e}), falling back to CSV...")

            ctx.raw_df = pd.read_csv(data_path)
            print(f"  Loaded {len(ctx.raw_df)} rows from {data_path}")

            # Data version fingerprint
            from pathlib import Path
            data_file = Path(data_path)
            if data_file.exists():
                ctx.data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()[:12]

        def step_eda(ctx: PipelineContext) -> None:
            generate_eda(ctx.raw_df)

        def step_feature_engineering(ctx: PipelineContext) -> None:
            ctx.raw_df, pop_stats = engineer_features_batch(ctx.raw_df)

            # Save population stats for inference
            stats_path = MODEL_DIR / "population_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(pop_stats, f, indent=2)
            print(f"  Saved population stats to {stats_path}")

        def step_preprocess(ctx: PipelineContext) -> None:
            ctx.X, ctx.y = preprocess(ctx.raw_df)

            # 3-way stratified split: 60% train / 20% calibration / 20% test
            from sklearn.model_selection import train_test_split
            X_dev, ctx.X_test, y_dev, ctx.y_test = train_test_split(
                ctx.X, ctx.y, test_size=0.2, stratify=ctx.y, random_state=42
            )
            ctx.X_train, ctx.X_cal, ctx.y_train, ctx.y_cal = train_test_split(
                X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
            )
            print(f"  Train: {len(ctx.X_train)} | Cal: {len(ctx.X_cal)} | Test: {len(ctx.X_test)}")

        def step_hyperparameter_optimization(ctx: PipelineContext) -> None:
            ctx.best_params = optimize_hyperparams(ctx.X_train, ctx.y_train, n_trials=config.n_trials)
            ctx.optimal_threshold = ctx.best_params.pop("_optimal_threshold", 0.5)

        def step_train(ctx: PipelineContext) -> None:
            ctx.model = train_model(ctx.X_train, ctx.y_train, ctx.best_params)
            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            ctx.model.save_model(str(MODEL_DIR / "xgb_attrition.json"))
            ctx.model_hash = hashlib.sha256(
                (MODEL_DIR / "xgb_attrition.json").read_bytes()
            ).hexdigest()[:12]

        def step_calibrate(ctx: PipelineContext) -> None:
            ctx.calibrated_model = calibrate_model(ctx.model, ctx.X_cal, ctx.y_cal)
            joblib.dump(ctx.calibrated_model, str(MODEL_DIR / "xgb_calibrated.joblib"))

        def step_evaluate(ctx: PipelineContext) -> None:
            ctx.metrics = evaluate(ctx.calibrated_model, ctx.X_test, ctx.y_test, ctx.optimal_threshold)

        def step_subpopulations(ctx: PipelineContext) -> None:
            evaluate_subpopulations(ctx.calibrated_model, ctx.X_test, ctx.y_test, ctx.raw_df, ctx.X.columns)

        def step_fairness_mitigation(ctx: PipelineContext) -> None:
            """Three-layer fairness mitigation: reweighing → constrained learning → threshold optimization."""
            from src.fairness_mitigation import FairnessMitigator
            mitigator = FairnessMitigator(sensitive_feature_name="Age_Group")
            fairness_result = mitigator.run(
                ctx.X_train, ctx.y_train,
                ctx.X_cal, ctx.y_cal,
                ctx.X_test, ctx.y_test,
                base_params=ctx.best_params,
                raw_df=ctx.raw_df,
            )
            # Store fair model for downstream steps
            if fairness_result.fair_model is not None:
                ctx.calibrated_model = fairness_result.fair_model
            ctx.quality_passed = fairness_result.quality_gate_passed
            ctx.quality_issues = fairness_result.issues

        def step_fairness(ctx: PipelineContext) -> None:
            run_fairness_audit(ctx.calibrated_model, ctx.X_test, ctx.y_test, ctx.raw_df, ctx.X.columns, ctx.optimal_threshold)

        def step_shap(ctx: PipelineContext) -> None:
            explain_model(ctx.model, ctx.X_train, ctx.X_test)

        def step_shap_drift(ctx: PipelineContext) -> None:
            shap_csv = OUT_DIR / "shap_global_importance.csv"
            if shap_csv.exists():
                current_importance = pd.read_csv(shap_csv)
                detect_shap_drift(current_importance)
            else:
                print("  [SKIP] No SHAP importance CSV found")

        def step_risk_framework(ctx: PipelineContext) -> None:
            build_risk_framework(ctx.calibrated_model, ctx.X_test, ctx.y_test, ctx.raw_df, ctx.X.columns)

        def step_drift_report(ctx: PipelineContext) -> None:
            generate_drift_report(ctx.X_train, ctx.X_test)

        def step_adversarial(ctx: PipelineContext) -> None:
            run_adversarial_robustness_test(ctx.model, ctx.X_test, ctx.y_test)

        def step_causal(ctx: PipelineContext) -> None:
            run_causal_validation(ctx.raw_df)

        def step_survival(ctx: PipelineContext) -> None:
            run_survival_analysis(ctx.raw_df)

        # Register steps in execution order
        steps_def = [
            ("load_data", "Loading Data", step_load_data),
            ("eda", "Exploratory Data Analysis", step_eda),
            ("feature_engineering", "Feature Engineering (SSoT)", step_feature_engineering),
            ("preprocess", "Preprocessing & Splitting", step_preprocess),
            ("hpo", "Hyperparameter Optimization (Optuna)", step_hyperparameter_optimization),
            ("train", "Training XGBoost Model", step_train),
            ("calibrate", "Probability Calibration (Platt Scaling)", step_calibrate),
            ("fairness_mitigation", "Fairness Mitigation (3-Layer)", step_fairness_mitigation),
            ("evaluate", "Model Evaluation", step_evaluate),
            ("subpopulations", "Subpopulation Analysis", step_subpopulations),
            ("fairness_audit", "Fairness Audit (Fairlearn)", step_fairness),
            ("shap", "SHAP Explainability", step_shap),
            ("shap_drift", "SHAP Attribution Drift", step_shap_drift),
            ("risk_framework", "Risk Framework & Cost Model", step_risk_framework),
            ("drift_report", "Data Drift Report (Evidently)", step_drift_report),
            ("adversarial", "Adversarial Robustness (Art. 15)", step_adversarial),
            ("causal", "Causal Validation (DoWhy)", step_causal),
            ("survival", "Survival Analysis (Cox PH)", step_survival),
        ]

        total = len(steps_def)
        for i, (name, display, func) in enumerate(steps_def, 1):
            self._steps.append(PipelineStep(name, display, func, i, total))

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        self._register_steps()
        results: list[StepResult] = []
        start = time.time()

        print("=" * 60)
        print("  HR Attrition Intelligence — Modular Pipeline")
        print(f"  Steps: {len(self._steps)} | Skip: {self.config.skip_steps or 'none'}")
        print("=" * 60)

        for step in self._steps:
            if step.name in self.config.skip_steps:
                print(f"\n[{step.step_number}/{step.total_steps}] {step.display_name}... SKIPPED")
                results.append(StepResult(name=step.name, status=StepStatus.SKIPPED))
                continue

            result = step.execute(self.context)
            results.append(result)

            if result.status == StepStatus.FAILED and self.config.fail_fast:
                print(f"\n  ✗ Pipeline halted at step '{step.name}': {result.error}")
                break

        total_duration = time.time() - start
        success = all(r.status in (StepStatus.SUCCESS, StepStatus.SKIPPED) for r in results)

        pipeline_result = PipelineResult(
            total_duration_seconds=total_duration,
            steps=results,
            success=success,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)
        summary = pipeline_result.summary
        print(f"  Duration: {summary['total_duration_seconds']:.1f}s")
        print(f"  Passed: {summary['passed']}/{summary['total_steps']}")
        if summary['failed'] > 0:
            print(f"  Failed: {summary['failed']}")
        if summary['skipped'] > 0:
            print(f"  Skipped: {summary['skipped']}")
        print(f"  Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
        print("=" * 60)

        return pipeline_result
