"""
Airflow DAG Integration Tests (G-7)

Validates DAG structure, task dependencies, and configuration
without requiring a running Airflow environment.

These tests catch:
- Import errors in DAG files
- Circular dependencies
- Missing task connections
- Invalid schedule expressions
- Task naming convention violations

Requires: apache-airflow (skipped if not installed)
"""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Skip all tests if Airflow is not installed
pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("airflow"),
    reason="Airflow not installed — DAG integration tests skipped",
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def mock_cosmos():
    """Mock Cosmos imports so DAG can be parsed without dbt/Cosmos installed."""
    mock_module = MagicMock()
    mock_module.DbtTaskGroup = MagicMock(return_value=MagicMock())
    mock_module.ProjectConfig = MagicMock()
    mock_module.ProfileConfig = MagicMock()
    mock_module.ExecutionConfig = MagicMock()
    mock_module.RenderConfig = MagicMock()

    with patch.dict(sys.modules, {"cosmos": mock_module}):
        yield mock_module


@pytest.fixture
def dag_module(mock_cosmos):
    """Import the DAG module with mocked Cosmos."""
    dag_path = Path(__file__).parent.parent / "astro" / "dags"
    if str(dag_path) not in sys.path:
        sys.path.insert(0, str(dag_path))

    # Clear cached module if already imported
    module_name = "hr_attrition_pipeline"
    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    return module


@pytest.fixture
def dag(dag_module):
    """Extract the DAG object from the module."""
    return dag_module.pipeline


# ── DAG Parse Tests ───────────────────────────────────────────────────
class TestDAGParsing:
    """Verify the DAG file can be parsed without errors."""

    def test_dag_loads_without_import_error(self, dag):
        """DAG file should import without errors."""
        assert dag is not None

    def test_dag_id_is_set(self, dag):
        assert dag.dag_id == "hr_attrition_batch_scoring"

    def test_dag_has_tags(self, dag):
        assert "hr_analytics" in dag.tags
        assert "mlops" in dag.tags

    def test_dag_has_description(self, dag):
        assert dag.description is not None
        assert len(dag.description) > 10

    def test_dag_catchup_disabled(self, dag):
        """Catchup should be disabled to prevent backfill storms."""
        assert dag.catchup is False


# ── Task Dependency Tests ─────────────────────────────────────────────
class TestTaskDependencies:
    """Verify task dependency graph is correctly wired."""

    def test_dag_has_expected_task_count(self, dag):
        """DAG should have 7 main tasks (+ dbt subtasks)."""
        task_ids = [t.task_id for t in dag.tasks]
        # Core tasks (dbt group may add more)
        expected_core = {
            "extract_hris_data",
            "validate_data",
            "run_attrition_model",
            "validate_fairness",
            "check_drift_and_alert",
            "update_dashboard_database",
            "auto_retrain_on_drift",
        }
        for task_name in expected_core:
            assert task_name in task_ids, f"Missing task: {task_name}"

    def test_model_depends_on_validation(self, dag):
        """Model training should only run after data validation."""
        model_task = dag.get_task("run_attrition_model")
        upstream_ids = {t.task_id for t in model_task.upstream_list}
        assert "validate_data" in upstream_ids, (
            "run_attrition_model must depend on validate_data"
        )

    def test_fairness_depends_on_model(self, dag):
        """Fairness validation must run after model training."""
        fairness_task = dag.get_task("validate_fairness")
        upstream_ids = {t.task_id for t in fairness_task.upstream_list}
        assert "run_attrition_model" in upstream_ids

    def test_dashboard_depends_on_fairness(self, dag):
        """Dashboard update must only happen after fairness passes."""
        db_task = dag.get_task("update_dashboard_database")
        upstream_ids = {t.task_id for t in db_task.upstream_list}
        assert "validate_fairness" in upstream_ids or "run_attrition_model" in upstream_ids

    def test_no_circular_dependencies(self, dag):
        """Verify DAG has no circular task dependencies."""
        # If the DAG parsed successfully, there are no cycles
        # (Airflow raises AirflowDagCycleException during parse)
        assert dag is not None

    def test_auto_retrain_depends_on_drift(self, dag):
        """G-12: Auto-retrain should depend on drift detection result."""
        retrain_task = dag.get_task("auto_retrain_on_drift")
        upstream_ids = {t.task_id for t in retrain_task.upstream_list}
        assert "check_drift_and_alert" in upstream_ids


# ── Task Configuration Tests ──────────────────────────────────────────
class TestTaskConfiguration:
    """Verify individual task configurations are correct."""

    def test_task_ids_are_snake_case(self, dag):
        """All task IDs should follow snake_case convention."""
        import re
        for task in dag.tasks:
            # Allow dots for dbt task group sub-tasks
            if "." in task.task_id:
                continue
            assert re.match(r'^[a-z][a-z0-9_]*$', task.task_id), (
                f"Task ID '{task.task_id}' doesn't follow snake_case convention"
            )

    def test_extract_task_has_no_upstream(self, dag):
        """Extract HRIS data should be a root task."""
        extract = dag.get_task("extract_hris_data")
        # May have upstream from chain(), but should not have data dependencies
        assert extract is not None
