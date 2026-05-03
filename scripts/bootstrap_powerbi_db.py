"""
Bootstrap Power BI Database Schema & Views.

Seeds the PostgreSQL database with:
1. analytics.predictions   — from risk_scores.csv + original dataset columns
2. prediction_logs         — empty table from SQLAlchemy ORM
3. human_overrides         — empty table from SQLAlchemy ORM
4. scoring_exclusions      — empty table from SQLAlchemy ORM
5. powerbi.*               — DirectQuery views via SQL file

Usage:
    python scripts/bootstrap_powerbi_db.py
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import create_engine, text


def get_engine():
    """Create SQLAlchemy engine from DATABASE_URL."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://hr_admin:change-me-in-production@localhost:5432/hr_analytics",
    )
    return create_engine(db_url)


def seed_predictions(engine) -> int:
    """Load risk_scores.csv into analytics.predictions, enriched with original dataset columns."""
    risk_path = PROJECT_ROOT / "outputs" / "risk_scores.csv"
    if not risk_path.exists():
        print(f"  [SKIP] {risk_path} not found — run the ML pipeline first.")
        return 0

    risk_df = pd.read_csv(risk_path)

    # Try to enrich with columns the DirectQuery views expect
    # (OverTime, YearsAtCompany, JobSatisfaction, EnvironmentSatisfaction,
    #  WorkLifeBalance, JobLevel, Age, Gender, Education, MaritalStatus)
    original_data_path = PROJECT_ROOT / "datasets" / "HR-Employee-Attrition.csv"
    if original_data_path.exists() and "EmployeeNumber" in risk_df.columns:
        original_df = pd.read_csv(original_data_path)
        extra_cols = [
            "EmployeeNumber", "OverTime", "YearsAtCompany",
            "JobSatisfaction", "EnvironmentSatisfaction",
            "WorkLifeBalance", "JobLevel", "Age", "Gender",
            "Education", "MaritalStatus",
        ]
        available = [c for c in extra_cols if c in original_df.columns]
        merge_df = original_df[available].drop_duplicates(subset=["EmployeeNumber"])
        before = len(risk_df.columns)
        risk_df = risk_df.merge(merge_df, on="EmployeeNumber", how="left", suffixes=("", "_orig"))
        # Drop any _orig duplicate columns
        risk_df = risk_df[[c for c in risk_df.columns if not c.endswith("_orig")]]
        print(f"  Enriched predictions: {before} → {len(risk_df.columns)} columns")

    # Also build the input_features JSON column for dim_employee view
    feature_cols = ["Age", "Gender", "Education", "MaritalStatus"]
    available_feature_cols = [c for c in feature_cols if c in risk_df.columns]
    if available_feature_cols:
        import json
        risk_df["input_features"] = risk_df[available_feature_cols].apply(
            lambda row: json.dumps(row.to_dict()), axis=1
        )

    # Create schema and write
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics;"))

    risk_df.to_sql(
        "predictions", engine, schema="analytics",
        if_exists="replace", index=False,
    )
    print(f"  [OK] analytics.predictions: {len(risk_df)} rows")
    return len(risk_df)


def create_orm_tables(engine) -> None:
    """Create prediction_logs, human_overrides, scoring_exclusions from ORM."""
    from src.models import Base
    Base.metadata.create_all(bind=engine)
    print("  [OK] ORM tables created (prediction_logs, human_overrides, scoring_exclusions)")


def run_directquery_views(engine) -> None:
    """Execute the DirectQuery views SQL file."""
    sql_path = PROJECT_ROOT / "powerbi" / "directquery_views.sql"
    if not sql_path.exists():
        print(f"  [SKIP] {sql_path} not found.")
        return

    sql_content = sql_path.read_text(encoding="utf-8")

    # Split into individual statements and execute each
    statements = [s.strip() for s in sql_content.split(";") if s.strip()]
    success, errors = 0, 0
    for stmt in statements:
        # Skip pure comment blocks
        lines = [l for l in stmt.split("\n") if l.strip() and not l.strip().startswith("--")]
        if not lines:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt + ";"))
            success += 1
        except Exception as e:
            errors += 1
            first_line = lines[0] if lines else stmt[:60]
            print(f"  [WARN] {first_line[:70]}... -> {e}")

    print(f"  [OK] DirectQuery views: {success} succeeded, {errors} failed")


def seed_rls_mapping(engine) -> None:
    """Seed the RLS manager-employee mapping table from the HR dataset."""
    try:
        from scripts.seed_rls_mapping import load_dataset, build_hierarchy, seed_to_csv
    except ImportError:
        # Direct import if running from project root
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seed_rls_mapping",
            str(PROJECT_ROOT / "scripts" / "seed_rls_mapping.py"),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            load_dataset = mod.load_dataset
            build_hierarchy = mod.build_hierarchy
            seed_to_csv = mod.seed_to_csv
        else:
            print("  [SKIP] Could not import seed_rls_mapping module")
            return

    employees = load_dataset()
    mappings = build_hierarchy(employees)

    # Always export CSV
    seed_to_csv(mappings)

    # Seed to PostgreSQL
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS powerbi;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS powerbi.manager_employee_map (
                manager_email TEXT NOT NULL,
                employee_id   TEXT NOT NULL,
                PRIMARY KEY (manager_email, employee_id)
            );
        """))
        conn.execute(text("TRUNCATE TABLE powerbi.manager_employee_map;"))

        for m in mappings:
            conn.execute(
                text(
                    "INSERT INTO powerbi.manager_employee_map (manager_email, employee_id) "
                    "VALUES (:email, :eid) ON CONFLICT DO NOTHING"
                ),
                {"email": m["manager_email"], "eid": m["employee_id"]},
            )

    print(f"  [OK] powerbi.manager_employee_map: {len(mappings)} rows")


def verify(engine) -> None:
    """Quick verification of what's in the database."""
    with engine.connect() as conn:
        # List schemas
        schemas = conn.execute(text(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('pg_catalog','pg_toast','information_schema') "
            "ORDER BY schema_name"
        )).fetchall()
        print(f"\n  Schemas: {[s[0] for s in schemas]}")

        # List user tables and views
        objects = conn.execute(text(
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog','pg_toast','information_schema') "
            "ORDER BY table_schema, table_type, table_name"
        )).fetchall()

        tables = [o for o in objects if o[2] == "BASE TABLE"]
        views = [o for o in objects if o[2] == "VIEW"]
        print(f"  Tables ({len(tables)}):")
        for t in tables:
            row_count = conn.execute(text(f'SELECT COUNT(*) FROM "{t[0]}"."{t[1]}"')).scalar()
            print(f"    {t[0]}.{t[1]}: {row_count} rows")
        print(f"  Views ({len(views)}):")
        for v in views:
            print(f"    {v[0]}.{v[1]}")


def main():
    print("=" * 60)
    print("  Power BI Database Bootstrap")
    print("=" * 60)

    engine = get_engine()

    # Test connectivity
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
            print(f"\n  Connected: {version[:60]}...")
    except Exception as e:
        print(f"\n  [FAIL] Connection failed: {e}")
        print("  Make sure PostgreSQL is running (podman compose up -d postgres)")
        sys.exit(1)

    print("\n── Step 1: Seed analytics.predictions ──")
    seed_predictions(engine)

    print("\n── Step 2: Create ORM tables ──")
    create_orm_tables(engine)

    print("\n── Step 3: Create DirectQuery views ──")
    run_directquery_views(engine)

    print("\n── Step 4: Seed RLS mapping ──")
    seed_rls_mapping(engine)

    print("\n── Verification ──")
    verify(engine)

    print("\n" + "=" * 60)
    print("  Bootstrap complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
