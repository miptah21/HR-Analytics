"""
Seed the Power BI Row-Level Security (RLS) mapping table.

Generates a realistic manager → employee hierarchy from the IBM HR dataset:
  - Assigns 1 "Department Head" per Department (highest JobLevel employee)
  - Assigns "Team Leads" per Department+JobRole (second-highest JobLevel)
  - Maps every employee to their Team Lead AND Department Head
  - Creates an "Executive" super-user who sees ALL employees

The mapping is stored in `powerbi.manager_employee_map`.

Usage:
    python scripts/seed_rls_mapping.py

    # Or with a custom database URL:
    DATABASE_URL=postgresql://user:pass@host:5432/db python scripts/seed_rls_mapping.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
from typing import Any


def load_dataset() -> list[dict[str, Any]]:
    """Load employee data from the original IBM HR dataset."""
    csv_path = PROJECT_ROOT / "datasets" / "HR-Employee-Attrition.csv"
    if not csv_path.exists():
        print(f"  [FAIL] Dataset not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Loaded {len(rows)} employees from dataset")
    return rows


def build_hierarchy(employees: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Build a manager-employee mapping based on Department + JobRole + JobLevel.

    Strategy:
      1. Group employees by Department
      2. The employee with the highest JobLevel in each Dept → Department Head
      3. Within each Dept+JobRole, the highest JobLevel → Team Lead
      4. Each employee is mapped to their Team Lead AND Department Head
      5. An "executive@company.com" super-user sees everyone

    Email format: firstname.lastname@company.com (simulated from EmployeeNumber)
    """
    # Group by department
    dept_groups: dict[str, list[dict]] = {}
    for emp in employees:
        dept = emp.get("Department", "Unknown")
        dept_groups.setdefault(dept, []).append(emp)

    mappings: list[dict[str, str]] = []
    dept_heads: dict[str, str] = {}  # dept → head_email
    all_employee_ids: list[str] = []

    for dept, members in dept_groups.items():
        # Sort by JobLevel descending, then by EmployeeNumber for determinism
        members.sort(
            key=lambda e: (-int(e.get("JobLevel", 1)), int(e.get("EmployeeNumber", 0)))
        )

        # Department Head = highest JobLevel in the department
        head = members[0]
        head_id = head["EmployeeNumber"]
        head_email = _employee_email(head)
        dept_heads[dept] = head_email

        # Group by JobRole within department
        role_groups: dict[str, list[dict]] = {}
        for m in members:
            role = m.get("JobRole", "Unknown")
            role_groups.setdefault(role, []).append(m)

        for role, role_members in role_groups.items():
            # Team Lead = highest JobLevel within the role group
            # (already sorted by JobLevel desc)
            lead = role_members[0]
            lead_email = _employee_email(lead)

            for emp in role_members:
                emp_id = emp["EmployeeNumber"]
                all_employee_ids.append(emp_id)

                # Map employee → Team Lead (if not themselves)
                if emp_id != lead["EmployeeNumber"]:
                    mappings.append({
                        "manager_email": lead_email,
                        "employee_id": emp_id,
                    })

                # Map employee → Department Head (if lead isn't the head)
                if lead["EmployeeNumber"] != head_id:
                    mappings.append({
                        "manager_email": head_email,
                        "employee_id": emp_id,
                    })

        # Department Head sees themselves
        mappings.append({
            "manager_email": head_email,
            "employee_id": head_id,
        })

    # Executive super-user sees ALL employees
    executive_email = "executive@company.com"
    for emp_id in all_employee_ids:
        mappings.append({
            "manager_email": executive_email,
            "employee_id": emp_id,
        })

    # Deduplicate
    seen = set()
    unique_mappings = []
    for m in mappings:
        key = (m["manager_email"], m["employee_id"])
        if key not in seen:
            seen.add(key)
            unique_mappings.append(m)

    return unique_mappings


def _employee_email(emp: dict) -> str:
    """Generate a deterministic email from employee data."""
    emp_num = emp.get("EmployeeNumber", "0")
    dept = emp.get("Department", "unknown").lower().replace(" ", "-").replace("&", "and")
    role = emp.get("JobRole", "unknown").lower().replace(" ", ".").replace("-", ".")
    return f"emp{emp_num}.{role}@{dept}.company.com"


def seed_to_postgres(mappings: list[dict[str, str]]) -> int:
    """Write mappings to powerbi.manager_employee_map in PostgreSQL."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("  [FAIL] sqlalchemy not installed. Run: uv pip install sqlalchemy psycopg2-binary")
        sys.exit(1)

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://hr_admin:change-me-in-production@localhost:5432/hr_analytics",
    )
    engine = create_engine(db_url)

    with engine.begin() as conn:
        # Ensure schema exists
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS powerbi;"))

        # Create table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS powerbi.manager_employee_map (
                manager_email TEXT NOT NULL,
                employee_id   TEXT NOT NULL,
                PRIMARY KEY (manager_email, employee_id)
            );
        """))

        # Clear existing mappings
        conn.execute(text("TRUNCATE TABLE powerbi.manager_employee_map;"))

        # Batch insert
        for m in mappings:
            conn.execute(
                text(
                    "INSERT INTO powerbi.manager_employee_map (manager_email, employee_id) "
                    "VALUES (:email, :eid) ON CONFLICT DO NOTHING"
                ),
                {"email": m["manager_email"], "eid": m["employee_id"]},
            )

    print(f"  [OK] powerbi.manager_employee_map: {len(mappings)} rows inserted")
    return len(mappings)


def seed_to_csv(mappings: list[dict[str, str]]) -> Path:
    """Write mappings to a CSV file (for non-PostgreSQL setups or Power BI Import)."""
    out_path = PROJECT_ROOT / "outputs" / "rls_manager_employee_map.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["manager_email", "employee_id"])
        writer.writeheader()
        writer.writerows(mappings)

    print(f"  [OK] CSV exported: {out_path} ({len(mappings)} rows)")
    return out_path


def print_summary(mappings: list[dict[str, str]]) -> None:
    """Print a summary of the generated hierarchy."""
    managers = set(m["manager_email"] for m in mappings)
    employees = set(m["employee_id"] for m in mappings)

    # Count reports per manager
    report_counts: dict[str, int] = {}
    for m in mappings:
        report_counts[m["manager_email"]] = report_counts.get(m["manager_email"], 0) + 1

    # Separate executive from others
    exec_email = "executive@company.com"
    line_managers = {k: v for k, v in report_counts.items() if k != exec_email}

    print(f"\n  -- RLS Hierarchy Summary --")
    print(f"  Total mappings:      {len(mappings)}")
    print(f"  Unique managers:     {len(managers)} (incl. executive)")
    print(f"  Unique employees:    {len(employees)}")
    print(f"  Line managers:       {len(line_managers)}")
    print(f"  Executive sees:      {report_counts.get(exec_email, 0)} employees")
    print(f"  Avg reports/manager: {sum(line_managers.values()) / max(len(line_managers), 1):.1f}")

    # Show top 5 managers by report count
    top = sorted(line_managers.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 managers by report count:")
    for email, count in top:
        print(f"    {email}: {count} reports")


def main():
    print("=" * 60)
    print("  RLS Mapping Seed Generator")
    print("=" * 60)

    # Load dataset
    print("\n-- Step 1: Load employee dataset --")
    employees = load_dataset()

    # Build hierarchy
    print("\n-- Step 2: Build manager-employee hierarchy --")
    mappings = build_hierarchy(employees)
    print(f"  Generated {len(mappings)} manager-employee mappings")

    # Always export CSV (works without PostgreSQL)
    print("\n-- Step 3: Export CSV --")
    seed_to_csv(mappings)

    # Try PostgreSQL if available
    print("\n-- Step 4: Seed PostgreSQL --")
    try:
        seed_to_postgres(mappings)
    except Exception as e:
        print(f"  [SKIP] PostgreSQL not available: {e}")
        print("  Use the CSV file to import into Power BI directly.")

    # Summary
    print_summary(mappings)

    print("\n" + "=" * 60)
    print("  Seed complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
