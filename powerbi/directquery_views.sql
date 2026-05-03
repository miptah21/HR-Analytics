-- ============================================================
-- Power BI DirectQuery Views (G-13)
-- ============================================================
-- These materialized views provide a clean, pre-optimized
-- interface for Power BI DirectQuery connections.
--
-- Run this SQL against the PostgreSQL analytics database:
--   psql -U hr_admin -d hr_analytics -f powerbi/directquery_views.sql
--
-- Power BI Connection:
--   Server:   <postgres-host> port 5432
--   Database: hr_analytics
--   Schema:   powerbi
-- ============================================================

CREATE SCHEMA IF NOT EXISTS powerbi;

-- ── Dim_Employee ─────────────────────────────────────────────────────
-- Dimension table for employee demographics
CREATE OR REPLACE VIEW powerbi.dim_employee AS
SELECT DISTINCT
    p."EmployeeNumber"::text AS employee_id,
    p."Age"::int AS age,
    p."Gender" AS gender,
    p."Education"::int AS education,
    p."MaritalStatus" AS marital_status,
    CASE
        WHEN p."Age"::int BETWEEN 18 AND 30 THEN '18-30'
        WHEN p."Age"::int BETWEEN 31 AND 40 THEN '31-40'
        WHEN p."Age"::int BETWEEN 41 AND 50 THEN '41-50'
        ELSE '51+'
    END AS age_group
FROM analytics.predictions p
WHERE p."EmployeeNumber" IS NOT NULL;


-- ── Dim_Job ──────────────────────────────────────────────────────────
-- Dimension table for job details
CREATE OR REPLACE VIEW powerbi.dim_job AS
SELECT DISTINCT
    p."JobRole" AS job_role,
    p."Department" AS department,
    p."JobLevel"::int AS job_level
FROM analytics.predictions p
WHERE p."JobRole" IS NOT NULL;


-- ── Fact_RiskScores ──────────────────────────────────────────────────
-- Core fact table for Power BI dashboards
CREATE OR REPLACE VIEW powerbi.fact_risk_scores AS
SELECT
    p."EmployeeNumber"::text AS employee_id,
    p."Department" AS department,
    p."JobRole" AS job_role,
    p."Predicted_Probability" AS predicted_probability,
    p."Risk_Tier" AS risk_tier,
    p."MonthlyIncome" AS monthly_income,
    p."MonthlyIncome" * 12 AS annual_salary,
    p."MonthlyIncome" * 12 * 1.5 AS replacement_cost,
    p."Expected_Loss" AS expected_loss,
    p."Actual"::int AS actual_attrition,
    p."OverTime" AS over_time,
    p."YearsAtCompany"::int AS years_at_company,
    p."JobSatisfaction"::int AS job_satisfaction,
    p."EnvironmentSatisfaction"::int AS environment_satisfaction,
    p."WorkLifeBalance"::int AS work_life_balance
FROM analytics.predictions p;


-- ── Fact_AuditTrail ──────────────────────────────────────────────────
-- Prediction audit trail for compliance reporting
CREATE OR REPLACE VIEW powerbi.fact_audit_trail AS
SELECT
    pl.id AS prediction_id,
    pl.timestamp AS prediction_timestamp,
    DATE(pl.timestamp) AS prediction_date,
    pl.employee_id,
    pl.requester_role,
    pl.attrition_probability,
    pl.risk_tier,
    pl.model_version,
    pl.generated_strategy IS NOT NULL AS has_strategy
FROM prediction_logs pl
ORDER BY pl.timestamp DESC;


-- ── Fact_Overrides ───────────────────────────────────────────────────
-- Human override history for compliance dashboards
CREATE OR REPLACE VIEW powerbi.fact_overrides AS
SELECT
    ho.id AS override_id,
    ho.timestamp AS override_timestamp,
    DATE(ho.timestamp) AS override_date,
    ho.employee_id,
    ho.original_risk_tier,
    ho.override_risk_tier,
    ho.override_reason,
    ho.overridden_by,
    CASE
        WHEN ho.original_risk_tier = 'High' AND ho.override_risk_tier = 'Low' THEN 'Downgrade (2 levels)'
        WHEN ho.original_risk_tier = 'High' AND ho.override_risk_tier = 'Medium' THEN 'Downgrade (1 level)'
        WHEN ho.original_risk_tier = 'Low' AND ho.override_risk_tier = 'High' THEN 'Upgrade (2 levels)'
        WHEN ho.original_risk_tier = 'Medium' AND ho.override_risk_tier = 'High' THEN 'Upgrade (1 level)'
        ELSE 'Same level'
    END AS override_direction
FROM human_overrides ho
ORDER BY ho.timestamp DESC;


-- ── Aggregate: Risk by Department ────────────────────────────────────
-- Pre-computed aggregation for faster Power BI rendering
CREATE OR REPLACE VIEW powerbi.agg_risk_by_department AS
SELECT
    p."Department" AS department,
    COUNT(*) AS headcount,
    COUNT(*) FILTER (WHERE p."Risk_Tier" = 'High') AS high_risk_count,
    COUNT(*) FILTER (WHERE p."Risk_Tier" = 'Medium') AS medium_risk_count,
    COUNT(*) FILTER (WHERE p."Risk_Tier" = 'Low') AS low_risk_count,
    ROUND(AVG(p."Predicted_Probability")::numeric, 4) AS avg_risk_probability,
    ROUND(SUM(p."Expected_Loss")::numeric, 2) AS total_value_at_risk
FROM analytics.predictions p
GROUP BY p."Department";


-- ── RLS Helper: Manager → Employee mapping ───────────────────────────
-- Power BI Row-Level Security requires a mapping table.
-- Populate this using: python scripts/seed_rls_mapping.py
CREATE TABLE IF NOT EXISTS powerbi.manager_employee_map (
    manager_email TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    PRIMARY KEY (manager_email, employee_id)
);

COMMENT ON TABLE powerbi.manager_employee_map IS
    'Row-Level Security mapping for Power BI. Managers see only their direct reports. '
    'Seed with: python scripts/seed_rls_mapping.py';


-- ── RLS-Filtered Risk Scores ─────────────────────────────────────────
-- This view is the MAIN table for RLS-protected Power BI reports.
-- Instead of connecting to fact_risk_scores directly, connect to this
-- view and apply the DAX filter: [manager_email] = USERPRINCIPALNAME()
--
-- When a manager queries this view, they only see employees assigned
-- to them in the manager_employee_map table.
CREATE OR REPLACE VIEW powerbi.rls_filtered_risk_scores AS
SELECT
    mem.manager_email,
    frs.*
FROM powerbi.fact_risk_scores frs
INNER JOIN powerbi.manager_employee_map mem
    ON frs.employee_id = mem.employee_id;


-- ── RLS Manager Scope ────────────────────────────────────────────────
-- Helper view: shows which employees each manager can see.
-- Useful for debugging RLS issues in Power BI Service.
CREATE OR REPLACE VIEW powerbi.rls_manager_scope AS
SELECT
    mem.manager_email,
    COUNT(DISTINCT mem.employee_id) AS visible_employee_count,
    STRING_AGG(DISTINCT frs.department, ', ' ORDER BY frs.department) AS departments,
    STRING_AGG(DISTINCT frs.risk_tier, ', ' ORDER BY frs.risk_tier) AS risk_tiers
FROM powerbi.manager_employee_map mem
LEFT JOIN powerbi.fact_risk_scores frs
    ON mem.employee_id = frs.employee_id
GROUP BY mem.manager_email
ORDER BY visible_employee_count DESC;


-- ── RLS Audit Trail (filtered) ───────────────────────────────────────
-- Prediction audit trail filtered by manager scope.
CREATE OR REPLACE VIEW powerbi.rls_filtered_audit_trail AS
SELECT
    mem.manager_email,
    fat.*
FROM powerbi.fact_audit_trail fat
INNER JOIN powerbi.manager_employee_map mem
    ON fat.employee_id = mem.employee_id;


-- ── RLS Overrides (filtered) ─────────────────────────────────────────
-- Human override history filtered by manager scope.
CREATE OR REPLACE VIEW powerbi.rls_filtered_overrides AS
SELECT
    mem.manager_email,
    fo.*
FROM powerbi.fact_overrides fo
INNER JOIN powerbi.manager_employee_map mem
    ON fo.employee_id = mem.employee_id;

