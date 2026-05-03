# Power BI Dashboard Architecture & DAX Formulas
**AI-Powered Employee Attrition Intelligence**

This document serves as the technical specification for building the Power BI Dashboard to visualize the XGBoost Attrition Model outputs.

---

## 1. Data Model Schema (Star Schema)

To ensure high performance and seamless interactive filtering, structure your Power BI data model into the following Fact and Dimension tables.

### Dimension Tables (Lookup Tables)
*   **`Dim_Employee`**: `EmployeeID`, `Age`, `Gender`, `Education`, `MaritalStatus`.
*   **`Dim_Job`**: `JobRole`, `Department`, `JobLevel`.
*   **`Dim_Date`**: Standard calendar table for tracking tenure and time-series metrics.

### Fact Tables (Data Tables)
*   **`Fact_EmployeeMetrics`**: `EmployeeID`, `MonthlyIncome`, `OverTime`, `DistanceFromHome`, `Compa_Ratio`, `Burnout_Risk`.
*   **`Fact_RiskScores`** *(Output from Python)*: `EmployeeID`, `Predicted_Probability`, `Risk_Tier`, `Expected_Loss`.
*   **`Fact_SHAP`** *(Output from Python)*: Unpivoted table containing `EmployeeID`, `Feature_Name`, `SHAP_Value`. (Used for the interactive Driver Analysis Tornado chart).

### RLS Tables
*   **`manager_employee_map`**: Mapping table for Row-Level Security (`manager_email`, `employee_id`).

### Relationships
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_EmployeeMetrics[EmployeeID]`
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_RiskScores[EmployeeID]`
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_SHAP[EmployeeID]`
*   `manager_employee_map[employee_id]` (*) <---> (1) `Dim_Employee[EmployeeID]`

---

## 2. Core DAX Measures (KPIs)

Create a dedicated "Measures Table" in Power BI to store these calculations.

### High-Level Executive KPIs

```dax
// 1. Total Active Headcount
Total Headcount = DISTINCTCOUNT(Fact_EmployeeMetrics[EmployeeID])

// 2. High-Risk Employees
High Risk Headcount = 
CALCULATE(
    [Total Headcount],
    Fact_RiskScores[Risk_Tier] = "High"
)

// 3. Predicted Attrition Rate (%)
Predicted Attrition Rate = 
DIVIDE([High Risk Headcount], [Total Headcount], 0)

// 4. Total Value of Talent at Risk ($)
Total Value at Risk = SUM(Fact_RiskScores[Expected_Loss])
```

### Risk Segmentation & Averages

```dax
// 5. Average Flight Risk Probability
Average Flight Risk = AVERAGE(Fact_RiskScores[Predicted_Probability])

// 6. Average Compa-Ratio
Average Compa Ratio = AVERAGE(Fact_EmployeeMetrics[Compa_Ratio])

// 7. Average Burnout Risk
Average Burnout Risk = AVERAGE(Fact_EmployeeMetrics[Burnout_Risk])
```

### Advanced Intelligence (What-If Parameter)
To allow HR to simulate interventions (e.g., "If we have a budget of $X, how much risk can we mitigate?"), create a What-If Parameter for `Retention Budget`.

```dax
// Assuming a parameter 'Retention Budget'[Retention Budget Value] exists
Expected Loss After Intervention = 
VAR CurrentLoss = [Total Value at Risk]
VAR Budget = 'Retention Budget'[Retention Budget Value]
// Assuming every $5,000 spent saves one high-risk employee's replacement cost
VAR EmployeesSaved = DIVIDE(Budget, 5000, 0)
VAR LossAvoided = EmployeesSaved * AVERAGE(Fact_RiskScores[Replacement_Cost])
RETURN 
MAX(0, CurrentLoss - LossAvoided)
```

---

## 3. Visualizations & Page Layout

### Page 1: Executive Overview
*   **Visuals**: KPI Scorecards (`Total Headcount`, `Predicted Attrition Rate`, `Total Value at Risk`).
*   **Chart**: Donut Chart showing `Headcount by Risk_Tier`.
*   **Chart**: Bar Chart showing `Total Value at Risk by Department`.

### Page 2: The "Flight Risk" Matrix (Segmenting Talent)
*   **Visual**: Scatter Plot.
    *   **X-Axis**: `PerformanceRating` (from Fact_EmployeeMetrics)
    *   **Y-Axis**: `Predicted_Probability` (from Fact_RiskScores)
    *   **Details**: `EmployeeID`
    *   **Color**: `Risk_Tier` (Red for High, Yellow for Medium, Green for Low)
    *   *Actionable Insight*: HR focuses entirely on the Top-Right quadrant (High Performers who are highly likely to leave).

### Page 3: Driver Analysis (The "Why")
*   **Visual**: Tornado Chart (or Clustered Bar Chart).
    *   **Y-Axis**: `Feature_Name` (from Fact_SHAP)
    *   **X-Axis**: Average `SHAP_Value` (Absolute value to show magnitude)
    *   *Actionable Insight*: This acts as the Global SHAP plot, showing HR leadership systemic issues like `OverTime` driving attrition.

### Page 4: Manager Action Center (Drill-Through)
*   Set up Drill-Through on `EmployeeID`.
*   **Visual**: Multi-row card showing Employee Name, Role, Current Salary, and Compa-Ratio.
*   **Visual**: Waterfall chart showing individual `SHAP_Values` for that specific employee (Local Interpretability).
*   **Visual**: Table recommending precise HR Actions based on their top SHAP driver.

---

## 4. Row-Level Security (RLS)

Because attrition risk is highly sensitive, different organizational roles should see only the data relevant to their scope.

### 4.1 Prerequisites

Before configuring RLS in Power BI, seed the mapping table:

```bash
# Generate the manager → employee hierarchy from the IBM HR dataset
python scripts/seed_rls_mapping.py

# This creates:
#   1. PostgreSQL: powerbi.manager_employee_map (if DB is running)
#   2. CSV: outputs/rls_manager_employee_map.csv (always)
```

The seed script automatically generates:
- **Department Heads**: Highest `JobLevel` per `Department` → sees all dept employees
- **Team Leads**: Highest `JobLevel` per `Department+JobRole` → sees role-group employees
- **Executive**: `executive@company.com` → sees ALL employees (super-user)

### 4.2 Power BI Desktop Configuration

#### Step 1: Import or Connect the Mapping Table

**Option A — DirectQuery (Recommended for production)**
1. In Power BI Desktop, **Get Data → PostgreSQL → DirectQuery**
2. Select `powerbi.rls_filtered_risk_scores` instead of `fact_risk_scores`
3. This view pre-joins the mapping table, so RLS filters automatically

**Option B — Import Mode (works without PostgreSQL)**
1. Import `outputs/rls_manager_employee_map.csv` as a new table
2. Create relationship: `manager_employee_map[employee_id]` → `Dim_Employee[EmployeeID]` (Many-to-One)

#### Step 2: Create RLS Roles

Go to **Modeling → Manage Roles** and create 4 roles:

| Role | DAX Filter Table | DAX Filter Expression | Description |
|------|-----------------|----------------------|-------------|
| `Executive` | — *(no filter)* | — | Full access to all data |
| `DepartmentHead` | `manager_employee_map` | `[manager_email] = USERPRINCIPALNAME()` | Department-scoped access |
| `LineManager` | `manager_employee_map` | `[manager_email] = USERPRINCIPALNAME()` | Team-scoped access |
| `HRAnalyst` | `fact_risk_scores` | `[monthly_income] = BLANK()` | Aggregates only, no individual financial data |

#### Step 3: DAX Filter Expressions

```dax
// ── Role: DepartmentHead ──────────────────────────────────────
// Applied on: manager_employee_map table
// Effect: Filters fact tables through the relationship chain
//   manager_employee_map → Dim_Employee → Fact_RiskScores
[manager_email] = USERPRINCIPALNAME()

// ── Role: LineManager ─────────────────────────────────────────
// Same filter, different scope (seed script controls which employees
// each manager sees — Team Leads see fewer than Dept Heads)
[manager_email] = USERPRINCIPALNAME()

// ── Role: HRAnalyst ───────────────────────────────────────────
// Applied on: fact_risk_scores table
// Masks individual financial data — analysts see risk tiers and
// probabilities but not salaries or exact dollar amounts
[monthly_income] = BLANK()

// ── Role: Executive ───────────────────────────────────────────
// No DAX filter — full access to all rows.
// The executive@company.com account is also mapped to ALL employees
// in the mapping table for DirectQuery compatibility.
```

#### Step 4: Test Roles Locally

1. Go to **Modeling → View as Roles**
2. Select a role (e.g., `LineManager`)
3. Enter a test email: pick one from the mapping CSV (e.g., `emp1171.sales.executive@sales.company.com`)
4. Verify the dashboard shows only that manager's employees

#### Step 5: Publish and Assign

1. **Publish** the report to Power BI Service
2. Navigate to the **Dataset → Security** tab
3. Assign Azure AD users/groups to each role:

| Azure AD Group | Power BI RLS Role | Access Scope |
|----------------|-------------------|--------------|
| `HR-Executive` | Executive | All employees, all metrics |
| `HR-DeptHeads` | DepartmentHead | Department employees only |
| `HR-TeamLeads` | LineManager | Direct reports only |
| `HR-Analysts`  | HRAnalyst | Aggregate metrics, no PII/salary |

### 4.3 RLS Data Flow (DirectQuery Mode)

```
User Login (Azure AD)
  │
  ├─ USERPRINCIPALNAME() = "jane.doe@company.com"
  │
  ├─ Power BI RLS Filter on manager_employee_map:
  │   [manager_email] = "jane.doe@company.com"
  │
  ├─ Relationship chain filters:
  │   manager_employee_map ──→ Dim_Employee ──→ Fact_RiskScores
  │                                          ──→ Fact_SHAP
  │                                          ──→ Fact_EmployeeMetrics
  │
  └─ Result: Jane sees only HER employees' risk data
```

### 4.4 Security Verification Queries

Run these against PostgreSQL to debug RLS scope issues:

```sql
-- Check how many employees each manager sees
SELECT * FROM powerbi.rls_manager_scope LIMIT 20;

-- Verify a specific manager's view
SELECT * FROM powerbi.rls_filtered_risk_scores
WHERE manager_email = 'executive@company.com'
LIMIT 5;

-- Count total mappings
SELECT COUNT(*) AS total_mappings,
       COUNT(DISTINCT manager_email) AS unique_managers,
       COUNT(DISTINCT employee_id) AS unique_employees
FROM powerbi.manager_employee_map;
```

### 4.5 RLS-Protected DAX Measures

These measures automatically respect RLS because they reference filtered tables:

```dax
// ── Scoped KPIs (auto-filtered by RLS) ───────────────────────
// These measures return different values per manager:
// - Executive sees company-wide totals
// - DepartmentHead sees department totals
// - LineManager sees team totals

My Team Headcount = 
DISTINCTCOUNT(rls_filtered_risk_scores[employee_id])

My Team Value at Risk = 
SUM(rls_filtered_risk_scores[expected_loss])

My Team High Risk = 
CALCULATE(
    [My Team Headcount],
    rls_filtered_risk_scores[risk_tier] = "High"
)

My Team Attrition Rate = 
DIVIDE([My Team High Risk], [My Team Headcount], 0)

// ── Manager Context Card ──────────────────────────────────────
// Shows "You are viewing: <X> employees across <Y> departments"
Scope Label = 
VAR EmployeeCount = [My Team Headcount]
VAR DeptCount = DISTINCTCOUNT(rls_filtered_risk_scores[department])
RETURN
"Viewing " & EmployeeCount & " employees across " & DeptCount & " departments"
```

---

## 5. Deployment Checklist

- [ ] Run `python scripts/seed_rls_mapping.py` (populates mapping table + CSV)
- [ ] Run `python scripts/bootstrap_powerbi_db.py` (creates views)
- [ ] Connect Power BI Desktop to PostgreSQL DirectQuery
- [ ] Import `rls_filtered_risk_scores`, `dim_employee`, `dim_job`, `manager_employee_map`
- [ ] Create relationships in Model View
- [ ] Configure 4 RLS roles (Executive, DepartmentHead, LineManager, HRAnalyst)
- [ ] Test with "View as Roles" in Power BI Desktop
- [ ] Publish to Power BI Service
- [ ] Assign Azure AD groups to RLS roles in Dataset Security
- [ ] Verify with test users from each role
