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

### Relationships
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_EmployeeMetrics[EmployeeID]`
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_RiskScores[EmployeeID]`
*   `Dim_Employee[EmployeeID]` (1) <---> (*) `Fact_SHAP[EmployeeID]`

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

Because attrition risk is highly sensitive, Line Managers should only see data for their direct reports.

1.  In Power BI Desktop, go to **Modeling > Manage Roles**.
2.  Create a role called `LineManager`.
3.  Apply the following DAX filter on the `Dim_Job` or `Dim_Employee` table:
    ```dax
    [ManagerEmail] = USERPRINCIPALNAME()
    ```
4.  Publish to Power BI Service and assign managers to this role. HR Directors can be assigned an `Executive` role with no filters.
