# AI-Powered Employee Attrition Intelligence System

> ⚠️ **Proof-of-Concept Notice:** This system is built on the [IBM HR Analytics Employee Attrition dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset), which is a **fictional, synthetically generated dataset** (1,470 rows). All metrics, fairness audits, and model outputs are demonstrations of MLOps architecture and should not be used for real employment decisions without integration of genuine HRIS data.

## Overview

An end-to-end Machine Learning system demonstrating production-grade MLOps architecture for predicting employee attrition risk, calculating expected financial impact, and generating retention scenarios using generative AI.

The system implements **2026 MLOps standards**, **Enterprise Security Guidelines**, and **EU AI Act compliance patterns (Annex III: High-Risk AI)**, with a **three-layer fairness remediation pipeline** ensuring equitable predictions across protected demographic groups.

---

## Key Features

### Machine Learning & Analytics
*   **Predictive Modeling:** XGBoost classifier with Optuna Bayesian HPO (30 trials), isotonic probability calibration, and optimal threshold selection via F2 score.
*   **Financial Cost Model:** Translates probabilities into "Value at Risk" using 1.5× annual salary replacement cost, producing per-employee `Expected_Loss` figures.
*   **Local Explainability:** SHAP TreeExplainer provides per-prediction "top risk drivers" with global summary and bar importance plots.
*   **Causal Validation (DoWhy):** Validates that SHAP-identified drivers (e.g., OverTime) have genuine causal support via backdoor adjustment + random common cause refutation.
*   **Survival Analysis (Lifelines):** Cox Proportional Hazards model answers "when will they leave?" — outputs median survival times and hazard ratios per feature.
*   **Generative AI Copilot:** Gemini 2.5 Flash generates actionable, human-centric retention strategies from SHAP risk drivers.

### Observability & Monitoring
*   **Data Drift Detection (Evidently AI):** Automated drift + data quality reports comparing training vs. inference distributions. HTML + JSON output.
*   **SHAP Attribution Drift:** Monitors feature importance stability via Spearman rank correlation, top-5 feature overlap, and per-feature magnitude changes. Leading indicator of concept drift.
*   **Drift-Triggered Retraining:** Airflow DAG auto-retrains when drift is detected, with model archival for rollback support.
*   **Data Version Tracking:** SHA-256 content-based fingerprinting of training data with full version history (`models/data_versions.json`).

### Security & Compliance
*   **Algorithmic Fairness (Five-Layer Pipeline):** Research-backed mitigation: (1) SMOTE-ENN hybrid augmentation for small subgroups, (2) cost-sensitive sample reweighing, (3) XGBoost retraining with group weights, (4) Fairlearn ThresholdOptimizer (Hardt et al. 2016) for Equalized Odds, (5) adaptive quality gates scaled to subgroup sample size with bootstrap confidence intervals.
*   **Fairness Quality Gate v3:** Adaptive pass/fail using sample-size-aware thresholds (EOD <= 0.50 for n_pos < 10, <= 0.35 for n_pos < 20, <= 0.25 for n_pos < 50, <= 0.15 for n_pos >= 50). Bootstrap 95% CIs flag statistically inconclusive metrics. Blocks MLflow registration on failure.
*   **Uplift-Aware Decision Engine:** Combines risk predictions with causal uplift (T-Learner CATE) to recommend interventions only where they are both warranted AND effective.
*   **Adversarial Robustness (Art. 15):** Gaussian noise injection, feature boundary stability, and deterministic consistency testing with ART integration.
*   **Subpopulation Performance (Art. 15):** Per-group AUC and F2 scores for Gender and Age bins to detect disparate impact.
*   **Audit Trail Database:** Every prediction persisted to PostgreSQL with full input/output traceability, model version, and requester identity (EU AI Act Art. 12).
*   **Human Override System (Art. 14):** HR professionals can override AI risk assessments with documented rationale. Override history tracked in `human_overrides` table.
*   **GDPR Right to Object (Art. 21):** Employees can be excluded from AI scoring via `scoring_exclusions` table with full audit trail.
*   **PII Protection:** Employee IDs are SHA-256 pseudonymized (salted hash) in audit logs for privacy-preserving traceability.
*   **RBAC:** Role-based API access — `admin`, `hr_partner`, `analyst`, `auditor` — with per-key role mapping via `RBAC_KEYS` env var.
*   **Rate Limiting:** SlowAPI-based request throttling on prediction endpoints.
*   **TLS Reverse Proxy:** Nginx with HTTPS termination, HSTS, CSP, X-Frame-Options, and 1MB request size limits.

### Infrastructure & Deployment
*   **Containerized Stack (Podman/Docker):** 4-service Compose deployment — PostgreSQL 15, FastAPI backend, React frontend (Nginx-served), TLS reverse proxy.
*   **Airflow Orchestration (Astronomer Cosmos):** DAG orchestrates: dlt extraction → dbt transform → Great Expectations validation → model training → fairness gate → dashboard upsert → drift check → conditional retraining.
*   **dbt Transformations:** 2-layer SQL pipeline — `staging/stg_hris` (view) → `marts/fct_attrition_features` (table) — running on PostgreSQL via Cosmos.
*   **Power BI DirectQuery:** Pre-built PostgreSQL star schema views (`powerbi.*`) for enterprise BI integration with Row-Level Security support.
*   **MLflow Model Registry:** Tracks model versions, parameters, and metrics. Quality gates block registration if AUC/F2/fairness thresholds fail.
*   **Modular Pipeline Orchestrator:** 18-step Python pipeline (`src/pipeline.py`) with configurable skip steps, fail-fast mode, and timing metadata.
*   **CI/CD (GitHub Actions):** 4-stage pipeline — backend lint+test → frontend test → Trivy container vulnerability scan + SBOM → Docker build.

---

## Architecture

```mermaid
graph TD
    subgraph Ingestion
        A[HRIS CSV] -->|dlt pipeline| B[(PostgreSQL<br/>raw_hris schema)]
    end

    subgraph Transformation
        B -->|Astronomer Cosmos| C[dbt: stg_hris]
        C --> D[dbt: fct_attrition_features]
    end

    subgraph ML Pipeline - 18 Steps
        D -->|Great Expectations| E{Quality Gate}
        E -->|Pass| F[Feature Engineering SSoT]
        F --> G[Optuna HPO + XGBoost]
        G --> H[Isotonic Calibration]
        H --> H2[Fairness Mitigation 3-Layer]
        H2 --> I[Evaluation + Subpopulations]
        I --> J[Fairness Audit]
        J --> K[SHAP Explainability]
        K --> L[Risk Framework + Cost Model]
        L --> M[Evidently Drift Report]
        M --> N[SHAP Drift Monitor]
        N --> O[Adversarial Robustness]
        O --> P[Causal Validation]
        P --> Q[Survival Analysis]
    end

    subgraph Persistence
        G -->|xgb_attrition.json| R[(Model Registry<br/>MLflow)]
        L -->|risk_scores.csv| S[(PostgreSQL<br/>analytics.predictions)]
        S -->|DirectQuery Views| T[Power BI]
    end

    subgraph Serving
        R --> U[FastAPI Backend]
        K --> U
        U --> V[(PostgreSQL<br/>Audit Trail)]
        U <-->|SHAP Drivers| W[Gemini 2.5 Flash]
    end

    subgraph Frontend
        X[React Dashboard] <-->|REST API| U
    end

    subgraph Orchestration
        Y[Airflow DAG] -->|Schedule: @daily| A
        Y --> E
        Y --> L
        N -->|Drift Detected| Z[Auto-Retrain]
    end
```

---

## Quick Start

### 1. Prerequisites
*   Python 3.13+
*   Node.js 20+ (or [Bun](https://bun.sh/))
*   [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
*   [Podman](https://podman.io/) or Docker (container engine for PostgreSQL)
*   [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli) (optional — for Airflow orchestration)

### 2. Setup

```bash
# Clone and set up Python backend
git clone <repository-url>
cd HR-Analytics
uv venv
uv sync

# Set up React frontend
cd frontend
bun install  # or npm install
cd ..
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure. **Never commit the actual `.env` file.**
```env
# ── Required ────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key_here

# ── Database (Docker Compose + backend + dlt + dbt) ────
POSTGRES_USER=hr_admin
POSTGRES_PASSWORD=change-me-in-production
POSTGRES_DB=hr_analytics
DATABASE_URL=postgresql://hr_admin:change-me-in-production@localhost:5432/hr_analytics

# ── dbt (used by profiles.yml via env_var()) ───────────
DBT_HOST=host.docker.internal   # Use 'localhost' outside Docker
DBT_PORT=5432

# ── API Security ───────────────────────────────────────
ENV=development                  # Set to 'production' to enforce API keys
API_SECRET_KEY=dev-key-change-in-production
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# ── RBAC (optional — overrides API_SECRET_KEY) ─────────
# Roles: admin, hr_partner, analyst, auditor
RBAC_KEYS={"sk-admin-xxx":"admin","sk-hr-yyy":"hr_partner"}
```

### 4. Start PostgreSQL (via Podman)

```bash
# Start the Podman machine (Windows/macOS only)
podman machine start

# Launch PostgreSQL container
podman compose up -d postgres

# Bootstrap: seed analytics.predictions, create ORM tables, create Power BI views
python scripts/bootstrap_powerbi_db.py
```

### 5. Run the Pipeline & Servers

```bash
# Option A: CLI entrypoint
python main.py train   # Run the full 17-step ML pipeline
python main.py serve   # Start FastAPI on port 8000

# Option B: Direct commands
uv run python -m src.train_attrition_model   # Train model
uv run uvicorn src.api:app --reload          # Start API with hot reload

# Start the Frontend (separate terminal)
cd frontend
bun dev  # or npm run dev → http://localhost:5173
```

### 6. Production Deployment (Full Stack)

```bash
# Generate self-signed TLS certificates (first time only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/certs/selfsigned.key \
  -out nginx/certs/selfsigned.crt \
  -subj "/CN=localhost"

# Build and start all 4 services: Nginx + Backend + Frontend + PostgreSQL
podman compose up -d

# Access:
#   https://localhost       → React Dashboard (via Nginx)
#   https://localhost/v1/   → FastAPI API (proxied)
#   https://localhost/docs  → Swagger UI (proxied)
```

### 7. Airflow Setup (Orchestration)

The Airflow project lives in `astro/` and is managed via the [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli).

```bash
cd astro

# Start local Airflow (spins up 5 containers: Postgres, Scheduler, DAG Processor, API Server, Triggerer)
astro dev start

# Access Airflow UI at http://localhost:8080
# Default credentials: admin / admin
```

**Required Airflow connections** (configure in Airflow UI → Admin → Connections):

| Connection ID | Type | Host | Port | Schema | Login | Password |
|--------------|------|------|------|--------|-------|----------|
| `postgres_default` | Postgres | `host.docker.internal` | `5432` | `hr_analytics` | `hr_admin` | `change-me-in-production` |

**Required environment variables** (set in `astro/.env` or Airflow UI → Admin → Variables):
```env
DATABASE_URL=postgresql://hr_admin:change-me-in-production@host.docker.internal:5432/hr_analytics
POSTGRES_USER=hr_admin
POSTGRES_PASSWORD=change-me-in-production
POSTGRES_DB=hr_analytics
```

The DAG `hr_attrition_batch_scoring` runs on `@daily` schedule with this task flow:

```mermaid
graph LR
    A[extract_hris_data<br/>dlt → raw_hris] --> B[dbt_transform<br/>Cosmos TaskGroup]
    B --> C[validate_data<br/>Great Expectations]
    C --> D[run_attrition_model<br/>17-step ML pipeline]
    D --> E[validate_fairness<br/>DPD/EOD gate]
    D --> F[check_drift_and_alert<br/>Evidently + SHAP drift]
    E --> G[update_dashboard_database<br/>→ analytics.predictions]
    F --> H[auto_retrain_on_drift<br/>conditional retrain]
```

> **Note:** The Airflow environment runs inside its own Docker containers (managed by Astro CLI). It connects to the *host* PostgreSQL via `host.docker.internal`. Make sure your main PostgreSQL container is running (`podman compose up -d postgres`) before starting Airflow.

---

## Project Structure

```
HR-Analytics/
├── .github/workflows/
│   └── mlops.yml                  # 4-stage CI/CD (lint → test → scan → build)
├── astro/                         # Astronomer Airflow project
│   ├── dags/
│   │   ├── hr_attrition_pipeline.py   # Batch scoring DAG (@daily)
│   │   └── dbt/                       # dbt project (Cosmos-managed)
│   │       ├── models/staging/        #   stg_hris.sql (view)
│   │       ├── models/marts/          #   fct_attrition_features.sql (table)
│   │       ├── dbt_project.yml
│   │       └── profiles.yml
│   └── include/
│       ├── hris_dlt_pipeline.py       # dlt extraction (CSV → PostgreSQL)
│       └── ge_validation.py           # Great Expectations quality checks
├── datasets/
│   └── HR-Employee-Attrition.csv      # IBM synthetic dataset (1,470 rows)
├── docs/
│   ├── AI_SYSTEM_CARD.md              # EU AI Act Annex IV documentation
│   ├── DATA_PROTECTION_IMPACT_ASSESSMENT.md  # GDPR Article 35 DPIA
│   └── RISK_MANAGEMENT_SYSTEM.md      # EU AI Act Article 9 risk register
├── frontend/                          # React 19 + Vite 8 + TypeScript 6
│   ├── src/
│   │   ├── App.tsx                    # 6-tab SPA (sidebar navigation)
│   │   ├── lib/api.ts                 # Typed API client (TanStack Query)
│   │   └── components/
│   │       ├── AnalyticsDashboard.tsx  #   Fleet-level risk heatmaps & cohorts
│   │       ├── AiEthicsDashboard.tsx   #   Fairness metrics + SHAP monitoring
│   │       ├── AuditTrail.tsx          #   Paginated audit log viewer
│   │       ├── DecisionCockpit.tsx     #   Individual scoring + What-If + Override
│   │       ├── DriftMonitor.tsx        #   Data & SHAP drift visualization
│   │       └── ErrorBoundary.tsx       #   Graceful error handling
│   ├── Dockerfile                     # Multi-stage: Bun build → Nginx serve
│   └── package.json                   # Tremor, Framer Motion, Lucide Icons
├── models/                            # Serialized model artifacts
│   ├── xgb_attrition.json             #   XGBoost model weights
│   ├── xgb_calibrated.joblib          #   Isotonic-calibrated model
│   ├── population_stats.json          #   Feature engineering statistics (SSoT)
│   ├── best_params.json               #   Optuna best hyperparameters
│   ├── training_metadata.json         #   Model provenance metadata
│   └── data_versions.json             #   SHA-256 data version history
├── nginx/
│   ├── nginx.conf                     # TLS reverse proxy + security headers
│   └── certs/                         # Self-signed TLS certificates
├── notebooks/
│   └── attrition_modeling.ipynb       # Exploratory analysis notebook
├── outputs/                           # Generated pipeline artifacts
│   ├── risk_scores.csv                #   Per-employee risk predictions
│   ├── risk_summary.csv               #   Aggregated risk tier statistics
│   ├── fairness_audit.json            #   DPD/EOD per protected attribute
│   ├── fairness_mitigation_report.json #   Five-layer mitigation before/after metrics + bootstrap CIs
│   ├── fairness_remediation_final.json #   Combined multi-attribute remediation results
│   ├── subpopulation_metrics.json     #   Per-group AUC/F2 scores
│   ├── shap_global_importance.csv     #   Feature importance rankings
│   ├── evidently_drift_report.html    #   Interactive drift dashboard
│   ├── evidently_drift_report.json    #   Machine-readable drift data
│   ├── shap_summary.png               #   SHAP beeswarm plot
│   ├── shap_bar.png                   #   Feature importance bar chart
│   ├── calibration_curve.png          #   Probability calibration plot
│   ├── confusion_matrix.png           #   Classification confusion matrix
│   └── eda_overview.png               #   Exploratory data analysis charts
├── powerbi/
│   ├── directquery_views.sql          # 6 PostgreSQL views for Power BI
│   └── PowerBI_Architecture_and_DAX.md  # Star schema design + DAX formulas
├── scripts/
│   ├── bootstrap_powerbi_db.py        # DB seeding + view creation utility
│   └── run_fairness_remediation.py    # Standalone fairness mitigation runner (per-attribute + combined)
├── src/
│   ├── api.py                         # FastAPI service (20 endpoints, 56KB)
│   ├── database.py                    # SQLAlchemy engine (SQLite/PostgreSQL)
│   ├── features.py                    # Unified Feature Engineering — Single Source of Truth
│   ├── fairness_mitigation.py         # Five-layer fairness remediation (SMOTE-ENN + reweighing + ThresholdOptimizer + bootstrap CI + adaptive gates)
│   ├── causal_uplift.py               # T-Learner causal uplift model (CATE estimation)
│   ├── models.py                      # ORM: PredictionLog, HumanOverride, ScoringExclusion
│   ├── pipeline.py                    # 18-step modular pipeline orchestrator
│   ├── data_version.py                # SHA-256 dataset fingerprinting & provenance
│   └── train_attrition_model.py       # Core ML pipeline (training → scoring → persistence)
├── tests/                             # Pytest test suite
│   ├── test_api.py                    #   API endpoint tests
│   ├── test_features.py               #   Feature engineering tests
│   ├── test_rbac.py                   #   RBAC authorization tests
│   ├── test_shap_drift.py             #   SHAP drift detection tests
│   ├── test_robustness_causal_survival.py  # Adversarial + causal + survival tests
│   ├── test_dag_integration.py        #   Airflow DAG integration tests
│   └── load_test.py                   #   Performance/load testing (excluded from CI)
├── main.py                            # CLI entrypoint (train / serve)
├── docker-compose.yml                 # 4-service Podman/Docker deployment
├── Dockerfile                         # Backend: Python 3.13-slim + uv + non-root user
└── pyproject.toml                     # uv dependencies (v1.1.0)
```

---

## Dashboard Tabs

The React frontend provides 6 views:

| Tab | Component | Description |
|-----|-----------|-------------|
| **Overview** | `App.tsx` | KPI scorecards (VaR, High Risk count, Fleet Risk %), SHAP bar chart, risk tier donut |
| **Analytics** | `AnalyticsDashboard` | Department heatmaps, cohort comparisons, trend analysis |
| **Drift Monitor** | `DriftMonitor` | Evidently data drift + SHAP attribution drift visualization |
| **Decision Cockpit** | `DecisionCockpit` | Individual employee scoring, What-If simulation, human override |
| **AI Ethics** | `AiEthicsDashboard` | Fairness audit results, SHAP global importance, compliance status |
| **Audit Trail** | `AuditTrail` | Paginated prediction log viewer (filterable by tier, employee, date) |

---

## API Endpoints

### Core Prediction & Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/predict` | Score an employee, log to audit trail, generate Gemini strategy |
| `GET` | `/v1/dashboard/summary` | Aggregate risk statistics (headcount, VaR, top drivers) |
| `GET` | `/v1/dashboard/employees` | Paginated employee risk scores with filters |
| `GET` | `/v1/dashboard/trends` | Historical risk trend data (configurable period) |
| `GET` | `/v1/dashboard/cohorts` | Cohort comparison by Department/JobRole/Age |
| `GET` | `/v1/dashboard/drift` | Live data drift + SHAP drift status |

### Compliance & Audit
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/audit-logs` | Paginated audit trail (filterable by tier/employee) |
| `POST` | `/v1/override` | Human override of AI risk assessment (Art. 14) |
| `GET` | `/v1/reports/export` | Export risk scores as CSV/JSON |

### GDPR (Right to Object)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/gdpr/exclude` | Exclude employee from AI scoring (Art. 21) |
| `DELETE` | `/v1/gdpr/exclude/{id}` | Revoke exclusion |
| `GET` | `/v1/gdpr/exclusions` | List active exclusions |

### System & MLOps
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | API health check (model loaded, stats loaded, version) |
| `GET` | `/v1/auth/whoami` | Current API key role and permissions |
| `POST` | `/v1/system/model/rollback` | Rollback to previous model version |
| `GET` | `/v1/system/models` | List available model versions |
| `POST` | `/v1/system/shadow/load` | Load a shadow model for A/B comparison |
| `DELETE` | `/v1/system/shadow` | Remove active shadow model |
| `GET` | `/v1/system/feature-store` | Feature store / population stats status |

---

## Power BI Integration

Pre-built PostgreSQL views optimized for Power BI DirectQuery mode:

| View | Description |
|------|-------------|
| `powerbi.dim_employee` | Employee demographics (age, gender, education, marital status, age group) |
| `powerbi.dim_job` | Job details (role, department, level) |
| `powerbi.fact_risk_scores` | Core risk metrics with financial impact + satisfaction scores |
| `powerbi.fact_audit_trail` | Prediction audit history for compliance reporting |
| `powerbi.fact_overrides` | Human override decisions with direction tracking |
| `powerbi.agg_risk_by_department` | Pre-aggregated department-level risk summary |
| `powerbi.manager_employee_map` | RLS mapping table for manager-scoped access |
| `powerbi.rls_filtered_risk_scores` | Risk scores pre-joined with manager mapping (use for RLS) |
| `powerbi.rls_filtered_audit_trail` | Audit trail filtered by manager scope |
| `powerbi.rls_filtered_overrides` | Override history filtered by manager scope |
| `powerbi.rls_manager_scope` | Debug view: employees visible per manager |

### Power BI Setup Guide

**Step 1: Bootstrap the database**
```bash
podman compose up -d postgres
python scripts/bootstrap_powerbi_db.py
```

**Step 2: Connect Power BI Desktop**
1. Open Power BI Desktop → **Get Data** → **PostgreSQL database**
2. Enter connection details:
   - Server: `localhost:5432`
   - Database: `hr_analytics`
3. Select **DirectQuery** mode (recommended for real-time data)
4. Authenticate with database credentials (`hr_admin` / `change-me-in-production`)
5. In the Navigator, expand the `powerbi` schema and select all views

**Step 3: Configure relationships**

Power BI should auto-detect these relationships (verify in Model View):
```
dim_employee.employee_id  (1) ←→ (*)  fact_risk_scores.employee_id
dim_job.job_role           (1) ←→ (*)  fact_risk_scores.job_role
```

**Step 4: Row-Level Security (RLS)**

First, seed the manager-employee mapping:
```bash
# Generate hierarchy from IBM HR dataset + export CSV
python scripts/seed_rls_mapping.py

# Output:
#   PostgreSQL: powerbi.manager_employee_map (4300+ rows)
#   CSV: outputs/rls_manager_employee_map.csv
```

Then configure 4 RLS roles in Power BI Desktop (**Modeling → Manage Roles**):

| Role | DAX Filter | Scope |
|------|-----------|-------|
| `Executive` | *(no filter)* | All employees |
| `DepartmentHead` | `[manager_email] = USERPRINCIPALNAME()` on `manager_employee_map` | Department employees |
| `LineManager` | `[manager_email] = USERPRINCIPALNAME()` on `manager_employee_map` | Direct reports only |
| `HRAnalyst` | `[monthly_income] = BLANK()` on `fact_risk_scores` | Aggregates only, no PII |

Use the RLS-filtered views for automatic scoping:

| View | Description |
|------|-------------|
| `powerbi.rls_filtered_risk_scores` | Risk scores pre-joined with manager mapping |
| `powerbi.rls_filtered_audit_trail` | Audit trail filtered by manager scope |
| `powerbi.rls_filtered_overrides` | Override history filtered by manager scope |
| `powerbi.rls_manager_scope` | Debug view: employees visible per manager |

**Starter DAX measures** (create in a Measures table):
```dax
Total Headcount = DISTINCTCOUNT(fact_risk_scores[employee_id])

High Risk Count =
    CALCULATE([Total Headcount], fact_risk_scores[risk_tier] = "High")

Predicted Attrition Rate =
    DIVIDE([High Risk Count], [Total Headcount], 0)

Total Value at Risk = SUM(fact_risk_scores[expected_loss])

// RLS-scoped measures (auto-filter per manager):
My Team Headcount = DISTINCTCOUNT(rls_filtered_risk_scores[employee_id])
My Team Value at Risk = SUM(rls_filtered_risk_scores[expected_loss])
```

See [`powerbi/PowerBI_Architecture_and_DAX.md`](powerbi/PowerBI_Architecture_and_DAX.md) for the full star schema design, all DAX formulas, RLS data flow diagrams, and deployment checklist.

---

## dbt Data Pipeline

The dbt project (`astro/dags/dbt/`) transforms raw HRIS data into ML-ready features:

```mermaid
graph LR
    A[(raw_hris.employee_attrition_records<br/>dlt-loaded)] -->|staging| B[stg_hris<br/>VIEW: type casting + renaming]
    B -->|marts| C[fct_attrition_features<br/>TABLE: 9 engineered features]
    C --> D[ML Pipeline]
    C --> E[Great Expectations<br/>12 quality checks]
```

### Engineered Features (fct_attrition_features)

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| `compa_ratio` | `income / avg_role_income` | Pay competitiveness vs. role average |
| `income_growth_gap` | `hike - median_level_hike` | Salary growth relative to peers at same level |
| `promotion_stagnation` | `years_since_promo / (tenure + 1)` | Career progression slowdown indicator |
| `burnout_risk` | `overtime × distance / WLB` | Composite work-life stress indicator |
| `manager_stability` | `years_with_mgr / (tenure + 1)` | Management relationship continuity |
| `engagement_index` | `avg(4 satisfaction scores)` | Overall employee engagement (1–4 scale) |
| `career_velocity` | `job_level / (total_years + 1)` | Advancement speed relative to experience |
| `loyalty_index` | `tenure / (total_years + 1)` | Organizational commitment indicator |
| `travel_burden` | `ordinal(business_travel)` | Travel frequency stress (0/1/2) |

### Great Expectations Quality Checks (12 rules)

| # | Check | Rule |
|---|-------|------|
| 1 | `years_at_company >= 0` | Tenure must be non-negative |
| 2 | `monthly_income > 0` | Salary must be positive |
| 3 | `distance_from_home not null` | No missing commute data |
| 4 | `years_in_role <= years_at_company` | Role tenure can't exceed total tenure |
| 5 | `over_time_yes ∈ {0, 1}` | Binary overtime flag |
| 6 | `attrition ∈ {0, 1}` | Binary target variable |
| 7 | `travel_burden ∈ {0, 1, 2}` | Valid ordinal values |
| 8 | `engagement_index ∈ [1, 4]` | Valid satisfaction range |
| 9 | `compa_ratio > 0` | Positive pay ratio |
| 10 | `burnout_risk >= 0` | Non-negative risk score |
| 11 | `promotion_stagnation >= 0` | Non-negative stagnation |
| 12 | `row_count >= 100` | Prevent empty/truncated tables |

---

## ML Pipeline Steps

The modular pipeline (`src/pipeline.py`) executes 18 steps in sequence:

| # | Step | Module | Output |
|---|------|--------|--------|
| 1 | Load Data | `pipeline.py` | DataFrame (from PostgreSQL mart or CSV fallback) |
| 2 | Exploratory Data Analysis | `train_attrition_model.py` | `outputs/eda_overview.png` |
| 3 | Feature Engineering (SSoT) | `features.py` | `models/population_stats.json` |
| 4 | Preprocessing & Splitting | `train_attrition_model.py` | 60/20/20 stratified split |
| 5 | Hyperparameter Optimization | Optuna (30 trials) | `models/best_params.json` |
| 6 | Training XGBoost | XGBoost | `models/xgb_attrition.json` |
| 7 | Probability Calibration | Isotonic/Platt | `models/xgb_calibrated.joblib` |
| **8** | **Fairness Mitigation (5-Layer)** | **`fairness_mitigation.py`** | **`outputs/fairness_mitigation_report.json`** |
| 9 | Model Evaluation | sklearn | F2, AUC, Brier score, confusion matrix |
| 10 | Subpopulation Analysis | Art. 15 | `outputs/subpopulation_metrics.json` |
| 11 | Fairness Audit | Fairlearn | `outputs/fairness_audit.json` |
| 12 | SHAP Explainability | SHAP | `outputs/shap_summary.png`, `shap_bar.png` |
| 13 | SHAP Attribution Drift | scipy | `outputs/shap_drift_report.json` |
| 14 | Risk Framework & Cost Model | — | `outputs/risk_scores.csv` → PostgreSQL |
| 15 | Data Drift Report | Evidently AI | `outputs/evidently_drift_report.html` |
| 16 | Adversarial Robustness | ART + custom | `outputs/adversarial_robustness_report.json` |
| 17 | Causal Validation | DoWhy | `outputs/causal_validation_report.json` |
| 18 | Survival Analysis | Lifelines | `outputs/survival_analysis_report.json` |

### Fairness Mitigation Pipeline (Step 8)

The five-layer mitigation addresses critical fairness violations discovered during audit, backed by deep research into small-subgroup statistical reliability:

```
+----------------------------------------------------+
|  Layer 1: DATA AUGMENTATION (SMOTE-ENN)            |
|  Synthetic oversampling + noise cleaning for Age   |
|  51+ minority (7 -> 35 positives in training)      |
+----------------------------------------------------+
|  Layer 2: COST-SENSITIVE REWEIGHING                |
|  Per-(group, label) sample weights equalize        |
|  representation: weight = P(g)*P(y) / P(g,y)      |
+----------------------------------------------------+
|  Layer 3: MODEL RETRAINING (XGBoost)               |
|  Retrained on augmented data with group weights    |
+----------------------------------------------------+
|  Layer 4: POST-PROCESSING (ThresholdOptimizer)     |
|  Fairlearn Equalized Odds (Hardt et al. 2016)      |
|  with degenerate-solution guard + manual fallback  |
+----------------------------------------------------+
|  Layer 5: ADAPTIVE QUALITY GATE + BOOTSTRAP CI     |
|  Sample-size-aware EOD/DPD thresholds + 1000-iter  |
|  bootstrap 95% CIs for uncertainty quantification  |
+----------------------------------------------------+
```

**Results (model registered as Staging):**

| Protected Attribute | EOD Before | EOD After | Adaptive Threshold | Bootstrap 95% CI | Status |
|---------------------|------------|-----------|-------------------|------------------|--------|
| **Age Group** | 0.556 | **0.246** | <= 0.50 (n_pos=6) | [0.18, 0.75] | PASS |
| **Gender** | 0.208 | **0.208** | <= 0.35 (n_pos=16) | [0.02, 0.52] | PASS |
| **MaritalStatus** | 0.560 | **0.270** | <= 0.50 (n_pos=4) | [0.10, 0.71] | PASS |

**Critical fix:** Age 51+ subgroup F2 improved from **0.00 -> 0.50** (model can now detect attrition risk for older employees). Global F2 improved from 0.42 -> **0.52** with subgroup-aware thresholds.

**EU AI Act compliance note:** The EU AI Act (Art. 10, Art. 15) does not prescribe numeric fairness thresholds. All bootstrap CIs are flagged as "statistically inconclusive" due to small subgroup sizes, which is the honest, documented answer — compliance is demonstrated through process rigor and mitigation documentation, not arbitrary numeric targets.

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/mlops.yml`) with 4 stages:

```
lint-and-test ──→ security-scan ──→ docker-build
                      ↑
frontend-test ────────┘
```

| Stage | Actions |
|-------|---------|
| **lint-and-test** | `ruff check` → `pytest` → module import validation |
| **frontend-test** | `bun install` → `bun run test` (Vitest + Testing Library) |
| **security-scan** | Trivy vulnerability scanner (CRITICAL/HIGH) + CycloneDX SBOM generation |
| **docker-build** | `docker compose build` (main branch only) |

---

## Compliance & Ethics (EU AI Act)

This system falls under Annex III of the EU AI Act (High-Risk AI Systems for Employment).

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Art. 9** | Risk Management System | [`docs/RISK_MANAGEMENT_SYSTEM.md`](docs/RISK_MANAGEMENT_SYSTEM.md) |
| **Art. 10** | Data Quality & Bias | Three-layer fairness mitigation (reweighing + ExponentiatedGradient + ThresholdOptimizer) + Great Expectations quality gates |
| **Art. 11** | Technical Documentation | [`docs/AI_SYSTEM_CARD.md`](docs/AI_SYSTEM_CARD.md) (Annex IV) |
| **Art. 12** | Record-Keeping | Full audit trail in PostgreSQL (input, output, timestamp, requester, model version) |
| **Art. 13** | Transparency | SHAP explanations + explicit "correlation ≠ causation" disclaimers + causal warnings |
| **Art. 14** | Human Oversight | Override system with documented rationale + scoring exclusions |
| **Art. 15** | Accuracy & Robustness | Subpopulation metrics, adversarial robustness testing, calibration curves, fairness Quality Gate v2 |
| **GDPR Art. 21** | Right to Object | Scoring exclusion system with activate/revoke lifecycle |
| **GDPR Art. 35** | DPIA | [`docs/DATA_PROTECTION_IMPACT_ASSESSMENT.md`](docs/DATA_PROTECTION_IMPACT_ASSESSMENT.md) |

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **ML Framework** | XGBoost + scikit-learn + Optuna | 3.2+ / 1.8+ / 4.8+ |
| **Explainability** | SHAP + DoWhy + Lifelines | 0.51+ / 0.11+ / 0.30+ |
| **Fairness** | Fairlearn + Adversarial Robustness Toolbox | 0.13+ / 1.18+ |
| **Observability** | Evidently AI + MLflow | 0.4.x / 2.21+ |
| **API** | FastAPI + SQLAlchemy + SlowAPI | 0.136+ / 2.0+ |
| **Frontend** | React 19 + Vite 8 + TypeScript 6 | — |
| **UI Library** | Tremor + Framer Motion + Tailwind CSS 3 | 3.18+ / 12+ |
| **Database** | PostgreSQL 15 (Podman) + SQLite (dev fallback) | — |
| **Orchestration** | Apache Airflow (Astronomer Cosmos) + dbt | — |
| **Data Ingestion** | dlt (data load tool) | — |
| **Data Quality** | Great Expectations | — |
| **Gen AI** | Google Gemini 2.5 Flash (`google-genai`) | 1.73+ |
| **CI/CD** | GitHub Actions + Trivy + Bun | — |
| **Proxy** | Nginx 1.27 (Alpine) | — |
| **Container** | Podman / Docker Compose | — |

---

## Known Limitations

*   **Synthetic Dataset:** The IBM HR Attrition dataset is fictional (1,470 rows). Metrics have high variance and do not validate real-world performance.
*   **Causal Inference:** DoWhy validation on synthetic data demonstrates methodology only — production deployment requires domain expert review of the causal DAG and real observational data.
*   **Calibration Constraints:** Isotonic calibration on small samples (~220) may overfit. Consider Platt scaling for datasets under 1,000 samples.
*   **Temporal Validation:** No time-series split is possible with this snapshot dataset. Production deployment requires longitudinal data.
*   **Fairness Statistical Power:** The 51+ age subgroup has only 28 test samples (6 positive cases). Bootstrap confidence intervals (1000 iterations) show EOD 95% CI widths of 0.50-0.60, confirming that point estimates are statistically inconclusive at this sample size. Adaptive quality gate thresholds (scaled to n_pos) account for this inherent limitation.
*   **Fairness-Performance Tradeoff:** The five-layer mitigation maintains global F2 at ~0.42-0.54 (improved from baseline via SMOTE augmentation) while achieving per-subgroup fairness. ThresholdOptimizer may produce degenerate all-zeros solutions on noisy data; the pipeline automatically detects and falls back to manual subgroup thresholds.
*   **Survival Analysis:** Uses `YearsAtCompany` as a duration proxy since true event timestamps are unavailable in the synthetic dataset.
*   **Feature Store:** The `/v1/system/feature-store` endpoint surfaces population statistics status; a full Feast-based feature store is architecturally planned but not yet implemented.

---

## Troubleshooting

<details>
<summary><b>Podman machine won't start / connection refused</b></summary>

```bash
# Check machine status
podman machine list

# Restart the machine
podman machine stop
podman machine start

# Verify connectivity
podman ps
```
</details>

<details>
<summary><b>PostgreSQL password authentication failed</b></summary>

The PostgreSQL image only sets credentials on first `initdb`. If you changed passwords in `.env` after the volume was created:
```bash
# Remove volume and recreate (WARNING: deletes all data)
podman compose down -v
podman compose up -d postgres
python scripts/bootstrap_powerbi_db.py
```
</details>

<details>
<summary><b>Port 5432 already in use</b></summary>

```bash
# Find what's using the port (Windows)
netstat -ano | findstr :5432

# Either stop the conflicting service or change the port in docker-compose.yml:
# ports:
#   - "5433:5432"  # Map to a different host port
```
</details>

<details>
<summary><b>Airflow can't connect to PostgreSQL</b></summary>

The Astro CLI runs Airflow inside its own Docker network. It needs `host.docker.internal` (not `localhost`) to reach the host PostgreSQL:
```env
# In astro/.env
DATABASE_URL=postgresql://hr_admin:change-me-in-production@host.docker.internal:5432/hr_analytics
```
Also ensure the main PostgreSQL container is running: `podman compose up -d postgres`
</details>

<details>
<summary><b>Power BI views show no data</b></summary>

1. Verify the bootstrap ran successfully:
   ```bash
   podman exec hr-analytics-postgres-1 psql -U hr_admin -d hr_analytics \
     -c "SELECT COUNT(*) FROM analytics.predictions;"
   ```
2. If 0 rows, re-run the bootstrap:
   ```bash
   python scripts/bootstrap_powerbi_db.py
   ```
3. If the `risk_scores.csv` is missing, run the ML pipeline first:
   ```bash
   python main.py train
   ```
</details>

<details>
<summary><b>dbt models fail to run</b></summary>

1. Ensure `raw_hris.employee_attrition_records` exists (run dlt pipeline first)
2. Check dbt profiles connection:
   ```bash
   cd astro/dags/dbt
   dbt debug --profiles-dir .
   ```
3. Verify env vars: `DBT_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
</details>

<details>
<summary><b>Frontend can't connect to API</b></summary>

1. Ensure the backend is running on port 8000
2. Check CORS: `ALLOWED_ORIGINS` in `.env` must include `http://localhost:5173`
3. For custom API URL, set `VITE_API_URL` in `frontend/.env`:
   ```env
   VITE_API_URL=http://localhost:8000
   ```
</details>

---

## License

MIT
