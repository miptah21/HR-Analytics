# AI System Card — HR Attrition Intelligence System

> **Document Version:** 1.1  
> **Last Updated:** 2026-05-02  
> **Classification:** Annex III, Item 4(a) — High-Risk AI System for Employment  
> **Applicable Regulation:** EU AI Act (Regulation (EU) 2024/1689)

---

## 1. System Description (Annex IV, §1)

### 1.1 Intended Purpose
This AI system predicts the probability that an employee will leave the organization voluntarily (attrition) within a forecast horizon. It is designed as a **decision-support tool** for HR Business Partners and people managers, providing:

- Risk probability scores per employee
- Financial impact estimates (expected replacement cost × probability)
- Correlated risk factors via SHAP values
- Scenario exploration via "What-If" simulation

### 1.2 Intended Users (Deployers)
- HR Business Partners with authority over retention programs
- People Managers receiving risk alerts for their direct reports
- HR Analytics teams interpreting fleet-level trends

### 1.3 NOT Intended For
- **Automated employment decisions** (hiring, firing, promotion, demotion)
- **Individual performance evaluation**
- **Surveillance of employee behavior**
- Deployment without human oversight

---

## 2. Technical Architecture (Annex IV, §2)

### 2.1 Model
- **Algorithm:** XGBoost Gradient Boosted Classifier
- **Optimization:** Optuna Bayesian hyperparameter search with F2-score objective
- **Calibration:** Sigmoid / Platt Scaling (CalibratedClassifierCV, 5-fold)
- **Threshold:** Optimized for F2 via precision-recall curve analysis

### 2.2 Feature Engineering
- Unified module (`src/features.py`) ensures zero train-serving skew
- Population statistics computed during training are serialized and reused at inference
- 15 engineered features including: engagement_index, burnout_risk, compa_ratio, promotion_stagnation

### 2.3 Explainability
- **Method:** SHAP (TreeExplainer) — local explanations per prediction
- **Limitation:** SHAP values explain the model's learned associations, not causal mechanisms. See Section 5.

### 2.4 Generative AI Component
- **Model:** Google Gemini 2.5 Flash
- **Purpose:** Generates natural-language retention strategy suggestions based on SHAP-identified correlated factors
- **Safeguards:** Prompt includes mandatory causal disclaimer; inputs sanitized against injection

---

## 3. Data Governance (Annex IV, §3 & Article 10)

### 3.1 Training Data
| Property | Value |
|----------|-------|
| **Dataset** | IBM HR Analytics Employee Attrition & Performance |
| **Source** | Kaggle (public domain, CC0) |
| **Nature** | **Fictional / Synthetically generated** by IBM data scientists |
| **Size** | 1,470 records, 35 features |
| **Target Variable** | Attrition (Binary: Yes/No) |
| **Class Balance** | ~16% positive (attrition) / ~84% negative (retention) |
| **Temporal Coverage** | Single snapshot — no longitudinal or time-series data |

### 3.2 Data Quality Measures
- Great Expectations validation suite (12 checks) including range, nullity, referential integrity
- Pandas fallback validation when GE is unavailable
- Evidently AI data drift monitoring between training and inference distributions

### 3.3 Data Limitations
1. **Fictional data** — not derived from any real organization
2. **Small sample size** — insufficient for robust subpopulation analysis
3. **No temporal dimension** — cannot validate out-of-time generalization
4. **Bias characteristics unknown** — synthetic generation process may embed or omit biases

---

## 4. Performance Metrics (Annex IV, §5 & Article 15)

### 4.1 Global Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | Logged per run | On test split (~20% of data) |
| F2-Score | Logged per run | Weighted toward recall (catching leavers) |
| Brier Score | Logged per run | Pre- and post-calibration |
| Calibrated AUC | Logged per run | Post-isotonic regression |

### 4.2 Subpopulation Metrics
AUC and F2 are computed per subgroup and logged to `outputs/subpopulation_metrics.json`:
- Gender (Male / Female)
- Age (18-30, 31-40, 41-50, 51+)

### 4.3 Fairness Metrics
| Metric | Threshold | Attributes Audited |
|--------|-----------|-------------------|
| Demographic Parity Difference | ≤ 0.10 | Gender, Age Group, Marital Status |
| Equalized Odds Difference | ≤ 0.10 | Gender, Age Group, Marital Status |

---

## 5. Risk Management (Article 9)

### 5.1 Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Causal misinterpretation** — HR acts on correlations as if they are causes | High | High | Causal disclaimers in UI, API, and LLM prompts; scenario framing; training materials |
| **Algorithmic bias** — model systematically flags protected groups | Medium | Critical | Fairlearn audit gate; quality gate blocks biased models from registration |
| **Data drift** — production data diverges from training distribution | Medium | High | Evidently AI monitoring; dashboard alerts |
| **Overtrust** — managers rely on AI instead of human judgment | Medium | High | Human-override capability; decision cockpit design; mandatory override logging |
| **Privacy violation** — sensitive employee data exposed | Low | Critical | RBAC with 4 roles; API key per-role isolation; audit logging; no PII in LLM prompts |
| **Model staleness** — model not retrained as organization evolves | High | Medium | Airflow-orchestrated retraining pipeline; drift detection |

### 5.2 Foreseeable Misuse Scenarios
1. **Termination justification** — Using risk scores as evidence to fire employees → Mitigated by explicit disclaimers and override requirements
2. **Surveillance** — Continuous monitoring of employee "loyalty" → Mitigated by batch-scoring design (not real-time surveillance)
3. **Discriminatory targeting** — Selectively applying retention interventions by demographic → Mitigated by fairness audits and subpopulation metrics

---

## 6. Human Oversight (Article 14)

### 6.1 Oversight Mechanisms
- **Decision Cockpit** — Interactive UI where HR reviews and approves/overrides every prediction
- **Human Override** — Documented override with mandatory justification (min 10 characters)
- **Audit Trail** — All predictions and overrides logged with timestamps and user attribution

### 6.2 Competence Requirements
Users of this system should have:
- Understanding that SHAP values are correlations, not causes
- Authority to approve or reject retention interventions
- Training on the system's limitations (see Section 5)

### 6.3 Role-Based Access Control (RBAC)
The system enforces fine-grained access control via API key → role mapping:

| Role | Predict | Override | Dashboard | Audit | Export | System |
|------|---------|----------|-----------|-------|--------|--------|
| **admin** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **hr_partner** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **analyst** | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **auditor** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

Every API request records the requester's role in the audit trail, enabling per-role usage analysis and compliance reporting.

---

## 7. Record-Keeping (Article 12)

### 7.1 Logged Data
Every prediction automatically logs:
- Timestamp (UTC)
- Input features (full employee profile)
- Output probability, risk tier, and financial estimate
- SHAP-based top risk drivers
- Model version hash
- Retention strategy text (if generated)

### 7.2 Override Records
Every human override logs:
- Original AI risk tier
- Overridden risk tier
- Human justification text
- Timestamp and (future: user ID)

### 7.3 Storage
- **Development:** SQLite (`hr_analytics.db`)
- **Production:** PostgreSQL with connection pooling

---

## 8. Robustness & Cybersecurity (Article 15)

### 8.1 Adversarial Robustness Testing
The training pipeline includes automated adversarial robustness tests:
- **Gaussian Noise Injection:** Tests prediction stability under random noise at ε = {0.01, 0.05, 0.10, 0.20}
- **Feature Boundary Stability:** Verifies that 1% perturbations of individual features do not flip predictions
- **Deterministic Consistency:** Confirms identical inputs produce identical outputs

Results are persisted to `outputs/adversarial_robustness_report.json` with each pipeline run.

### 8.2 Cybersecurity Controls
- API key authentication with OWASP security headers
- Rate limiting (10 req/min predict, 30 req/min override)
- Pydantic input validation with enum constraints
- Error message sanitization (no internal paths or stack traces)
- LLM prompt injection defense (regex sanitization)

### 8.3 Known Limitations
- Full data poisoning and model evasion testing (via IBM ART) is recommended for production
- Network-level security (TLS, VPN, WAF) is deployment-dependent

---

## 9. Causal Inference Validation

### 9.1 Purpose
SHAP values measure feature *attribution* (correlation-based), NOT causal effects. Interventions based solely on SHAP can be ineffective or harmful. The causal validation layer validates whether SHAP-identified drivers have genuine causal support.

### 9.2 Methodology
- **Framework:** DoWhy (Microsoft Research)
- **Estimator:** Backdoor adjustment with linear regression
- **Refutation:** Random common cause (placebo test) to verify robustness
- **Treatments Tested:** OverTime (binary), with confounders: Age, MonthlyIncome, JobLevel, YearsAtCompany, TotalWorkingYears, DistanceFromHome, JobSatisfaction

### 9.3 Important Disclaimer
On the current synthetic dataset, causal estimates validate *methodology* only. Real-world causal claims require genuine HRIS data with appropriate temporal structure.

---

## 10. Survival Analysis (Time-to-Event)

### 10.1 Purpose
Binary classification answers "will they leave?" but not "when?" Survival analysis adds a temporal dimension, enabling:
- Urgency-aware intervention prioritization
- Budget allocation across time horizons
- Multi-method validation (complementing XGBoost)

### 10.2 Methodology
- **Model:** Cox Proportional Hazards (penalized, λ=0.1)
- **Duration Proxy:** YearsAtCompany (synthetic data limitation)
- **Event Indicator:** Attrition (Yes/No)
- **Output:** Hazard ratios with 95% confidence intervals
- **Validation:** Concordance index, Kaplan-Meier survival curves by income cohort

---

## 11. Data Protection Impact Assessment (DPIA)

A formal DPIA has been prepared per GDPR Article 35. See: [`docs/DATA_PROTECTION_IMPACT_ASSESSMENT.md`](./DATA_PROTECTION_IMPACT_ASSESSMENT.md)

Key conclusions:
- Processing is based on legitimate interest (Art. 6(1)(f))
- System is advisory-only — no automated decision-making under Art. 22
- Residual risks are manageable with implemented controls
- DPO sign-off required before production deployment

---

## 12. Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-05-02 | 1.1 | Added Art. 15 robustness testing, causal inference, survival analysis, DPIA |
| 2026-05-02 | 1.0 | Initial AI System Card created |

