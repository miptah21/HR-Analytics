# Risk Management System — EU AI Act Article 9

**System:** HR Attrition Intelligence System  
**Version:** 1.0.0  
**Last Review:** 2026-04-25  
**Next Scheduled Review:** 2026-07-25  
**Owner:** HR Analytics Engineering Team

---

## 1. System Purpose & Classification

This system predicts employee attrition risk using machine learning (XGBoost) to enable proactive retention interventions. Under the EU AI Act, AI systems used in **employment management** (Annex III, Section 4) are classified as **high-risk**.

### Intended Use
- Score individual employees for flight risk probability
- Generate explainable risk drivers (SHAP) for HR professionals
- Suggest retention strategies via LLM (Gemini) as advisory input
- Provide human-in-the-loop override capability for all predictions

### Users (Deployers)
- HR Business Partners and People Analytics teams
- The system is **advisory only** — no automated employment decisions are made

---

## 2. Risk Identification (Art. 9(2)(a))

| Risk ID | Category | Description | Affected Right | Likelihood | Impact |
|---------|----------|-------------|----------------|------------|--------|
| R-01 | **Misclassification** | False negative: high-risk employee labeled Low, no intervention offered | Right to fair treatment | Medium | High |
| R-02 | **Misclassification** | False positive: stable employee labeled High, subject to unnecessary intervention | Right to dignity, privacy | Medium | Medium |
| R-03 | **Proxy Discrimination** | Model uses features correlated with protected attributes (e.g., DistanceFromHome → socioeconomic status) | Non-discrimination | Medium | High |
| R-04 | **Automation Bias** | HR professionals over-rely on AI score, reducing independent judgment | Right to human oversight | Medium | High |
| R-05 | **Privacy** | Prediction inputs and audit logs contain sensitive employee data | Right to data protection | Low | High |
| R-06 | **Psychological Harm** | Employee learns they are "High Risk" — negative impact on morale and self-perception | Right to dignity | Low | Medium |
| R-07 | **Data Staleness** | Model trained on historical data becomes inaccurate as workforce dynamics change | Accuracy requirement | Medium | Medium |
| R-08 | **Adversarial Gaming** | Managers manipulate input features to alter predictions | System integrity | Low | Medium |

---

## 3. Risk Mitigation Measures (Art. 9(2)(d))

| Risk ID | Mitigation | Status |
|---------|------------|--------|
| R-01 | Optimize for F2 (recall-weighted), not accuracy. Calibrate probabilities with Isotonic Regression on held-out set. | ✅ Implemented |
| R-02 | Use three-tier system (Low/Medium/High) with human review required for all tiers. Cost model quantifies financial impact to prevent overreaction. | ✅ Implemented |
| R-03 | Fairlearn auditing computes DPD/EOD across Gender, Age, Marital Status. SHAP disclaimer warns about correlation ≠ causation. Quality gate blocks model publishing if DPD > 0.1. | ✅ Implemented |
| R-04 | "Override AI" button in dashboard records human disagreements. Mandatory override reason field. Training guidance for HR deployers. | ✅ Implemented |
| R-05 | Employee IDs can be anonymized. API key authentication. Audit database access restricted. PII not exposed in SHAP explanations. | ✅ Implemented |
| R-06 | System is deployer-facing only (HR team, not employees). Scores are never communicated directly to the scored individual. | ✅ Policy control |
| R-07 | Evidently AI drift detection runs on every batch. Drift check task in Airflow DAG parses report and alerts on significant drift. Daily retraining schedule. | ✅ Implemented |
| R-08 | All predictions are logged with full input features and model version for forensic review. Anomalous input patterns can be detected post-hoc. | ✅ Implemented |

---

## 4. Residual Risk Assessment (Art. 9(4))

| Risk ID | Residual Risk | Acceptable? | Justification |
|---------|---------------|-------------|---------------|
| R-01 | F2 ≈ 0.52 means ~48% of actual leavers may be missed | ⚠ Conditionally | Acceptable for advisory system with human oversight. NOT acceptable for automated decisions. Must improve with real-world data. |
| R-02 | Some false positives are inevitable in any probabilistic system | ✅ Yes | Mitigated by cost model and human review requirement. |
| R-03 | Complete elimination of proxy discrimination is impossible with observational data | ⚠ Conditionally | Fairlearn gate + SHAP disclaimer + human review provide defense-in-depth. Causal inference should be added for production. |
| R-04 | Some degree of automation bias is inherent in AI-assisted systems | ✅ Yes | Override mechanism and training materials mitigate this. |
| R-05 | Database contains sensitive data | ✅ Yes | Standard infosec controls (encryption at rest, access control). |

---

## 5. Post-Market Monitoring Plan (Art. 9(2)(c) + Art. 72)

| Monitoring Activity | Frequency | Tool | Owner |
|---------------------|-----------|------|-------|
| Data drift detection | Daily (automated) | Evidently AI | ML Engineering |
| Fairness audit (DPD/EOD) | Every retrain | Fairlearn | ML Engineering |
| Model performance vs. actual attrition | Quarterly | Manual analysis | People Analytics |
| Human override rate analysis | Monthly | SQL on human_overrides table | People Analytics |
| Prediction audit trail review | Quarterly | SQL on prediction_logs table | Compliance |
| System security audit | Semi-annually | External auditor | InfoSec |

---

## 6. Testing Summary (Art. 9(6)-(7))

| Test Type | Scope | Status |
|-----------|-------|--------|
| Unit tests | Feature engineering (features.py) | ✅ Passing |
| Integration tests | API endpoints, audit logging | ✅ Implemented |
| Fairness tests | DPD/EOD across 3 protected attributes | ✅ Passing |
| Data quality tests | dbt schema tests + Great Expectations | ✅ Implemented |
| Calibration validation | Brier score comparison (raw vs calibrated) | ✅ Passing |
| Drift detection | Evidently DataDrift + DataQuality presets | ✅ Automated |
| **Adversarial robustness** | Noise injection, boundary stability, determinism | ✅ Automated |
| **Causal validation** | DoWhy ATE estimation with refutation tests | ✅ Automated |
| **Survival analysis** | Cox PH hazard ratios + Kaplan-Meier curves | ✅ Automated |
| **Quality gate (EOD)** | Equalized Odds Difference now blocks biased models | ✅ Enforced |

---

## 7. Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-02 | G-3: Production DB enforcement — SQLite blocked in production | HR Analytics Engineering |
| 2026-05-02 | G-6: Model rollback mechanism via hot-swap API endpoint | HR Analytics Engineering |
| 2026-05-02 | G-8: GDPR Art. 21 opt-out endpoint for scoring exclusion | HR Analytics Engineering |
| 2026-05-02 | G-9: TLS termination via nginx reverse proxy | HR Analytics Engineering |
| 2026-05-02 | G-10: PII masking (SHA-256 hashing) in audit trail | HR Analytics Engineering |
| 2026-05-02 | G-12: Auto-retrain trigger from drift detection in Airflow DAG | HR Analytics Engineering |
| 2026-05-02 | G-14: Locust load testing script added | HR Analytics Engineering |
| 2026-05-02 | G-15: Trivy container vulnerability scanning in CI/CD | HR Analytics Engineering |
| 2026-05-02 | Added adversarial robustness testing (EU AI Act Art. 15) | HR Analytics Engineering |
| 2026-05-02 | Added causal inference validation (DoWhy) | HR Analytics Engineering |
| 2026-05-02 | Added survival analysis (Cox PH + Kaplan-Meier) | HR Analytics Engineering |
| 2026-05-02 | Quality gate updated to check EOD in addition to DPD | HR Analytics Engineering |
| 2026-05-02 | DPIA document created (GDPR Art. 35) | HR Analytics Engineering |
| 2026-04-25 | Initial Risk Management System document created | HR Analytics Engineering |
| 2026-04-25 | Calibration data leakage fixed (train/cal/test split) | HR Analytics Engineering |
| 2026-04-25 | Human override persistence implemented | HR Analytics Engineering |
| 2026-04-25 | Drift check task added to Airflow DAG | HR Analytics Engineering |

