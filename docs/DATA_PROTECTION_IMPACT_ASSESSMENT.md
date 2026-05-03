# Data Protection Impact Assessment (DPIA)
**AI-Powered Employee Attrition Intelligence System**

> **Document Status:** Living Document — update with each significant system change  
> **GDPR Article:** Article 35 — Data protection impact assessment  
> **Last Updated:** 2026-05-02  
> **DPO Review Required:** Yes (before production deployment)

---

## 1. System Description

### 1.1 Purpose
This AI system predicts employee attrition risk using machine learning to enable proactive retention interventions. It processes employee demographic, compensation, and satisfaction data to generate:
- Individual attrition probability scores (0–100%)
- Risk tier classifications (High / Medium / Low)
- Financial exposure estimates (replacement cost × probability)
- Feature-level explanations (SHAP values)
- AI-generated retention strategy recommendations (Gemini 2.5 Flash)

### 1.2 Data Processing Activities

| Activity | Data Categories | Legal Basis | Retention |
|----------|----------------|-------------|-----------|
| Model training | Demographics, compensation, job details, satisfaction scores | Legitimate interest (Art. 6(1)(f)) | Duration of model lifecycle |
| Real-time prediction | Individual employee record | Legitimate interest (Art. 6(1)(f)) | Indefinite (audit trail) |
| SHAP explanation | Derived from prediction inputs | Same as prediction | Same as prediction |
| LLM strategy generation | Anonymized feature summary | Legitimate interest (Art. 6(1)(f)) | Same as prediction |
| Human override | Override decision + justification | Legitimate interest (Art. 6(1)(f)) | Indefinite (audit trail) |
| Audit logging | Full input/output + metadata | Legal obligation (Art. 6(1)(c)) | 7 years minimum |

### 1.3 Data Subjects
- Current employees of the deploying organization
- Data is NOT used for automated decision-making under Art. 22 — the system is advisory only with mandatory human oversight

---

## 2. Necessity & Proportionality Assessment

### 2.1 Necessity
Employee attrition costs organizations 50–200% of annual salary per departure. Proactive identification of at-risk employees enables targeted interventions that benefit both the organization and the employee (e.g., workload rebalancing, career development, compensation review).

### 2.2 Proportionality
- **Data minimization:** The system uses only 19 input features (vs. potentially hundreds available in HRIS). No PII identifiers (name, email, SSN) are used in the model.
- **Purpose limitation:** Data is processed solely for attrition risk assessment and retention planning.
- **Storage limitation:** Audit logs are retained for regulatory compliance; model training data is version-controlled and can be deleted upon request.

### 2.3 Alternatives Considered

| Alternative | Assessment | Decision |
|------------|-----------|----------|
| Manual manager judgment only | Inconsistent, biased, not scalable | Rejected — but human oversight retained |
| Anonymous aggregate analytics only | Insufficient for individual interventions | Rejected — individual scoring needed |
| Opt-in employee self-assessment | Low participation, selection bias | Rejected — but employees informed |

---

## 3. Risk Assessment

### 3.1 Risks to Data Subjects

| Risk | Likelihood | Severity | Inherent Risk | Mitigation | Residual Risk |
|------|-----------|----------|---------------|------------|---------------|
| **Discriminatory treatment** based on protected attributes | Medium | High | **HIGH** | Fairlearn fairness audit (DPD + EOD), quality gate blocks biased models | Medium |
| **Stigmatization** of "high-risk" employees | Medium | High | **HIGH** | Human override mechanism, advisory-only system, causal disclaimers | Medium |
| **Unauthorized access** to sensitive predictions | Low | High | **MEDIUM** | API key auth, rate limiting, OWASP headers, error sanitization | Low |
| **Inaccurate predictions** leading to wrong interventions | Medium | Medium | **MEDIUM** | Probability calibration, SHAP explanations, human review mandatory | Low |
| **Re-identification** from model explanations | Low | Medium | **LOW** | SHAP values are relative, not absolute; no PII in model | Low |
| **Data breach** of prediction database | Low | High | **MEDIUM** | PostgreSQL with TDE recommended, access controls | Low (with TDE) |
| **Chilling effect** on employee behavior | Medium | Medium | **MEDIUM** | Transparent communication, employee access rights | Medium |

### 3.2 Risks from AI/ML Processing

| Risk | Mitigation Implemented |
|------|----------------------|
| Model memorization of training data | Regularization (L1/L2), limited tree depth, early stopping |
| Concept drift degrading accuracy | Evidently AI drift monitoring, Airflow alerting |
| Adversarial manipulation of inputs | Pydantic input validation, adversarial robustness testing (ART) |
| LLM hallucination in retention strategies | Prompt sanitization, causal disclaimers, human review |
| Correlation mistaken for causation | DoWhy causal validation layer, explicit SHAP disclaimers |

---

## 4. Data Subject Rights

### 4.1 Rights Implementation

| Right | Implementation Status |
|-------|--------------------|
| **Right to be informed** (Art. 13–14) | Employees must be informed that AI is used for risk assessment |
| **Right of access** (Art. 15) | API endpoint can return prediction history per employee ID |
| **Right to rectification** (Art. 16) | Incorrect employee data can be corrected in HRIS; model re-runs |
| **Right to erasure** (Art. 17) | Audit logs may be exempted for legal obligation; prediction data deletable |
| **Right to object** (Art. 21) | Employees can request exclusion from scoring — must be honored |
| **Right to human intervention** (Art. 22) | Override mechanism provides human review of any AI assessment |
| **Right to explanation** (Art. 22(3)) | SHAP-based local explanations provided with every prediction |

### 4.2 Process for Exercising Rights
1. Employee submits request to HR/DPO
2. DPO validates identity and right
3. Technical team executes (data export, deletion, or scoring exclusion)
4. Response within 30 days per GDPR Art. 12(3)

---

## 5. Technical & Organizational Measures

### 5.1 Security Controls
- [x] API authentication (key-based, RBAC planned)
- [x] Rate limiting (slowapi: 10 req/min predict, 30 req/min override)
- [x] Input validation (Pydantic with enum constraints)
- [x] OWASP security headers
- [x] Error message sanitization (no stack traces, no internal paths)
- [x] CORS restriction to allowed origins
- [x] **DONE:** Encryption in transit (TLS via nginx reverse proxy)
- [ ] **TODO:** Encryption at rest (PostgreSQL TDE)
- [x] **DONE:** Role-Based Access Control (RBAC) — 4 roles enforced

### 5.2 Organizational Controls
- [x] AI System Card documenting system purpose and limitations
- [x] Risk Management System (documented and maintained)
- [x] Human override mechanism with mandatory justification
- [x] Causal disclaimers on all AI-generated recommendations
- [ ] **TODO:** Employee notification policy
- [x] **DONE:** GDPR Art. 21 opt-out mechanism (scoring exclusion endpoint)
- [ ] **TODO:** Annual DPIA review schedule
- [ ] **TODO:** DPO sign-off before production launch

---

## 6. Consultation

### 6.1 Stakeholders Consulted

| Stakeholder | Status | Notes |
|------------|--------|-------|
| Data Protection Officer (DPO) | **PENDING** | Must review before production |
| HR Leadership | **PENDING** | Must approve use case and interventions |
| Works Council / Employee Representatives | **PENDING** | Required in many EU jurisdictions |
| IT Security Team | **PENDING** | Must validate infrastructure controls |
| Legal Counsel | **PENDING** | Must confirm legal basis and jurisdiction |

### 6.2 Supervisory Authority Consultation
Under GDPR Art. 36, prior consultation with the supervisory authority is required if the DPIA indicates high residual risk that cannot be sufficiently mitigated. Based on the current assessment, prior consultation is **not required** provided all TODO mitigations are implemented before production deployment.

---

## 7. Decision

| Decision | Justification |
|----------|--------------|
| **PROCEED with mitigations** | The system provides significant benefits for employee retention. Residual risks are manageable with the implemented controls and pending TODO items. Production deployment should only proceed after all Priority 1 mitigations are complete and DPO sign-off is obtained. |

---

## Appendix A: Review Schedule

| Review Trigger | Action |
|---------------|--------|
| Every 12 months | Full DPIA review |
| Significant system change | Targeted DPIA update |
| New data category added | Data minimization re-assessment |
| Fairness audit failure | Immediate risk re-assessment |
| Data breach incident | Emergency DPIA review |
| Regulatory guidance update | Compliance re-mapping |
