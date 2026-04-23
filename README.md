# AI-Powered Employee Attrition Intelligence System 🚀

This repository contains an end-to-end Machine Learning, Power BI, and API architecture designed to shift Human Resources from reactive attrition reporting to proactive, data-driven retention intelligence. 

## 📖 The Case Study Narrative (Problem -> Approach -> Insights -> Impact)

### 🔴 The Problem
Employee turnover is a silent multi-million-dollar liability. According to Gallup and SHRM, the cost to replace an employee ranges from **50% to 200% of their annual salary** due to recruitment costs, onboarding, and lost productivity. HR departments traditionally rely on lagging indicators (e.g., quarterly attrition reports), preventing intervention before a high-performing employee decides to leave.

### 🧠 The Approach
We built an end-to-end Decision Intelligence system utilizing the **IBM HR Analytics Dataset**. 
1. **Domain-Specific Feature Engineering**: Instead of relying on raw data, we engineered psychological and organizational features such as `Burnout_Risk`, `Promotion_Stagnation`, and `Compa_Ratio` (Salary Competitiveness).
2. **Predictive Modeling**: We utilized **XGBoost** with **Optuna** Bayesian Hyperparameter optimization. To handle class imbalance (16% attrition), we utilized Cost-Sensitive Learning (`scale_pos_weight`), prioritizing Recall to catch flight risks without creating fake synthetic data.
3. **Explainable AI (SHAP)**: We wrapped the model in SHAP (SHapley Additive exPlanations) to translate "black-box" probabilities into actionable, localized insights for HR managers.

### 💡 The Insights
Our ML pipeline and EDA uncovered non-obvious organizational truths:
*   **The Overtime Trap**: Overtime is the single strongest predictor of attrition, driving a 30.5% attrition rate compared to just 10.4% for non-overtime workers.
*   **Income & Seniority Anchor**: `TotalWorkingYears` and `MonthlyIncome` act as the strongest retention anchors. 
*   **SHAP Transparency**: Our model proved mathematically that intervention should focus on reducing overtime and assessing competitive pay (`Compa_Ratio`) rather than generic "pizza party" retention strategies.

### 💥 The Business Impact
The model successfully identified **$7.17M in total organizational value at risk**. By translating ML probabilities into a Risk Tier Framework (Low/Medium/High) and connecting it to a financial Expected Loss model, HR can surgically deploy a retention budget. **For example, utilizing a $200k targeted retention budget on the top 40 high-risk employees could yield over $1M in turnover cost savings.**

---

## 🏗️ System Architecture

1. **`datasets/`**: IBM HR Analytics Dataset.
2. **`src/train_attrition_model.py`**: The production MLOps pipeline featuring data preprocessing, engineered features, Optuna tuning, and SHAP visualization exports.
3. **`src/api.py`**: A **FastAPI** application offering a real-time `/predict` endpoint. It accepts a JSON payload, dynamically calculates features, scores the employee, and returns the top 3 localized SHAP drivers to explain *why* the employee is at risk.
4. **`powerbi/`**: Architecture diagrams and DAX documentation for the enterprise dashboard, featuring what-if budget simulation parameters and Row-Level Security (RLS).
5. **`notebooks/`**: Interactive Jupyter Notebook for exploratory data analysis and model prototyping.

---

## 🚀 How to Pitch This Project (Portfolio Guide)

If you are using this repository for your portfolio, here is how to position it:

### 📝 On Your CV / Resume
> **AI HR Analytics & Decision Intelligence System**
> * Engineered an end-to-end XGBoost machine learning pipeline to predict employee attrition, optimizing for F2-Score via Optuna to catch 53%+ of flight risks.
> * Designed custom organizational features (`Burnout_Risk`, `Compa_Ratio`) and integrated SHAP explainability, moving the business from "black-box" models to transparent, action-oriented HR interventions.
> * Built a financial cost-impact framework estimating $7.1M in at-risk talent, and deployed a real-time FastAPI endpoint to serve predictions directly to business intelligence dashboards.

### 💼 On LinkedIn
**Headline hook:** *Predictive analytics isn't just for marketing—it's time we apply it to our most valuable asset: our people.*
**The post:** Highlight the "Business Impact" section above. Share one of the SHAP visual charts generated in the `outputs/` folder and explain how Explainable AI (SHAP) is the only way to build trust with non-technical stakeholders like an HR Director.

### 🎤 In Interviews
**When asked:** *"Tell me about a time you solved a business problem with data."*
**Your answer:** Do not start by talking about XGBoost or Python. Start by talking about the **Cost of Turnover**. Explain that you wanted to build a system that didn't just guess who was leaving, but calculated the *financial risk* of them leaving, and provided the exact reason *why* they were leaving (via SHAP) so the business knew exactly how to intervene. Then, explain your architecture.
