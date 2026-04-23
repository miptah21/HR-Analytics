import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uvicorn

# ── Paths & Globals ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_attrition.json"

app = FastAPI(
    title="HR Attrition Intelligence API",
    description="Real-time predictive scoring and explainability API for HR Systems.",
    version="1.0.0"
)

# Global variables for model and explainer
model = None
explainer = None
# For Compa-Ratio calculations, we'd normally query a database. 
# Here we mock the average income by role for demonstration.
MOCK_ROLE_INCOMES = {
    "Sales Executive": 6500,
    "Research Scientist": 3200,
    "Laboratory Technician": 3200,
    "Manufacturing Director": 7300,
    "Healthcare Representative": 7500,
    "Manager": 17000,
    "Sales Representative": 2600,
    "Research Director": 16000,
    "Human Resources": 4200
}

# ── Pydantic Schema ───────────────────────────────────────────────────
class EmployeeData(BaseModel):
    EmployeeID: str
    Age: int
    JobRole: str
    JobLevel: int
    MonthlyIncome: float
    PercentSalaryHike: float
    OverTime: str  # "Yes" or "No"
    DistanceFromHome: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    TotalWorkingYears: int
    JobSatisfaction: int
    EnvironmentSatisfaction: int
    RelationshipSatisfaction: int
    JobInvolvement: int
    BusinessTravel: str

class PredictionResponse(BaseModel):
    EmployeeID: str
    Risk_Probability: float
    Risk_Tier: str
    Expected_Financial_Loss: float
    Top_Risk_Drivers: dict
    Retention_Strategy: str = ""

# ── Startup Event ─────────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, explainer
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Please run train_attrition_model.py first.")
    
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    explainer = shap.TreeExplainer(model)
    print("Model and SHAP explainer loaded successfully.")

# ── Feature Engineering logic ─────────────────────────────────────────
def engineer_api_features(data: EmployeeData) -> pd.DataFrame:
    """Replicates the exact feature engineering pipeline used during training."""
    
    # Calculate custom metrics
    avg_income = MOCK_ROLE_INCOMES.get(data.JobRole, data.MonthlyIncome)
    compa_ratio = data.MonthlyIncome / avg_income if avg_income else 1.0
    
    promotion_stagnation = data.YearsSinceLastPromotion / (data.YearsAtCompany + 1)
    burnout_risk = (1 if data.OverTime == "Yes" else 0) * data.DistanceFromHome / max(data.WorkLifeBalance, 1)
    manager_stability = data.YearsWithCurrManager / (data.YearsAtCompany + 1)
    engagement_index = (data.JobSatisfaction + data.EnvironmentSatisfaction + 
                        data.RelationshipSatisfaction + data.JobInvolvement) / 4
    career_velocity = data.JobLevel / (data.TotalWorkingYears + 1)
    loyalty_index = data.YearsAtCompany / (data.TotalWorkingYears + 1)
    
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    travel_burden = travel_map.get(data.BusinessTravel, 1)

    # Note: In a true production environment, you must ensure the DataFrame 
    # columns exactly match the one-hot encoded columns the XGBoost model expects.
    # For this blueprint, we construct a simplified representation of the numeric features.
    
    features = {
        'Age': data.Age,
        'DistanceFromHome': data.DistanceFromHome,
        'MonthlyIncome': data.MonthlyIncome,
        'NumCompaniesWorked': 1, # Default/Mock
        'PercentSalaryHike': data.PercentSalaryHike,
        'TotalWorkingYears': data.TotalWorkingYears,
        'YearsAtCompany': data.YearsAtCompany,
        'YearsInCurrentRole': data.YearsInCurrentRole,
        'YearsSinceLastPromotion': data.YearsSinceLastPromotion,
        'YearsWithCurrManager': data.YearsWithCurrManager,
        'Compa_Ratio': compa_ratio,
        'Promotion_Stagnation': promotion_stagnation,
        'Burnout_Risk': burnout_risk,
        'Manager_Stability': manager_stability,
        'Engagement_Index': engagement_index,
        'Career_Velocity': career_velocity,
        'Loyalty_Index': loyalty_index,
        'Travel_Burden': travel_burden,
        'OverTime_Yes': 1 if data.OverTime == "Yes" else 0
    }
    
    # Create DataFrame (missing one-hot columns will be filled with 0 if necessary)
    df = pd.DataFrame([features])
    
    # Align columns with XGBoost expectation (mocking the align process)
    expected_cols = model.get_booster().feature_names
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder to match model
    df = df[expected_cols]
    return df

# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(employee: EmployeeData):
    try:
        # 1. Feature Engineering
        X_infer = engineer_api_features(employee)
        
        # 2. Prediction
        proba = float(model.predict_proba(X_infer)[0][1])
        
        # 3. Decision Logic & Cost Model
        if proba >= 0.60:
            tier = "High"
        elif proba >= 0.30:
            tier = "Medium"
        else:
            tier = "Low"
            
        annual_salary = employee.MonthlyIncome * 12
        replacement_cost = annual_salary * 1.5
        expected_loss = round(proba * replacement_cost, 2)
        
        # 4. SHAP Local Interpretability (Why is this specific person at risk?)
        shap_vals = explainer.shap_values(X_infer)[0]
        
        # Zip feature names with their SHAP impact and sort by absolute magnitude
        feature_impacts = list(zip(X_infer.columns, shap_vals))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Extract the top 3 drivers pushing the score up (positive SHAP values)
        top_drivers = {feat: round(float(val), 4) for feat, val in feature_impacts if val > 0}
        top_3_drivers = dict(list(top_drivers.items())[:3])
        
        # 5. LLM Retention Copilot
        retention_strategy = "Continue regular check-ins."
        if tier in ["High", "Medium"]:
            try:
                import os
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    You are an expert HR Business Partner. 
                    An employee in the '{employee.JobRole}' role is a high flight risk.
                    Top predictive drivers for their departure: {top_3_drivers}.
                    Write a brief, 2-sentence actionable retention strategy for their manager.
                    Do not mention 'AI' or 'SHAP'. Focus purely on human management tactics.
                    """
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    retention_strategy = response.text.strip()
            except Exception as e:
                print(f"LLM Generation Failed: {e}")

        # Construct final response
        return PredictionResponse(
            EmployeeID=employee.EmployeeID,
            Risk_Probability=round(proba, 4),
            Risk_Tier=tier,
            Expected_Financial_Loss=expected_loss,
            Top_Risk_Drivers=top_3_drivers,
            Retention_Strategy=retention_strategy
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "operational", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
