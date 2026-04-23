"""
AI-Powered Employee Attrition Intelligence System
Production Pipeline: Feature Engineering → XGBoost → SHAP → Cost Model → Risk Scoring
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, fbeta_score, 
                             roc_auc_score, confusion_matrix, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "HR-Employee-Attrition.csv"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# ── 1. FEATURE ENGINEERING ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create HR-domain-specific features grounded in organizational psychology."""
    df = df.copy()
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

    # Salary Competitiveness (Compa-Ratio): pay vs role peers
    df['Avg_Role_Income'] = df.groupby('JobRole')['MonthlyIncome'].transform('mean')
    df['Compa_Ratio'] = df['MonthlyIncome'] / df['Avg_Role_Income']

    # Promotion Stagnation: years stuck without upward mobility
    df['Promotion_Stagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    # Burnout Risk: overtime × commute burden / work-life balance
    df['Burnout_Risk'] = ((df['OverTime'] == 'Yes').astype(int) 
                          * df['DistanceFromHome'] / df['WorkLifeBalance'])

    # Manager Stability: tenure with current manager relative to company tenure
    df['Manager_Stability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)

    # Engagement Index: composite of satisfaction surveys
    df['Engagement_Index'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                              df['RelationshipSatisfaction'] + df['JobInvolvement']) / 4

    # Career Velocity: job level relative to total working years
    df['Career_Velocity'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1)

    # Income Growth Gap: salary hike vs expected growth
    df['Income_Growth_Gap'] = df['PercentSalaryHike'] - df.groupby('JobLevel')['PercentSalaryHike'].transform('median')

    # Loyalty Index: company tenure vs total career
    df['Loyalty_Index'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

    # Travel Burden (ordinal encode)
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df['Travel_Burden'] = df['BusinessTravel'].map(travel_map)

    # Drop constant/redundant columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Avg_Role_Income']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df


# ── 2. PREPROCESSING ──────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Split features/target and encode categoricals."""
    y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X, y


# ── 3. OPTUNA HYPERPARAMETER TUNING ───────────────────────────────────
def optimize_hyperparams(X_train, y_train, n_trials: int = 30):
    """Bayesian hyperparameter search with Optuna using stratified 3-fold CV."""
    scale_w = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_w,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(fbeta_score(y_train.iloc[val_idx], preds, beta=2))
        return np.mean(scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"  Best F2-Score (CV): {study.best_value:.4f}")
    return study.best_params


# ── 4. MODEL TRAINING ─────────────────────────────────────────────────
def train_model(X_train, y_train, best_params: dict):
    """Train final XGBoost with optimized hyperparameters."""
    scale_w = (y_train == 0).sum() / (y_train == 1).sum()
    params = {**best_params, 'scale_pos_weight': scale_w, 
              'eval_metric': 'logloss', 'random_state': 42}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ── 5. EVALUATION ─────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    """Full evaluation with classification report, F2, AUC, and confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    auc = roc_auc_score(y_test, y_proba)
    
    print(classification_report(y_test, y_pred))
    print(f"  F2-Score: {f2:.4f}  |  ROC-AUC: {auc:.4f}")
    
    # Save confusion matrix
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', 
                cmap='Blues', ax=ax, xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    
    return {'f2': f2, 'auc': auc, 'report': report}


# ── 6. SHAP EXPLAINABILITY ────────────────────────────────────────────
def explain_model(model, X_train, X_test):
    """Generate global and local SHAP explanations."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("Global SHAP: Systemic Attrition Drivers")
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'shap_summary.png', dpi=150)
    plt.close()

    # Bar importance
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, max_display=15)
    plt.title("Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'shap_bar.png', dpi=150)
    plt.close()

    return explainer, shap_values


# ── 7. RISK SCORING & COST MODEL ──────────────────────────────────────
def build_risk_framework(model, X_test, y_test, df_original, X_columns):
    """Assign risk tiers and calculate financial impact."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    risk_df = pd.DataFrame({
        'Predicted_Probability': y_proba,
        'Actual': y_test.values
    }, index=X_test.index)
    
    # Risk tiers
    risk_df['Risk_Tier'] = pd.cut(risk_df['Predicted_Probability'],
                                   bins=[0, 0.3, 0.6, 1.0],
                                   labels=['Low', 'Medium', 'High'])
    
    # Attach income for cost model (from original df)
    if 'MonthlyIncome' in df_original.columns:
        risk_df['MonthlyIncome'] = df_original.loc[X_test.index, 'MonthlyIncome'].values
        risk_df['Annual_Salary'] = risk_df['MonthlyIncome'] * 12
        risk_df['Replacement_Cost'] = risk_df['Annual_Salary'] * 1.5
        risk_df['Expected_Loss'] = risk_df['Predicted_Probability'] * risk_df['Replacement_Cost']
    
    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(OUT_DIR / 'risk_scores.csv', index=False)
    
    # Summary stats
    summary = risk_df.groupby('Risk_Tier', observed=True).agg(
        Count=('Predicted_Probability', 'count'),
        Avg_Risk=('Predicted_Probability', 'mean'),
        Total_Expected_Loss=('Expected_Loss', 'sum')
    ).round(2)
    print("\n-- Risk Tier Summary --")
    print(summary)
    print(f"\n  Total Value at Risk: ${risk_df['Expected_Loss'].sum():,.0f}")
    
    summary.to_csv(OUT_DIR / 'risk_summary.csv')
    return risk_df


# ── 8. EDA CHARTS ─────────────────────────────────────────────────────
def generate_eda(df: pd.DataFrame):
    """Generate key exploratory charts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Attrition rate by OverTime
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ot = df.groupby('OverTime')['Attrition'].mean() * 100
    axes[0].bar(ot.index, ot.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Attrition Rate by OverTime')
    axes[0].set_ylabel('Attrition %')
    
    # Attrition by Department
    dept = df.groupby('Department')['Attrition'].mean() * 100
    axes[1].barh(dept.index, dept.values, color='#3498db')
    axes[1].set_title('Attrition Rate by Department')
    axes[1].set_xlabel('Attrition %')
    
    # Income distribution by attrition
    for label, group in df.groupby('Attrition'):
        tag = 'Left' if label == 1 else 'Stayed'
        axes[2].hist(group['MonthlyIncome'], bins=30, alpha=0.6, label=tag)
    axes[2].set_title('Income Distribution by Attrition')
    axes[2].legend()
    axes[2].set_xlabel('Monthly Income')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'eda_overview.png', dpi=150)
    plt.close(fig)
    print("  EDA charts saved.")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HR Attrition Intelligence System - Production Pipeline")
    print("=" * 60)
    
    # Load & Engineer
    print("\n[1/7] Loading data & engineering HR features...")
    raw_df = pd.read_csv(DATA_PATH)
    df = engineer_features(raw_df)
    print(f"  Dataset: {df.shape[0]} employees, {df.shape[1]} features")
    print(f"  Attrition rate: {df['Attrition'].mean():.1%}")
    
    # EDA
    print("\n[2/7] Generating EDA charts...")
    generate_eda(df)
    
    # Preprocess
    print("\n[3/7] Preprocessing...")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Hyperparameter Tuning
    print("\n[4/7] Optimizing hyperparameters (Optuna, 30 trials)...")
    best_params = optimize_hyperparams(X_train, y_train, n_trials=30)
    print(f"  Best params: {best_params}")
    
    # Train
    print("\n[5/7] Training final model...")
    model = train_model(X_train, y_train, best_params)
    
    # Evaluate
    print("\n[6/7] Evaluating model...")
    metrics = evaluate(model, X_test, y_test)
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / 'xgb_attrition.json'))
    with open(MODEL_DIR / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  Model saved to {MODEL_DIR / 'xgb_attrition.json'}")
    
    # SHAP
    print("\n[7/7] Generating SHAP explanations...")
    explain_model(model, X_train, X_test)
    
    # Risk Framework & Cost Model
    print("\n-- Building Risk Framework & Cost Model --")
    risk_df = build_risk_framework(model, X_test, y_test, df, X.columns)
    
    print("\n" + "=" * 60)
    print("  Pipeline complete. All outputs saved to ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
