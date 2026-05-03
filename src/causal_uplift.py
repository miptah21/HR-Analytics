import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path

class TLearnerUplift(BaseEstimator, ClassifierMixin):
    """
    A custom T-Learner (Two-Model Estimator) for Causal Inference (Uplift Modeling).
    This native implementation uses two XGBoost models, avoiding complex dependencies
    like econml or causalml which often conflict with modern python environments.
    
    The T-Learner trains:
    - Model 0: on the Control group (e.g., Did not receive high salary hike)
    - Model 1: on the Treatment group (e.g., Received high salary hike)
    
    CATE (Conditional Average Treatment Effect) is computed as:
    E[Y | X, T=1] - E[Y | X, T=0]
    
    In the context of Attrition (where Y=1 means leaving):
    - A NEGATIVE CATE means the treatment REDUCES attrition (good!).
    - A POSITIVE CATE means the treatment INCREASES attrition (bad!).
    """
    def __init__(self, treatment_col: str, **xgb_kwargs):
        self.treatment_col = treatment_col
        # We use default XGBoost parameters, but these can be tuned.
        # We explicitly set eval_metric to suppress warnings
        default_params = {
            "eval_metric": "logloss",
            "random_state": 42,
            "use_label_encoder": False
        }
        default_params.update(xgb_kwargs)
        
        self.model_0 = xgb.XGBClassifier(**default_params)
        self.model_1 = xgb.XGBClassifier(**default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.treatment_col not in X.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found in X.")
            
        # Split data into treatment (1) and control (0)
        treatment_mask = X[self.treatment_col] == 1
        
        X_1 = X[treatment_mask].drop(columns=[self.treatment_col])
        y_1 = y[treatment_mask]
        
        X_0 = X[~treatment_mask].drop(columns=[self.treatment_col])
        y_0 = y[~treatment_mask]
        
        # Train both models
        if len(y_1) > 0:
            self.model_1.fit(X_1, y_1)
        else:
            raise ValueError("No positive treatment samples found.")
            
        if len(y_0) > 0:
            self.model_0.fit(X_0, y_0)
        else:
            raise ValueError("No control samples found.")
            
        return self
        
    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate the Conditional Average Treatment Effect (CATE).
        Returns the difference in probability of Y=1 between Treatment and Control.
        """
        X_features = X.copy()
        if self.treatment_col in X_features.columns:
            X_features = X_features.drop(columns=[self.treatment_col])
            
        # Predict probability of Attrition=1 under Treatment
        prob_1 = self.model_1.predict_proba(X_features)[:, 1]
        
        # Predict probability of Attrition=1 under Control
        prob_0 = self.model_0.predict_proba(X_features)[:, 1]
        
        # CATE = P(Y=1|T=1) - P(Y=1|T=0)
        return prob_1 - prob_0

    def recommend_intervention(self, X: pd.DataFrame) -> tuple[bool, float]:
        """
        Returns whether the intervention should be applied based on CATE.
        Since Y=1 is Attrition, we want CATE < 0 (Intervention reduces attrition).
        Returns:
            (should_intervene: bool, expected_probability_reduction: float)
        """
        cate = self.predict_cate(X)[0]
        # If CATE < -0.05, it means the intervention drops attrition risk by at least 5%
        should_intervene = cate < -0.05 
        return should_intervene, abs(cate)

    def save(self, path: str | Path):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str | Path) -> 'TLearnerUplift':
        return joblib.load(path)
