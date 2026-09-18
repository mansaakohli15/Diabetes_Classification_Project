"""
Inference Module for Single-Patient and Batch Diabetes Risk Prediction.
"""

from typing import Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.models import load_model_pipeline
from src.data_loader import FEATURE_COLUMNS


def categorize_risk(probability: float) -> Tuple[str, str]:
    """
    Categorize patient risk based on predicted probability.
    """
    if probability < 0.30:
        return "Low Risk", "Normal diagnostic indicators. Standard routine screening recommended."
    elif probability < 0.65:
        return "Moderate Risk", "Elevated indicators detected. Follow-up lifestyle modification & HbA1c screening advised."
    else:
        return "High Risk", "High probability of diabetes/metabolic syndrome. Immediate clinical consultation advised."


class DiabetesPredictor:
    """
    Production-ready inference engine for diabetes risk prediction.
    """

    def __init__(self, model_pipeline: Union[Pipeline, str, None] = None):
        if isinstance(model_pipeline, str) or model_pipeline is None:
            self.pipeline = load_model_pipeline(model_pipeline)
        else:
            self.pipeline = model_pipeline

    def predict_single(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference for an individual patient record.
        """
        for col in FEATURE_COLUMNS:
            if col not in patient_data:
                raise ValueError(f"Missing feature '{col}' in patient input.")

        df_input = pd.DataFrame([patient_data])[FEATURE_COLUMNS]
        
        prediction = int(self.pipeline.predict(df_input)[0])
        probability = float(self.pipeline.predict_proba(df_input)[0][1])
        risk_level, clinical_note = categorize_risk(probability)

        return {
            "prediction": prediction,
            "prediction_label": "Diabetes Positive (High Risk)" if prediction == 1 else "Diabetes Negative (Low Risk)",
            "diabetes_probability": round(probability, 4),
            "risk_tier": risk_level,
            "clinical_recommendation": clinical_note,
        }

    def predict_batch(self, df_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a batch DataFrame of patients.
        """
        for col in FEATURE_COLUMNS:
            if col not in df_patients.columns:
                raise ValueError(f"Batch dataset missing required feature: '{col}'")

        df_features = df_patients[FEATURE_COLUMNS].copy()
        preds = self.pipeline.predict(df_features)
        probas = self.pipeline.predict_proba(df_features)[:, 1]

        df_out = df_patients.copy()
        df_out["Predicted_Outcome"] = preds
        df_out["Diabetes_Probability"] = np.round(probas, 4)
        df_out["Risk_Category"] = [categorize_risk(p)[0] for p in probas]

        return df_out


if __name__ == "__main__":
    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 145.0,
        "BloodPressure": 82.0,
        "SkinThickness": 30.0,
        "Insulin": 180.0,
        "BMI": 33.5,
        "DiabetesPedigreeFunction": 0.65,
        "Age": 45,
    }
    predictor = DiabetesPredictor()
    result = predictor.predict_single(sample_patient)
    print("Sample Inference Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
