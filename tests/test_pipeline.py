"""
Unit and Integration Tests for Diabetes Classification Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, validate_dataset, get_data_splits, FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import (
    build_preprocessing_pipeline,
    ZeroToNaNTransformer,
    MedicalFeatureEngineer,
)
from src.models import (
    get_candidate_models,
    benchmark_models,
    tune_best_model,
    save_model_pipeline,
    load_model_pipeline,
)
from src.predict import DiabetesPredictor, categorize_risk
from src.evaluation import evaluate_model_performance


class TestDiabetesPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a synthetic mini-dataset for fast reproducible testing
        np.random.seed(42)
        n_samples = 50
        cls.test_df = pd.DataFrame({
            "Pregnancies": np.random.randint(0, 10, n_samples),
            "Glucose": np.random.choice([0, 90, 120, 150, 180], n_samples),
            "BloodPressure": np.random.choice([0, 60, 75, 85, 95], n_samples),
            "SkinThickness": np.random.choice([0, 15, 25, 35], n_samples),
            "Insulin": np.random.choice([0, 50, 100, 200], n_samples),
            "BMI": np.random.choice([0.0, 22.5, 28.0, 34.0], n_samples),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.2, n_samples),
            "Age": np.random.randint(20, 70, n_samples),
            "Outcome": np.random.choice([0, 1], n_samples),
        })

    def test_data_validation(self):
        """Test dataset schema validation."""
        self.assertTrue(validate_dataset(self.test_df))

        # Test invalid schema
        invalid_df = self.test_df.drop(columns=["Glucose"])
        with self.assertRaises(ValueError):
            validate_dataset(invalid_df)

    def test_data_splits(self):
        """Test stratified train-test splitting."""
        X_train, X_test, y_train, y_test = get_data_splits(self.test_df, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train) + len(X_test), len(self.test_df))
        self.assertEqual(len(X_train), 40)
        self.assertEqual(len(X_test), 10)

    def test_zero_to_nan_transformer(self):
        """Test replacement of biological zeros with NaN."""
        transformer = ZeroToNaNTransformer(zero_cols=["Glucose", "BMI"])
        transformed = transformer.transform(self.test_df)
        self.assertFalse((transformed["Glucose"] == 0).any())
        self.assertFalse((transformed["BMI"] == 0).any())

    def test_feature_engineering(self):
        """Test clinical feature generation."""
        fe = MedicalFeatureEngineer(add_interaction_features=True)
        transformed = fe.transform(self.test_df)
        expected_cols = [
            "Insulin_Glucose_Ratio",
            "Age_Over_35",
            "Is_Obese",
            "Metabolic_Risk",
            "Pregnancy_Age_Ratio",
        ]
        for col in expected_cols:
            self.assertIn(col, transformed.columns)

    def test_preprocessing_pipeline(self):
        """Test full preprocessing pipeline transformation."""
        pipeline = build_preprocessing_pipeline()
        X = self.test_df[FEATURE_COLUMNS]
        X_proc = pipeline.fit_transform(X)
        self.assertFalse(X_proc.isnull().any().any())
        self.assertGreater(X_proc.shape[1], X.shape[1])

    def test_model_training_and_serialization(self):
        """Test model training, evaluation, saving, and loading."""
        X_train, X_test, y_train, y_test = get_data_splits(self.test_df, test_size=0.2, random_state=42)
        
        # Build and train pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        # Evaluation
        metrics = evaluate_model_performance(pipeline, X_test, y_test)
        self.assertIn("Accuracy", metrics)
        self.assertIn("ROC-AUC", metrics)
        self.assertTrue(0.0 <= metrics["Accuracy"] <= 1.0)

        # Save and load
        temp_model_path = os.path.join(os.path.dirname(__file__), "temp_test_model.joblib")
        save_model_pipeline(pipeline, temp_model_path)
        self.assertTrue(os.path.exists(temp_model_path))

        loaded_pipeline = load_model_pipeline(temp_model_path)
        self.assertIsNotNone(loaded_pipeline)

        # Test Predictor
        predictor = DiabetesPredictor(loaded_pipeline)
        sample_patient = {
            "Pregnancies": 1,
            "Glucose": 110.0,
            "BloodPressure": 70.0,
            "SkinThickness": 20.0,
            "Insulin": 80.0,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.35,
            "Age": 28,
        }
        res = predictor.predict_single(sample_patient)
        self.assertIn("prediction", res)
        self.assertIn("diabetes_probability", res)
        self.assertIn("risk_tier", res)

        # Cleanup
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    def test_risk_categorization(self):
        """Test risk tier boundaries."""
        tier_low, _ = categorize_risk(0.15)
        self.assertEqual(tier_low, "Low Risk")

        tier_mod, _ = categorize_risk(0.45)
        self.assertEqual(tier_mod, "Moderate Risk")

        tier_high, _ = categorize_risk(0.85)
        self.assertEqual(tier_high, "High Risk")


if __name__ == "__main__":
    unittest.main()
