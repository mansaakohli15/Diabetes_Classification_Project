"""
Preprocessing and Feature Engineering Module.
Handles biological zero-to-NaN conversions, robust statistical imputation,
domain-specific medical feature engineering, and scikit-learn compatible pipelines.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Columns where a 0 is biologically impossible and represents missing data
ZERO_AS_NAN_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to engineer clinical risk indicators
    from physiological metrics for diabetes classification.
    """

    def __init__(self, add_interaction_features: bool = True):
        self.add_interaction_features = add_interaction_features
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derives clinical features:
        1. BMI Risk Tier (Normal: <25, Overweight: 25-30, Obese: >=30)
        2. Glucose to Insulin Index
        3. High Risk Age (> 35 years)
        4. Metabolic Syndrome Flag (BMI >= 30 and BloodPressure >= 80)
        5. Pregnancy to Age Ratio
        """
        if isinstance(X, np.ndarray):
            # If passed as numpy array, convert back to DataFrame if possible
            raise TypeError("MedicalFeatureEngineer requires a pandas DataFrame with named columns.")

        X_out = X.copy()

        if self.add_interaction_features:
            # 1. Insulin / Glucose interaction (indicator of insulin sensitivity)
            X_out["Insulin_Glucose_Ratio"] = X_out["Insulin"] / (X_out["Glucose"] + 1e-5)

            # 2. Age risk indicator (clinical diabetes risk rises significantly past 35)
            X_out["Age_Over_35"] = (X_out["Age"] >= 35).astype(float)

            # 3. High BMI Risk Category (Obesity indicator: BMI >= 30)
            X_out["Is_Obese"] = (X_out["BMI"] >= 30.0).astype(float)

            # 4. Metabolic Risk Combination (Obesity + Hypertension)
            X_out["Metabolic_Risk"] = (
                (X_out["BMI"] >= 30.0) & (X_out["BloodPressure"] >= 80.0)
            ).astype(float)

            # 5. Pregnancy-to-Age Ratio (Parity rate)
            X_out["Pregnancy_Age_Ratio"] = X_out["Pregnancies"] / (X_out["Age"] + 1.0)

        self.feature_names_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Replaces biological zeros in specified physiological columns with NaN.
    """

    def __init__(self, zero_cols: Optional[List[str]] = None):
        self.zero_cols = zero_cols or ZERO_AS_NAN_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.zero_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].replace(0, np.nan)
        return X_out


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Imputer wrapper that preserves pandas DataFrame column headers and types.
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = self.imputer.transform(X)
        return pd.DataFrame(X_imp, columns=self.columns_, index=X.index)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Scaler wrapper that preserves pandas DataFrame structure.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)


def build_preprocessing_pipeline(
    imputation_strategy: str = "median",
    scaler_type: str = "robust",
    add_features: bool = True,
) -> Pipeline:
    """
    Build complete scikit-learn preprocessing pipeline.
    
    Steps:
    1. Replace biological 0s with NaN.
    2. Impute missing values with statistical medians.
    3. Generate clinical features.
    4. Robust / Standard feature scaling.
    """
    pipeline = Pipeline(
        steps=[
            ("zero_to_nan", ZeroToNaNTransformer()),
            ("imputer", DataFrameImputer(strategy=imputation_strategy)),
            ("feature_engineer", MedicalFeatureEngineer(add_interaction_features=add_features)),
            ("scaler", DataFrameScaler(scaler_type=scaler_type)),
        ]
    )
    return pipeline


if __name__ == "__main__":
    from data_loader import load_dataset, get_data_splits
    
    data = load_dataset()
    X_train, X_test, y_train, y_test = get_data_splits(data)
    
    pipe = build_preprocessing_pipeline()
    X_train_proc = pipe.fit_transform(X_train)
    X_test_proc = pipe.transform(X_test)
    
    print("Preprocessing successful!")
    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Engineered feature count: {X_train_proc.shape[1]}")
    print("Features:", list(X_train_proc.columns))
