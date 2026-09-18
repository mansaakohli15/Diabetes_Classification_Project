"""
Data Loader Module for Diabetes Classification.
Handles data fetching from remote URL or local cache with validation.
"""

import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np

DEFAULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DEFAULT_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "diabetes.csv")

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(
    source_path: Optional[str] = None,
    save_raw_copy: bool = True
) -> pd.DataFrame:
    """
    Load the PIMA Indians Diabetes dataset from a local path or remote URL.
    
    Parameters:
    -----------
    source_path : str, optional
        Path to local CSV or URL. If None, tries local cache then remote URL.
    save_raw_copy : bool
        Whether to cache the downloaded data to data/raw/diabetes.csv.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with standard column names.
    """
    df = None
    
    # 1. If explicit path is passed
    if source_path:
        if os.path.exists(source_path):
            df = pd.read_csv(source_path)
        elif source_path.startswith("http://") or source_path.startswith("https://"):
            df = pd.read_csv(source_path, names=ALL_COLUMNS, header=0 if "Pregnancies" in pd.read_csv(source_path, nrows=1).columns else None)
        else:
            raise FileNotFoundError(f"Source path not found: {source_path}")
            
    # 2. Try default local path
    elif os.path.exists(DEFAULT_LOCAL_PATH):
        try:
            df = pd.read_csv(DEFAULT_LOCAL_PATH)
            # If loaded without headers, assign names
            if df.columns.tolist() != ALL_COLUMNS and len(df.columns) == len(ALL_COLUMNS):
                if not any(c in df.columns for c in ["Pregnancies", "Outcome"]):
                    df = pd.read_csv(DEFAULT_LOCAL_PATH, names=ALL_COLUMNS)
        except Exception:
            df = None

    # 3. Fallback to remote URL
    if df is None:
        try:
            df = pd.read_csv(DEFAULT_URL, names=ALL_COLUMNS)
            if save_raw_copy:
                os.makedirs(os.path.dirname(DEFAULT_LOCAL_PATH), exist_ok=True)
                df.to_csv(DEFAULT_LOCAL_PATH, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch dataset from {DEFAULT_URL}: {e}")

    # Standardize column names if headerless
    if len(df.columns) == len(ALL_COLUMNS) and "Outcome" not in df.columns:
        df.columns = ALL_COLUMNS

    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate dataset structure and column integrity.
    """
    for col in ALL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    if df.empty:
        raise ValueError("Dataset is empty.")
        
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains non-binary values.")
        
    return True


def get_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target vector into train and test sets with stratification.
    """
    from sklearn.model_selection import train_test_split
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    data = load_dataset()
    print("Dataset successfully loaded:")
    print(f"Shape: {data.shape}")
    print(f"Class distribution:\n{data['Outcome'].value_counts(normalize=True)}")
