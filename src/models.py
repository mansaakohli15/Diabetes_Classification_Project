"""
Model Training, Benchmarking, and Hyperparameter Tuning Module.
Integrates preprocessing with multiple classifier algorithms and evaluates performance.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC

from src.preprocessing import build_preprocessing_pipeline

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate classification models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }


def benchmark_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified cross-validation across all candidate models inside the preprocessing pipeline.
    """
    models = get_candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    results = []
    
    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", clf),
        ])
        
        cv_res = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        
        results.append({
            "Model": name,
            "Accuracy": np.mean(cv_res["test_accuracy"]),
            "Precision": np.mean(cv_res["test_precision"]),
            "Recall": np.mean(cv_res["test_recall"]),
            "F1-Score": np.mean(cv_res["test_f1"]),
            "ROC-AUC": np.mean(cv_res["test_roc_auc"]),
            "Std ROC-AUC": np.std(cv_res["test_roc_auc"]),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_results


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "Random Forest",
    random_state: int = 42,
) -> Pipeline:
    """
    Fine-tunes the selected model architecture using Grid Search CV.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if model_type == "Random Forest":
        pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", RandomForestClassifier(random_state=random_state, class_weight="balanced")),
        ])
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "Gradient Boosting":
        pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", GradientBoostingClassifier(random_state=random_state)),
        ])
        param_grid = {
            "classifier__n_estimators": [80, 120, 160],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4],
            "classifier__subsample": [0.8, 1.0],
        }
    else:  # Logistic Regression fallback
        pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")),
        ])
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
        }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def save_model_pipeline(
    pipeline: Pipeline,
    filepath: Optional[str] = None
) -> str:
    """
    Serialize the trained scikit-learn pipeline to disk.
    """
    if filepath is None:
        os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(DEFAULT_MODEL_DIR, "best_diabetes_pipeline.joblib")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    joblib.dump(pipeline, filepath)
    return filepath


def load_model_pipeline(filepath: Optional[str] = None) -> Pipeline:
    """
    Load a serialized scikit-learn pipeline from disk.
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_MODEL_DIR, "best_diabetes_pipeline.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at {filepath}. Please train a model first.")
    return joblib.load(filepath)
