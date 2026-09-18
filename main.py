"""
Main CLI Entrypoint for Diabetes Classification Project.
Usage:
    python main.py --train --evaluate --export-plots
    python main.py --benchmark
    python main.py --predict
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, get_data_splits, FEATURE_COLUMNS
from src.preprocessing import build_preprocessing_pipeline
from src.models import benchmark_models, tune_best_model, save_model_pipeline, load_model_pipeline
from src.evaluation import (
    evaluate_model_performance,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_feature_importance,
    plot_correlation_heatmap,
)
from src.predict import DiabetesPredictor


def run_benchmark(data_path: str = None):
    print("=" * 60)
    print("RUNNING MULTI-MODEL BENCHMARK (Stratified 5-Fold CV)")
    print("=" * 60)
    data = load_dataset(data_path)
    X = data[FEATURE_COLUMNS]
    y = data["Outcome"]

    benchmark_df = benchmark_models(X, y)
    print("\nBenchmark Results Summary:")
    print(benchmark_df.to_string(index=False))

    plot_path = plot_model_comparison(benchmark_df)
    print(f"\n[Saved] Benchmark comparison plot -> {plot_path}")
    return benchmark_df


def run_training_pipeline(data_path: str = None, export_plots: bool = True, model_choice: str = "Random Forest"):
    print("=" * 60)
    print("STARTING END-TO-END TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1/5] Loading & validating PIMA Diabetes Dataset...")
    data = load_dataset(data_path)
    print(f"Loaded {len(data)} patient records with {data.shape[1]} features.")

    # 2. Split Data (Stratified)
    print("\n[Step 2/5] Splitting data into 80% Train and 20% Test sets (Stratified)...")
    X_train, X_test, y_train, y_test = get_data_splits(data, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Model Benchmark & Hyperparameter Tuning
    print(f"\n[Step 3/5] Tuning {model_choice} with GridSearchCV...")
    best_pipeline = tune_best_model(X_train, y_train, model_type=model_choice)
    
    # 4. Save Model Artifact
    model_path = save_model_pipeline(best_pipeline)
    print(f"\n[Step 4/5] [Saved] Best model pipeline serialized -> {model_path}")

    # 5. Comprehensive Evaluation
    print("\n[Step 5/5] Evaluating model on Holdout Test Set...")
    metrics = evaluate_model_performance(best_pipeline, X_test, y_test)

    print("\n" + "=" * 40)
    print("FINAL TEST METRICS:")
    print("=" * 40)
    print(f" Accuracy            : {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f" Precision           : {metrics['Precision']:.4f}")
    print(f" Recall (Sensitivity): {metrics['Recall (Sensitivity)']:.4f}")
    print(f" Specificity         : {metrics['Specificity']:.4f}")
    print(f" F1-Score            : {metrics['F1-Score']:.4f}")
    if "ROC-AUC" in metrics:
        print(f" ROC-AUC             : {metrics['ROC-AUC']:.4f}")
        print(f" PR-AUC              : {metrics['PR-AUC']:.4f}")

    if export_plots:
        print("\nGenerating publication-grade visualization reports in reports/figures/...")
        # Correlation
        corr_path = plot_correlation_heatmap(data)
        print(f"  - Correlation Heatmap: {corr_path}")
        
        # Benchmark
        bench_df = benchmark_models(data[FEATURE_COLUMNS], data["Outcome"])
        bench_path = plot_model_comparison(bench_df)
        print(f"  - Model Benchmark:     {bench_path}")

        # Confusion Matrix
        cm_path = plot_confusion_matrix(np.array(metrics["Confusion Matrix"]))
        print(f"  - Confusion Matrix:    {cm_path}")

        # ROC & PR
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        roc_path = plot_roc_pr_curves(y_test, y_proba)
        print(f"  - ROC & PR Curves:     {roc_path}")

        # Feature Importance
        feat_path = plot_feature_importance(best_pipeline)
        print(f"  - Feature Importance:  {feat_path}")

    print("\n" + "=" * 60)
    print("SUCCESS: Pipeline execution complete!")
    print("=" * 60)
    return best_pipeline, metrics


def run_sample_prediction():
    print("=" * 60)
    print("RUNNING SAMPLE PATIENT INFERENCE")
    print("=" * 60)
    predictor = DiabetesPredictor()
    
    sample_patients = [
        {
            "Name": "Patient A (High Risk Profile)",
            "Data": {
                "Pregnancies": 6,
                "Glucose": 168.0,
                "BloodPressure": 90.0,
                "SkinThickness": 35.0,
                "Insulin": 220.0,
                "BMI": 38.2,
                "DiabetesPedigreeFunction": 0.85,
                "Age": 52,
            }
        },
        {
            "Name": "Patient B (Low Risk Profile)",
            "Data": {
                "Pregnancies": 1,
                "Glucose": 92.0,
                "BloodPressure": 68.0,
                "SkinThickness": 18.0,
                "Insulin": 65.0,
                "BMI": 22.4,
                "DiabetesPedigreeFunction": 0.22,
                "Age": 24,
            }
        }
    ]

    for p in sample_patients:
        res = predictor.predict_single(p["Data"])
        print(f"\n--- {p['Name']} ---")
        for k, v in res.items():
            print(f"  {k:24}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Production Diabetes Classification Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-model cross-validation benchmark")
    parser.add_argument("--export-plots", action="store_true", help="Export high-res evaluation plots to reports/figures")
    parser.add_argument("--predict", action="store_true", help="Run inference on sample patient cases")
    parser.add_argument("--data", type=str, default=None, help="Path to custom CSV dataset")
    parser.add_argument("--model", type=str, default="Random Forest", choices=["Random Forest", "Gradient Boosting", "Logistic Regression"], help="Model type to train")

    args = parser.parse_args()

    if len(sys.argv) == 1 or (args.train and args.evaluate):
        run_training_pipeline(data_path=args.data, export_plots=True, model_choice=args.model)
    elif args.benchmark:
        run_benchmark(data_path=args.data)
    elif args.train:
        run_training_pipeline(data_path=args.data, export_plots=args.export_plots, model_choice=args.model)
    elif args.predict:
        run_sample_prediction()
    else:
        run_training_pipeline(data_path=args.data, export_plots=True, model_choice=args.model)


if __name__ == "__main__":
    main()
