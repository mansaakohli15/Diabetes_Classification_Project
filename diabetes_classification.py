"""
Diabetes Classification Project
Main Execution Script for End-to-End Model Training and Visualization.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import run_training_pipeline, run_sample_prediction

if __name__ == "__main__":
    print("Executing Diabetes Classification Pipeline...")
    pipeline, metrics = run_training_pipeline(export_plots=True)
    print("\nRunning Sample Diagnostic Predictions:")
    run_sample_prediction()