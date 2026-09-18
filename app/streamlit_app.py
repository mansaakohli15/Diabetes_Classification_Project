"""
Streamlit Web Application: Clinical Diabetes Risk Assessment & Prediction Platform.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import DiabetesPredictor
from src.data_loader import FEATURE_COLUMNS, DEFAULT_LOCAL_PATH, load_dataset
from src.models import DEFAULT_MODEL_DIR, tune_best_model, save_model_pipeline, get_data_splits

# Streamlit Page Config
st.set_page_config(
    page_title="Diabetes Risk AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-moderate {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        color: #92400E;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        color: #065F46;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_or_train_model():
    model_path = os.path.join(DEFAULT_MODEL_DIR, "best_diabetes_pipeline.joblib")
    if not os.path.exists(model_path):
        data = load_dataset()
        X_train, _, y_train, _ = get_data_splits(data)
        pipe = tune_best_model(X_train, y_train, model_type="Random Forest")
        save_model_pipeline(pipe, model_path)
    return DiabetesPredictor(model_path)


def main():
    st.markdown('<div class="main-header">🩺 Diabetes Risk AI Assessment Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production Machine Learning decision-support system for early diabetes risk screening.</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/medical-heart.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode:",
        ["🎯 Single Patient Screening", "📁 Batch CSV Screening", "📊 Model Performance & EDA", "ℹ️ About the Project"],
    )

    predictor = get_or_train_model()

    # Mode 1: Single Patient Screening
    if app_mode == "🎯 Single Patient Screening":
        st.subheader("Patient Clinical Profile Input")
        st.write("Enter the patient's diagnostic values or load a sample preset below:")

        # Presets
        preset = st.selectbox(
            "Quick Load Presets:",
            ["Custom Input", "Sample 1: High Risk Case", "Sample 2: Low Risk Healthy", "Sample 3: Borderline Case"]
        )

        # Preset values
        defaults = {
            "Pregnancies": 2,
            "Glucose": 120.0,
            "BloodPressure": 70.0,
            "SkinThickness": 20.0,
            "Insulin": 80.0,
            "BMI": 27.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 33,
        }

        if preset == "Sample 1: High Risk Case":
            defaults = {
                "Pregnancies": 6,
                "Glucose": 172.0,
                "BloodPressure": 88.0,
                "SkinThickness": 38.0,
                "Insulin": 230.0,
                "BMI": 39.4,
                "DiabetesPedigreeFunction": 0.88,
                "Age": 54,
            }
        elif preset == "Sample 2: Low Risk Healthy":
            defaults = {
                "Pregnancies": 1,
                "Glucose": 88.0,
                "BloodPressure": 65.0,
                "SkinThickness": 15.0,
                "Insulin": 50.0,
                "BMI": 21.8,
                "DiabetesPedigreeFunction": 0.20,
                "Age": 23,
            }
        elif preset == "Sample 3: Borderline Case":
            defaults = {
                "Pregnancies": 3,
                "Glucose": 135.0,
                "BloodPressure": 78.0,
                "SkinThickness": 28.0,
                "Insulin": 110.0,
                "BMI": 29.8,
                "DiabetesPedigreeFunction": 0.52,
                "Age": 41,
            }

        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=int(defaults["Pregnancies"]))
            glucose = st.slider("Fasting Glucose (mg/dL)", min_value=40.0, max_value=260.0, value=float(defaults["Glucose"]), step=1.0)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=40.0, max_value=140.0, value=float(defaults["BloodPressure"]), step=1.0)

        with col2:
            skin_thickness = st.slider("Triceps Skin Thickness (mm)", min_value=5.0, max_value=80.0, value=float(defaults["SkinThickness"]), step=1.0)
            insulin = st.slider("2-Hour Serum Insulin (µU/mL)", min_value=10.0, max_value=700.0, value=float(defaults["Insulin"]), step=1.0)
            bmi = st.slider("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=65.0, value=float(defaults["BMI"]), step=0.1)

        with col3:
            dpf = st.slider("Diabetes Pedigree Function", min_value=0.05, max_value=2.50, value=float(defaults["DiabetesPedigreeFunction"]), step=0.01)
            age = st.slider("Age (years)", min_value=18, max_value=100, value=int(defaults["Age"]), step=1)

        patient_input = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        if st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True):
            result = predictor.predict_single(patient_input)
            prob = result["diabetes_probability"]
            risk_tier = result["risk_tier"]

            st.divider()
            st.subheader("Diagnostic Risk Assessment Results")

            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Predicted Outcome", "Positive (Diabetic)" if result["prediction"] == 1 else "Negative (Non-Diabetic)")
            with mcol2:
                st.metric("Risk Probability", f"{prob * 100:.1f}%")
            with mcol3:
                st.metric("Risk Classification", risk_tier)

            st.progress(prob, text=f"Probability Score: {prob*100:.1f}%")

            if risk_tier == "High Risk":
                st.markdown(f'<div class="risk-high">⚠️ <b>{risk_tier}:</b> {result["clinical_recommendation"]}</div>', unsafe_allow_html=True)
            elif risk_tier == "Moderate Risk":
                st.markdown(f'<div class="risk-moderate">⚠️ <b>{risk_tier}:</b> {result["clinical_recommendation"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ <b>{risk_tier}:</b> {result["clinical_recommendation"]}</div>', unsafe_allow_html=True)

            # Clinical Insights
            with st.expander("🔬 View Detailed Factor Breakdown", expanded=True):
                st.write("**Patient Input Parameters:**")
                st.json(patient_input)
                st.info("💡 **Clinical Note:** In this predictive model, Glucose levels, BMI, and Age exhibit the highest predictive weights. Patients presenting both BMI >= 30 and elevated glucose are categorized with heightened metabolic risk.")

    # Mode 2: Batch CSV Screening
    elif app_mode == "📁 Batch CSV Screening":
        st.subheader("Batch Patient Diagnostic Screening")
        st.write("Upload a CSV file with patient records to generate batch risk classifications.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # Provide sample download template
        sample_df = pd.DataFrame([
            {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 27.5, "DiabetesPedigreeFunction": 0.45, "Age": 33},
            {"Pregnancies": 6, "Glucose": 168, "BloodPressure": 88, "SkinThickness": 35, "Insulin": 210, "BMI": 38.0, "DiabetesPedigreeFunction": 0.85, "Age": 52},
            {"Pregnancies": 0, "Glucose": 85, "BloodPressure": 60, "SkinThickness": 15, "Insulin": 45, "BMI": 21.0, "DiabetesPedigreeFunction": 0.18, "Age": 22},
        ])
        st.download_button(
            "📥 Download Sample CSV Template",
            sample_df.to_csv(index=False),
            file_name="diabetes_batch_template.csv",
            mime="text/csv",
        )

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df_upload)} patient records:")
                st.dataframe(df_upload.head())

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        df_res = predictor.predict_batch(df_upload)
                        st.success("Batch screening complete!")
                        st.dataframe(df_res)

                        # Summary stats
                        pos_count = (df_res["Predicted_Outcome"] == 1).sum()
                        total = len(df_res)
                        st.info(f"📊 Summary: **{pos_count} / {total} ({pos_count/total*100:.1f}%)** patients flagged with high diabetes risk.")

                        csv_out = df_res.to_csv(index=False)
                        st.download_button(
                            "💾 Download Classified Results CSV",
                            csv_out,
                            file_name="diabetes_risk_predictions.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

    # Mode 3: Performance & EDA
    elif app_mode == "📊 Model Performance & EDA":
        st.subheader("Model Benchmarks & Clinical Explainability")
        
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
        
        tab1, tab2, tab3 = st.tabs(["🏆 Multi-Model Benchmark", "📈 ROC & PR Curves / Confusion Matrix", "🧬 Feature Importance & Correlations"])
        
        with tab1:
            bench_img = os.path.join(figures_dir, "model_comparison.png")
            if os.path.exists(bench_img):
                st.image(bench_img, caption="5-Fold Cross-Validation Performance Comparison Across Algorithms", use_container_width=True)
            else:
                st.info("Run `python main.py --train --export-plots` to generate visualization figures.")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                cm_img = os.path.join(figures_dir, "confusion_matrix.png")
                if os.path.exists(cm_img):
                    st.image(cm_img, caption="Holdout Test Set Confusion Matrix", use_container_width=True)
            with col2:
                roc_img = os.path.join(figures_dir, "roc_pr_curves.png")
                if os.path.exists(roc_img):
                    st.image(roc_img, caption="Receiver Operating Characteristic & Precision-Recall Curves", use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fi_img = os.path.join(figures_dir, "feature_importance.png")
                if os.path.exists(fi_img):
                    st.image(fi_img, caption="Ranked Feature Importance (Random Forest)", use_container_width=True)
            with col2:
                corr_img = os.path.join(figures_dir, "correlation_matrix.png")
                if os.path.exists(corr_img):
                    st.image(corr_img, caption="Physiological Feature Correlation Matrix", use_container_width=True)

    # Mode 4: About
    elif app_mode == "ℹ️ About the Project":
        st.subheader("About the Diabetes Classification Platform")
        st.markdown("""
        ### Clinical Decision Support System (CDSS)
        This project implements an end-to-end machine learning pipeline built on the **Pima Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases).

        #### Key Highlights:
        - **Domain-Aware Data Imputation**: Resolves biological zero values in physiological metrics (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI).
        - **Clinical Feature Engineering**: Computes BMI risk categories, insulin-to-glucose sensitivity ratios, and age-related risk thresholds.
        - **Ensemble Learning**: Random Forest Classifier optimized with Stratified K-Fold Cross-Validation and Hyperparameter Grid Search.
        - **High-Recall Medical Objective**: Calibrated to minimize False Negatives in clinical risk screening.
        """)


if __name__ == "__main__":
    main()
