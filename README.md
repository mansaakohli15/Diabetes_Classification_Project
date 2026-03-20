<<<<<<< HEAD
# Diabetes Classification Project

## Problem Statement
The goal of this project is to **predict whether a person has diabetes** based on medical measurements such as pregnancies, glucose level, blood pressure, BMI, and more. This is a **binary classification problem** where the output is either `0` (No Diabetes) or `1` (Diabetes).

## Dataset
We used the **Pima Indians Diabetes Dataset**, which contains 768 records and 8 input features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target)

Dataset source: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

## Algorithm Used
We used a **Random Forest Classifier**, which is an ensemble learning method that combines multiple decision trees for better accuracy and stability.

- Number of trees: 100
- Random state: 42

## Results
The dataset was split into **80% training** and **20% testing**.  

- **Accuracy:** 72.08%
- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|-------|---------|--------|
| 0     | 0.79     | 0.78  | 0.78    | 99     |
| 1     | 0.61     | 0.62  | 0.61    | 55     |

- **Macro Average:** Precision 0.70, Recall 0.70, F1-score 0.70
- **Weighted Average:** Precision 0.72, Recall 0.72, F1-score 0.72

- **Confusion Matrix and Feature Importance** plots are included in the Python script.

## Observations
- The model predicts **non-diabetic patients more accurately** than diabetic patients (precision 0.79 vs 0.61).  
- The most important features affecting prediction are **Glucose**, **BMI**, and **Age**.  

## How to Run
1. Make sure Python is installed.  
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```
Run the Python script:

python diabetes_classification.py

Plots for Confusion Matrix and Feature Importance will appear during execution.
=======
# Diabetes_Classification_Project
Diabetes Classification Project using Random Forest. Predicts whether a person has diabetes based on medical measurements. Includes dataset analysis, model training, evaluation, and feature importance visualization.
>>>>>>> 28daade7a34d94dee3bf936ca9bb0a0660ab151d
