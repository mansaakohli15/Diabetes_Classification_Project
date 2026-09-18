"""
Evaluation and Visualization Module.
Computes clinical & statistical metrics and generates publication-grade visualizations.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def evaluate_model_performance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Computes a full suite of classification and clinical metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall (Sensitivity)": float(rec),
        "Specificity": float(spec),
        "F1-Score": float(f1),
        "Confusion Matrix": cm.tolist(),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
    }

    if y_proba is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_test, y_proba))
        metrics["PR-AUC"] = float(average_precision_score(y_test, y_proba))

    return metrics


def plot_model_comparison(
    benchmark_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comparison plot across candidate models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "model_comparison.png")

    melted = benchmark_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 5.5), dpi=300)
    palette = ["#2b5c8f", "#38908f", "#b2e061", "#ff6f69"]
    ax = sns.barplot(
        data=melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
    )
    plt.title("Model Performance Benchmark (Stratified 5-Fold CV)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classifier Model", fontsize=11, fontweight="bold")
    plt.ylabel("Cross-Validation Score", fontsize=11, fontweight="bold")
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """
    Generate normalized and raw count confusion matrix.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    annot = np.array([
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"],
    ])

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["No Diabetes (0)", "Diabetes (1)"],
        yticklabels=["No Diabetes (0)", "Diabetes (1)"],
        ax=ax,
    )
    plt.title("Confusion Matrix (Counts & Ratios)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Condition", fontsize=11, fontweight="bold")
    plt.ylabel("True Condition", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_pr_curves(
    y_test: pd.Series,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """
    Generate ROC Curve and Precision-Recall Curve side-by-side.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # ROC
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")

    # PR
    ax2.plot(rec, prec, color="#2ca02c", lw=2.5, label=f"PR Curve (AP = {pr_auc:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight="bold")
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_importance(
    pipeline: Pipeline,
    save_path: Optional[str] = None
) -> str:
    """
    Extract and plot feature importances from classifier inside pipeline.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "feature_importance.png")

    clf = pipeline.named_steps["classifier"]
    
    # Extract feature names after preprocessor
    fe_step = pipeline.named_steps["preprocessor"].named_steps.get("feature_engineer")
    if fe_step and hasattr(fe_step, "feature_names_") and fe_step.feature_names_:
        feature_names = fe_step.feature_names_
    else:
        feature_names = [f"Feature {i}" for i in range(13)]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return ""

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(9, 6), dpi=300)
    palette = sns.color_palette("viridis", len(df_imp))
    plt.barh(df_imp["Feature"], df_imp["Importance"], color=palette)
    plt.title("Clinical Feature Importance Ranking", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> str:
    """
    Plot feature correlation matrix.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "correlation_matrix.png")

    plt.figure(figsize=(10, 8), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.5,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
