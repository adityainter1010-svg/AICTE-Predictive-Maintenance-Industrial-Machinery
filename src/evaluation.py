"""
src/evaluation.py
Compute and visualise classification metrics for trained models.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute metrics for binary (failure) classification."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            metrics["ROC-AUC"] = None
    metrics["Classification Report"] = classification_report(
        y_true, y_pred, target_names=["No Failure", "Failure"], zero_division=0
    )
    return metrics


def compute_multiclass_metrics(y_true, y_pred, class_names=None) -> dict:
    """Compute metrics for multi-class (failure type) classification."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision (Macro)": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall (Macro)": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1 Score (Macro)": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1 Score (Weighted)": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    try:
        metrics["ROC-AUC (OvR Macro)"] = round(
            roc_auc_score(y_true, np.eye(len(set(y_true)))[y_pred], multi_class="ovr", average="macro"),
            4
        )
    except Exception:
        metrics["ROC-AUC (OvR Macro)"] = None

    metrics["Classification Report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    return metrics


def evaluate_all_models(model_results: dict, X_test, y_test, task: str = "binary", class_names=None) -> pd.DataFrame:
    """
    Evaluate all trained models on held-out test data and return a comparison DataFrame.
    task: 'binary' | 'multiclass'
    """
    rows = []
    for name, info in model_results.items():
        model = info["model"]
        y_pred = model.predict(X_test)

        if task == "binary":
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            m = compute_binary_metrics(y_test, y_pred, y_proba)
            row = {
                "Model": name,
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1 Score": m["F1 Score"],
                "ROC-AUC": m.get("ROC-AUC"),
                "CV F1 (Mean)": round(info.get("cv_f1_mean", 0), 4),
                "CV F1 (Std)": round(info.get("cv_f1_std", 0), 4),
            }
        else:
            m = compute_multiclass_metrics(y_test, y_pred, class_names)
            row = {
                "Model": name,
                "Accuracy": m["Accuracy"],
                "Precision (Macro)": m["Precision (Macro)"],
                "Recall (Macro)": m["Recall (Macro)"],
                "F1 (Macro)": m["F1 Score (Macro)"],
                "F1 (Weighted)": m["F1 Score (Weighted)"],
                "CV F1 (Mean)": round(info.get("cv_f1_mean", 0), 4),
                "CV F1 (Std)": round(info.get("cv_f1_std", 0), 4),
            }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "F1 Score" if task == "binary" else "F1 (Macro)", ascending=False
    ).reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix") -> plt.Figure:
    """Return a Matplotlib Figure of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(set(y_true))), max(5, len(set(y_true)) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title="ROC Curve") -> plt.Figure:
    """Return a Matplotlib Figure of the ROC curve (binary classification)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, metric_col="F1 Score") -> plt.Figure:
    """Bar chart comparing all models on a chosen metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(comparison_df["Model"], comparison_df[metric_col], color=colors[:len(comparison_df)])
    ax.set_ylabel(metric_col)
    ax.set_title(f"Model Comparison – {metric_col}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> plt.Figure:
    """Return a horizontal bar chart of feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importances_.")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), importances[indices], color="#1f77b4")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
