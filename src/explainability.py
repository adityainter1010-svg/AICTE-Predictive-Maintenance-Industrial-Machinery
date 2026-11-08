"""
src/explainability.py
SHAP-based model explainability: global feature importance and per-prediction explanations.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# Lazy import SHAP to avoid slow startup when not needed
_shap = None


def _get_shap():
    global _shap
    if _shap is None:
        import shap
        _shap = shap
    return _shap


# ── Explainer Factory ─────────────────────────────────────────────────────────

def get_explainer(model, X_background: np.ndarray):
    """
    Return the appropriate SHAP explainer for the model type.
    Uses TreeExplainer for tree-based models, LinearExplainer for linear models,
    and KernelExplainer as fallback.
    """
    shap = _get_shap()

    model_type = type(model).__name__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_type in ("RandomForestClassifier", "DecisionTreeClassifier",
                          "XGBClassifier", "GradientBoostingClassifier",
                          "ExtraTreesClassifier"):
            return shap.TreeExplainer(model)
        if model_type in ("LogisticRegression", "LinearSVC", "RidgeClassifier"):
            # Use a small background sample for efficiency
            background = shap.sample(X_background, min(100, X_background.shape[0]))
            return shap.LinearExplainer(model, background)
        # Generic fallback
        background = shap.sample(X_background, min(100, X_background.shape[0]))
        return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(explainer, X: np.ndarray):
    """Compute SHAP values. Returns raw shap_values array."""
    shap = _get_shap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X)
    return shap_values


# ── Global Importance Plots ───────────────────────────────────────────────────

def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list,
                      class_idx: int = 1, title: str = "SHAP Summary Plot") -> plt.Figure:
    """
    Beeswarm / dot summary plot.
    For binary classification, class_idx=1 shows impact on the 'Failure' class.
    """
    shap = _get_shap()
    fig, ax = plt.subplots(figsize=(10, 7))

    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
    elif len(np.array(shap_values).shape) == 3:
        vals = np.array(shap_values)[:, :, class_idx]

    shap.summary_plot(vals, X, feature_names=feature_names, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, feature_names: list,
                  class_idx: int = 1, top_n: int = 15,
                  title: str = "Mean |SHAP| Feature Importance") -> plt.Figure:
    """Horizontal bar of mean absolute SHAP values."""
    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
    elif len(np.array(shap_values).shape) == 3:
        vals = np.array(shap_values)[:, :, class_idx]

    mean_abs = np.abs(vals).mean(axis=0)
    indices = np.argsort(mean_abs)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_names[i] for i in indices], mean_abs[indices], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Individual Prediction Explanation ────────────────────────────────────────

def plot_shap_waterfall(explainer, X_instance: np.ndarray,
                        feature_names: list, class_idx: int = 1,
                        title: str = "SHAP Explanation – Single Prediction") -> plt.Figure:
    """
    Waterfall plot for a single prediction instance.
    Shows how each feature pushed the prediction up or down.
    """
    shap = _get_shap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values_single = explainer.shap_values(X_instance)

    if isinstance(shap_values_single, list):
        sv = shap_values_single[class_idx] if len(shap_values_single) > class_idx else shap_values_single[0]
        base = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif len(np.array(shap_values_single).shape) == 3:
        sv = np.array(shap_values_single)[0, :, class_idx]
        base = explainer.expected_value[class_idx] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    else:
        sv = shap_values_single[0] if len(shap_values_single.shape) > 1 else shap_values_single
        base = explainer.expected_value if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

    # Build SHAP Explanation object for waterfall
    explanation = shap.Explanation(
        values=sv,
        base_values=float(base),
        data=X_instance[0],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()


def get_top_contributing_features(shap_values_single, feature_names: list,
                                   class_idx: int = 1, top_n: int = 5) -> pd.DataFrame:
    """Return a DataFrame of the top N features contributing to a single prediction."""
    if isinstance(shap_values_single, list):
        sv = shap_values_single[class_idx]
    elif len(np.array(shap_values_single).shape) == 3:
        sv = np.array(shap_values_single)[0, :, class_idx]
    else:
        sv = shap_values_single[0] if len(shap_values_single.shape) > 1 else shap_values_single

    df = pd.DataFrame({"Feature": feature_names, "SHAP Value": sv})
    df["Abs"] = df["SHAP Value"].abs()
    df = df.nlargest(top_n, "Abs").drop(columns="Abs")
    df["Impact"] = df["SHAP Value"].apply(lambda v: "↑ Increases Risk" if v > 0 else "↓ Decreases Risk")
    return df.reset_index(drop=True)
