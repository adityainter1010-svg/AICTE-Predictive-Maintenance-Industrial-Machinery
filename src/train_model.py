"""
src/train_model.py
Trains four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost)
for both binary Machine Failure and multi-class Failure Type prediction.
Saves the best model for each task to disk.
"""

import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.preprocessing import load_dataset, prepare_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_failure_models():
    """Return a dict of candidate models for binary failure prediction."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE, C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=10,   # handles imbalance
            eval_metric="logloss",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }


def get_failure_type_models(n_classes: int):
    """Return a dict of candidate models for multi-class failure type prediction."""
    xgb_kwargs = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    # For multi-class XGBoost, set objective only when n_classes > 2
    if n_classes > 2:
        xgb_kwargs["objective"] = "multi:softprob"
        xgb_kwargs["num_class"] = n_classes

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=config.RANDOM_STATE, C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


# ── SMOTE resampling ──────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state=config.RANDOM_STATE):
    """Apply SMOTE to training data to handle class imbalance."""
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE applied. Before: %s  After: %s",
            dict(zip(*np.unique(y_train, return_counts=True))),
            dict(zip(*np.unique(y_res, return_counts=True))),
        )
        return X_res, y_res
    except Exception as exc:
        logger.warning("SMOTE failed (%s). Proceeding without resampling.", exc)
        return X_train, y_train


# ── Cross-validation Evaluation ───────────────────────────────────────────────

def cross_validate_model(model, X, y, cv=None):
    """Return mean ± std of F1-macro via stratified k-fold CV."""
    cv = cv or StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ── Training Loops ────────────────────────────────────────────────────────────

def train_failure_models(X_train, y_train):
    """Train all candidate models for binary failure prediction and return results."""
    X_res, y_res = apply_smote(X_train, y_train)
    results = {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, model in get_failure_models().items():
        logger.info("Training failure model: %s", name)
        model.fit(X_res, y_res)
        mean_f1, std_f1 = cross_validate_model(model, X_res, y_res, cv=cv)
        results[name] = {"model": model, "cv_f1_mean": mean_f1, "cv_f1_std": std_f1}
        logger.info("  CV F1-macro: %.4f ± %.4f", mean_f1, std_f1)

    return results


def train_failure_type_models(X_train, y_train, n_classes: int):
    """Train all candidate models for multi-class failure type prediction."""
    X_res, y_res = apply_smote(X_train, y_train)
    results = {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, model in get_failure_type_models(n_classes).items():
        logger.info("Training failure-type model: %s", name)
        model.fit(X_res, y_res)
        mean_f1, std_f1 = cross_validate_model(model, X_res, y_res, cv=cv)
        results[name] = {"model": model, "cv_f1_mean": mean_f1, "cv_f1_std": std_f1}
        logger.info("  CV F1-macro: %.4f ± %.4f", mean_f1, std_f1)

    return results


def select_best_model(results: dict, metric="cv_f1_mean"):
    """Return the name and model with the highest metric score."""
    best_name = max(results, key=lambda k: results[k][metric])
    return best_name, results[best_name]["model"]


# ── Save / Load Models ────────────────────────────────────────────────────────

def save_models(failure_model, failure_type_model, all_failure_results=None, all_type_results=None):
    """Persist best models and training metadata to disk."""
    joblib.dump(failure_model, config.FAILURE_MODEL_PATH)
    joblib.dump(failure_type_model, config.FAILURE_TYPE_MODEL_PATH)
    logger.info("Models saved: %s  %s", config.FAILURE_MODEL_PATH, config.FAILURE_TYPE_MODEL_PATH)

    # Save CV summary for dashboard display
    summary = {}
    if all_failure_results:
        summary["failure"] = {
            k: {"cv_f1_mean": v["cv_f1_mean"], "cv_f1_std": v["cv_f1_std"]}
            for k, v in all_failure_results.items()
        }
    if all_type_results:
        summary["failure_type"] = {
            k: {"cv_f1_mean": v["cv_f1_mean"], "cv_f1_std": v["cv_f1_std"]}
            for k, v in all_type_results.items()
        }
    summary_path = config.MODEL_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved: %s", summary_path)


def load_failure_model():
    if not config.FAILURE_MODEL_PATH.exists():
        raise FileNotFoundError("Failure model not found. Run train_model.py first.")
    return joblib.load(config.FAILURE_MODEL_PATH)


def load_failure_type_model():
    if not config.FAILURE_TYPE_MODEL_PATH.exists():
        raise FileNotFoundError("Failure type model not found. Run train_model.py first.")
    return joblib.load(config.FAILURE_TYPE_MODEL_PATH)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_training():
    """End-to-end training pipeline. Call this directly or via CLI."""
    logger.info("=== Starting Training Pipeline ===")

    # 1. Load and prepare data
    df = load_dataset()
    (
        X_train, X_test,
        y_f_train, y_f_test,
        y_t_train, y_t_test,
        preprocessor, le, feature_names,
    ) = prepare_data(df)

    n_classes = len(le.classes_)
    logger.info("Failure type classes (%d): %s", n_classes, le.classes_.tolist())

    # 2. Train failure prediction models
    failure_results = train_failure_models(X_train, y_f_train)
    best_failure_name, best_failure_model = select_best_model(failure_results)
    logger.info("Best failure model: %s (CV F1=%.4f)", best_failure_name, failure_results[best_failure_name]["cv_f1_mean"])

    # 3. Train failure type classification models
    type_results = train_failure_type_models(X_train, y_t_train, n_classes)
    best_type_name, best_type_model = select_best_model(type_results)
    logger.info("Best failure type model: %s (CV F1=%.4f)", best_type_name, type_results[best_type_name]["cv_f1_mean"])

    # 4. Save best models
    save_models(best_failure_model, best_type_model, failure_results, type_results)

    return {
        "failure_results": failure_results,
        "type_results": type_results,
        "best_failure_model": best_failure_model,
        "best_type_model": best_type_model,
        "X_test": X_test,
        "y_f_test": y_f_test,
        "y_t_test": y_t_test,
        "le": le,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    run_training()
