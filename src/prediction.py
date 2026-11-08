"""
src/prediction.py
Real-time prediction module: takes raw sensor readings -> preprocesses -> predicts.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.preprocessing import load_preprocessor, load_label_encoder, load_feature_names
from src.train_model import load_failure_model, load_failure_type_model

logger = logging.getLogger(__name__)

# ── Lazy-loaded singletons (loaded once per Streamlit session) ────────────────
_preprocessor = None
_label_encoder = None
_failure_model = None
_failure_type_model = None
_feature_names = None


def _load_all():
    global _preprocessor, _label_encoder, _failure_model, _failure_type_model, _feature_names
    if _preprocessor is None:
        _preprocessor = load_preprocessor()
    if _label_encoder is None:
        _label_encoder = load_label_encoder()
    if _failure_model is None:
        _failure_model = load_failure_model()
    if _failure_type_model is None:
        _failure_type_model = load_failure_type_model()
    if _feature_names is None:
        _feature_names = load_feature_names()


def _build_input_df(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    product_type: str,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature schema."""
    row = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": product_type.upper().strip(),
    }
    return pd.DataFrame([row])


def predict(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    product_type: str,
) -> dict:
    """
    Run both failure and failure-type prediction on a single sensor reading.

    Returns
    -------
    dict with keys:
        failure_prediction  : int  (0 = No Failure, 1 = Failure)
        failure_probability : float  (probability of class 1)
        failure_type_label  : str   (e.g. 'Tool Wear Failure')
        failure_type_code   : str   (e.g. 'TWF')
        failure_type_probs  : dict  {class_label: probability}
        feature_values      : dict  raw sensor values used
    """
    _load_all()

    input_df = _build_input_df(air_temp, process_temp, rotational_speed, torque, tool_wear, product_type)

    # Validate product type
    valid_types = {"L", "M", "H"}
    if input_df["Type"].iloc[0] not in valid_types:
        raise ValueError(f"product_type must be one of {valid_types}, got '{product_type}'.")

    X = _preprocessor.transform(input_df)

    # Binary failure prediction
    failure_pred = int(_failure_model.predict(X)[0])
    if hasattr(_failure_model, "predict_proba"):
        failure_prob = float(_failure_model.predict_proba(X)[0][1])
    else:
        failure_prob = float(failure_pred)

    # Multi-class failure type prediction
    type_pred_encoded = int(_failure_type_model.predict(X)[0])
    failure_type_code = _label_encoder.inverse_transform([type_pred_encoded])[0]
    failure_type_label = config.FAILURE_TYPE_MAP.get(failure_type_code, failure_type_code)

    # Per-class probabilities for failure type
    type_probs = {}
    if hasattr(_failure_type_model, "predict_proba"):
        proba = _failure_type_model.predict_proba(X)[0]
        for idx, cls in enumerate(_label_encoder.classes_):
            type_probs[config.FAILURE_TYPE_MAP.get(cls, cls)] = round(float(proba[idx]), 4)

    return {
        "failure_prediction": failure_pred,
        "failure_probability": round(failure_prob, 4),
        "failure_type_code": failure_type_code,
        "failure_type_label": failure_type_label,
        "failure_type_probs": type_probs,
        "feature_values": {
            "Air Temp (K)": air_temp,
            "Process Temp (K)": process_temp,
            "Rotational Speed (rpm)": rotational_speed,
            "Torque (Nm)": torque,
            "Tool Wear (min)": tool_wear,
            "Product Type": product_type,
        },
        "feature_names": _feature_names,
        "processed_input": X,
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a DataFrame containing raw sensor readings.
    Expected columns match config.ALL_FEATURES.
    """
    _load_all()

    required = config.ALL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    X = _preprocessor.transform(df[required])

    failure_preds = _failure_model.predict(X)
    type_preds_enc = _failure_type_model.predict(X)
    type_labels = _label_encoder.inverse_transform(type_preds_enc)

    result_df = df.copy()
    result_df["Predicted_Failure"] = failure_preds
    result_df["Predicted_Failure_Type_Code"] = type_labels
    result_df["Predicted_Failure_Type"] = [
        config.FAILURE_TYPE_MAP.get(c, c) for c in type_labels
    ]

    if hasattr(_failure_model, "predict_proba"):
        result_df["Failure_Probability"] = _failure_model.predict_proba(X)[:, 1]

    return result_df


def get_risk_level(failure_probability: float) -> tuple[str, str]:
    """Map probability to a risk level label and colour."""
    if failure_probability >= 0.75:
        return "Critical", "#d62728"
    if failure_probability >= 0.50:
        return "High", "#ff7f0e"
    if failure_probability >= 0.25:
        return "Medium", "#ffdd57"
    return "Low", "#2ca02c"
