"""
src/preprocessing.py
Loads the dataset, engineers features, and builds sklearn preprocessing pipelines.
All fit operations are performed on training data only to prevent data leakage.

Actual Kaggle dataset schema:
  Columns: UDI, Product ID, Type, Air temperature [K], Process temperature [K],
           Rotational speed [rpm], Torque [Nm], Tool wear [min], Target, Failure Type
  - "Target"       : 0 = Normal, 1 = Failure  (binary)
  - "Failure Type" : "No Failure" / "Tool Wear Failure" / "Heat Dissipation Failure"
                     / "Power Failure" / "Overstrain Failure" / "Random Failures"
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset(path=None) -> pd.DataFrame:
    """Load CSV dataset and return a raw DataFrame."""
    path = Path(path or config.DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from:\n"
            "  https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification\n"
            "Save as 'data/predictive_maintenance.csv' in the project folder."
        )
    df = pd.read_csv(path)
    logger.info("Dataset loaded: %d rows x %d columns", *df.shape)
    return df


# ── Column normalisation ──────────────────────────────────────────────────────

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle different column naming conventions between dataset versions.
    The Kaggle dataset uses 'Target' and 'Failure Type' (full strings).
    Some versions use 'Machine failure' and binary TWF/HDF/PWF/OSF/RNF columns.
    This function normalises to a single consistent schema.
    """
    df = df.copy()

    # Rename 'Target' -> 'Machine failure' if needed for legacy compatibility
    if "Target" in df.columns and config.TARGET_FAILURE == "Target":
        pass  # already correct
    elif "Target" in df.columns and "Machine failure" not in df.columns:
        df.rename(columns={"Target": "Machine failure"}, inplace=True)

    # Rename 'Failure Type' if it contains full-name strings — keep as-is
    return df


# ── Data Inspection ──────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict for the EDA / dashboard."""
    target_col = config.TARGET_FAILURE
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_stats": df.describe().round(4).to_dict(),
        "failure_dist": df[target_col].value_counts().to_dict() if target_col in df.columns else {},
    }


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw sensor readings."""
    df = df.copy()

    # Temperature difference – indicator for heat dissipation failure
    if "Air temperature [K]" in df.columns and "Process temperature [K]" in df.columns:
        df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Power = Torque × AngularSpeed (ω = rpm × 2π/60)
    if "Rotational speed [rpm]" in df.columns and "Torque [Nm]" in df.columns:
        df["Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)

    # Tool-wear bins
    if "Tool wear [min]" in df.columns:
        df["Wear_Category"] = pd.cut(
            df["Tool wear [min]"],
            bins=[-1, 60, 120, 180, 240, 300],
            labels=["Very Low", "Low", "Medium", "High", "Critical"],
        )

    return df


def derive_failure_type_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a clean 'Failure Type' column exists.
    
    The Kaggle dataset already has a 'Failure Type' column with full string labels.
    If the dataset instead has binary TWF/HDF/PWF/OSF/RNF columns, derive the label.
    """
    df = df.copy()
    type_col = config.TARGET_FAILURE_TYPE    # "Failure Type"
    target_col = config.TARGET_FAILURE       # "Target"

    # Case 1: Dataset already has 'Failure Type' as a string column (actual Kaggle dataset)
    # pandas 3.x uses dtype 'str' (StringDtype) instead of object for string columns
    if type_col in df.columns and (
        df[type_col].dtype == object
        or str(df[type_col].dtype) in ("string", "str", "object")
        or hasattr(df[type_col].dtype, "kind") and df[type_col].dtype.kind in ("O", "U", "S")
    ):
        # Normalise any whitespace
        df[type_col] = df[type_col].astype(str).str.strip()
        return df

    # Case 2: Binary indicator columns exist — derive label from them
    fail_cols = [c for c in config.FAILURE_TYPE_COLUMNS if c in df.columns]
    if fail_cols:
        def _label(row):
            if row.get(target_col, 0) == 0:
                return "No Failure"
            active = [c for c in fail_cols if row.get(c, 0) == 1]
            if len(active) == 0:
                return "No Failure"
            if len(active) == 1:
                return config.FAILURE_TYPE_MAP.get(active[0], active[0])
            return "Multiple Failures"
        df[type_col] = df.apply(_label, axis=1)
        return df

    # Case 3: No failure type info — derive from Target only
    df[type_col] = df[target_col].map({0: "No Failure", 1: "No Failure"}) if target_col in df.columns else "No Failure"
    return df


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def build_preprocessor(numerical_features=None, categorical_features=None):
    """Return an unfitted ColumnTransformer preprocessor."""
    numerical_features = numerical_features or config.NUMERICAL_FEATURES
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    try:
        cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    except TypeError:
        cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor, numerical_features=None, categorical_features=None):
    """Recover feature names after ColumnTransformer fitting."""
    numerical_features = numerical_features or config.NUMERICAL_FEATURES
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    cat_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            try:
                ohe = transformer.named_steps["ohe"]
                cat_names = list(ohe.get_feature_names_out(cols))
            except Exception:
                cat_names = []
    return numerical_features + cat_names


# ── Full Preprocessing Flow ───────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Full pipeline:
      engineer features → ensure Failure Type column → split → fit preprocessor.

    Returns:
      X_train_proc, X_test_proc,
      y_failure_train, y_failure_test,
      y_type_train, y_type_test,
      preprocessor, label_encoder, feature_names
    """
    df = engineer_features(df)
    df = derive_failure_type_label(df)

    target_col = config.TARGET_FAILURE       # "Target"
    type_col   = config.TARGET_FAILURE_TYPE  # "Failure Type"

    # Validate required columns
    required = config.ALL_FEATURES + [target_col, type_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=required)

    X         = df[config.ALL_FEATURES]
    y_failure = df[target_col].astype(int)
    y_type_raw = df[type_col].astype(str)

    # Encode failure type labels
    le = LabelEncoder()
    y_type = le.fit_transform(y_type_raw)

    logger.info("Failure type classes (%d): %s", len(le.classes_), le.classes_.tolist())
    logger.info("Class distribution: %s", dict(zip(*np.unique(y_type, return_counts=True))))

    # Stratified split on binary failure label
    X_train, X_test, y_f_train, y_f_test, y_t_train, y_t_test = train_test_split(
        X, y_failure, y_type,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_failure,
    )

    # Fit preprocessor on training data ONLY
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = get_feature_names_after_preprocessing(preprocessor)

    # Persist artefacts
    joblib.dump(preprocessor,  config.PREPROCESSOR_PATH)
    joblib.dump(le,             config.LABEL_ENCODER_PATH)
    joblib.dump(feature_names,  config.FEATURE_NAMES_PATH)

    logger.info(
        "Data prepared. Train: %d  Test: %d  Features: %d",
        X_train_proc.shape[0], X_test_proc.shape[0], X_train_proc.shape[1],
    )

    return (
        X_train_proc, X_test_proc,
        y_f_train.values, y_f_test.values,
        y_t_train, y_t_test,
        preprocessor, le, feature_names,
    )


def load_preprocessor():
    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Preprocessor not found. Run train_model.py first.")
    return joblib.load(config.PREPROCESSOR_PATH)


def load_label_encoder():
    if not config.LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError("Label encoder not found. Run train_model.py first.")
    return joblib.load(config.LABEL_ENCODER_PATH)


def load_feature_names():
    if not config.FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError("Feature names not found. Run train_model.py first.")
    return joblib.load(config.FEATURE_NAMES_PATH)
