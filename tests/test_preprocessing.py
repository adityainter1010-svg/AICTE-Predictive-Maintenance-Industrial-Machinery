"""
tests/test_preprocessing.py
Unit tests for src/preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.preprocessing import (
    inspect_dataset,
    engineer_features,
    derive_failure_type_label,
    build_preprocessor,
    get_feature_names_after_preprocessing,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame matching the Kaggle dataset schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"L{i:05d}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]": np.random.normal(300, 2, n),
        "Process temperature [K]": np.random.normal(310, 1.5, n),
        "Rotational speed [rpm]": np.random.randint(1200, 2900, n).astype(float),
        "Torque [Nm]": np.random.normal(40, 10, n),
        "Tool wear [min]": np.random.randint(0, 250, n).astype(float),
        "Machine failure": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "TWF": np.zeros(n, dtype=int),
        "HDF": np.zeros(n, dtype=int),
        "PWF": np.zeros(n, dtype=int),
        "OSF": np.zeros(n, dtype=int),
        "RNF": np.zeros(n, dtype=int),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInspectDataset:
    def test_returns_dict(self, sample_df):
        summary = inspect_dataset(sample_df)
        assert isinstance(summary, dict)

    def test_shape_key(self, sample_df):
        summary = inspect_dataset(sample_df)
        assert summary["shape"] == sample_df.shape

    def test_no_missing_in_synthetic(self, sample_df):
        summary = inspect_dataset(sample_df)
        assert sum(summary["missing_values"].values()) == 0

    def test_duplicates_count(self, sample_df):
        summary = inspect_dataset(sample_df)
        assert summary["duplicates"] == 0


class TestEngineerFeatures:
    def test_temp_diff_created(self, sample_df):
        df = engineer_features(sample_df)
        assert "Temp_Diff" in df.columns

    def test_temp_diff_values(self, sample_df):
        df = engineer_features(sample_df)
        expected = sample_df["Process temperature [K]"] - sample_df["Air temperature [K]"]
        pd.testing.assert_series_equal(df["Temp_Diff"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)

    def test_power_created(self, sample_df):
        df = engineer_features(sample_df)
        assert "Power_W" in df.columns

    def test_power_nonnegative(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["Power_W"] >= 0).all()

    def test_wear_category_created(self, sample_df):
        df = engineer_features(sample_df)
        assert "Wear_Category" in df.columns

    def test_original_columns_preserved(self, sample_df):
        df = engineer_features(sample_df)
        for col in config.NUMERICAL_FEATURES:
            assert col in df.columns


class TestDeriveFailureTypeLabel:
    def test_column_created(self, sample_df):
        df = derive_failure_type_label(sample_df)
        assert config.TARGET_FAILURE_TYPE in df.columns

    def test_no_failure_when_machine_failure_zero(self, sample_df):
        df = sample_df.copy()
        df["Machine failure"] = 0
        df = derive_failure_type_label(df)
        assert (df[config.TARGET_FAILURE_TYPE] == "No Failure").all()

    def test_twf_label_when_only_twf_set(self, sample_df):
        df = sample_df.copy()
        df["Machine failure"] = 1
        df["TWF"] = 1
        df["HDF"] = 0; df["PWF"] = 0; df["OSF"] = 0; df["RNF"] = 0
        df = derive_failure_type_label(df)
        assert (df[config.TARGET_FAILURE_TYPE] == "TWF").all()

    def test_multiple_when_two_flags_set(self, sample_df):
        df = sample_df.copy()
        df["Machine failure"] = 1
        df["TWF"] = 1; df["HDF"] = 1
        df["PWF"] = 0; df["OSF"] = 0; df["RNF"] = 0
        df = derive_failure_type_label(df)
        assert (df[config.TARGET_FAILURE_TYPE] == "Multiple").all()

    def test_valid_label_values(self, sample_df):
        df = derive_failure_type_label(sample_df)
        valid = {"No Failure", "TWF", "HDF", "PWF", "OSF", "RNF", "Multiple"}
        assert set(df[config.TARGET_FAILURE_TYPE].unique()).issubset(valid)


class TestPreprocessor:
    def test_preprocessor_builds(self):
        pp = build_preprocessor()
        assert pp is not None

    def test_preprocessor_transforms(self, sample_df):
        df = engineer_features(sample_df)
        X = df[config.ALL_FEATURES]
        pp = build_preprocessor()
        X_proc = pp.fit_transform(X)
        assert X_proc.shape[0] == len(df)
        assert X_proc.shape[1] > len(config.NUMERICAL_FEATURES)  # OHE expands categoricals

    def test_no_nan_after_transform(self, sample_df):
        df = engineer_features(sample_df)
        X = df[config.ALL_FEATURES]
        pp = build_preprocessor()
        X_proc = pp.fit_transform(X)
        assert not np.isnan(X_proc).any()

    def test_feature_names_length(self, sample_df):
        df = engineer_features(sample_df)
        X = df[config.ALL_FEATURES]
        pp = build_preprocessor()
        pp.fit_transform(X)
        names = get_feature_names_after_preprocessing(pp)
        # Must include at least the numerical features
        assert len(names) >= len(config.NUMERICAL_FEATURES)
