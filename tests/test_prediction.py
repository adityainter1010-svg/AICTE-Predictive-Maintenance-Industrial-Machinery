"""
tests/test_prediction.py
Unit tests for src/prediction.py  (mocked – no actual model files required).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.prediction import get_risk_level, _build_input_df


# ── Tests for get_risk_level ───────────────────────────────────────────────────

class TestGetRiskLevel:
    def test_low_risk(self):
        level, color = get_risk_level(0.10)
        assert level == "Low"
        assert color == "#2ca02c"

    def test_medium_risk(self):
        level, color = get_risk_level(0.35)
        assert level == "Medium"
        assert color == "#ffdd57"

    def test_high_risk(self):
        level, color = get_risk_level(0.60)
        assert level == "High"
        assert color == "#ff7f0e"

    def test_critical_risk(self):
        level, color = get_risk_level(0.80)
        assert level == "Critical"
        assert color == "#d62728"

    def test_boundary_low_medium(self):
        level, _ = get_risk_level(0.25)
        assert level == "Medium"

    def test_boundary_medium_high(self):
        level, _ = get_risk_level(0.50)
        assert level == "High"

    def test_zero_probability(self):
        level, _ = get_risk_level(0.0)
        assert level == "Low"

    def test_full_probability(self):
        level, _ = get_risk_level(1.0)
        assert level == "Critical"


# ── Tests for _build_input_df ──────────────────────────────────────────────────

class TestBuildInputDf:
    def test_returns_dataframe(self):
        df = _build_input_df(298.0, 308.0, 1500.0, 40.0, 50.0, "M")
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self):
        df = _build_input_df(298.0, 308.0, 1500.0, 40.0, 50.0, "M")
        assert len(df) == 1

    def test_correct_columns(self):
        df = _build_input_df(298.0, 308.0, 1500.0, 40.0, 50.0, "L")
        for col in config.ALL_FEATURES:
            assert col in df.columns

    def test_product_type_uppercase(self):
        df = _build_input_df(298.0, 308.0, 1500.0, 40.0, 50.0, "l")
        assert df["Type"].iloc[0] == "L"

    def test_values_stored_correctly(self):
        df = _build_input_df(299.1, 309.5, 1600.0, 45.3, 120.0, "H")
        assert df["Air temperature [K]"].iloc[0] == pytest.approx(299.1)
        assert df["Torque [Nm]"].iloc[0] == pytest.approx(45.3)
        assert df["Type"].iloc[0] == "H"


# ── Tests for recommendations ─────────────────────────────────────────────────

class TestRecommendations:
    def test_no_failure_recommendation(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("No Failure")
        assert rec["urgency"] == "Routine"
        assert len(rec["actions"]) > 0

    def test_twf_recommendation(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("TWF")
        assert rec["urgency"] == "Immediate"
        assert "tool" in rec["actions"][0].lower() or "stop" in rec["actions"][0].lower() or "🔴" in rec["actions"][0]

    def test_hdf_recommendation(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("HDF")
        assert rec["urgency"] in ("Urgent", "Immediate")

    def test_recommendation_has_required_keys(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("PWF")
        for key in ["urgency", "actions", "timeline", "failure_type_label", "urgency_color"]:
            assert key in rec

    def test_tool_wear_sensor_advice(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("No Failure", {"Tool wear [min]": 230})
        all_text = " ".join(rec["actions"])
        assert "wear" in all_text.lower() or "230" in all_text

    def test_temp_diff_advice(self):
        from src.recommendations import get_recommendations
        rec = get_recommendations("HDF", {
            "Air temperature [K]": 295.0,
            "Process temperature [K]": 315.0,  # diff = 20 K > threshold
        })
        all_text = " ".join(rec["actions"])
        assert "temperature" in all_text.lower() or "cooling" in all_text.lower()


# ── Tests for failure classification helpers ──────────────────────────────────

class TestFailureClassification:
    def test_description_returned(self):
        from src.failure_classification import get_failure_type_description
        desc = get_failure_type_description("TWF")
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_color_returned(self):
        from src.failure_classification import get_failure_type_color
        color = get_failure_type_color("HDF")
        assert color.startswith("#")

    def test_unknown_code_returns_default(self):
        from src.failure_classification import get_failure_type_description
        desc = get_failure_type_description("UNKNOWN_CODE")
        assert isinstance(desc, str)

    def test_failure_indicator_summary(self):
        from src.failure_classification import failure_indicator_summary
        import pandas as pd
        df = pd.DataFrame({
            "TWF": [1, 0, 0, 1],
            "HDF": [0, 1, 0, 0],
            "PWF": [0, 0, 0, 0],
            "OSF": [0, 0, 0, 0],
            "RNF": [0, 0, 0, 0],
        })
        summary = failure_indicator_summary(df)
        assert len(summary) == 5
        twf_row = summary[summary["Code"] == "TWF"].iloc[0]
        assert twf_row["Count"] == 2
