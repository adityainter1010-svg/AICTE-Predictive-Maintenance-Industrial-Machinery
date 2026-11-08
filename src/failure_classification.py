"""
src/failure_classification.py
Helpers specific to multi-class failure type classification and display.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


FAILURE_DESCRIPTIONS = {
    "No Failure": "Machine is operating within normal parameters.",
    "Tool Wear Failure": (
        "Tool Wear Failure: The cutting tool has exceeded safe wear limits. "
        "Immediate tool replacement is required."
    ),
    "Heat Dissipation Failure": (
        "Heat Dissipation Failure: Insufficient cooling leads to thermal overload. "
        "Check cooling systems and reduce process temperature."
    ),
    "Power Failure": (
        "Power Failure: Abnormal power consumption detected. "
        "Inspect electrical supply and motor health."
    ),
    "Overstrain Failure": (
        "Overstrain Failure: Torque x tool wear exceeds machine capacity. "
        "Reduce load or replace worn components."
    ),
    "Random Failures": (
        "Random Failure: Stochastic component failure not associated with specific sensor pattern. "
        "Schedule general inspection."
    ),
    "Multiple Failures": (
        "Multiple Failures: More than one failure mode detected simultaneously. "
        "Immediate shutdown and full diagnostic recommended."
    ),
    # Short-code aliases for legacy compatibility
    "TWF": "Tool Wear Failure: Cutting tool has exceeded safe wear limits.",
    "HDF": "Heat Dissipation Failure: Insufficient cooling detected.",
    "PWF": "Power Failure: Abnormal power consumption detected.",
    "OSF": "Overstrain Failure: Machine load exceeds capacity.",
    "RNF": "Random Failure: Stochastic component failure.",
}

FAILURE_COLORS = {
    "No Failure":               "#2ca02c",
    "Tool Wear Failure":        "#d62728",
    "Heat Dissipation Failure": "#ff7f0e",
    "Power Failure":            "#1f77b4",
    "Overstrain Failure":       "#9467bd",
    "Random Failures":          "#8c564b",
    "Multiple Failures":        "#e377c2",
    # Short-code aliases
    "TWF": "#d62728",
    "HDF": "#ff7f0e",
    "PWF": "#1f77b4",
    "OSF": "#9467bd",
    "RNF": "#8c564b",
    "Multiple": "#e377c2",
}


def get_failure_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with failure type counts and percentages."""
    if config.TARGET_FAILURE_TYPE not in df.columns:
        raise ValueError(f"Column '{config.TARGET_FAILURE_TYPE}' not found. Run derive_failure_type_label() first.")

    counts = df[config.TARGET_FAILURE_TYPE].value_counts().reset_index()
    counts.columns = ["Failure Type", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)
    counts["Label"] = counts["Failure Type"].map(config.FAILURE_TYPE_MAP)
    return counts


def get_failure_type_description(code: str) -> str:
    """Return human-readable description for a failure type code."""
    return FAILURE_DESCRIPTIONS.get(code, "Unknown failure type.")


def get_failure_type_color(code: str) -> str:
    """Return hex colour string for a failure type code."""
    return FAILURE_COLORS.get(code, "#7f7f7f")


def failure_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the five binary failure indicator columns."""
    fail_cols = [c for c in config.FAILURE_TYPE_COLUMNS if c in df.columns]
    summary_rows = []
    for col in fail_cols:
        total = int(df[col].sum())
        pct = round(total / len(df) * 100, 3)
        summary_rows.append({
            "Code": col,
            "Full Name": config.FAILURE_TYPE_MAP.get(col, col),
            "Count": total,
            "Percentage (%)": pct,
        })
    return pd.DataFrame(summary_rows)
