"""
pages/6_Sensor_Analytics.py
Real-time sensor analytics, threshold monitoring, and trend visualisation page.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Sensor Analytics", page_icon="📡", layout="wide")


@st.cache_data(show_spinner=False)
def get_data():
    try:
        from src.preprocessing import load_dataset, engineer_features, derive_failure_type_label
        df = load_dataset()
        df = engineer_features(df)
        df = derive_failure_type_label(df)
        return df, None
    except FileNotFoundError as e:
        return None, str(e)


def add_threshold_line(fig, value, name, color="#d62728", dash="dash"):
    fig.add_hline(y=value, line_dash=dash, line_color=color,
                  annotation_text=name, annotation_position="top left",
                  annotation_font_color=color)
    return fig


def main():
    st.title("📡 Sensor Analytics & Threshold Monitoring")
    st.markdown("Deep-dive into individual sensor behaviour, failure correlations, and operational thresholds.")

    df, err = get_data()
    if err or df is None:
        st.error(f"**Dataset not available.** {err or ''}")
        st.info("Download the dataset from Kaggle and place it in `data/predictive_maintenance.csv`.")
        return

    sensor_cols = config.NUMERICAL_FEATURES
    available   = [c for c in sensor_cols if c in df.columns]

    # ── KPIs ────────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Sensor Fleet Overview")
    cols = st.columns(len(available))
    for i, col in enumerate(available):
        mean_val  = df[col].mean()
        fail_mean = df[df[config.TARGET_FAILURE] == 1][col].mean() if col in df.columns else mean_val
        delta     = round(fail_mean - mean_val, 2)
        cols[i].metric(
            col.split("[")[0].strip(),
            f"{mean_val:.1f}",
            delta=f"{delta:+.2f} at failure",
            delta_color="inverse",
        )

    st.divider()

    # ── Sidebar controls ────────────────────────────────────────────────────────
    st.sidebar.markdown("## 📡 Sensor Controls")
    selected_sensor = st.sidebar.selectbox("Select Sensor", available)
    product_filter  = st.sidebar.multiselect("Filter by Product Type", ["L", "M", "H"], default=["L", "M", "H"])
    show_failures   = st.sidebar.checkbox("Highlight Failures", value=True)
    sample_size     = st.sidebar.slider("Sample size for scatter", 500, 5000, 2000, step=500)

    # Apply product type filter
    df_filtered = df[df["Type"].isin(product_filter)] if "Type" in df.columns else df

    # ── 1. Sensor Distribution vs Failure ────────────────────────────────────────
    st.markdown(f"## 📈 {selected_sensor} – Failure vs Normal Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df_filtered, x=selected_sensor,
            color=config.TARGET_FAILURE if show_failures else None,
            color_discrete_map={0: "#2ca02c", 1: "#d62728"},
            nbins=60, barmode="overlay", opacity=0.7,
            title=f"{selected_sensor} Distribution",
            labels={str(config.TARGET_FAILURE): "Machine Failure"},
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.box(
            df_filtered, x=config.TARGET_FAILURE if show_failures else "Type",
            y=selected_sensor,
            color=config.TARGET_FAILURE if show_failures else "Type",
            color_discrete_map={0: "#2ca02c", 1: "#d62728"},
            title=f"{selected_sensor} – Box Plot",
            points="outliers",
        )
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # ── 2. Sensor Statistics Table ────────────────────────────────────────────────
    st.markdown("## 📋 Sensor Statistics by Failure Status")
    stat_rows = []
    for col in available:
        normal  = df[df[config.TARGET_FAILURE] == 0][col]
        failure = df[df[config.TARGET_FAILURE] == 1][col]
        stat_rows.append({
            "Sensor": col.split("[")[0].strip(),
            "Normal Mean": round(normal.mean(), 3),
            "Normal Std":  round(normal.std(), 3),
            "Failure Mean": round(failure.mean(), 3),
            "Failure Std":  round(failure.std(), 3),
            "Δ (Failure−Normal)": round(failure.mean() - normal.mean(), 3),
            "% Difference": round((failure.mean() - normal.mean()) / normal.mean() * 100, 2),
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 3. Threshold Monitoring ──────────────────────────────────────────────────
    st.markdown("## 🚦 Threshold Monitoring")
    st.markdown("Sensor values shown against operational thresholds. Red zones indicate danger levels.")

    thresh_configs = {
        "Tool wear [min]": {
            "warning": config.TOOL_WEAR_WARNING_THRESHOLD,
            "critical": config.TOOL_WEAR_CRITICAL_THRESHOLD,
            "unit": "min",
        },
        "Torque [Nm]": {
            "warning": config.TORQUE_WARNING_THRESHOLD,
            "critical": None,
            "unit": "Nm",
        },
    }

    for sensor, thresholds in thresh_configs.items():
        if sensor not in df.columns:
            continue
        sample = df_filtered.sample(min(sample_size, len(df_filtered)), random_state=42)
        fig = px.scatter(
            sample, x=sample.index, y=sensor,
            color=config.TARGET_FAILURE,
            color_discrete_map={0: "#2ca02c", 1: "#d62728"},
            title=f"{sensor} – Threshold Monitoring",
            opacity=0.5, size_max=4,
        )
        # Add threshold lines
        if thresholds.get("warning"):
            add_threshold_line(fig, thresholds["warning"],
                               f"Warning ({thresholds['warning']} {thresholds['unit']})", "#ff7f0e")
        if thresholds.get("critical"):
            add_threshold_line(fig, thresholds["critical"],
                               f"Critical ({thresholds['critical']} {thresholds['unit']})", "#d62728")

        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Breach statistics
        n_total = len(df_filtered)
        col1, col2, col3 = st.columns(3)
        if thresholds.get("warning"):
            n_warn = int((df_filtered[sensor] >= thresholds["warning"]).sum())
            col1.metric(f"Above Warning ({thresholds['warning']})",
                        f"{n_warn:,}", delta=f"{n_warn/n_total*100:.1f}%")
        if thresholds.get("critical"):
            n_crit = int((df_filtered[sensor] >= thresholds["critical"]).sum())
            col2.metric(f"Above Critical ({thresholds['critical']})",
                        f"{n_crit:,}", delta=f"{n_crit/n_total*100:.1f}%")

    st.divider()

    # ── 4. Temperature Differential Analysis ─────────────────────────────────────
    st.markdown("## 🌡️ Temperature Differential Analysis")
    st.markdown("Heat Dissipation Failure (HDF) is triggered when Process−Air temperature differential drops too low.")

    if "Temp_Diff" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_filtered, x="Temp_Diff",
                color=config.TARGET_FAILURE if show_failures else None,
                color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                nbins=50, barmode="overlay", opacity=0.7,
                title="Temperature Differential Distribution",
            )
            add_threshold_line(fig, config.TEMP_DIFF_WARNING_THRESHOLD,
                               f"Warning ({config.TEMP_DIFF_WARNING_THRESHOLD} K)", "#ff7f0e")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # HDF failure specifically
            hdf_col = "HDF" if "HDF" in df.columns else None
            if hdf_col:
                fig2 = px.scatter(
                    df_filtered.sample(min(2000, len(df_filtered)), random_state=42),
                    x="Temp_Diff", y="Torque [Nm]" if "Torque [Nm]" in df.columns else "Rotational speed [rpm]",
                    color=hdf_col,
                    color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                    title="Temp Diff vs Torque (HDF Highlighted)",
                    opacity=0.6,
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 5. Power Analysis ──────────────────────────────────────────────────────────
    st.markdown("## ⚡ Power Analysis")
    st.markdown("Power = Torque × Rotational Speed (× 2π/60). Power failures occur at extreme values.")

    if "Power_W" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_filtered, x="Power_W",
                color=config.TARGET_FAILURE if show_failures else None,
                color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                nbins=60, barmode="overlay", opacity=0.7,
                title="Computed Power Distribution (Watts)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            pwf_col = "PWF" if "PWF" in df.columns else None
            if pwf_col:
                fig2 = px.scatter(
                    df_filtered.sample(min(2000, len(df_filtered)), random_state=42),
                    x="Power_W", y="Tool wear [min]" if "Tool wear [min]" in df.columns else "Torque [Nm]",
                    color=pwf_col,
                    color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                    title="Power vs Tool Wear (PWF Highlighted)",
                    opacity=0.6,
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 6. Sensor Correlation with Each Failure Type ───────────────────────────────
    st.markdown("## 🔗 Sensor Correlation per Failure Type")

    fail_cols = [c for c in config.FAILURE_TYPE_COLUMNS if c in df.columns]
    if fail_cols:
        corr_data = {}
        for f_col in fail_cols:
            corr_data[config.FAILURE_TYPE_MAP.get(f_col, f_col)] = [
                round(df[sensor].corr(df[f_col]), 3) for sensor in available
            ]

        corr_df = pd.DataFrame(corr_data, index=[s.split("[")[0].strip() for s in available])
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdYlGn",
            zmin=-0.3, zmax=0.3,
            title="Sensor × Failure Type Correlation",
            text_auto=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── 7. Product Type Analysis ──────────────────────────────────────────────────
    st.markdown("## 🏭 Failure Rate by Product Type")
    if "Type" in df.columns:
        type_stats = df.groupby("Type").agg(
            Total=(config.TARGET_FAILURE, "count"),
            Failures=(config.TARGET_FAILURE, "sum"),
        ).reset_index()
        type_stats["Failure Rate (%)"] = (type_stats["Failures"] / type_stats["Total"] * 100).round(2)
        type_stats["Product Type"] = type_stats["Type"].map({"L": "Low Quality", "M": "Medium Quality", "H": "High Quality"})

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(type_stats, x="Product Type", y="Failure Rate (%)",
                         color="Product Type",
                         color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
                         text="Failure Rate (%)",
                         title="Failure Rate by Product Type")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.dataframe(type_stats[["Product Type", "Total", "Failures", "Failure Rate (%)"]],
                         use_container_width=True, hide_index=True)

            # Per-type failure breakdown
            st.markdown("#### Failure Type by Product")
            for f_col in fail_cols[:3]:
                type_f = df.groupby("Type")[f_col].sum().reset_index()
                type_f.columns = ["Type", f_col]
                st.markdown(f"**{config.FAILURE_TYPE_MAP.get(f_col, f_col)}:** " +
                            ", ".join([f"{r['Type']}: {r[f_col]}" for _, r in type_f.iterrows()]))


main()
