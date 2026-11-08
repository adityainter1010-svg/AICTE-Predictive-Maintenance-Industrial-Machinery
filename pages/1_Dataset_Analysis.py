"""
pages/1_Dataset_Analysis.py
Dataset Overview and Exploratory Data Analysis page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Dataset Analysis", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def get_data():
    try:
        from src.preprocessing import load_dataset, engineer_features, derive_failure_type_label, inspect_dataset
        df = load_dataset()
        df = engineer_features(df)
        df = derive_failure_type_label(df)
        summary = inspect_dataset(df)
        return df, summary, None
    except FileNotFoundError as e:
        return None, None, str(e)


def main():
    st.title("📊 Dataset Overview & Exploratory Data Analysis")

    df, summary, err = get_data()

    if err or df is None:
        st.error(f"**Dataset not available.** {err or ''}")
        st.info("""
**How to download the dataset:**
1. Go to: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
2. Download `predictive_maintenance.csv`
3. Place it in the `data/` folder of this project
4. Restart the app
        """)
        return

    # ── Dataset Info ─────────────────────────────────────────────────────────
    st.markdown("## 📋 Dataset Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(sum(summary["missing_values"].values())))
    c4.metric("Duplicate Rows", summary["duplicates"])

    # Column breakdown
    with st.expander("📌 Column Details", expanded=False):
        dtype_df = pd.DataFrame({
            "Column": list(summary["dtypes"].keys()),
            "Type": list(summary["dtypes"].values()),
            "Missing": [summary["missing_values"].get(c, 0) for c in summary["dtypes"]],
            "Missing %": [summary["missing_pct"].get(c, 0) for c in summary["dtypes"]],
        })
        st.dataframe(dtype_df, use_container_width=True)

    with st.expander("📄 Sample Data (First 10 Rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Statistical Summary ───────────────────────────────────────────────────
    st.markdown("## 📈 Statistical Summary")
    num_cols = config.NUMERICAL_FEATURES + ["Temp_Diff", "Power_W"]
    available_num = [c for c in num_cols if c in df.columns]
    st.dataframe(df[available_num].describe().round(3), use_container_width=True)

    # ── Class Distribution ────────────────────────────────────────────────────
    st.markdown("## ⚖️ Class Distribution")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Machine Failure Distribution")
        fd = df[config.TARGET_FAILURE].value_counts().reset_index()
        fd.columns = ["Failure", "Count"]
        fd["Label"] = fd["Failure"].map({0: "No Failure", 1: "Failure"})
        fig = px.pie(fd, names="Label", values="Count",
                     color_discrete_sequence=["#2ca02c", "#d62728"],
                     title="Binary Failure Distribution")
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Failure Type Distribution")
        if config.TARGET_FAILURE_TYPE in df.columns:
            ftd = df[config.TARGET_FAILURE_TYPE].value_counts().reset_index()
            ftd.columns = ["Type", "Count"]
            ftd["Label"] = ftd["Type"].map(lambda x: config.FAILURE_TYPE_MAP.get(x, x))
            fig2 = px.bar(ftd, x="Label", y="Count", color="Label",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          title="Failure Type Distribution")
            fig2.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Individual Failure Indicator Counts ───────────────────────────────────
    st.markdown("### Individual Failure Indicator Counts")
    from src.failure_classification import failure_indicator_summary
    fi_df = failure_indicator_summary(df)
    if not fi_df.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(fi_df, use_container_width=True)
        with c2:
            fig = px.bar(fi_df, x="Code", y="Count", color="Full Name",
                         text="Count",
                         title="Individual Failure Type Counts")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=True, height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ── Sensor Distributions ───────────────────────────────────────────────────
    st.markdown("## 📉 Sensor Feature Distributions")
    sensor_cols = [c for c in config.NUMERICAL_FEATURES if c in df.columns]

    cols_per_row = 3
    for i in range(0, len(sensor_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(sensor_cols[i:i + cols_per_row]):
            with row_cols[j]:
                fig = px.histogram(df, x=col, color=config.TARGET_FAILURE,
                                   color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                                   nbins=50, barmode="overlay",
                                   title=f"{col} Distribution",
                                   labels={str(config.TARGET_FAILURE): "Machine Failure"})
                fig.update_layout(height=300, showlegend=(j == 0))
                st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown("## 🔥 Correlation Heatmap")
    corr_cols = available_num + [config.TARGET_FAILURE]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax,
                vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Box Plots ──────────────────────────────────────────────────────────────
    st.markdown("## 📦 Sensor Values by Machine Failure Status")
    for col in sensor_cols:
        fig = px.box(df, x=config.TARGET_FAILURE, y=col,
                     color=config.TARGET_FAILURE,
                     color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                     title=f"{col} by Failure Status",
                     labels={str(config.TARGET_FAILURE): "Machine Failure (0=Normal, 1=Failure)"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: Torque vs Rotational Speed ───────────────────────────────────
    st.markdown("## 🔵 Scatter: Torque vs Rotational Speed (Coloured by Failure)")
    if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
        fig = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="Rotational speed [rpm]", y="Torque [Nm]",
            color=config.TARGET_FAILURE,
            color_continuous_scale=["#2ca02c", "#d62728"],
            opacity=0.6,
            title="Torque vs Rotational Speed (Failure Highlighted)",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Engineered Feature: Power ─────────────────────────────────────────────
    if "Power_W" in df.columns:
        st.markdown("## ⚡ Engineered Feature: Power (Torque × ω)")
        fig = px.histogram(df, x="Power_W", color=config.TARGET_FAILURE,
                           color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                           nbins=60, barmode="overlay",
                           title="Power Distribution by Failure Status")
        st.plotly_chart(fig, use_container_width=True)

    # ── Pairplot (sampled) ────────────────────────────────────────────────────
    st.markdown("## 🔗 Feature Pair Relationships")
    sample = df.sample(min(1000, len(df)), random_state=42)
    pair_cols = sensor_cols[:4]
    if len(pair_cols) >= 2:
        fig = px.scatter_matrix(
            sample, dimensions=pair_cols,
            color=config.TARGET_FAILURE,
            color_discrete_map={0: "#2ca02c", 1: "#d62728"},
            title="Sensor Feature Pair Plot (1000-record sample)",
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=2, opacity=0.5))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)


main()
