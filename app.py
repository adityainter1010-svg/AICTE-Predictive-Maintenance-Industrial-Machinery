"""
app.py – Main Streamlit entry point for the Predictive Maintenance Dashboard.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800; color: #1a1a2e;
        text-align: center; padding: 0.5rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7f8fa; border-radius: 10px; padding: 1rem 1.5rem;
        border-left: 5px solid #3b82d4; margin-bottom: 1rem;
    }
    .status-ok  { color: #2ca02c; font-weight: bold; }
    .status-warn{ color: #ff7f0e; font-weight: bold; }
    .status-err { color: #d62728; font-weight: bold; }
    .section-title { font-size: 1.4rem; font-weight: 700; margin: 1.2rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached Loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Load dataset with caching; returns None if unavailable."""
    try:
        from src.preprocessing import load_dataset, engineer_features, derive_failure_type_label
        df = load_dataset()
        df = engineer_features(df)
        df = derive_failure_type_label(df)
        return df
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models from disk with caching."""
    try:
        from src.train_model import load_failure_model, load_failure_type_model
        from src.preprocessing import load_preprocessor, load_label_encoder, load_feature_names
        return {
            "failure_model": load_failure_model(),
            "type_model": load_failure_type_model(),
            "preprocessor": load_preprocessor(),
            "label_encoder": load_label_encoder(),
            "feature_names": load_feature_names(),
            "loaded": True,
        }
    except FileNotFoundError:
        return {"loaded": False}


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(df, models):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png", width=90)
        st.markdown("## ⚙️ Predictive Maintenance")
        st.markdown(f"**Version:** {config.APP_VERSION}")
        st.divider()

        # Dataset status
        if df is not None and not isinstance(df, tuple):
            st.markdown(f"📊 **Dataset:** {len(df):,} records")
            st.markdown(f"🔢 **Features:** {len(config.ALL_FEATURES)}")
            failure_rate = df[config.TARGET_FAILURE].mean() * 100
            st.markdown(f"⚠️ **Failure Rate:** {failure_rate:.1f}%")
        else:
            st.warning("⚠️ Dataset not loaded")

        # Model status
        if models.get("loaded"):
            st.markdown('<p class="status-ok">✅ Models Loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-err">❌ Models not trained</p>', unsafe_allow_html=True)
            if st.button("🚀 Train Models Now", type="primary"):
                with st.spinner("Training models… this may take 1–3 minutes"):
                    try:
                        from src.train_model import run_training
                        run_training()
                        st.success("Training complete! Reload the page.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        st.divider()

        # IBM status
        api_key_set = bool(config.IBM_API_KEY and config.IBM_API_KEY not in ("", "your_ibm_api_key_here"))
        proj_id_set = bool(config.IBM_WATSON_STUDIO_PROJECT_ID and config.IBM_WATSON_STUDIO_PROJECT_ID != "your_project_id_here")

        ibm_status = "✅ Enabled" if config.IBM_INTEGRATION_ENABLED else "⚙️ Configured" if api_key_set else "🔲 Disabled"
        st.markdown(f"**IBM Cloud:** {ibm_status}")

        if api_key_set:
            masked = config.IBM_API_KEY[:6] + "…" + config.IBM_API_KEY[-4:]
            st.markdown(f"🔑 **API Key:** `{masked}`")
        if proj_id_set:
            st.markdown(f"📁 **Project:** `{config.IBM_WATSON_STUDIO_PROJECT_ID[:18]}…`")

        if st.button("🔗 Test IBM Connection"):
            from src.ibm_integration import test_ibm_connection
            result = test_ibm_connection()
            overall = result.get("overall_status", "Unknown")
            if "✅" in overall:
                st.success(overall)
            elif "disabled" in overall.lower():
                st.info(overall)
            else:
                st.error(overall)

        st.divider()
        st.caption("AICTE Problem Statement #39\nPredictive Maintenance of Industrial Machinery")


# ── Home Page ──────────────────────────────────────────────────────────────────

def render_home(df, models):
    st.markdown('<div class="main-header">⚙️ Predictive Maintenance of Industrial Machinery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AICTE Problem Statement #39 · AI/ML-Based Proactive Maintenance System</div>', unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    if df is not None and not isinstance(df, tuple):
        failure_rate = df[config.TARGET_FAILURE].mean() * 100
        failure_count = int(df[config.TARGET_FAILURE].sum())
        type_col = config.TARGET_FAILURE_TYPE
        n_types = df[type_col].nunique() - 1 if type_col in df.columns else 0

        col1.metric("📋 Total Records", f"{len(df):,}")
        col2.metric("⚠️ Failure Cases", f"{failure_count:,}")
        col3.metric("📈 Failure Rate", f"{failure_rate:.2f}%")
        col4.metric("🔧 Failure Types", str(n_types))
        col5.metric("📊 Features", str(len(config.ALL_FEATURES)))
    else:
        col1.metric("📋 Total Records", "–")
        col2.metric("⚠️ Failure Cases", "–")
        col3.metric("📈 Failure Rate", "–")
        col4.metric("🔧 Failure Types", "–")
        col5.metric("📊 Features", str(len(config.ALL_FEATURES)))

    st.divider()

    # Project overview
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 🎯 Project Objective")
        st.markdown("""
Develop a machine learning–based predictive maintenance system for industrial machinery that:

- **Detects** potential machine failures **before** they occur using real-time sensor data.
- **Classifies** the type of failure (Tool Wear, Heat Dissipation, Power, Overstrain, Random).
- **Explains** predictions using SHAP (SHapley Additive exPlanations).
- **Recommends** targeted maintenance actions based on the failure category.
- **Integrates** with **IBM Cloud Lite** (Watson Machine Learning + Cloud Object Storage).

This reduces unplanned downtime, cuts maintenance costs, and extends machine lifespan.
        """)

    with c2:
        st.markdown("### 🛠️ Technology Stack")
        tech = {
            "Python": "3.10+",
            "Scikit-learn": "ML pipeline",
            "XGBoost": "Best classifier",
            "SHAP": "Explainability",
            "Streamlit": "Dashboard",
            "IBM WML": "Cloud scoring",
            "IBM COS": "Model storage",
            "Pandas / NumPy": "Data processing",
            "Plotly / Seaborn": "Visualisation",
        }
        for k, v in tech.items():
            st.markdown(f"- **{k}** – {v}")

    st.divider()

    # Failure type descriptions
    st.markdown("### ⚠️ Failure Types Covered")
    from src.failure_classification import FAILURE_DESCRIPTIONS, FAILURE_COLORS
    cols = st.columns(3)
    for i, (code, desc) in enumerate(FAILURE_DESCRIPTIONS.items()):
        if code == "No Failure":
            continue
        color = FAILURE_COLORS.get(code, "#888")
        cols[i % 3].markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<strong>{config.FAILURE_TYPE_MAP.get(code, code)}</strong><br>'
            f'<small>{desc[:120]}…</small></div>',
            unsafe_allow_html=True,
        )

    # IBM Cloud status banner
    api_key_set = bool(config.IBM_API_KEY and config.IBM_API_KEY not in ("", "your_ibm_api_key_here"))
    proj_id_set = bool(config.IBM_WATSON_STUDIO_PROJECT_ID and config.IBM_WATSON_STUDIO_PROJECT_ID != "your_project_id_here")

    if api_key_set or proj_id_set:
        st.markdown("### ☁️ IBM Cloud Lite – Integration Status")
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("IBM API Key",       "✅ Configured" if api_key_set else "❌ Missing")
        ic2.metric("Watson Studio",     "✅ Project Set" if proj_id_set else "⚙️ Not set",
                   delta=config.IBM_WATSON_STUDIO_PROJECT_ID if proj_id_set else None)
        ic3.metric("Integration",       "✅ Active" if config.IBM_INTEGRATION_ENABLED else "⚙️ Standby")
        st.caption("👉 Go to **5 IBM Cloud** page for full details, connection tests, and WML deployment.")
        st.divider()

    # Maintenance log analyser (IBM Watson NLP)
    st.markdown("### 🛠️ Maintenance Log Analyser (IBM Watson NLP)")
    st.markdown("Enter a free-text maintenance observation. The system extracts failure type, severity, and recommendations.")

    from src.ibm_watson_nlp import analyse_maintenance_log, SAMPLE_LOGS
    log_text = st.text_area(
        "Maintenance Log Entry",
        value=SAMPLE_LOGS[0],
        height=100,
        help="Describe the machine symptom or observation in plain English.",
    )
    use_nlu = st.checkbox("Use IBM Watson NLU (requires IBM API key + NLU service)", value=False)

    if st.button("🔍 Analyse Log", type="secondary"):
        with st.spinner("Analysing…"):
            insights = analyse_maintenance_log(log_text, use_ibm_nlu=use_nlu)

        a1, a2, a3 = st.columns(3)
        a1.metric("Predicted Failure", insights["predicted_type_label"])
        a2.metric("Severity",          insights["severity"].capitalize())
        a3.metric("Urgency",           insights["urgency"])

        if insights.get("detected_failure_types"):
            st.info("**Detected failure indicators:** " + ", ".join(insights["detected_failure_types"]))
        if insights.get("nlu_keywords"):
            st.caption("**NLU Keywords:** " + ", ".join(insights["nlu_keywords"]))

        st.markdown("**Recommended Actions:**")
        for i, action in enumerate(insights.get("recommendations", [])[:5], 1):
            st.markdown(f"{i}. {action}")

        st.caption(f"Analysis method: `{insights.get('analysis_method', 'local')}`")

    st.divider()

    # Dataset availability notice
    if df is None or isinstance(df, tuple):
        st.warning("""
**📥 Dataset Required**

Please download the dataset from Kaggle and place it at `data/predictive_maintenance.csv`:

👉 https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

Then train the models using: `python -m src.train_model`
        """)

    # Model training notice
    if not models.get("loaded"):
        st.info("**🧠 Models not yet trained.** Use the sidebar button or run `python -m src.train_model`.")

    # Training summary if available
    summary_path = config.MODEL_DIR / "training_summary.json"
    if summary_path.exists():
        st.divider()
        st.markdown("### 📊 Latest Training Summary (Cross-Validation F1-Macro)")
        with open(summary_path) as f:
            summary = json.load(f)

        for task, results in summary.items():
            label = "Machine Failure" if task == "failure" else "Failure Type"
            df_summary = pd.DataFrame([
                {"Model": k, "CV F1 Mean": v["cv_f1_mean"], "CV F1 Std": v["cv_f1_std"]}
                for k, v in results.items()
            ]).sort_values("CV F1 Mean", ascending=False)

            st.markdown(f"**{label} Models**")
            fig = px.bar(df_summary, x="Model", y="CV F1 Mean",
                         error_y="CV F1 Std", color="Model",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         text="CV F1 Mean")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=350, showlegend=False, yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    df_result = load_data()
    if isinstance(df_result, tuple):
        df, err = None, df_result[1]
    elif df_result is None:
        df, err = None, "Dataset file not found."
    else:
        df, err = df_result, None

    models = load_models()

    render_sidebar(df, models)
    render_home(df, models)


if __name__ == "__main__":
    main()
