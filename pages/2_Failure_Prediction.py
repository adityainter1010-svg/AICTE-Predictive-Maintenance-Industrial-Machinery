"""
pages/2_Failure_Prediction.py
Interactive machine failure and failure-type prediction page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Failure Prediction", page_icon="🔮", layout="wide")


@st.cache_resource(show_spinner=False)
def load_all_models():
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
    except FileNotFoundError as e:
        return {"loaded": False, "error": str(e)}


def render_risk_gauge(probability: float) -> go.Figure:
    """Render a half-donut risk gauge."""
    from src.prediction import get_risk_level
    level, color = get_risk_level(probability)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"Failure Risk Level: <b>{level}</b>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#d4edda"},
                {"range": [25, 50], "color": "#fff3cd"},
                {"range": [50, 75], "color": "#ffd6a5"},
                {"range": [75, 100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def main():
    st.title("🔮 Machine Failure Prediction")
    st.markdown("Enter real-time sensor readings to predict machine failure and identify the failure type.")

    models = load_all_models()
    if not models.get("loaded"):
        st.error(f"**Models not loaded.** {models.get('error', '')}")
        st.info("Train the models first: `python -m src.train_model` from the project folder.")
        return

    st.divider()

    # ── Input Form ─────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Sensor Input Panel")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            air_temp = st.number_input(
                "Air Temperature (K)", min_value=290.0, max_value=310.0,
                value=298.1, step=0.1, help="Typical range: 295–305 K")
            process_temp = st.number_input(
                "Process Temperature (K)", min_value=300.0, max_value=320.0,
                value=308.6, step=0.1, help="Typical range: 305–315 K")

        with c2:
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)", min_value=1000.0, max_value=3000.0,
                value=1551.0, step=1.0, help="Typical range: 1168–2886 rpm")
            torque = st.number_input(
                "Torque (Nm)", min_value=1.0, max_value=100.0,
                value=42.8, step=0.1, help="Typical range: 3.8–76.6 Nm")

        with c3:
            tool_wear = st.number_input(
                "Tool Wear (minutes)", min_value=0.0, max_value=260.0,
                value=0.0, step=1.0, help="Accumulated tool wear in minutes (0–253)")
            product_type = st.selectbox(
                "Product Type", options=["L", "M", "H"],
                help="L=Low quality, M=Medium quality, H=High quality",
            )

        submitted = st.form_submit_button("🔍 Predict Failure", type="primary", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────────────────────────
    if submitted:
        try:
            from src.prediction import predict, get_risk_level
            from src.recommendations import get_recommendations, format_recommendations_markdown
            from src.failure_classification import get_failure_type_description, get_failure_type_color

            with st.spinner("Running prediction…"):
                result = predict(air_temp, process_temp, rotational_speed, torque, tool_wear, product_type)

            failure_pred = result["failure_prediction"]
            failure_prob = result["failure_probability"]
            failure_type_code = result["failure_type_code"]
            failure_type_label = result["failure_type_label"]
            risk_level, risk_color = get_risk_level(failure_prob)

            st.divider()
            st.markdown("## 📊 Prediction Results")

            # Main result cards
            c1, c2, c3 = st.columns(3)
            status_label = "⚠️ FAILURE PREDICTED" if failure_pred == 1 else "✅ NORMAL OPERATION"
            status_color = "#d62728" if failure_pred == 1 else "#2ca02c"
            c1.markdown(
                f'<div style="background:{status_color}20;border:2px solid {status_color};'
                f'border-radius:10px;padding:1rem;text-align:center;">'
                f'<h3 style="color:{status_color};margin:0">{status_label}</h3>'
                f'<p style="margin:0;font-size:0.9rem">Failure Probability: <b>{failure_prob*100:.1f}%</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            type_color = get_failure_type_color(failure_type_code)
            c2.markdown(
                f'<div style="background:{type_color}20;border:2px solid {type_color};'
                f'border-radius:10px;padding:1rem;text-align:center;">'
                f'<h3 style="color:{type_color};margin:0">🔧 {failure_type_label}</h3>'
                f'<p style="margin:0;font-size:0.9rem">Predicted Failure Category</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            c3.markdown(
                f'<div style="background:{risk_color}20;border:2px solid {risk_color};'
                f'border-radius:10px;padding:1rem;text-align:center;">'
                f'<h3 style="color:{risk_color};margin:0">🚦 {risk_level} Risk</h3>'
                f'<p style="margin:0;font-size:0.9rem">Risk Assessment Level</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge + Failure type probabilities
            g1, g2 = st.columns([1, 1])
            with g1:
                st.plotly_chart(render_risk_gauge(failure_prob), use_container_width=True)
            with g2:
                if result["failure_type_probs"]:
                    st.markdown("#### Failure Type Probabilities")
                    prob_df = pd.DataFrame(
                        list(result["failure_type_probs"].items()),
                        columns=["Failure Type", "Probability"]
                    ).sort_values("Probability", ascending=True)
                    fig = px.bar(prob_df, x="Probability", y="Failure Type",
                                 orientation="h",
                                 color="Probability",
                                 color_continuous_scale=["#2ca02c", "#ffdd57", "#d62728"],
                                 text="Probability")
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig.update_layout(height=350, showlegend=False, xaxis_range=[0, 1.05])
                    st.plotly_chart(fig, use_container_width=True)

            # Sensor values used
            st.markdown("#### 📋 Sensor Readings Used for Prediction")
            sensor_df = pd.DataFrame(list(result["feature_values"].items()), columns=["Parameter", "Value"])
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)

            # Failure type description
            desc = get_failure_type_description(failure_type_code)
            st.info(f"**Failure Type Description:** {desc}")

            # Maintenance recommendations
            st.divider()
            st.markdown("### 🛠️ Maintenance Recommendations")
            rec = get_recommendations(failure_type_code, {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            })

            urgency_color = rec["urgency_color"]
            st.markdown(
                f'<div style="background:{urgency_color}15;border-left:5px solid {urgency_color};'
                f'padding:0.8rem 1rem;border-radius:4px;margin-bottom:1rem;">'
                f'<strong>Urgency:</strong> {rec["urgency"]} | '
                f'<strong>Timeline:</strong> {rec["timeline"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            for i, action in enumerate(rec["actions"], 1):
                st.markdown(f"{i}. {action}")

            # IBM WML scoring (if enabled)
            if config.IBM_INTEGRATION_ENABLED:
                st.divider()
                st.markdown("### ☁️ IBM Watson Machine Learning Scoring")
                with st.spinner("Scoring via IBM WML endpoint…"):
                    try:
                        from src.ibm_integration import score_with_wml
                        feature_vector = list(result["processed_input"][0])
                        wml_result = score_with_wml(feature_vector)
                        if wml_result.get("status") == "success":
                            st.success("IBM WML scoring successful!")
                            st.json(wml_result.get("prediction", wml_result.get("raw")))
                        else:
                            st.info(f"IBM WML: {wml_result.get('reason', 'Disabled')}")
                    except Exception as exc:
                        st.warning(f"IBM WML scoring unavailable: {exc}")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)

    # ── Batch Prediction ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📂 Batch Prediction from CSV")
    st.markdown("Upload a CSV with sensor columns to run bulk predictions.")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", key="batch_csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            from src.prediction import batch_predict
            with st.spinner("Running batch predictions…"):
                result_df = batch_predict(batch_df)
            st.success(f"✅ Predictions complete for {len(result_df):,} records.")
            st.dataframe(result_df.head(50), use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", data=csv,
                               file_name="batch_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


main()
