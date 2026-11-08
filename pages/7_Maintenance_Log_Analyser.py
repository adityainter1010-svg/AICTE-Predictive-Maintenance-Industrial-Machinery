"""
pages/7_Maintenance_Log_Analyser.py
AI-powered maintenance log analyser using IBM Watson NLP + local keyword analysis.
Analyses free-text maintenance observations to classify failure types,
assess severity, and generate IBM watsonx.ai maintenance reports.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Maintenance Log Analyser", page_icon="📝", layout="wide")

st.markdown("""
<style>
.log-card {
    background:#f8f9fa; border-left:4px solid #0f62fe;
    padding:1rem 1.2rem; border-radius:6px; margin-bottom:0.8rem;
}
.sev-critical { color:#d62728; font-weight:bold; }
.sev-warning  { color:#ff7f0e; font-weight:bold; }
.sev-info     { color:#2ca02c; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


def severity_color(sev: str) -> str:
    return {"critical": "#d62728", "warning": "#ff7f0e", "info": "#2ca02c"}.get(sev, "#888")


def main():
    st.title("📝 Maintenance Log Analyser")
    st.markdown("""
Analyse free-text maintenance observations using:
- **Local keyword matching** – instant, no API required
- **IBM Watson NLU** – advanced entity extraction, sentiment, categories *(requires Watson NLU service)*
- **IBM watsonx.ai (Granite)** – AI-generated maintenance reports *(requires Watson Studio Project ID)*
    """)

    from src.ibm_watson_nlp import (
        analyse_maintenance_log, generate_report_with_watsonx,
        SAMPLE_LOGS, FAILURE_KEYWORDS,
    )

    api_key_set  = bool(config.IBM_API_KEY and config.IBM_API_KEY not in ("", "your_ibm_api_key_here"))
    project_set  = bool(config.IBM_WATSON_STUDIO_PROJECT_ID and config.IBM_WATSON_STUDIO_PROJECT_ID != "your_project_id_here")

    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Log Analysis",
        "📋 Batch Log Processing",
        "🤖 AI Report Generator",
    ])

    # ── Tab 1: Single Log Analysis ────────────────────────────────────────────
    with tab1:
        st.markdown("### Analyse a Single Maintenance Log Entry")

        col1, col2 = st.columns([2, 1])
        with col1:
            log_input = st.text_area(
                "Enter Maintenance Log",
                height=130,
                help="Describe what was observed on the machine – symptoms, sensor readings, actions taken.",
                placeholder="e.g. Machine #A12 experienced overheating. Coolant flow rate dropped. Thermal sensors showed readings above 320K…",
            )

        with col2:
            st.markdown("**Sample Logs:**")
            for i, sample in enumerate(SAMPLE_LOGS):
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    st.session_state["log_sample"] = sample
                    st.rerun()

        if "log_sample" in st.session_state and not log_input:
            log_input = st.session_state["log_sample"]

        use_nlu = st.checkbox(
            "🔬 Use IBM Watson NLU",
            value=False,
            disabled=not api_key_set,
            help="Requires IBM_API_KEY and a Watson NLU service instance."
        )
        if not api_key_set:
            st.caption("⚙️ IBM API key not configured. Running local keyword analysis.")

        if st.button("🔍 Analyse Log Entry", type="primary", disabled=not log_input):
            with st.spinner("Analysing maintenance log…"):
                insights = analyse_maintenance_log(log_input or "", use_ibm_nlu=use_nlu)

            st.divider()
            st.markdown("## 📊 Analysis Results")

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            sev_col = severity_color(insights["severity"])
            c1.metric("Predicted Failure Type", insights["predicted_type_label"])
            c2.metric("Severity Level", insights["severity"].capitalize())
            c3.metric("Urgency", insights["urgency"])
            c4.metric("Analysis Method",
                      "IBM Watson NLU" if insights["analysis_method"] == "ibm_watson_nlu" else "Local Keywords")

            # Detected failure types
            if insights.get("detected_failure_types"):
                st.info("**Detected Failure Indicators:** " + " | ".join(insights["detected_failure_types"]))

            # Type score chart
            if insights.get("type_scores"):
                st.markdown("#### Failure Type Score Breakdown")
                score_df = pd.DataFrame(
                    list(insights["type_scores"].items()),
                    columns=["Failure Type", "Keyword Matches"]
                ).sort_values("Keyword Matches", ascending=True)
                fig = px.bar(score_df, x="Keyword Matches", y="Failure Type",
                             orientation="h", color="Keyword Matches",
                             color_continuous_scale=["#2ca02c", "#ffdd57", "#d62728"],
                             title="Failure Type Keyword Match Scores")
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # IBM NLU results
            if insights.get("nlu_keywords"):
                st.markdown("#### 🔬 IBM Watson NLU Insights")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Extracted Keywords:**")
                    kw_df = pd.DataFrame({"Keyword": insights["nlu_keywords"]})
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                with c2:
                    if insights.get("nlu_entities"):
                        st.markdown("**Entities Detected:**")
                        ent_df = pd.DataFrame(insights["nlu_entities"])
                        st.dataframe(ent_df, use_container_width=True, hide_index=True)

                if insights.get("nlu_sentiment"):
                    sent = insights["nlu_sentiment"]
                    label = sent.get("label", "N/A")
                    score = sent.get("score", 0)
                    st.metric("Document Sentiment", label, delta=f"score: {score:.3f}")

            # Recommendations
            st.markdown("#### 🛠️ Recommended Maintenance Actions")
            urgency_color_map = {
                "Routine":   "#2ca02c",
                "Urgent":    "#ff7f0e",
                "Immediate": "#d62728",
                "High":      "#ff7f0e",
                "Critical – Immediate Shutdown": "#7f0000",
            }
            urgency = insights.get("urgency", "Routine")
            urg_color = urgency_color_map.get(urgency, "#888")
            st.markdown(
                f'<div style="background:{urg_color}15;border-left:4px solid {urg_color};'
                f'padding:0.6rem 1rem;border-radius:4px;margin-bottom:0.8rem;">'
                f'<b>Urgency: {urgency}</b></div>',
                unsafe_allow_html=True,
            )
            for i, action in enumerate(insights.get("recommendations", []), 1):
                st.markdown(f"{i}. {action}")

    # ── Tab 2: Batch Log Processing ───────────────────────────────────────────
    with tab2:
        st.markdown("### Batch Process Multiple Maintenance Logs")
        st.markdown("Analyse all sample logs at once, or upload a CSV with a `log_text` column.")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📋 Analyse All Sample Logs", type="primary"):
                with st.spinner("Processing all sample logs…"):
                    results = []
                    for log in SAMPLE_LOGS:
                        ins = analyse_maintenance_log(log, use_ibm_nlu=False)
                        results.append({
                            "Log (truncated)": log[:80] + "…",
                            "Predicted Failure": ins["predicted_type_label"],
                            "Severity": ins["severity"].capitalize(),
                            "Urgency": ins["urgency"],
                            "Failure Types Found": ", ".join(ins.get("detected_failure_types", ["None"])),
                        })

                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Summary chart
                sev_counts = result_df["Severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig = px.pie(sev_counts, names="Severity", values="Count",
                             color="Severity",
                             color_discrete_map={"Critical": "#d62728", "Warning": "#ff7f0e", "Info": "#2ca02c"},
                             title="Severity Distribution Across Sample Logs")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Upload CSV Logs")
            uploaded = st.file_uploader("Upload CSV (must have `log_text` column)", type="csv")
            if uploaded:
                try:
                    log_df = pd.read_csv(uploaded)
                    if "log_text" not in log_df.columns:
                        st.error("CSV must have a `log_text` column.")
                    else:
                        with st.spinner(f"Analysing {len(log_df)} logs…"):
                            rows = []
                            for text in log_df["log_text"].fillna("").tolist():
                                ins = analyse_maintenance_log(text)
                                rows.append({
                                    "Predicted Failure": ins["predicted_type_label"],
                                    "Severity": ins["severity"],
                                    "Urgency": ins["urgency"],
                                })
                            out_df = pd.concat([log_df, pd.DataFrame(rows)], axis=1)
                        st.dataframe(out_df, use_container_width=True)
                        st.download_button(
                            "⬇️ Download Results",
                            out_df.to_csv(index=False).encode("utf-8"),
                            "log_analysis_results.csv",
                            "text/csv",
                        )
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")

    # ── Tab 3: AI Report Generator ────────────────────────────────────────────
    with tab3:
        st.markdown("### 🤖 IBM watsonx.ai Maintenance Report Generator")
        st.markdown(f"""
Uses **IBM Granite-13B-Instruct** model via `watsonx.ai` to generate a professional
maintenance report based on predicted failure type and sensor readings.

**Watson Studio Project ID:** `{config.IBM_WATSON_STUDIO_PROJECT_ID or 'Not configured'}`
        """)

        if not project_set:
            st.warning("Watson Studio Project ID not configured. Set `IBM_WATSON_STUDIO_PROJECT_ID` in `.env`.")

        c1, c2 = st.columns(2)
        with c1:
            failure_type = st.selectbox("Failure Type", list(config.FAILURE_TYPE_MAP.values())[1:])
            air_temp = st.number_input("Air Temperature (K)", value=298.1, step=0.1)
            process_temp = st.number_input("Process Temperature (K)", value=308.6, step=0.1)

        with c2:
            rpm = st.number_input("Rotational Speed (rpm)", value=1551.0, step=1.0)
            torque = st.number_input("Torque (Nm)", value=42.8, step=0.1)
            tool_wear = st.number_input("Tool Wear (min)", value=150.0, step=1.0)

        sensor_vals = {
            "Air Temperature (K)": air_temp,
            "Process Temperature (K)": process_temp,
            "Rotational Speed (rpm)": rpm,
            "Torque (Nm)": torque,
            "Tool Wear (min)": tool_wear,
        }

        if st.button("🤖 Generate AI Maintenance Report", type="primary"):
            if not project_set:
                st.error("Watson Studio Project ID required for watsonx.ai generation.")
            elif not api_key_set:
                st.error("IBM API Key required.")
            else:
                with st.spinner("Generating report via IBM watsonx.ai Granite model…"):
                    report = generate_report_with_watsonx(failure_type, sensor_vals)

                st.markdown("#### 📄 Generated Maintenance Report")
                st.markdown(
                    f'<div class="log-card">{report}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️ Download Report",
                    report.encode("utf-8"),
                    "maintenance_report.txt",
                    "text/plain",
                )

        st.divider()
        st.markdown("#### 📖 About IBM watsonx.ai")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| **Model** | IBM Granite-13B-Instruct v2 |
| **Project ID** | `{config.IBM_WATSON_STUDIO_PROJECT_ID or 'Not set'}` |
| **Endpoint** | `{config.IBM_WATSONX_URL}` |
| **Plan** | IBM Lite (free tier) |
| **Token limit** | 350 tokens per request |

IBM Granite is IBM's enterprise-grade foundation model, available through watsonx.ai.
It is optimised for instruction-following tasks like report generation, summarisation,
and classification.
        """)


main()
