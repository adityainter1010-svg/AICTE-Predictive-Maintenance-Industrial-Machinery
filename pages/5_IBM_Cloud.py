"""
pages/5_IBM_Cloud.py
IBM Cloud Lite Integration page – shows connection status, Watson NLP analysis,
model deployment info, and Cloud Object Storage operations.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="IBM Cloud Integration", page_icon="☁️", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.ibm-card {
    background: #f0f4ff; border-radius: 10px; padding: 1.2rem 1.5rem;
    border-left: 5px solid #0f62fe; margin-bottom: 1rem;
}
.ibm-title { color: #0f62fe; font-size: 1.1rem; font-weight: 700; }
.status-badge-ok   { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.85rem; }
.status-badge-warn { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.85rem; }
.status-badge-err  { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_key_is_set() -> bool:
    key = config.IBM_API_KEY
    return bool(key and key not in ("", "your_ibm_api_key_here"))


def project_id_is_set() -> bool:
    pid = config.IBM_WATSON_STUDIO_PROJECT_ID
    return bool(pid and pid not in ("", "your_project_id_here"))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    st.title("☁️ IBM Cloud Lite Integration")
    st.markdown("""
This page demonstrates the live IBM Cloud Lite integration built into the
Predictive Maintenance system. Two services are used:

| Service | Plan | Purpose |
|---|---|---|
| **IBM Watson Machine Learning** | Lite (Free) | Remote model scoring via REST API |
| **IBM Cloud Object Storage** | Lite (Free) | Remote model artefact storage |
| **IBM Watson Studio** | Lite (Free) | Project hosting & AutoAI experiments |
    """)

    st.divider()

    # ── 1. Configuration Status ─────────────────────────────────────────────────
    st.markdown("## 🔧 Configuration Status")

    col1, col2, col3, col4 = st.columns(4)
    api_ok   = api_key_is_set()
    proj_ok  = project_id_is_set()
    wml_ok   = bool(config.IBM_WML_URL)
    integ_on = config.IBM_INTEGRATION_ENABLED

    col1.metric("API Key",       "✅ Set"     if api_ok   else "❌ Missing",
                delta="Configured" if api_ok else "Set IBM_API_KEY in .env")
    col2.metric("Project ID",    "✅ Set"     if proj_ok  else "❌ Missing",
                delta=config.IBM_WATSON_STUDIO_PROJECT_ID[:18] + "…" if proj_ok else "Not set")
    col3.metric("WML Endpoint",  "✅ Ready"   if wml_ok   else "❌ Not set",
                delta=config.IBM_WML_URL.replace("https://", "") if wml_ok else "")
    col4.metric("Integration",   "✅ Enabled" if integ_on else "⚙️ Disabled",
                delta="Active" if integ_on else "Set IBM_INTEGRATION_ENABLED=true")

    st.divider()

    # ── 2. Credential Details ───────────────────────────────────────────────────
    st.markdown("## 📋 Active IBM Credentials")

    cred_data = {
        "Parameter": [
            "IBM API Key",
            "Watson Studio Project ID",
            "WML Endpoint URL",
            "WML Space ID",
            "WML Deployment ID",
            "COS Endpoint",
            "COS Bucket",
            "Integration Enabled",
        ],
        "Value": [
            ("✅ " + config.IBM_API_KEY[:8] + "…" + config.IBM_API_KEY[-4:]) if api_ok else "❌ Not configured",
            ("✅ " + config.IBM_WATSON_STUDIO_PROJECT_ID) if proj_ok else "❌ Not configured",
            config.IBM_WML_URL or "Not set",
            ("✅ " + config.IBM_WML_SPACE_ID[:16] + "…") if config.IBM_WML_SPACE_ID and config.IBM_WML_SPACE_ID != "your_wml_space_id_here" else "⚙️ Not set",
            ("✅ " + config.IBM_WML_DEPLOYMENT_ID[:16] + "…") if config.IBM_WML_DEPLOYMENT_ID and config.IBM_WML_DEPLOYMENT_ID != "your_wml_deployment_id_here" else "⚙️ Not set",
            config.IBM_COS_ENDPOINT,
            config.IBM_COS_BUCKET,
            "✅ True" if integ_on else "⚙️ False",
        ],
        "Status": [
            "✅" if api_ok else "❌",
            "✅" if proj_ok else "❌",
            "✅",
            "✅" if (config.IBM_WML_SPACE_ID and config.IBM_WML_SPACE_ID != "your_wml_space_id_here") else "⚙️",
            "✅" if (config.IBM_WML_DEPLOYMENT_ID and config.IBM_WML_DEPLOYMENT_ID != "your_wml_deployment_id_here") else "⚙️",
            "✅",
            "✅",
            "✅" if integ_on else "⚙️",
        ]
    }
    st.dataframe(pd.DataFrame(cred_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── 3. Live Connection Test ─────────────────────────────────────────────────
    st.markdown("## 🔗 Live IBM Cloud Connection Test")

    if not api_ok:
        st.warning("**IBM API key not configured.** Add your key to `.env` as `IBM_API_KEY=<your_key>` and restart.")
    else:
        if st.button("🚀 Run IBM Connection Test", type="primary"):
            with st.spinner("Testing IBM Cloud connectivity…"):
                try:
                    from src.ibm_integration import test_ibm_connection
                    result = test_ibm_connection()

                    status = result.get("overall_status", "Unknown")
                    if "✅" in status:
                        st.success(f"**{status}**")
                    elif "disabled" in status.lower():
                        st.info(f"**{status}**")
                    else:
                        st.error(f"**{status}**")

                    # Display test results as clean table
                    test_rows = []
                    for k, v in result.items():
                        if k == "overall_status":
                            continue
                        test_rows.append({"Test": k.replace("_", " ").title(), "Result": str(v)})
                    st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

                except Exception as exc:
                    st.error(f"Connection test failed: {exc}")

    st.divider()

    # ── 4. IAM Token Test ───────────────────────────────────────────────────────
    st.markdown("## 🔑 IBM IAM Authentication Test")
    st.markdown("Test whether the API key can successfully obtain an IAM bearer token.")

    if not api_ok:
        st.warning("Configure `IBM_API_KEY` first.")
    else:
        if st.button("🔑 Test IAM Token Exchange"):
            with st.spinner("Contacting IBM IAM endpoint…"):
                try:
                    from src.ibm_integration import get_iam_token
                    token = get_iam_token()
                    st.success("✅ IAM token obtained successfully!")
                    # Show first/last 12 chars only for security
                    st.code(f"Token (truncated): {token[:12]}…{token[-12:]}", language="text")
                    st.info(f"Token length: {len(token)} characters")
                except Exception as exc:
                    st.error(f"❌ IAM authentication failed: {exc}")
                    st.markdown("""
**Troubleshooting:**
- Verify the API key is correct in `.env`
- Ensure your IBM Cloud account is active
- Check network connectivity to `iam.cloud.ibm.com`
                    """)

    st.divider()

    # ── 5. Watson Machine Learning ──────────────────────────────────────────────
    st.markdown("## 🤖 Watson Machine Learning (WML) – Model Deployment")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
### How WML Integration Works

1. **Local Training** – Models are trained locally using scikit-learn/XGBoost
2. **Model Upload** – Best model `.pkl` is uploaded to IBM Cloud Object Storage
3. **Deployment** – Model is deployed as a REST endpoint in Watson Machine Learning
4. **Live Scoring** – The Failure Prediction page calls the WML endpoint for cloud-based inference

### WML Lite Plan Limits
- 20 capacity unit-hours / month free
- Up to 5 deployed models
- REST API v4 scoring
- Compatible with scikit-learn, XGBoost, TensorFlow
        """)

    with c2:
        st.markdown("### WML Service Architecture")
        fig = go.Figure()

        # Nodes
        nodes = {
            "Sensor Data": (0.1, 0.8),
            "Local ML Training\n(sklearn/XGBoost)": (0.35, 0.8),
            "IBM COS\n(Model Storage)": (0.65, 0.8),
            "IBM WML\n(Deployment)": (0.9, 0.8),
            "Streamlit\nDashboard": (0.5, 0.4),
            "REST API\nScoring": (0.75, 0.4),
        }
        for name, (x, y) in nodes.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=40, color="#0f62fe" if "IBM" in name else "#6ea6ff"),
                text=[name], textposition="bottom center",
                showlegend=False,
            ))

        # Arrows via annotations
        edges = [
            (0.1, 0.8, 0.3, 0.8),
            (0.4, 0.8, 0.6, 0.8),
            (0.65, 0.8, 0.88, 0.8),
            (0.9, 0.8, 0.76, 0.42),
            (0.5, 0.4, 0.73, 0.4),
        ]
        for x0, y0, x1, y1 in edges:
            fig.add_annotation(ax=x0, ay=y0, x=x1, y=y1,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=2, arrowcolor="#0f62fe", arrowwidth=2)

        fig.update_layout(
            height=320, showlegend=False,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0.2, 1.0]),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=60, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # WML Scoring test
    st.markdown("### 🎯 Test WML Endpoint Scoring")
    st.markdown("If a deployment is configured, test scoring a sample input through the WML REST API.")

    if not (config.IBM_WML_DEPLOYMENT_ID and config.IBM_WML_DEPLOYMENT_ID != "your_wml_deployment_id_here"):
        st.info("""
**WML Deployment not configured yet.**

To enable live WML scoring:
1. Open [IBM Watson Studio](https://dataplatform.cloud.ibm.com/)
2. Go to your project → **Assets** → **Models**
3. Promote the trained model to a Deployment Space
4. Create an **Online Deployment** → copy the Deployment ID
5. Add `IBM_WML_DEPLOYMENT_ID=<your_id>` to `.env`
6. Set `IBM_INTEGRATION_ENABLED=true` in `.env`
        """)
    else:
        if st.button("🧪 Test WML Scoring"):
            with st.spinner("Sending test vector to WML endpoint…"):
                try:
                    from src.ibm_integration import score_with_wml
                    # Sample feature vector: [air_temp, process_temp, rpm, torque, tool_wear, Type_H, Type_L, Type_M]
                    test_vector = [298.1, 308.6, 1551.0, 42.8, 0.0, 0.0, 1.0, 0.0]
                    result = score_with_wml(test_vector)
                    if result.get("status") == "success":
                        st.success("✅ WML scoring successful!")
                        st.json(result.get("prediction", result.get("raw", {})))
                    else:
                        st.info(f"Status: {result}")
                except Exception as exc:
                    st.error(f"WML scoring failed: {exc}")

    st.divider()

    # ── 6. Cloud Object Storage ─────────────────────────────────────────────────
    st.markdown("## 🪣 IBM Cloud Object Storage (COS)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### COS Usage in This Project
- Stores trained model `.pkl` files remotely
- Enables model sharing across environments
- Backup for local model files
- Compatible with any S3-compatible client

### COS Lite Plan Limits
- 25 GB storage free
- 20,000 GET requests / month
- 2,000 PUT requests / month
        """)

    with col2:
        st.markdown("### Model Files for COS Upload")
        model_files = [
            config.FAILURE_MODEL_PATH,
            config.FAILURE_TYPE_MODEL_PATH,
            config.PREPROCESSOR_PATH,
            config.LABEL_ENCODER_PATH,
            config.FEATURE_NAMES_PATH,
        ]
        file_data = []
        for p in model_files:
            size = p.stat().st_size if p.exists() else 0
            file_data.append({
                "File": p.name,
                "Local Status": "✅ Exists" if p.exists() else "❌ Missing",
                "Size (KB)": round(size / 1024, 1) if size else 0,
            })
        st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)

    if st.button("☁️ Upload Models to IBM COS"):
        if not api_ok:
            st.warning("Configure IBM_API_KEY first.")
        else:
            with st.spinner("Uploading model files to IBM Cloud Object Storage…"):
                try:
                    from src.ibm_integration import upload_models_to_cos
                    result = upload_models_to_cos()
                    if result.get("status") == "success":
                        st.success(f"✅ Uploaded {len(result.get('uploaded', []))} files to COS bucket `{config.IBM_COS_BUCKET}`")
                        st.json(result)
                    elif result.get("status") == "skipped":
                        st.info(f"Skipped: {result.get('reason')}")
                    else:
                        st.error(f"Upload failed: {result}")
                except Exception as exc:
                    st.error(f"COS upload failed: {exc}")

    st.divider()

    # ── 7. IBM Watson Studio Project ────────────────────────────────────────────
    st.markdown("## 🧪 IBM Watson Studio Project")

    if proj_ok:
        st.success(f"✅ Watson Studio Project configured: `{config.IBM_WATSON_STUDIO_PROJECT_ID}`")
        st.markdown(f"""
**Project ID:** `{config.IBM_WATSON_STUDIO_PROJECT_ID}`

This project ID links the predictive maintenance application to an IBM Watson Studio project,
enabling:
- **AutoAI experiments** – automated model selection and hyperparameter tuning
- **Notebook execution** – run the EDA notebook in IBM cloud environment
- **Model versioning** – track model iterations in the project assets
- **Collaboration** – share results with team members
        """)

        st.markdown("### Useful IBM Watson Studio Links")
        col1, col2, col3 = st.columns(3)
        col1.link_button(
            "📊 Open Watson Studio",
            "https://dataplatform.cloud.ibm.com/",
        )
        col2.link_button(
            "📁 View Project",
            f"https://dataplatform.cloud.ibm.com/projects/{config.IBM_WATSON_STUDIO_PROJECT_ID}",
        )
        col3.link_button(
            "🤖 Watson Machine Learning",
            "https://cloud.ibm.com/catalog/services/machine-learning",
        )
    else:
        st.info("**Watson Studio Project ID not configured.** Add `IBM_WATSON_STUDIO_PROJECT_ID` to `.env`.")

    st.divider()

    # ── 8. IBM Cloud Setup Guide ─────────────────────────────────────────────────
    st.markdown("## 📖 IBM Cloud Lite Setup Guide")

    with st.expander("Step-by-Step Setup Instructions", expanded=False):
        st.markdown("""
### Step 1 – Create IBM Cloud Free Account
1. Visit: https://cloud.ibm.com/registration
2. Sign up with an email address (**no credit card required** for Lite plan)
3. Verify your email and log in

### Step 2 – Create Watson Studio (Lite)
1. Go to IBM Cloud Catalog → search **Watson Studio**
2. Select **Lite** plan → click **Create**
3. Click **Launch Watson Studio**
4. Create a new **Project** → note the **Project ID** from the URL

### Step 3 – Create Watson Machine Learning (Lite)
1. IBM Cloud Catalog → search **Machine Learning**
2. Select **Lite** plan → click **Create**
3. Go to **Service credentials** → **New credential** → copy the **API key**
4. In Watson Studio → project **Settings** → associate the WML service

### Step 4 – Deploy a Model in Watson Studio
1. In Watson Studio project → **Assets** → **Add asset** → **Model**
2. Upload the trained `failure_model.pkl` file
3. **Promote** the model → select a **Deployment Space**
4. In the Deployment Space → **Deploy** → **Online** deployment
5. Copy the **Deployment ID** and **Space ID**

### Step 5 – Create Cloud Object Storage (Lite)
1. IBM Cloud Catalog → **Object Storage** → Lite plan → Create
2. Create a **bucket** named `predictive-maintenance-models`
3. **Service credentials** → **New credential** → enable **HMAC** → copy keys

### Step 6 – Update .env
```
IBM_API_KEY=<your_api_key>
IBM_WML_SPACE_ID=<your_space_id>
IBM_WML_DEPLOYMENT_ID=<your_deployment_id>
IBM_WATSON_STUDIO_PROJECT_ID=1b36f8d5-0bbb-4db1-a8e2-aecf9457aa5b
IBM_COS_API_KEY=<your_cos_api_key>
IBM_COS_SERVICE_INSTANCE_ID=<your_cos_instance_id>
IBM_INTEGRATION_ENABLED=true
```
        """)

    # ── 9. Architecture Diagram ──────────────────────────────────────────────────
    st.markdown("## 🏗️ Full System Architecture")
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTIVE MAINTENANCE SYSTEM                 │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Kaggle CSV   │───▶│ Preprocessing │───▶│ Feature Engineer │   │
│  │ (10K records)│    │ (scaling/OHE) │    │ Temp_Diff, Power │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ML LAYER                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ SMOTE        │    │ 4 Classifiers│    │ Best Model       │   │
│  │ Resampling   │───▶│ LR/DT/RF/XGB │───▶│ (by CV F1-macro) │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  IBM CLOUD LAYER (Lite)                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Watson Studio│    │ WML Lite     │    │ Cloud Object     │   │
│  │ Project ID:  │    │ Deployment   │    │ Storage (Lite)   │   │
│  │ 1b36f8d5…   │    │ REST API v4  │    │ Model Artefacts  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  PRESENTATION LAYER                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Streamlit    │    │ SHAP         │    │ Maintenance      │   │
│  │ Dashboard    │    │ Explainability│   │ Recommendations  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
    """)


main()
