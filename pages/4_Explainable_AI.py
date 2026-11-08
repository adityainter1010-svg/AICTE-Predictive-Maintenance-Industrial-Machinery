"""
pages/4_Explainable_AI.py
SHAP-based explainability page – global and per-prediction explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def load_shap_resources():
    try:
        from src.train_model import load_failure_model
        from src.preprocessing import (load_preprocessor, load_label_encoder,
                                       load_feature_names, load_dataset,
                                       engineer_features, derive_failure_type_label)
        from sklearn.model_selection import train_test_split

        df = load_dataset()
        df = engineer_features(df)
        df = derive_failure_type_label(df)

        X = df[config.ALL_FEATURES]
        y_failure = df[config.TARGET_FAILURE].astype(int)
        y_type_raw = df[config.TARGET_FAILURE_TYPE]
        le = load_label_encoder()
        y_type = le.transform(y_type_raw)

        _, X_test, _, y_f_test, _, _ = train_test_split(
            X, y_failure, y_type,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_failure,
        )

        preprocessor = load_preprocessor()
        X_test_proc = preprocessor.transform(X_test)
        feature_names = load_feature_names()
        failure_model = load_failure_model()

        return {
            "model": failure_model,
            "X_test": X_test_proc,
            "y_f_test": y_f_test.values,
            "feature_names": feature_names,
            "loaded": True,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def main():
    st.title("🧠 Explainable AI – SHAP Analysis")
    st.markdown("""
SHAP (SHapley Additive exPlanations) provides transparent, model-agnostic explanations
for individual predictions and global feature importance.
    """)

    res = load_shap_resources()
    if not res.get("loaded"):
        st.error(f"**Resources not loaded.** {res.get('error', '')}")
        st.info("Train the models first: `python -m src.train_model`")
        return

    model = res["model"]
    X_test = res["X_test"]
    feature_names = res["feature_names"]

    from src.explainability import get_explainer, compute_shap_values, plot_shap_summary, plot_shap_bar, get_top_contributing_features

    # Sample for SHAP (limit for speed)
    max_shap_samples = 300
    idx = np.random.RandomState(42).choice(X_test.shape[0], min(max_shap_samples, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]

    tab1, tab2, tab3 = st.tabs([
        "🌍 Global Feature Importance",
        "🔍 Individual Prediction",
        "📖 How SHAP Works",
    ])

    with tab1:
        st.markdown("### Global SHAP Feature Importance")
        st.markdown(f"Computing SHAP values on {len(X_sample)} test samples…")

        with st.spinner("Computing SHAP values (this may take ~30 seconds)…"):
            try:
                explainer = get_explainer(model, X_sample)
                shap_values = compute_shap_values(explainer, X_sample)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("#### Mean |SHAP| – Bar Chart")
                    fig_bar = plot_shap_bar(shap_values, feature_names, class_idx=1,
                                            title="Mean |SHAP| Feature Importance (Failure Class)")
                    st.pyplot(fig_bar)
                    plt.close()

                with c2:
                    st.markdown("#### SHAP Summary (Beeswarm) Plot")
                    try:
                        fig_sum = plot_shap_summary(shap_values, X_sample, feature_names, class_idx=1,
                                                     title="SHAP Summary – Feature Impact on Failure Prediction")
                        st.pyplot(fig_sum)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Beeswarm plot error: {e}")

            except Exception as exc:
                st.error(f"SHAP computation failed: {exc}")
                st.exception(exc)

    with tab2:
        st.markdown("### Individual Prediction Explanation")
        st.markdown("Choose a test sample index to explain the model's decision for that specific instance.")

        sample_idx = st.slider("Test Sample Index", min_value=0,
                                max_value=min(99, X_test.shape[0] - 1), value=0)
        X_instance = X_test[sample_idx:sample_idx + 1]
        actual_label = "Failure" if res["y_f_test"][sample_idx] == 1 else "No Failure"
        predicted_label = "Failure" if model.predict(X_instance)[0] == 1 else "No Failure"
        proba = model.predict_proba(X_instance)[0][1] if hasattr(model, "predict_proba") else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Label", actual_label)
        c2.metric("Predicted Label", predicted_label)
        if proba is not None:
            c3.metric("Failure Probability", f"{proba*100:.1f}%")

        with st.spinner("Computing SHAP for selected instance…"):
            try:
                explainer_single = get_explainer(model, X_sample)
                shap_values_single = compute_shap_values(explainer_single, X_instance)

                # Top contributing features table
                top_df = get_top_contributing_features(shap_values_single, feature_names, class_idx=1, top_n=8)
                st.markdown("#### Top Contributing Features")
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                # Waterfall plot
                try:
                    from src.explainability import plot_shap_waterfall
                    fig_wf = plot_shap_waterfall(explainer_single, X_instance, feature_names,
                                                  class_idx=1, title=f"SHAP Waterfall – Sample {sample_idx}")
                    st.pyplot(fig_wf)
                    plt.close()
                except Exception as wf_exc:
                    st.warning(f"Waterfall plot unavailable: {wf_exc}")

                # Feature values for this sample
                st.markdown("#### Feature Values for Selected Sample")
                fv_df = pd.DataFrame({"Feature": feature_names, "Value": X_instance[0]})
                st.dataframe(fv_df.round(4), use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"SHAP explanation failed: {exc}")

    with tab3:
        st.markdown("### 📖 Understanding SHAP")
        st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a unified framework for interpreting machine learning model predictions.

#### Key Concepts

| Concept | Explanation |
|---|---|
| **SHAP Value** | How much each feature contributed to pushing the prediction above/below the baseline |
| **Base Value** | Average model output across all training samples |
| **Positive SHAP** | Feature pushes prediction toward **Failure** |
| **Negative SHAP** | Feature pushes prediction toward **No Failure** |
| **Mean \|SHAP\|** | Global feature importance across all samples |

#### Plots Explained

- **Bar Chart**: Global average absolute SHAP values – shows which features matter most overall.
- **Beeswarm Plot**: Each dot is one sample; colour shows feature value (red=high, blue=low).
  Shows both importance AND direction of impact.
- **Waterfall Plot**: Single-prediction explanation – shows the chain of feature contributions
  starting from the base value to the final prediction.

#### Why SHAP for Predictive Maintenance?
- Identifies **which sensor** triggered a failure alert.
- Helps engineers understand whether temperature, torque, or tool wear is the primary driver.
- Builds **trust** in model decisions before maintenance actions are taken.
- Enables **sensor calibration** decisions based on feature importance.
        """)


main()
