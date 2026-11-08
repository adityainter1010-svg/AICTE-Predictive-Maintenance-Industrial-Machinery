"""
pages/3_Model_Evaluation.py
Comprehensive model evaluation and comparison page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        from src.train_model import load_failure_model, load_failure_type_model
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

        _, X_test, _, y_f_test, _, y_t_test = train_test_split(
            X, y_failure, y_type,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_failure,
        )

        preprocessor = load_preprocessor()
        X_test_proc = preprocessor.transform(X_test)
        feature_names = load_feature_names()

        return {
            "failure_model": load_failure_model(),
            "type_model": load_failure_type_model(),
            "X_test": X_test_proc,
            "y_f_test": y_f_test.values,
            "y_t_test": y_t_test,
            "le": le,
            "feature_names": feature_names,
            "loaded": True,
        }
    except FileNotFoundError as e:
        return {"loaded": False, "error": str(e)}
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def main():
    st.title("📈 Model Evaluation & Performance Comparison")

    res = load_resources()
    if not res.get("loaded"):
        st.error(f"**Resources not loaded.** {res.get('error', '')}")
        st.info("Train the models first: `python -m src.train_model`")
        return

    failure_model = res["failure_model"]
    type_model = res["type_model"]
    X_test = res["X_test"]
    y_f_test = res["y_f_test"]
    y_t_test = res["y_t_test"]
    le = res["le"]
    feature_names = res["feature_names"]

    from src.evaluation import (
        compute_binary_metrics, compute_multiclass_metrics,
        plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Machine Failure Model",
        "🟠 Failure Type Model",
        "🏆 Model Comparison",
        "📊 Feature Importance",
    ])

    # ── Tab 1: Binary Failure Model ────────────────────────────────────────────
    with tab1:
        st.markdown("### Binary Classification – Machine Failure (Normal vs Failure)")

        y_pred_f = failure_model.predict(X_test)
        y_proba_f = failure_model.predict_proba(X_test)[:, 1] if hasattr(failure_model, "predict_proba") else None

        m = compute_binary_metrics(y_f_test, y_pred_f, y_proba_f)

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
        c2.metric("Precision", f"{m['Precision']:.4f}")
        c3.metric("Recall", f"{m['Recall']:.4f}")
        c4.metric("F1 Score", f"{m['F1 Score']:.4f}")
        c5.metric("ROC-AUC", f"{m.get('ROC-AUC', 'N/A')}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_f_test, y_pred_f,
                                           class_names=["No Failure", "Failure"],
                                           title="Machine Failure – Confusion Matrix")
            st.pyplot(fig_cm)
            plt.close()

        with c2:
            if y_proba_f is not None:
                st.markdown("#### ROC Curve")
                fig_roc = plot_roc_curve(y_f_test, y_proba_f, title="ROC Curve – Machine Failure")
                st.pyplot(fig_roc)
                plt.close()

        st.markdown("#### Classification Report")
        st.code(m["Classification Report"])

    # ── Tab 2: Failure Type Model ───────────────────────────────────────────────
    with tab2:
        st.markdown("### Multi-Class Classification – Failure Type")

        class_names = le.classes_.tolist()
        y_pred_t = type_model.predict(X_test)

        m2 = compute_multiclass_metrics(y_t_test, y_pred_t, class_names)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m2['Accuracy']:.4f}")
        c2.metric("Precision (Macro)", f"{m2['Precision (Macro)']:.4f}")
        c3.metric("Recall (Macro)", f"{m2['Recall (Macro)']:.4f}")
        c4.metric("F1 Macro", f"{m2['F1 Score (Macro)']:.4f}")

        st.markdown("#### Confusion Matrix")
        readable_names = [config.FAILURE_TYPE_MAP.get(c, c) for c in class_names]
        fig_cm2 = plot_confusion_matrix(y_t_test, y_pred_t,
                                        class_names=readable_names,
                                        title="Failure Type – Confusion Matrix")
        st.pyplot(fig_cm2)
        plt.close()

        st.markdown("#### Classification Report")
        st.code(m2["Classification Report"])

    # ── Tab 3: Model Comparison (from saved summary) ───────────────────────────
    with tab3:
        st.markdown("### 🏆 Model Comparison (Cross-Validation Results)")

        summary_path = config.MODEL_DIR / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            for task, results in summary.items():
                label = "Machine Failure" if task == "failure" else "Failure Type"
                df_summary = pd.DataFrame([
                    {
                        "Model": k,
                        "CV F1 Mean": v["cv_f1_mean"],
                        "CV F1 Std": v["cv_f1_std"],
                        "CI Lower": max(0, v["cv_f1_mean"] - v["cv_f1_std"]),
                        "CI Upper": min(1, v["cv_f1_mean"] + v["cv_f1_std"]),
                    }
                    for k, v in results.items()
                ]).sort_values("CV F1 Mean", ascending=False)

                st.markdown(f"#### {label} – CV F1-Macro Comparison")
                st.dataframe(df_summary[["Model", "CV F1 Mean", "CV F1 Std"]].round(4),
                             use_container_width=True, hide_index=True)

                fig = go.Figure()
                colors = px.colors.qualitative.Set2
                for i, row in df_summary.iterrows():
                    fig.add_trace(go.Bar(
                        name=row["Model"],
                        x=[row["Model"]],
                        y=[row["CV F1 Mean"]],
                        error_y=dict(type="data", array=[row["CV F1 Std"]], visible=True),
                        marker_color=colors[list(df_summary.index).index(i) % len(colors)],
                        text=[f"{row['CV F1 Mean']:.4f}"],
                        textposition="outside",
                    ))
                fig.update_layout(
                    title=f"{label} – CV F1-Macro (± 1 Std Dev)",
                    showlegend=False, yaxis_range=[0, 1.15], height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training summary not available. Please run training first.")

        # Radar chart of best model metrics
        st.markdown("#### Best Model – Full Metrics Radar")
        radar_metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        y_pred_f = failure_model.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        radar_values = [
            accuracy_score(y_f_test, y_pred_f),
            precision_score(y_f_test, y_pred_f, zero_division=0),
            recall_score(y_f_test, y_pred_f, zero_division=0),
            f1_score(y_f_test, y_pred_f, zero_division=0),
        ]
        fig = go.Figure(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            fillcolor="rgba(59,130,212,0.2)",
            line=dict(color="#3b82d4", width=2),
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          height=400, title="Best Failure Model – Performance Radar")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Feature Importance ──────────────────────────────────────────────
    with tab4:
        st.markdown("### 📊 Feature Importance (Best Models)")

        c1, c2 = st.columns(2)
        with c1:
            try:
                from src.evaluation import plot_feature_importance
                fig_fi = plot_feature_importance(failure_model, feature_names, top_n=15)
                st.markdown("#### Machine Failure Model – Feature Importances")
                st.pyplot(fig_fi)
                plt.close()
            except Exception as e:
                st.info(f"Feature importance not available for this model type. ({e})")

        with c2:
            try:
                fig_fi2 = plot_feature_importance(type_model, feature_names, top_n=15)
                st.markdown("#### Failure Type Model – Feature Importances")
                st.pyplot(fig_fi2)
                plt.close()
            except Exception as e:
                st.info(f"Feature importance not available for this model type. ({e})")


main()
