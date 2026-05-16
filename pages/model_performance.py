"""Streamlit Model Performance page: compare all 3 trained models."""
import json
import pathlib
import sys

import pandas as pd
import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_METRICS_PATH  = _ROOT / "models" / "all_model_metrics.json"
_METADATA_PATH = _ROOT / "models" / "model_metadata.json"
_DECILE_PATH   = _ROOT / "reports" / "decile_analysis.csv"
_PLOTS = {
    "Confusion Matrix":  _ROOT / "plots" / "confusion_matrix.png",
    "ROC Curve":         _ROOT / "plots" / "roc_curve.png",
    "Feature Importance":_ROOT / "plots" / "feature_importance.png",
    "Lift Curve":        _ROOT / "plots" / "lift_curve.png",
}


def main():
    st.title("Model Performance Comparison")

    if not _METRICS_PATH.exists():
        st.warning(
            "No model metrics found. Run `python run_training.py` first.",
            icon="⚠️",
        )
        return

    with open(_METRICS_PATH) as f:
        data = json.load(f)

    best_model = data.get("best_model", "")
    models_data = data.get("models", {})
    best_cv = models_data.get(best_model, {}).get("cv_score", 0.0)

    # --- Best model banner ---
    st.success(
        f"Best model: **{best_model.replace('_', ' ').title()}** "
        f"(CV Recall: {best_cv:.4f})",
        icon="🏆",
    )

    # ---------------------------------------------------------------
    # Section 1 — Metrics comparison table
    # ---------------------------------------------------------------
    st.header("All Models — Test Set Metrics")
    rows = []
    for name, m in models_data.items():
        rows.append({
            "Model":          name.replace("_", " ").title(),
            "Recall":         f"{m['recall']:.4f}",
            "Precision":      f"{m['precision']:.4f}",
            "F1":             f"{m['f1']:.4f}",
            "ROC-AUC":        f"{m['roc_auc']:.4f}",
            "KS Stat":        f"{m.get('ks_statistic', 0.0):.4f}",
            "Accuracy":       f"{m['accuracy']:.4f}",
            "CV Score (Recall)": f"{m.get('cv_score', 0.0):.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------
    # Section 2 — Threshold Analysis
    # ---------------------------------------------------------------
    st.header("Threshold Analysis")
    threshold_analysis = models_data.get(best_model, {}).get("threshold_analysis")

    if threshold_analysis:
        def_t  = threshold_analysis.get("threshold_default", 0.5)
        f1_t   = threshold_analysis.get("threshold_f1", 0.5)
        opt_t  = threshold_analysis.get("threshold_recall", 0.5)
        def_m  = threshold_analysis.get("metrics_at_default", {})
        f1_m   = threshold_analysis.get("metrics_at_f1", {})
        opt_m  = threshold_analysis.get("metrics_at_recall", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"Default (t = {def_t:.2f})")
            st.metric("Recall",    f"{def_m.get('recall', 0):.4f}")
            st.metric("Precision", f"{def_m.get('precision', 0):.4f}")
            st.metric("F1",        f"{def_m.get('f1', 0):.4f}")

        with col2:
            st.subheader(f"F1-Optimal (t = {f1_t:.3f})")
            st.metric("Recall",    f"{f1_m.get('recall', 0):.4f}")
            st.metric("Precision", f"{f1_m.get('precision', 0):.4f}")
            st.metric("F1",        f"{f1_m.get('f1', 0):.4f}")

        with col3:
            st.subheader(f"Recall≥0.80 (t = {opt_t:.3f})")
            st.metric("Recall",    f"{opt_m.get('recall', 0):.4f}")
            st.metric("Precision", f"{opt_m.get('precision', 0):.4f}")
            st.metric("F1",        f"{opt_m.get('f1', 0):.4f}")

        st.info(
            f"The API and Live Predictor use the **Recall≥0.80 threshold "
            f"(t = {opt_t:.3f})** to reduce missed churners.",
            icon="ℹ️",
        )
    else:
        st.info("Threshold analysis not found — retrain to generate it.", icon="ℹ️")

    # ---------------------------------------------------------------
    # Section 3 — Decile Analysis
    # ---------------------------------------------------------------
    st.header("Decile Analysis")
    st.markdown(
        "Customers sorted by predicted churn probability (highest risk first), "
        "split into 10 equal groups. Shows how well the model concentrates "
        "actual churners in the top deciles."
    )
    if _DECILE_PATH.exists():
        decile_df = pd.read_csv(_DECILE_PATH)
        styled = decile_df.style.background_gradient(
            subset=["Churn_Rate_%", "Cumulative_Capture_Rate_%"],
            cmap="YlOrRd",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Decile analysis not found — run `python run_training.py`.", icon="ℹ️")

    # ---------------------------------------------------------------
    # Section 4 — Lift Curve
    # ---------------------------------------------------------------
    st.header("Cumulative Lift Curve")
    if _PLOTS["Lift Curve"].exists():
        st.image(str(_PLOTS["Lift Curve"]), use_container_width=True)
    else:
        st.info("Lift curve not found — run `python run_training.py`.", icon="ℹ️")

    # ---------------------------------------------------------------
    # Section 5 — Evaluation Plots
    # ---------------------------------------------------------------
    st.header("Best Model Evaluation Plots")
    col1, col2 = st.columns(2)
    with col1:
        if _PLOTS["Confusion Matrix"].exists():
            st.subheader("Confusion Matrix")
            st.image(str(_PLOTS["Confusion Matrix"]))
        else:
            st.info("Confusion matrix plot not found.")

    with col2:
        if _PLOTS["ROC Curve"].exists():
            st.subheader("ROC Curve")
            st.image(str(_PLOTS["ROC Curve"]))
        else:
            st.info("ROC curve plot not found.")

    if _PLOTS["Feature Importance"].exists():
        st.subheader("Feature Importance")
        st.image(str(_PLOTS["Feature Importance"]), use_container_width=True)
    else:
        st.info("Feature importance plot not found.")


main()
