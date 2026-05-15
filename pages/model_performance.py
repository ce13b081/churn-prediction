"""Streamlit Model Performance page: compare all 3 trained models."""
import json
import pathlib
import sys

import pandas as pd
import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_METRICS_PATH = _ROOT / "models" / "all_model_metrics.json"
_PLOTS = {
    "Confusion Matrix": _ROOT / "plots" / "confusion_matrix.png",
    "ROC Curve": _ROOT / "plots" / "roc_curve.png",
    "Feature Importance": _ROOT / "plots" / "feature_importance.png",
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

    # --- Best model banner ---
    st.success(
        f"Best model: **{best_model.replace('_', ' ').title()}** "
        f"(CV ROC-AUC: {models_data[best_model]['cv_roc_auc']:.4f})",
        icon="🏆",
    )

    # --- Metrics comparison table ---
    st.header("All Models — Test Set Metrics")
    rows = []
    for name, m in models_data.items():
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "CV ROC-AUC": f"{m['cv_roc_auc']:.4f}",
        })

    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # --- Evaluation plots ---
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
