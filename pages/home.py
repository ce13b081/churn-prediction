"""Streamlit Home page: project overview and dataset summary."""
import json
import pathlib
import sys

import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_data
from src.eda import get_class_distribution, get_summary_stats

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_data
def _load():
    return load_data()


def main():
    st.title("Customer Churn Prediction")
    st.markdown(
        """
        A production-ready machine learning application that predicts whether a telecom
        customer is likely to churn, using the **IBM Telco Customer Churn** dataset.
        """
    )

    # Dataset summary
    st.header("Dataset Overview")
    df = _load()
    dist = get_class_distribution(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Features", str(df.shape[1] - 1))
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Non-Churn Customers", f"{dist.get('No', 0):,}")

    st.subheader("Numeric Feature Summary")
    st.dataframe(get_summary_stats(df), use_container_width=True)

    # Model info (if trained)
    meta_path = _ROOT / "models" / "model_metadata.json"
    if meta_path.exists():
        st.header("Trained Model")
        with open(meta_path) as f:
            meta = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model", meta.get("model_name", "—").replace("_", " ").title())
        c2.metric("CV ROC-AUC", f"{meta.get('cv_roc_auc', 0):.4f}")
        c3.metric("Trained At", meta.get("saved_at", "—")[:19])
    else:
        st.info("Model not yet trained. Run `python run_training.py` to train.", icon="ℹ️")

    # Navigation guide
    st.header("Navigation Guide")
    st.markdown(
        """
        | Page | Description |
        |------|-------------|
        | **Home** | Project overview and dataset summary (this page) |
        | **EDA** | Interactive exploratory data analysis charts |
        | **Model Performance** | Comparison of all 3 models, ROC curves, confusion matrix |
        | **Live Predictor** | Fill in customer details and get an instant churn prediction |
        """
    )

    st.header("Tech Stack")
    st.markdown(
        "scikit-learn · XGBoost · FastAPI · Streamlit · Plotly · pandas · pytest"
    )


main()
