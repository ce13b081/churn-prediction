"""Streamlit EDA page: interactive exploratory data analysis."""
import pathlib
import sys

import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_data
from src.eda import (
    get_churn_rate_by_feature,
    plot_categorical_counts,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numerical_distributions,
)
from src.preprocessing import get_feature_columns


@st.cache_data
def _load():
    return load_data()


def main():
    st.title("Exploratory Data Analysis")
    df = _load()
    cols = get_feature_columns()

    # --- Churn distribution ---
    st.header("Churn Distribution")
    st.plotly_chart(plot_churn_distribution(df), use_container_width=True)

    # --- Numerical distributions ---
    st.header("Numerical Feature Distributions")
    st.plotly_chart(
        plot_numerical_distributions(df, ["tenure", "MonthlyCharges", "TotalCharges"]),
        use_container_width=True,
    )

    # --- Categorical feature drill-down ---
    st.header("Categorical Features by Churn")
    selected_col = st.sidebar.selectbox(
        "Select categorical feature",
        options=cols["categorical"],
        index=0,
    )
    st.subheader(f"{selected_col} vs Churn")
    st.plotly_chart(plot_categorical_counts(df, selected_col), use_container_width=True)

    # Churn rate table
    st.subheader(f"Churn Rate by {selected_col}")
    rate_df = get_churn_rate_by_feature(df, selected_col)
    rate_df["churn_rate"] = rate_df["churn_rate"].round(1).astype(str) + "%"
    st.dataframe(rate_df, use_container_width=True, hide_index=True)

    # --- Correlation heatmap ---
    st.header("Feature Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)


main()
