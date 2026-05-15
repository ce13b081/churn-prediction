"""Plotly-based EDA functions for the Telco churn dataset.

All functions accept a cleaned DataFrame (from load_data()) and return
Plotly figures suitable for st.plotly_chart() in Streamlit.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts for the Churn column."""
    return df["Churn"].value_counts()


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.select_dtypes(include="number").describe().round(2)


def get_churn_rate_by_feature(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return churn rate (%) per category of a given column."""
    grouped = df.groupby(col)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    grouped.columns = [col, "churn_rate"]
    return grouped.sort_values("churn_rate", ascending=False)


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of churn vs. non-churn customers."""
    dist = get_class_distribution(df)
    fig = go.Figure(go.Pie(
        labels=dist.index.tolist(),
        values=dist.values.tolist(),
        hole=0.4,
        marker_colors=["#2196F3", "#F44336"],
    ))
    fig.update_layout(
        title="Customer Churn Distribution",
        legend_title="Churn",
    )
    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    cols: list = None,
) -> go.Figure:
    """Overlapping histograms per numeric column, split by churn status."""
    if cols is None:
        cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)

    colors = {"No": "#2196F3", "Yes": "#F44336"}
    for i, col in enumerate(cols, start=1):
        for churn_val, color in colors.items():
            subset = df[df["Churn"] == churn_val][col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=f"Churn={churn_val}",
                    marker_color=color,
                    opacity=0.7,
                    showlegend=(i == 1),
                    legendgroup=churn_val,
                ),
                row=1,
                col=i,
            )

    fig.update_layout(
        title="Numerical Feature Distributions by Churn",
        barmode="overlay",
        height=400,
    )
    return fig


def plot_categorical_counts(df: pd.DataFrame, col: str) -> go.Figure:
    """Grouped bar chart: category counts split by churn status."""
    counts = (
        df.groupby([col, "Churn"]).size().reset_index(name="count")
    )
    fig = px.bar(
        counts,
        x=col,
        y="count",
        color="Churn",
        barmode="group",
        color_discrete_map={"No": "#2196F3", "Yes": "#F44336"},
        title=f"{col} by Churn Status",
    )
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of Pearson correlations among numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=450,
    )
    return fig
