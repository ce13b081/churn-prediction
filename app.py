"""Streamlit multi-page application entry point.

Run with:
    streamlit run app.py
"""
import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/eda.py", title="EDA", icon="📊"),
    st.Page("pages/model_performance.py", title="Model Performance", icon="📈"),
    st.Page("pages/live_predictor.py", title="Live Predictor", icon="🔮"),
])
pg.run()
