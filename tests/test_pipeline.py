"""Tests for the data loading and preprocessing pipeline."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_data
from src.preprocessing import build_pipeline, build_preprocessor, get_feature_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_df():
    """Load the actual dataset (downloads if needed)."""
    return load_data()


@pytest.fixture
def synthetic_df():
    """Minimal in-memory DataFrame for pipeline tests (no network required)."""
    cols = get_feature_columns()
    n = 20
    data = {}
    for col in cols["numerical"]:
        data[col] = np.random.rand(n).tolist()
    for col in cols["categorical"]:
        data[col] = ["Yes"] * n  # arbitrary but valid

    data["Churn"] = (["Yes"] * 10) + (["No"] * 10)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


def test_load_data_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) > 1000


def test_load_data_drops_customerid(raw_df):
    assert "customerID" not in raw_df.columns


def test_total_charges_is_numeric(raw_df):
    assert pd.api.types.is_float_dtype(raw_df["TotalCharges"]), (
        "TotalCharges should be float after load_data()"
    )


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def test_feature_columns_complete():
    cols = get_feature_columns()
    assert "numerical" in cols
    assert "categorical" in cols
    assert "target" in cols
    assert len(cols["numerical"]) > 0
    assert len(cols["categorical"]) > 0


def test_build_preprocessor_fits(synthetic_df):
    cols = get_feature_columns()
    X = synthetic_df[cols["numerical"] + cols["categorical"]]
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(X)
    assert result.ndim == 2
    assert result.shape[0] == len(X)


def test_build_pipeline_fits_predicts(synthetic_df):
    cols = get_feature_columns()
    X = synthetic_df[cols["numerical"] + cols["categorical"]]
    y = (synthetic_df["Churn"] == "Yes").astype(int)

    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_pipeline_handles_unknown_categories(synthetic_df):
    """OHE with handle_unknown='ignore' should not raise on unseen categories."""
    cols = get_feature_columns()
    X = synthetic_df[cols["numerical"] + cols["categorical"]]
    y = (synthetic_df["Churn"] == "Yes").astype(int)

    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X, y)

    # Introduce an unseen category value
    X_new = X.copy()
    X_new.iloc[0, len(cols["numerical"])] = "UnseenCategory"

    # Should not raise
    probs = pipeline.predict_proba(X_new)
    assert probs.shape[0] == len(X_new)
