"""Inference utilities shared by the API and Streamlit live predictor."""
import json
import pathlib

import joblib
import pandas as pd

from src.features import engineer_features, load_feature_thresholds
from src.preprocessing import get_feature_columns

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Raw input columns (no engineered cols — those are derived in predict_single)
_RAW_NUMERICAL = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def get_model_path() -> pathlib.Path:
    """Resolve the saved model path relative to the project root."""
    return PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: pathlib.Path = None):
    """Load the trained pipeline from disk."""
    if model_path is None:
        model_path = get_model_path()
    return joblib.load(model_path)


def predict_single(pipeline, input_dict: dict, threshold: float | None = None) -> dict:
    """Run inference on a single customer record.

    The API sends only raw features (no engineered columns). This function
    engineers features internally using stored training thresholds before
    calling the pipeline.

    Args:
        pipeline: Fitted sklearn/imblearn Pipeline.
        input_dict: Flat dict of raw feature_name -> value (from API schema).
        threshold: Decision threshold. If None, loads optimal_threshold from
                   model_metadata.json; defaults to 0.5 if not found.

    Returns:
        {churn: bool, probability: float, confidence: str}
    """
    # --- 1. Load feature engineering thresholds ---
    thresholds_path = PROJECT_ROOT / "models" / "feature_thresholds.json"
    try:
        thresholds = load_feature_thresholds(thresholds_path)
    except FileNotFoundError:
        raise RuntimeError(
            "feature_thresholds.json not found — run `python run_training.py` first."
        )

    # --- 2. Build single-row DataFrame from raw input columns only ---
    cols = get_feature_columns()
    raw_cat = [c for c in cols["categorical"] if c != "contract_x_internet"]
    row = {col: [input_dict.get(col)] for col in _RAW_NUMERICAL + raw_cat}
    df = pd.DataFrame(row)

    # --- 3. Engineer features using stored training thresholds ---
    df_eng, _ = engineer_features(df, thresholds=thresholds)

    # --- 4. Reorder columns to match training order ---
    all_feature_cols = cols["numerical"] + cols["categorical"]
    df_eng = df_eng[all_feature_cols]

    # --- 5. Predict probability ---
    prob = float(pipeline.predict_proba(df_eng)[0, 1])

    # --- 6. Resolve threshold ---
    if threshold is None:
        meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                threshold = json.load(f).get("optimal_threshold", 0.5)
        else:
            threshold = 0.5

    churn = prob >= threshold

    if prob < 0.3:
        confidence = "Low"
    elif prob < 0.6:
        confidence = "Medium"
    else:
        confidence = "High"

    return {"churn": churn, "probability": prob, "confidence": confidence}


def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on a DataFrame of customer records (already feature-engineered).

    Returns the input DataFrame with 'churn_predicted' and
    'churn_probability' columns appended.
    """
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(bool)
    result = df.copy()
    result["churn_predicted"] = preds
    result["churn_probability"] = probs
    return result
