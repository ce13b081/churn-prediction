"""Inference utilities shared by the API and Streamlit live predictor."""
import pathlib

import joblib
import pandas as pd

from src.preprocessing import get_feature_columns

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def get_model_path() -> pathlib.Path:
    """Resolve the saved model path relative to the project root."""
    return PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: pathlib.Path = None):
    """Load the trained pipeline from disk."""
    if model_path is None:
        model_path = get_model_path()
    return joblib.load(model_path)


def predict_single(pipeline, input_dict: dict) -> dict:
    """Run inference on a single customer record.

    Args:
        pipeline: Fitted sklearn Pipeline.
        input_dict: Flat dict of feature_name -> value.

    Returns:
        {churn: bool, probability: float, confidence: str}
    """
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"]

    # Build a single-row DataFrame in the exact training column order
    row = {col: [input_dict.get(col)] for col in feature_cols}
    df = pd.DataFrame(row)

    prob = float(pipeline.predict_proba(df)[0, 1])
    churn = prob >= 0.5

    if prob < 0.3:
        confidence = "Low"
    elif prob < 0.6:
        confidence = "Medium"
    else:
        confidence = "High"

    return {"churn": churn, "probability": prob, "confidence": confidence}


def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on a DataFrame of customer records.

    Returns the input DataFrame with 'churn_predicted' and
    'churn_probability' columns appended.
    """
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(bool)
    result = df.copy()
    result["churn_predicted"] = preds
    result["churn_probability"] = probs
    return result
