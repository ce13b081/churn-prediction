"""Feature engineering for the Telco Customer Churn dataset.

Call engineer_features() after load_data() and before building X/y for training.
At inference time, pass the stored thresholds dict so binary flags are consistent
with what the model was trained on.
"""
import json
import pathlib

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def engineer_features(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Add engineered features to the DataFrame.

    Args:
        df: Raw DataFrame from load_data() (customerID already dropped).
        thresholds: If None (training mode), compute q75 of MonthlyCharges from df
                    and return it in the second element of the tuple.
                    If provided (inference mode), use stored values for consistency.

    Returns:
        (df_engineered, thresholds_dict)
        - df_engineered: Copy of df with 5 new columns added.
        - thresholds_dict: {"monthly_charges_q75": float}
    """
    df = df.copy()

    # --- Numerical interaction feature ---
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]

    # --- Ratio feature (avg monthly spend over customer lifetime) ---
    # +1 avoids division by zero for tenure=0; NaN in TotalCharges propagates
    # correctly and will be handled by the downstream median imputer.
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # --- Binary flag: high-value customer (top 25% by monthly charges) ---
    if thresholds is None:
        # Training mode: compute threshold from the passed data
        q75 = float(df["MonthlyCharges"].quantile(0.75))
        thresholds = {"monthly_charges_q75": q75}
    else:
        q75 = thresholds["monthly_charges_q75"]

    df["high_value_customer"] = (df["MonthlyCharges"] >= q75).astype(int)

    # --- Binary flag: long-tenure customer ---
    df["long_tenure"] = (df["tenure"] > 24).astype(int)

    # --- Categorical interaction: contract type × internet service ---
    df["contract_x_internet"] = (
        df["Contract"].astype(str) + "_" + df["InternetService"].astype(str)
    )

    return df, thresholds


def save_feature_thresholds(thresholds: dict, path: pathlib.Path) -> None:
    """Persist feature thresholds to a JSON file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Feature thresholds saved to {path}")


def load_feature_thresholds(path: pathlib.Path) -> dict:
    """Load feature thresholds from a JSON file.

    Raises FileNotFoundError if the file does not exist — the caller should
    surface a clear message asking the user to run run_training.py first.
    """
    with open(path) as f:
        return json.load(f)
