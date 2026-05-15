"""Download and load the Telco Customer Churn dataset."""
import pathlib

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def download_data(force: bool = False) -> pathlib.Path:
    """Download the dataset if not already present."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH

    print(f"Downloading dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, stream=True, timeout=30)
    response.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {DATA_PATH}")
    return DATA_PATH


def load_data() -> pd.DataFrame:
    """Load the Telco churn dataset, downloading if necessary.

    Returns a DataFrame with:
    - customerID dropped
    - TotalCharges converted to float (NaN where blank)
    - Churn as 'Yes'/'No' strings
    """
    download_data()
    df = pd.read_csv(DATA_PATH)

    # Drop non-predictive ID column
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges is stored as string with spaces for missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df
