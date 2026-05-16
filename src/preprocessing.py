"""sklearn preprocessing pipeline for the Telco churn dataset."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns() -> dict:
    """Single source of truth for all feature column names.

    Returns a dict with keys: 'numerical', 'categorical', 'target'.
    """
    return {
        "numerical": [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            # Engineered numerical features
            "tenure_x_monthly", "avg_monthly_spend",
            "high_value_customer", "long_tenure",
        ],
        "categorical": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            # Engineered categorical feature
            "contract_x_internet",
        ],
        "target": "Churn",
    }


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for numerical and categorical features."""
    cols = get_feature_columns()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, cols["numerical"]),
            ("categorical", categorical_pipeline, cols["categorical"]),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """Wrap a classifier in a full preprocessing + classification pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Return OHE-expanded feature names after the preprocessor has been fit."""
    cols = get_feature_columns()
    numerical_names = cols["numerical"]
    categorical_names = list(
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(cols["categorical"])
    )
    return numerical_names + categorical_names
