"""Pydantic v2 request/response models for the churn prediction API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer churn prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.10,
                "TotalCharges": 2140.00,
            }
        }
    )

    # Numerical features
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)

    # Binary categorical features
    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]

    # Multi-class categorical features
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    churn_predicted: bool
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    model_name: str


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    status: str
    model_loaded: bool
    model_path: str
    version: str
