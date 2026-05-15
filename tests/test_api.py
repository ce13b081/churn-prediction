"""Tests for the FastAPI prediction endpoints."""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MODEL_PATH = _ROOT / "models" / "best_model.joblib"

VALID_PAYLOAD = {
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


@pytest.fixture(scope="module")
def client():
    """Create a TestClient; skip all tests if the model hasn't been trained yet."""
    if not _MODEL_PATH.exists():
        pytest.skip(
            "Model not found. Run `python run_training.py` before executing API tests."
        )

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict tests — valid payload
# ---------------------------------------------------------------------------


def test_predict_returns_200_with_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "churn_predicted" in data
    assert "churn_probability" in data
    assert "risk_level" in data


def test_predict_probability_in_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_level_valid(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    risk = response.json()["risk_level"]
    assert risk in ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# /predict tests — invalid payload
# ---------------------------------------------------------------------------


def test_predict_invalid_payload_returns_422(client):
    """Missing required fields should return 422 Unprocessable Entity."""
    response = client.post("/predict", json={"gender": "Male"})
    assert response.status_code == 422


def test_predict_invalid_contract_value_returns_422(client):
    """Supplying an invalid Literal value should return 422."""
    bad_payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
