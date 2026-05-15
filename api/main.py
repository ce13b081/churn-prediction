"""FastAPI application for customer churn prediction."""
import json
import pathlib
import sys
from contextlib import asynccontextmanager

# Ensure project root is on the path so 'src' imports work
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.predict import load_model, predict_single

_MODEL_PATH = _PROJECT_ROOT / "models" / "best_model.joblib"
_METADATA_PATH = _PROJECT_ROOT / "models" / "model_metadata.json"
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release on shutdown."""
    if _MODEL_PATH.exists():
        app.state.model = load_model(_MODEL_PATH)
        if _METADATA_PATH.exists():
            with open(_METADATA_PATH) as f:
                app.state.metadata = json.load(f)
        else:
            app.state.metadata = {"model_name": "unknown"}
    else:
        app.state.model = None
        app.state.metadata = {}
    yield
    # No cleanup needed


app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability using a trained ML model.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Check API and model health."""
    return HealthResponse(
        status="ok",
        model_loaded=app.state.model is not None,
        model_path=str(_MODEL_PATH),
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run `python run_training.py` first to "
                "train and save the model."
            ),
        )

    try:
        input_dict = features.model_dump()
        result = predict_single(app.state.model, input_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    prob = result["probability"]
    if prob < 0.3:
        risk_level = "Low"
    elif prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return PredictionResponse(
        churn_predicted=result["churn"],
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        model_name=app.state.metadata.get("model_name", "unknown"),
    )
