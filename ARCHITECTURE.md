# Architecture & Design Decisions

## Overview

This document captures the full project plan, module responsibilities, and architectural decisions made when building the Customer Churn Prediction ML project.

---

## Project Structure

```
churn-prediction/
├── data/                         # Telco CSV (auto-downloaded, gitignored)
├── models/                       # Saved model + metrics JSON (gitignored)
├── plots/                        # PNG evaluation plots (gitignored)
├── src/
│   ├── data_loader.py            # Dataset download & loading
│   ├── eda.py                    # Plotly EDA functions
│   ├── preprocessing.py          # sklearn ColumnTransformer pipeline
│   ├── train.py                  # Model training + hyperparameter tuning
│   ├── evaluate.py               # Metrics + matplotlib PNG plots
│   └── predict.py                # Inference utilities
├── api/
│   ├── main.py                   # FastAPI app (/health, /predict)
│   └── schemas.py                # Pydantic v2 request/response models
├── pages/
│   ├── home.py                   # Streamlit: project overview & stats
│   ├── eda.py                    # Streamlit: interactive EDA charts
│   ├── model_performance.py      # Streamlit: model comparison & plots
│   └── live_predictor.py         # Streamlit: real-time prediction form
├── tests/
│   ├── test_pipeline.py          # pytest: data loader & pipeline tests
│   └── test_api.py               # pytest: FastAPI endpoint tests
├── app.py                        # Streamlit entry point (st.navigation)
├── run_training.py               # End-to-end training orchestrator
├── requirements.txt
├── README.md
└── ARCHITECTURE.md               # This file
```

---

## Module Responsibilities

### `src/data_loader.py`
- Downloads the IBM Telco Customer Churn CSV from GitHub if not cached locally
- Drops the non-predictive `customerID` column
- Converts `TotalCharges` from string to float (`pd.to_numeric(errors='coerce')`) — the raw CSV stores blank values as spaces
- Returns a DataFrame with `Churn` as `"Yes"/"No"` strings; binarisation happens in `run_training.py`

### `src/preprocessing.py`
- **Single source of truth** for all feature column names via `get_feature_columns()`
- 4 numerical features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- 15 categorical features: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- Numerical branch: `SimpleImputer(median)` → `StandardScaler`
- Categorical branch: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- `build_pipeline(classifier)` wraps any sklearn-compatible classifier into a full `Pipeline`

### `src/train.py`
- Defines three base classifiers: Logistic Regression, Random Forest, XGBoost
- `RandomizedSearchCV` with `n_iter=20` and `StratifiedKFold(n_splits=5)` per model
- Primary scoring metric: `roc_auc`
- Saves best pipeline to `models/best_model.joblib` and writes `model_metadata.json`

### `src/evaluate.py`
- Sets `matplotlib.use('Agg')` at module level for headless PNG generation
- Generates three plots for the best model: confusion matrix, ROC curve, feature importance
- Feature importance source varies by model type:
  - Logistic Regression: `abs(coef_[0])`
  - Random Forest / XGBoost: `feature_importances_`
- `evaluate_all_models()` computes test-set metrics for all three models, returned as a dict for serialisation to `all_model_metrics.json`

### `src/eda.py`
- Pure functions returning Plotly `go.Figure` objects — no side effects, no file I/O
- Consumed exclusively by the Streamlit EDA page
- Separated from `evaluate.py` to keep interactive UI charts (Plotly) decoupled from static training report plots (matplotlib)

### `src/predict.py`
- Shared inference layer used by both the FastAPI endpoint and optionally Streamlit
- `predict_single()` reconstructs a single-row DataFrame in the exact column order defined by `get_feature_columns()` before calling `pipeline.predict_proba()`
- This column-order guarantee prevents silent feature misalignment at inference time

### `api/schemas.py`
- Pydantic v2 models with `Literal` type constraints for all categorical features
- Enforces valid category values at the API boundary before any data reaches the model
- Uses `ConfigDict(json_schema_extra={"example": {...}})` to populate the FastAPI `/docs` UI with a realistic sample payload

### `api/main.py`
- Uses the `@asynccontextmanager lifespan` pattern (FastAPI 0.93+) instead of the deprecated `@app.on_event`
- Model loaded once into `app.state.model` at startup; shared across all requests without reloading
- `CORSMiddleware(allow_origins=["*"])` allows the Streamlit frontend to call the API during local development
- Returns `HTTP 503` if the model file is missing, `HTTP 422` automatically from Pydantic validation failures, `HTTP 500` for unexpected prediction errors

### `pages/live_predictor.py`
- Calls the FastAPI endpoint via `requests.post("http://localhost:8000/predict")` rather than importing `src.predict` directly
- Rationale: validates the full API stack end-to-end; avoids loading the model into memory twice (once in API, once in Streamlit)
- Wraps the HTTP call in `try/except` with a clear error message guiding the user to start the API if it is not running

### `run_training.py`
- Entry point at the project root; uses `sys.path.insert(0, project_root)` so `from src.xxx import yyy` works without `pip install -e .`
- Binarises target here: `y = (df['Churn'] == 'Yes').astype(int)` — kept outside the pipeline because the API receives features only, not labels
- Serialises all model metrics to `models/all_model_metrics.json` for the Streamlit Model Performance page

---

## Key Architectural Decisions

### 1. `sys.path.insert` over editable install
**Choice:** Entry-point files (`run_training.py`, `api/main.py`, `app.py`) insert the project root into `sys.path` at startup.  
**Why:** Avoids requiring `pip install -e .` or a `setup.py`/`pyproject.toml`. The project is structured as an application, not a library.  
**Trade-off:** Slightly fragile if scripts are invoked from unexpected working directories. Mitigated by using `pathlib.Path(__file__).parent` for all path resolution.

### 2. ROC-AUC as the primary CV metric
**Choice:** `scoring='roc_auc'` in `RandomizedSearchCV`.  
**Why:** The Telco dataset has ~26% churn rate. Accuracy would be misleadingly high even for a model that always predicts "No Churn". ROC-AUC measures ranking quality across all decision thresholds and is robust to class imbalance.  
**Trade-off:** The deployed threshold is still 0.5 (probability ≥ 0.5 → churn). A business-optimised threshold (e.g. maximising F1 or minimising false negatives) could be tuned post-training.

### 3. RandomizedSearchCV over GridSearchCV
**Choice:** `RandomizedSearchCV(n_iter=20)` with continuous distributions (`loguniform`, `uniform`) for numeric hyperparameters.  
**Why:** Exhaustive grid search over the same space would require hundreds of fits per model. With 20 iterations × 5 folds × 3 models = 300 total fits, training completes in ~50 seconds.  
**Trade-off:** Does not guarantee finding the global optimum. Acceptable for a 7,043-row dataset where differences between models are small.

### 4. Matplotlib (static PNGs) for training reports, Plotly for Streamlit
**Choice:** `src/evaluate.py` saves PNG files; `src/eda.py` returns interactive Plotly figures.  
**Why:** Training runs headlessly (no display). `matplotlib.use('Agg')` enables PNG generation without a GUI. Streamlit pages benefit from interactive Plotly charts (zoom, hover, tooltips).  
**Trade-off:** Two charting libraries in the project. Kept cleanly separated by module boundary.

### 5. Target binarisation outside the pipeline
**Choice:** `y = (df['Churn'] == 'Yes').astype(int)` in `run_training.py`, not inside the sklearn Pipeline.  
**Why:** The prediction API receives feature vectors only — it never sees a target column. Putting target encoding inside the pipeline would require stripping it at inference time, adding unnecessary complexity.

### 6. Pydantic `Literal` type constraints for all categoricals
**Choice:** Every categorical field in `CustomerFeatures` uses `Literal[...]` with the exact valid values from the dataset.  
**Why:** Catches invalid inputs at the API boundary before they reach the model. Invalid values cause silent `handle_unknown='ignore'` zeroing in OHE — the model would silently produce a prediction based on missing information.  
**Trade-off:** Schema must be updated if the dataset's valid categories change.

### 7. Only the best model is saved; all metrics are retained
**Choice:** `joblib.dump` saves only `best_model.joblib`. All three models' test-set metrics are written to `all_model_metrics.json`.  
**Why:** Avoids storing 3 large model files. The metrics JSON gives the Streamlit Model Performance page full visibility into all three models without loading them.

### 8. StratifiedKFold for cross-validation
**Choice:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.  
**Why:** Preserves the ~26% churn class ratio in each fold, preventing folds where the minority class is underrepresented from producing misleading CV scores.

---

## Data Flow

```
Telco CSV (raw)
    │
    ▼
data_loader.load_data()
    │  drop customerID, coerce TotalCharges to float
    ▼
run_training.py
    │  train/test split (80/20, stratified)
    ▼
preprocessing.build_pipeline(classifier)
    │  numerical: impute → scale
    │  categorical: impute → OHE
    ▼
train.train_all_models()          ← RandomizedSearchCV × 3 models
    │
    ▼
evaluate.evaluate_all_models()    ← metrics on held-out test set
    │
    ├── plots/ (PNG files)
    ├── models/best_model.joblib
    ├── models/model_metadata.json
    └── models/all_model_metrics.json

At inference time:

HTTP POST /predict  (CustomerFeatures JSON)
    │
    ▼
api/main.py         ← Pydantic validation
    │
    ▼
predict.predict_single()
    │  reconstruct single-row DataFrame in training column order
    ▼
pipeline.predict_proba()
    │
    ▼
PredictionResponse  (churn_predicted, churn_probability, risk_level)
```

---

## Test Strategy

| Test file | Scope | Network required |
|-----------|-------|-----------------|
| `tests/test_pipeline.py` | Data loader, preprocessor, pipeline | Yes (downloads CSV once) |
| `tests/test_api.py` | FastAPI /health and /predict endpoints | No (uses TestClient) |

- API tests use `fastapi.testclient.TestClient` (synchronous) — no `pytest-asyncio` needed
- API tests skip automatically if `models/best_model.joblib` is missing (model must be trained first)
- Pipeline tests include a `synthetic_df` fixture for offline tests that exercise the preprocessor without network access

---

## Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads dataset, ~50 seconds)
python run_training.py

# 3. Run tests
pytest tests/ -v

# 4. Start API (Terminal 1)
python -m uvicorn api.main:app --reload --port 8000

# 5. Start Streamlit (Terminal 2)
python -m streamlit run app.py
```

---

## Results (Training Run — 2026-05-16)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|-------|----------|-----------|--------|----|---------|------------|
| Logistic Regression | 0.8048 | — | — | 0.6032 | 0.8412 | 0.8463 |
| Random Forest | 0.8013 | — | — | 0.5783 | 0.8422 | 0.8461 |
| **XGBoost** ✓ | **0.8070** | — | — | **0.5854** | **0.8481** | **0.8485** |

**Selected model:** XGBoost (highest CV and test ROC-AUC)  
**Dataset:** 7,043 rows · 19 features · 26.5% churn rate  
**Train/test split:** 5,634 / 1,409 (stratified)
