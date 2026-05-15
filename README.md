# Customer Churn Prediction

A production-ready end-to-end machine learning project for predicting telecom customer churn.

**Stack:** scikit-learn · XGBoost · FastAPI · Streamlit · Plotly · pandas · pytest

---

## Project Structure

```
churn-prediction/
├── data/                         # Telco CSV (auto-downloaded)
├── models/                       # Saved model + metrics JSON
├── plots/                        # PNG evaluation plots
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
│   ├── home.py                   # Streamlit: project overview
│   ├── eda.py                    # Streamlit: interactive EDA
│   ├── model_performance.py      # Streamlit: model comparison
│   └── live_predictor.py         # Streamlit: real-time prediction form
├── tests/
│   ├── test_pipeline.py          # pytest: pipeline & data loader tests
│   └── test_api.py               # pytest: FastAPI endpoint tests
├── app.py                        # Streamlit entry point
├── run_training.py               # End-to-end training orchestrator
└── requirements.txt
```

---

## Setup

```bash
cd "churn-prediction"
pip install -r requirements.txt
```

---

## Step 1 — Train Models

Downloads the dataset automatically, trains all 3 models with 5-fold cross-validation and hyperparameter tuning, saves the best model and evaluation plots.

```bash
python run_training.py
```

Expected output:
- `models/best_model.joblib` — trained sklearn Pipeline
- `models/model_metadata.json` — best model name + CV score
- `models/all_model_metrics.json` — metrics for all 3 models
- `plots/confusion_matrix.png`
- `plots/roc_curve.png`
- `plots/feature_importance.png`

> Training takes approximately 3–5 minutes with hyperparameter tuning.

---

## Step 2 — Run Tests

Requires the model to be trained first (`run_training.py`).

```bash
pytest tests/ -v
```

---

## Step 3 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### API Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "model_path": "...", "version": "1.0.0"}
```

#### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 2140.00
  }'
```

```json
{
  "churn_predicted": true,
  "churn_probability": 0.7823,
  "risk_level": "High",
  "model_name": "xgboost"
}
```

---

## Step 4 — Start the Streamlit App

Open a **second terminal** while the API is running:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 with 4 pages:

| Page | Description |
|------|-------------|
| **Home** | Project overview, dataset stats, model summary |
| **EDA** | Interactive charts — churn distribution, histograms, categorical breakdowns, correlation heatmap |
| **Model Performance** | Metrics table comparing all 3 models, ROC curve, confusion matrix, feature importance |
| **Live Predictor** | Fill in customer details and get a real-time prediction with probability and risk level |

---

## Model Details

### Models Trained

| Model | Tuned Hyperparameters |
|-------|----------------------|
| Logistic Regression | C, solver |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

### Training Strategy

- **Cross-validation:** Stratified 5-fold (preserves ~26% churn class imbalance)
- **Hyperparameter search:** `RandomizedSearchCV` with 20 iterations per model
- **Selection metric:** ROC-AUC (preferred over accuracy for imbalanced classification)
- **Best model saved:** The pipeline with the highest CV ROC-AUC is serialized with `joblib`

### Dataset

- **Source:** IBM Telco Customer Churn (7,043 rows, 19 features)
- **Target:** `Churn` (Yes/No → 1/0)
- **Preprocessing:** Median imputation for numerics, mode imputation + OneHotEncoding for categoricals, StandardScaler for numerics

---

## Evaluation Results

> Run `python run_training.py` to populate this section with actual results.

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
