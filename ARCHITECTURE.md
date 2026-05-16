# Architecture & Design Decisions

## Overview

This document captures the full project plan, module responsibilities, and architectural decisions made when building the Customer Churn Prediction ML project.

**Phase 1** — Baseline pipeline: LR / RF / XGBoost with ROC-AUC scoring, FastAPI, Streamlit, pytest.  
**Phase 2** — Performance improvements: feature engineering, Optuna XGBoost tuning, Recall as primary metric, KS statistic, decile/lift analysis, threshold optimisation.

---

## Project Structure

```
churn-prediction/
├── data/                         # Telco CSV (auto-downloaded, gitignored)
├── models/                       # Saved model + metrics JSON (gitignored)
│   ├── best_model.joblib
│   ├── model_metadata.json
│   ├── all_model_metrics.json
│   └── feature_thresholds.json   # q75 threshold for inference-time feature engineering
├── plots/                        # PNG evaluation plots (gitignored)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── lift_curve.png            # NEW — cumulative lift vs random baseline
├── reports/                      # Analysis outputs (gitignored)
│   └── decile_analysis.csv       # NEW — 10-decile churn capture table
├── src/
│   ├── data_loader.py            # Dataset download & loading
│   ├── eda.py                    # Plotly EDA functions
│   ├── features.py               # NEW — feature engineering + threshold persistence
│   ├── preprocessing.py          # sklearn ColumnTransformer pipeline
│   ├── train.py                  # Model training + Optuna XGBoost + SMOTE
│   ├── evaluate.py               # Metrics, KS stat, decile analysis, plots
│   └── predict.py                # Inference utilities (engineers features at runtime)
├── api/
│   ├── main.py                   # FastAPI app (/health, /predict)
│   └── schemas.py                # Pydantic v2 request/response models
├── pages/
│   ├── home.py                   # Streamlit: project overview & stats
│   ├── eda.py                    # Streamlit: interactive EDA charts
│   ├── model_performance.py      # Streamlit: model comparison, threshold, decile, lift
│   └── live_predictor.py         # Streamlit: real-time prediction form
├── tests/
│   ├── test_pipeline.py          # pytest: data loader & pipeline tests
│   └── test_api.py               # pytest: FastAPI endpoint tests
├── app.py                        # Streamlit entry point (st.navigation)
├── run_training.py               # End-to-end training orchestrator (10 steps)
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

### `src/features.py` *(added Phase 2)*
- **Feature engineering** — works on a copy of the DataFrame; never mutates the caller's frame
- `engineer_features(df, thresholds=None) -> tuple[pd.DataFrame, dict]`:
  - **Training mode** (`thresholds=None`): computes `monthly_charges_q75 = df['MonthlyCharges'].quantile(0.75)`, returns it in the thresholds dict
  - **Inference mode** (`thresholds` provided): uses the stored q75 value for consistent binary flags
  - New **numerical** features: `tenure_x_monthly`, `avg_monthly_spend` (TotalCharges / (tenure+1)), `high_value_customer` (int flag), `long_tenure` (int flag)
  - New **categorical** feature: `contract_x_internet` (Contract + '_' + InternetService string concat)
- `save_feature_thresholds(thresholds, path)` → `models/feature_thresholds.json`
- `load_feature_thresholds(path) -> dict` → raises `FileNotFoundError` naturally if missing

### `src/preprocessing.py`
- **Single source of truth** for all feature column names via `get_feature_columns()`
- **8 numerical features**: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `tenure_x_monthly`, `avg_monthly_spend`, `high_value_customer`, `long_tenure` *(last 4 added Phase 2)*
- **16 categorical features**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `contract_x_internet` *(last one added Phase 2)*
- Numerical branch: `SimpleImputer(median)` → `StandardScaler`
- Categorical branch: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- `build_pipeline(classifier)` wraps any sklearn-compatible classifier into a full sklearn `Pipeline`

### `src/train.py` *(updated Phase 2)*
- Defines three model families: Logistic Regression, Random Forest, XGBoost
- **LR + RF**: `RandomizedSearchCV(n_iter=20, scoring='recall', StratifiedKFold(5))` — `class_weight=[None, 'balanced']` added to both param grids
- **XGBoost**: `tune_xgboost_optuna()` replaces RandomizedSearchCV:
  - Optuna study (`direction='maximize'`) over 8 hyperparameters including `scale_pos_weight` (range: 1.0 to 2× neg/pos ratio)
  - `n_jobs=1` inside objective to prevent nested-parallelism deadlocks
  - Compares scale_pos_weight variant vs SMOTE variant (each fit with best Optuna params); keeps whichever achieves higher CV recall
- `build_smote_pipeline(classifier) -> ImbPipeline`: uses `imblearn.pipeline.Pipeline` (skips SMOTE during `predict()`) — **never** use `sklearn.Pipeline` with SMOTE
- Primary scoring metric changed from `roc_auc` → **`recall`**
- `save_model(..., optimal_threshold)`: writes `model_metadata.json` with `cv_score` (recall) and `optimal_threshold`

### `src/evaluate.py` *(updated Phase 2)*
- Sets `matplotlib.use('Agg')` at module level for headless PNG generation
- `evaluate_model()`: now includes **KS Statistic** via `scipy.stats.ks_2samp(y_prob[y==1], y_prob[y==0])`
- `evaluate_all_models()`: renamed `cv_roc_auc` → `cv_score` to reflect recall scoring
- **`find_optimal_threshold(y_test, y_prob, target_recall=0.80) -> dict`** *(new)*:
  - Returns three threshold variants: default (0.5), F1-optimal, recall≥0.80
  - Uses `precision_recall_curve`; applies `[:-1]` slicing on precisions/recalls to align with the thresholds array (len(thresholds) == len(precisions)−1)
- **`compute_decile_analysis(y_test, y_prob, save_path) -> pd.DataFrame`** *(new)*:
  - Sorts customers by predicted probability descending; `pd.qcut(index, q=10)` into deciles
  - Columns: Decile, Total_Customers, Actual_Churners, Churn_Rate_%, Cumulative_Churners, Cumulative_Capture_Rate_%
  - Saves to `reports/decile_analysis.csv`
- **`plot_lift_curve(decile_df, save_path)`** *(new)*: model cumulative capture rate vs random baseline
- Existing plots retained: confusion matrix, ROC curve, feature importance (top 20)

### `src/eda.py`
- Pure functions returning Plotly `go.Figure` objects — no side effects, no file I/O
- Consumed exclusively by the Streamlit EDA page
- Separated from `evaluate.py` to keep interactive UI charts (Plotly) decoupled from static training report plots (matplotlib)

### `src/predict.py` *(updated Phase 2)*
- Shared inference layer used by both the FastAPI endpoint and the Streamlit live predictor
- `predict_single(pipeline, input_dict, threshold=None)`:
  1. Loads `models/feature_thresholds.json`; raises `RuntimeError` with actionable message if missing
  2. Builds a single-row DataFrame from **raw** columns only (excludes engineered cols)
  3. Calls `engineer_features(df, thresholds=thresholds)` to add the 5 derived features
  4. Reorders columns to match `get_feature_columns()` training order before calling `predict_proba`
  5. Resolves threshold: uses passed value → `model_metadata.json["optimal_threshold"]` → 0.5 fallback
- `predict_batch()`: batch inference on already-engineered DataFrames (unchanged)

### `api/schemas.py`
- Pydantic v2 models with `Literal` type constraints for all categorical features
- Enforces valid category values at the API boundary before any data reaches the model
- Uses `ConfigDict(json_schema_extra={"example": {...}})` to populate the FastAPI `/docs` UI with a realistic sample payload

### `api/main.py` *(updated Phase 2)*
- Uses the `@asynccontextmanager lifespan` pattern (FastAPI 0.93+) instead of the deprecated `@app.on_event`
- Model and metadata both loaded into `app.state` at startup
- `/predict` endpoint now reads `optimal_threshold` from `app.state.metadata` and passes it to `predict_single()` — ensures the API uses the recall-optimised threshold, not 0.5
- Returns `HTTP 503` if model missing, `HTTP 422` from Pydantic validation, `HTTP 500` for prediction errors

### `pages/model_performance.py` *(updated Phase 2)*
- **Section 1 — Metrics table**: added KS Stat column; renamed "CV ROC-AUC" → "CV Score (Recall)"
- **Section 2 — Threshold Analysis** *(new)*: 3-column layout comparing Default (0.5), F1-Optimal, and Recall≥0.80 thresholds with precision/recall/F1 at each
- **Section 3 — Decile Analysis** *(new)*: reads `reports/decile_analysis.csv`; renders with `background_gradient(cmap="YlOrRd")` on Churn_Rate_% and Cumulative_Capture_Rate_%
- **Section 4 — Lift Curve** *(new)*: displays `plots/lift_curve.png`
- **Section 5 — Evaluation Plots**: unchanged — confusion matrix, ROC curve, feature importance

### `pages/live_predictor.py`
- Calls the FastAPI endpoint via `requests.post("http://localhost:8000/predict")` rather than importing `src.predict` directly
- Rationale: validates the full API stack end-to-end; avoids loading the model into memory twice
- Wraps the HTTP call in `try/except` with a clear error message if the API is not running

### `run_training.py` *(rewritten Phase 2)*
- 10-step orchestration:
  1. `load_data()` — downloads/loads Telco CSV
  2. `engineer_features(df)` + `save_feature_thresholds()` — creates 5 new features, saves q75 threshold
  3. Build X / y from engineered df; `train_test_split(stratify=y)`
  4. `train_all_models()` — LR + RF via RandomizedSearchCV; XGBoost via Optuna (50 trials)
  5. `evaluate_all_models()` — test-set metrics including KS statistic
  6. `select_best_model()` — by CV recall score
  7. `find_optimal_threshold()` — target recall ≥ 0.80
  8. `compute_decile_analysis()` → `reports/decile_analysis.csv`
  9. Generate plots: lift curve, confusion matrix, ROC curve, feature importance
  10. `save_model()` + serialize `all_model_metrics.json` with threshold analysis for best model
- Final summary table prints: Recall, Precision, F1, ROC-AUC, KS Stat, CV Recall

---

## Key Architectural Decisions

### 1. `sys.path.insert` over editable install
**Choice:** Entry-point files (`run_training.py`, `api/main.py`, `app.py`) insert the project root into `sys.path` at startup.  
**Why:** Avoids requiring `pip install -e .` or a `setup.py`/`pyproject.toml`. The project is structured as an application, not a library.  
**Trade-off:** Slightly fragile if scripts are invoked from unexpected working directories. Mitigated by using `pathlib.Path(__file__).parent` for all path resolution.

### 2. Recall as primary CV metric *(changed in Phase 2)*
**Choice:** `scoring='recall'` in `RandomizedSearchCV` and Optuna objectives.  
**Why:** In churn, missing an actual churner (false negative) is far costlier than a false alarm. A model that maximises ROC-AUC can still produce low recall at the default 0.5 threshold. Optimising directly for recall during CV aligns training with the business objective.  
**Trade-off:** Optimising recall can hurt precision. Mitigated by reporting all metrics and offering three threshold options (default, F1-optimal, recall≥0.80) in the Streamlit UI.

### 3. Optuna for XGBoost, RandomizedSearchCV for LR/RF *(added Phase 2)*
**Choice:** `tune_xgboost_optuna()` uses Optuna with 50 trials; LR and RF keep `RandomizedSearchCV(n_iter=20)`.  
**Why:** Optuna's sequential model-based optimisation (TPE sampler) is more sample-efficient than random search for XGBoost's large, correlated hyperparameter space (8 params including `scale_pos_weight`). LR and RF have simpler spaces where random search with 20 iterations is sufficient.  
**Trade-off:** Optuna adds ~2–5 min training time. `n_jobs=1` inside the Optuna objective prevents nested-parallelism deadlocks — outer parallelism is handled by Optuna itself.

### 4. scale_pos_weight vs SMOTE comparison *(added Phase 2)*
**Choice:** Both variants are trained and evaluated; the one with higher CV recall is kept.  
**Why:** There is no universally superior approach for class imbalance. `scale_pos_weight` adjusts the loss function directly; SMOTE generates synthetic minority samples. Runtime comparison costs one extra fit per XGBoost run and ensures the best strategy for this dataset is always selected.  
**Critical detail:** SMOTE must use `imblearn.pipeline.Pipeline` (`ImbPipeline`), which skips the sampler during `predict()`. Using `sklearn.Pipeline` with SMOTE would apply oversampling at inference time, corrupting predictions.

### 5. Feature engineering at inference time via stored thresholds *(added Phase 2)*
**Choice:** `models/feature_thresholds.json` stores the training-time q75 of MonthlyCharges; `predict_single()` loads this file and passes thresholds into `engineer_features()`.  
**Why:** Derived binary flags (`high_value_customer`) depend on dataset-level statistics that are unknown at inference time. Using the training q75 ensures the feature means the same thing in production as it did during training.  
**Trade-off:** Adds a file dependency for inference. A clear `RuntimeError` with an actionable message is raised if the file is missing.

### 6. Threshold optimisation post-training *(added Phase 2)*
**Choice:** Three thresholds are computed and stored: default (0.5), F1-optimal, recall≥0.80. The API uses the recall≥0.80 threshold by default.  
**Why:** The classification threshold is a business decision, not a model parameter. Different teams (marketing, retention) may require different precision/recall trade-offs. Storing all three gives the dashboard flexibility without retraining.  
**Implementation note:** `precision_recall_curve` returns arrays of length `n_thresholds+1`; always use `precisions[:-1]`, `recalls[:-1]` when indexing alongside the `thresholds` array.

### 7. Matplotlib (static PNGs) for training reports, Plotly for Streamlit
**Choice:** `src/evaluate.py` saves PNG files; `src/eda.py` returns interactive Plotly figures.  
**Why:** Training runs headlessly (no display). `matplotlib.use('Agg')` enables PNG generation without a GUI. Streamlit pages benefit from interactive Plotly charts (zoom, hover, tooltips).  
**Trade-off:** Two charting libraries in the project. Kept cleanly separated by module boundary.

### 8. Target binarisation outside the pipeline
**Choice:** `y = (df['Churn'] == 'Yes').astype(int)` in `run_training.py`, not inside the sklearn Pipeline.  
**Why:** The prediction API receives feature vectors only — it never sees a target column. Putting target encoding inside the pipeline would require stripping it at inference time, adding unnecessary complexity.

### 9. Pydantic `Literal` type constraints for all categoricals
**Choice:** Every categorical field in `CustomerFeatures` uses `Literal[...]` with the exact valid values from the dataset.  
**Why:** Catches invalid inputs at the API boundary before they reach the model. Invalid values cause silent `handle_unknown='ignore'` zeroing in OHE — the model would silently produce a prediction based on missing information.  
**Trade-off:** Schema must be updated if the dataset's valid categories change.

### 10. Only the best model is saved; all metrics are retained
**Choice:** `joblib.dump` saves only `best_model.joblib`. All three models' test-set metrics are written to `all_model_metrics.json`.  
**Why:** Avoids storing 3 large model files. The metrics JSON gives the Streamlit Model Performance page full visibility into all three models without loading them.

### 11. StratifiedKFold for cross-validation
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
features.engineer_features(df)          ← Phase 2: 5 new features derived
    │  tenure_x_monthly, avg_monthly_spend
    │  high_value_customer, long_tenure
    │  contract_x_internet
    │  save models/feature_thresholds.json
    ▼
run_training.py
    │  train/test split (80/20, stratified)
    ▼
preprocessing.build_pipeline(classifier)
    │  numerical: impute → scale  (8 features)
    │  categorical: impute → OHE  (16 features)
    ▼
train.train_all_models()
    │  LR + RF: RandomizedSearchCV (recall scoring)
    │  XGBoost: Optuna 50 trials → scale_pos_weight vs SMOTE comparison
    ▼
evaluate.evaluate_all_models()          ← recall, precision, F1, ROC-AUC, KS stat
    │
evaluate.find_optimal_threshold()       ← default / F1-optimal / recall≥0.80
    │
evaluate.compute_decile_analysis()      ← reports/decile_analysis.csv
    │
    ├── plots/ (confusion_matrix, roc_curve, feature_importance, lift_curve)
    ├── models/best_model.joblib
    ├── models/model_metadata.json       ← optimal_threshold, cv_score (recall)
    ├── models/feature_thresholds.json
    └── models/all_model_metrics.json    ← all 3 models + threshold_analysis

At inference time:

HTTP POST /predict  (CustomerFeatures JSON — raw features only)
    │
    ▼
api/main.py         ← Pydantic validation; reads optimal_threshold from metadata
    │
    ▼
predict.predict_single()
    │  load feature_thresholds.json
    │  build raw-only DataFrame
    │  engineer_features(df, thresholds) → adds 5 derived features
    │  reorder to training column order
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
- `synthetic_df` fixture in `test_pipeline.py` provides all 8 numerical + 16 categorical columns including engineered features, so the preprocessor can fit without network access
- `contract_x_internet` in the fixture uses a realistic value (`"Month-to-month_Fiber optic"`) so OHE does not encounter an empty vocabulary

---

## Execution Order

```bash
# 1. Install dependencies (includes optuna, imbalanced-learn)
pip install -r requirements.txt

# 2. Train all models (~2–8 minutes with Optuna 50 trials)
python run_training.py

# 3. Run tests
pytest tests/ -v

# 4. Start API (Terminal 1)
python -m uvicorn api.main:app --reload --port 8000

# 5. Start Streamlit (Terminal 2)
python -m streamlit run app.py
```

---

## Results (Training Run — Phase 2, 2026-05-16)

| Model | Recall | Precision | F1 | ROC-AUC | KS Stat | CV Recall |
|-------|--------|-----------|----|---------|---------|-----------|
| Logistic Regression | 0.8153 | 0.5016 | 0.6217 | 0.8484 | 0.5165 | 0.7885 |
| Random Forest | 0.7834 | 0.5273 | 0.6303 | 0.8464 | 0.5355 | 0.7808 |
| **XGBoost** ✓ | **0.9118** | **0.4372** | **0.5910** | **0.8463** | **0.5460** | **0.8245** |

**Selected model:** XGBoost (highest CV recall — 0.8245)  
**Dataset:** 7,043 rows · 24 features (8 numerical, 16 categorical) · 26.5% churn rate  
**Train/test split:** 5,634 / 1,409 (stratified)  
**Optimal threshold (recall≥0.80):** 0.645  
**Training time:** ~160 seconds

### Threshold Analysis (XGBoost best model)

| Threshold | Recall | Precision | F1 |
|-----------|--------|-----------|----|
| Default (0.50) | 0.9118 | 0.4372 | 0.5910 |
| F1-Optimal | varies | varies | varies |
| Recall≥0.80 (0.645) | ≥0.80 | higher | varies |

> **Phase 1 baseline for comparison:** XGBoost CV ROC-AUC 0.8485, test Recall 0.5854, F1 0.5854 (4+15 features, ROC-AUC scoring).  
> Phase 2 recall improvement: **+32.6 pp** (0.5854 → 0.9118) at default threshold.
