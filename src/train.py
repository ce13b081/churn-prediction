"""Model training with RandomizedSearchCV hyperparameter tuning."""
import json
import pathlib
from datetime import datetime

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.preprocessing import build_pipeline

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_models() -> dict:
    """Return a dict of base classifier instances."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": XGBClassifier(
            random_state=42,
            verbosity=0,
            objective="binary:logistic",
        ),
    }


def get_param_grids() -> dict:
    """Return hyperparameter search spaces for each model.

    All keys use the 'classifier__' prefix for pipeline compatibility.
    """
    return {
        "logistic_regression": {
            "classifier__C": loguniform(1e-3, 10),
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": randint(2, 10),
            "classifier__min_samples_leaf": randint(1, 5),
        },
        "xgboost": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": randint(3, 10),
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
    }


def train_all_models(
    X_train,
    y_train,
    cv: int = 5,
    n_iter: int = 20,
) -> dict:
    """Train all models with RandomizedSearchCV.

    Returns a dict:
        {model_name: {best_estimator, best_params, best_score, cv_results}}
    """
    models = get_models()
    param_grids = get_param_grids()
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipeline = build_pipeline(clf)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_strategy,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        results[name] = {
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": {
                "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
            },
        }
        print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")

    return results


def select_best_model(results: dict) -> tuple:
    """Return (name, pipeline) with the highest best_score."""
    best_name = max(results, key=lambda k: results[k]["best_score"])
    return best_name, results[best_name]["best_estimator"]


def save_model(pipeline, path: pathlib.Path, model_name: str = "", cv_score: float = 0.0):
    """Save the pipeline to disk with joblib and write metadata JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

    metadata = {
        "model_name": model_name,
        "cv_roc_auc": cv_score,
        "saved_at": datetime.now().isoformat(),
        "model_path": str(path),
    }
    meta_path = path.parent / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {path}")
    print(f"Metadata saved to {meta_path}")


def load_model(path: pathlib.Path):
    """Load a pipeline from disk."""
    return joblib.load(path)
