"""Model training: RandomizedSearchCV for LR/RF, Optuna for XGBoost."""
import json
import pathlib
from datetime import datetime

import joblib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.preprocessing import build_pipeline, build_preprocessor

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_smote_pipeline(classifier) -> ImbPipeline:
    """Wrap classifier in preprocessor → SMOTE → classifier pipeline.

    Uses imblearn.Pipeline so SMOTE is skipped during predict() calls.
    """
    return ImbPipeline([
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Param grids for LR and RF (XGBoost handled by Optuna)
# ---------------------------------------------------------------------------


def get_models() -> dict:
    """Return base classifier instances for LR and RF."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
    }


def get_param_grids() -> dict:
    """Return hyperparameter search spaces for LR and RF."""
    return {
        "logistic_regression": {
            "classifier__C": loguniform(1e-3, 10),
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": randint(2, 10),
            "classifier__min_samples_leaf": randint(1, 5),
            "classifier__class_weight": [None, "balanced"],
        },
    }


# ---------------------------------------------------------------------------
# Optuna XGBoost tuning
# ---------------------------------------------------------------------------


def tune_xgboost_optuna(
    X_train,
    y_train,
    n_trials: int = 50,
    cv: int = 5,
) -> dict:
    """Tune XGBoost with Optuna, comparing scale_pos_weight vs SMOTE variants.

    Returns the same result schema as train_all_models entries:
        {best_estimator, best_params, best_score, cv_results}
    where best_score is CV Recall of the winning variant.
    """
    neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, neg_pos_ratio * 2),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        }
        clf = XGBClassifier(
            random_state=42, verbosity=0, objective="binary:logistic", **params
        )
        pipeline = build_pipeline(clf)
        # n_jobs=1 inside objective to avoid nested parallelism deadlocks
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_strat, scoring="recall", n_jobs=1
        )
        return scores.mean()

    print("\nTuning XGBoost with Optuna ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"  Optuna best CV Recall: {study.best_value:.4f}")

    # --- Variant A: scale_pos_weight (best Optuna params as-is) ---
    clf_spw = XGBClassifier(
        random_state=42, verbosity=0, objective="binary:logistic", **best_params
    )
    pipeline_spw = build_pipeline(clf_spw)
    score_spw = cross_val_score(
        pipeline_spw, X_train, y_train, cv=cv_strat, scoring="recall", n_jobs=-1
    ).mean()
    pipeline_spw.fit(X_train, y_train)
    print(f"  scale_pos_weight variant CV Recall: {score_spw:.4f}")

    # --- Variant B: SMOTE (drop scale_pos_weight, let SMOTE balance classes) ---
    smote_params = {k: v for k, v in best_params.items() if k != "scale_pos_weight"}
    clf_smote = XGBClassifier(
        random_state=42, verbosity=0, objective="binary:logistic", **smote_params
    )
    pipeline_smote = build_smote_pipeline(clf_smote)
    score_smote = cross_val_score(
        pipeline_smote, X_train, y_train, cv=cv_strat, scoring="recall", n_jobs=-1
    ).mean()
    pipeline_smote.fit(X_train, y_train)
    print(f"  SMOTE variant CV Recall:             {score_smote:.4f}")

    # Keep whichever scores higher recall
    if score_spw >= score_smote:
        best_pipeline = pipeline_spw
        best_score = score_spw
        variant = "scale_pos_weight"
    else:
        best_pipeline = pipeline_smote
        best_score = score_smote
        variant = "SMOTE"
    print(f"  Selected variant: {variant}")

    return {
        "best_estimator": best_pipeline,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": {
            "mean_test_score": [best_score],
            "std_test_score": [0.0],
        },
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_all_models(
    X_train,
    y_train,
    cv: int = 5,
    n_iter: int = 20,
    n_optuna_trials: int = 50,
) -> dict:
    """Train all three models.

    LR and RF use RandomizedSearchCV(scoring='recall').
    XGBoost uses Optuna with scale_pos_weight vs SMOTE comparison.

    Returns:
        {model_name: {best_estimator, best_params, best_score, cv_results}}
    """
    models = get_models()
    param_grids = get_param_grids()
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    # --- LR and RF via RandomizedSearchCV ---
    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipeline = build_pipeline(clf)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring="recall",
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
        print(f"  Best CV Recall: {search.best_score_:.4f}")

    # --- XGBoost via Optuna ---
    results["xgboost"] = tune_xgboost_optuna(
        X_train, y_train, n_trials=n_optuna_trials, cv=cv
    )

    return results


def select_best_model(results: dict, metric: str = "best_score") -> tuple:
    """Return (name, pipeline) with the highest value for the given metric key."""
    best_name = max(results, key=lambda k: results[k][metric])
    return best_name, results[best_name]["best_estimator"]


def save_model(
    pipeline,
    path: pathlib.Path,
    model_name: str = "",
    cv_score: float = 0.0,
    optimal_threshold: float = 0.5,
):
    """Save the pipeline to disk with joblib and write metadata JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

    metadata = {
        "model_name": model_name,
        "cv_score": cv_score,           # CV Recall (primary metric)
        "optimal_threshold": optimal_threshold,
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
