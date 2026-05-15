"""End-to-end training orchestrator.

Run this script from the project root:
    python run_training.py

It will:
1. Download the Telco churn dataset (if not already cached)
2. Split into train/test sets
3. Train Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
4. Evaluate all models on the test set
5. Save evaluation plots (confusion matrix, ROC curve, feature importance)
6. Save the best model and all model metrics
"""
import json
import pathlib
import sys
import time

# Ensure src/ is importable from anywhere
_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.evaluate import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.preprocessing import get_feature_columns, get_feature_names_out
from src.train import save_model, select_best_model, train_all_models

MODELS_DIR = _ROOT / "models"
PLOTS_DIR = _ROOT / "plots"


def main():
    start = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"Churn rate: {churn_rate:.1f}%")

    # ------------------------------------------------------------------
    # 2. Prepare features and target
    # ------------------------------------------------------------------
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"]
    X = df[feature_cols]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Train all models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Training models with hyperparameter tuning")
    print("=" * 60)
    results = train_all_models(X_train, y_train, cv=5, n_iter=20)

    # ------------------------------------------------------------------
    # 4. Evaluate all models on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating models on test set")
    print("=" * 60)
    all_metrics = evaluate_all_models(results, X_test, y_test)

    # ------------------------------------------------------------------
    # 5. Select best model
    # ------------------------------------------------------------------
    best_name, best_pipeline = select_best_model(results)
    print(f"\nBest model: {best_name}")
    print(f"  CV ROC-AUC:   {results[best_name]['best_score']:.4f}")
    print(f"  Test ROC-AUC: {all_metrics[best_name]['roc_auc']:.4f}")
    print(f"  Test F1:      {all_metrics[best_name]['f1']:.4f}")
    print(f"  Test Accuracy:{all_metrics[best_name]['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 6. Generate evaluation plots for the best model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Generating evaluation plots")
    print("=" * 60)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    best_metrics = all_metrics[best_name]
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        labels=["No Churn", "Churn"],
        save_path=PLOTS_DIR / "confusion_matrix.png",
    )
    plot_roc_curve(
        y_test,
        best_metrics["y_prob"],
        save_path=PLOTS_DIR / "roc_curve.png",
    )

    # Extract feature names from the fitted preprocessor
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_out(preprocessor)
    plot_feature_importance(
        best_pipeline,
        feature_names,
        save_path=PLOTS_DIR / "feature_importance.png",
        top_n=20,
    )

    # ------------------------------------------------------------------
    # 7. Save best model and all metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Saving model and metrics")
    print("=" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_model(
        best_pipeline,
        MODELS_DIR / "best_model.joblib",
        model_name=best_name,
        cv_score=results[best_name]["best_score"],
    )

    # Serialize all_metrics (strip numpy arrays for JSON)
    serializable_metrics = {}
    for name, m in all_metrics.items():
        serializable_metrics[name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "cv_roc_auc": m.get("cv_roc_auc", 0.0),
        }

    metrics_path = MODELS_DIR / "all_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {"best_model": best_name, "models": serializable_metrics},
            f,
            indent=2,
        )
    print(f"All model metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s\n")
    print("Model comparison (test set):")
    header = f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10} {'CV ROC-AUC':>12}"
    print(header)
    print("-" * len(header))
    for name, m in serializable_metrics.items():
        marker = " <-- BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['f1']:>8.4f} "
            f"{m['roc_auc']:>10.4f} {m['cv_roc_auc']:>12.4f}{marker}"
        )

    print("\nNext steps:")
    print("  pytest tests/ -v")
    print("  uvicorn api.main:app --reload --port 8000")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
