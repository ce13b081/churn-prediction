"""End-to-end training orchestrator.

Run this script from the project root:
    python run_training.py

Steps:
1.  Download / load Telco churn dataset
2.  Engineer features (interaction, ratio, binary flags, categorical interaction)
3.  Train/test split
4.  Train LR + RF (RandomizedSearchCV, recall scoring) and
    XGBoost (Optuna, scale_pos_weight vs SMOTE)
5.  Evaluate all models on test set (recall, precision, F1, ROC-AUC, KS stat)
6.  Select best model by CV recall
7.  Find optimal decision threshold (target recall ≥ 0.80)
8.  Compute decile / lift analysis
9.  Generate evaluation plots
10. Save model, metrics, and artefacts
"""
import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.evaluate import (
    compute_decile_analysis,
    evaluate_all_models,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_lift_curve,
    plot_roc_curve,
)
from src.features import engineer_features, save_feature_thresholds
from src.preprocessing import get_feature_columns, get_feature_names_out
from src.train import save_model, select_best_model, train_all_models

MODELS_DIR = _ROOT / "models"
PLOTS_DIR = _ROOT / "plots"
REPORTS_DIR = _ROOT / "reports"


def main():
    start = time.time()

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STEP 1: Loading data")
    print("=" * 65)
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"Churn rate: {churn_rate:.1f}%")

    # ------------------------------------------------------------------
    # STEP 2: Feature engineering
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 2: Engineering features")
    print("=" * 65)
    df_eng, thresholds = engineer_features(df, thresholds=None)
    print(f"Monthly charges q75 threshold: {thresholds['monthly_charges_q75']:.2f}")
    print(f"New features: tenure_x_monthly, avg_monthly_spend, "
          f"high_value_customer, long_tenure, contract_x_internet")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_feature_thresholds(thresholds, MODELS_DIR / "feature_thresholds.json")

    # ------------------------------------------------------------------
    # STEP 3: Prepare X, y and split
    # ------------------------------------------------------------------
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"]
    X = df_eng[feature_cols]
    y = (df_eng["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Feature count: {len(feature_cols)} "
          f"({len(cols['numerical'])} numerical, {len(cols['categorical'])} categorical)")

    # ------------------------------------------------------------------
    # STEP 4: Train all models
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 4: Training models (primary metric: Recall)")
    print("=" * 65)
    results = train_all_models(X_train, y_train, cv=5, n_iter=20, n_optuna_trials=50)

    # ------------------------------------------------------------------
    # STEP 5: Evaluate all models on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 5: Evaluating models on test set")
    print("=" * 65)
    all_metrics = evaluate_all_models(results, X_test, y_test)

    # ------------------------------------------------------------------
    # STEP 6: Select best model (by CV recall = best_score)
    # ------------------------------------------------------------------
    best_name, best_pipeline = select_best_model(results)
    best_metrics = all_metrics[best_name]
    print(f"\nBest model: {best_name}")
    print(f"  CV Recall:    {results[best_name]['best_score']:.4f}")
    print(f"  Test Recall:  {best_metrics['recall']:.4f}")
    print(f"  Test F1:      {best_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"  Test KS Stat: {best_metrics['ks_statistic']:.4f}")

    # ------------------------------------------------------------------
    # STEP 7: Threshold optimisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 7: Finding optimal decision threshold")
    print("=" * 65)
    threshold_results = find_optimal_threshold(
        y_test, best_metrics["y_prob"], target_recall=0.80
    )
    opt_t = threshold_results["threshold_recall"]
    opt_m = threshold_results["metrics_at_recall"]
    def_m = threshold_results["metrics_at_default"]
    f1_m  = threshold_results["metrics_at_f1"]

    print(f"  Default (0.50):  Recall={def_m['recall']:.4f}  Precision={def_m['precision']:.4f}  F1={def_m['f1']:.4f}")
    print(f"  F1-optimal ({threshold_results['threshold_f1']:.3f}): Recall={f1_m['recall']:.4f}  Precision={f1_m['precision']:.4f}  F1={f1_m['f1']:.4f}")
    print(f"  Recall>=0.80 ({opt_t:.3f}): Recall={opt_m['recall']:.4f}  Precision={opt_m['precision']:.4f}  F1={opt_m['f1']:.4f}")

    # ------------------------------------------------------------------
    # STEP 8: Decile analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 8: Computing decile analysis")
    print("=" * 65)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    decile_df = compute_decile_analysis(
        y_test,
        best_metrics["y_prob"],
        save_path=REPORTS_DIR / "decile_analysis.csv",
    )
    print(decile_df.to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 9: Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 9: Generating evaluation plots")
    print("=" * 65)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_lift_curve(decile_df, save_path=PLOTS_DIR / "lift_curve.png")
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        labels=["No Churn", "Churn"],
        save_path=PLOTS_DIR / "confusion_matrix.png",
    )
    plot_roc_curve(y_test, best_metrics["y_prob"], save_path=PLOTS_DIR / "roc_curve.png")

    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_out(preprocessor)
    plot_feature_importance(
        best_pipeline,
        feature_names,
        save_path=PLOTS_DIR / "feature_importance.png",
        top_n=20,
    )

    # ------------------------------------------------------------------
    # STEP 10: Save model and all metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 10: Saving model and metrics")
    print("=" * 65)
    save_model(
        best_pipeline,
        MODELS_DIR / "best_model.joblib",
        model_name=best_name,
        cv_score=results[best_name]["best_score"],
        optimal_threshold=opt_t,
    )

    # Serialize metrics (strip numpy arrays)
    serializable_metrics = {}
    for name, m in all_metrics.items():
        entry = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "cv_score": m.get("cv_score", 0.0),
            "ks_statistic": m.get("ks_statistic", 0.0),
        }
        if name == best_name:
            entry["threshold_analysis"] = threshold_results
        serializable_metrics[name] = entry

    metrics_path = MODELS_DIR / "all_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_model": best_name, "models": serializable_metrics}, f, indent=2)
    print(f"All model metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"Total time: {elapsed:.1f}s\n")

    header = (f"{'Model':<25} {'Recall':>8} {'Precision':>10} "
              f"{'F1':>8} {'ROC-AUC':>10} {'KS Stat':>9} {'CV Recall':>10}")
    print(header)
    print("-" * len(header))
    for name, m in serializable_metrics.items():
        marker = " <-- BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['recall']:>8.4f} {m['precision']:>10.4f} "
            f"{m['f1']:>8.4f} {m['roc_auc']:>10.4f} "
            f"{m['ks_statistic']:>9.4f} {m['cv_score']:>10.4f}{marker}"
        )

    print(f"\nOptimal threshold (recall>=0.80): {opt_t:.4f}")
    print(f"  → Recall={opt_m['recall']:.4f}  Precision={opt_m['precision']:.4f}  F1={opt_m['f1']:.4f}")

    print("\nNext steps:")
    print("  pytest tests/ -v")
    print("  uvicorn api.main:app --reload --port 8000")
    print("  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
