"""Model evaluation: metrics, threshold optimisation, decile analysis, and plots."""
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Compute classification metrics for a fitted pipeline.

    Returns a dict with:
        accuracy, precision, recall, f1, roc_auc, ks_statistic,
        classification_report (str), confusion_matrix (ndarray),
        y_pred (ndarray), y_prob (ndarray)
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    ks_stat, _ = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0])

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "ks_statistic": float(ks_stat),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_all_models(results: dict, X_test, y_test) -> dict:
    """Evaluate every model in the training results dict.

    Returns a dict: {model_name: metrics_dict}
    """
    all_metrics = {}
    for name, res in results.items():
        print(f"Evaluating {name} on test set ...")
        metrics = evaluate_model(res["best_estimator"], X_test, y_test)
        metrics["cv_score"] = res["best_score"]  # CV Recall
        all_metrics[name] = metrics
        print(
            f"  Recall={metrics['recall']:.4f}  "
            f"Precision={metrics['precision']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  "
            f"KS={metrics['ks_statistic']:.4f}"
        )
    return all_metrics


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------


def _metrics_at_threshold(y_test, y_prob, threshold: float, ks_stat: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ks_statistic": ks_stat,
    }


def find_optimal_threshold(
    y_test,
    y_prob,
    target_recall: float = 0.80,
) -> dict:
    """Find classification thresholds optimised for recall and F1.

    Returns a dict with three threshold scenarios:
        threshold_default (0.5), threshold_f1 (max F1), threshold_recall (recall >= target).
    Each scenario includes precision, recall, f1, ks_statistic at that threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    # len(thresholds) == len(precisions) - 1; use [:-1] for alignment
    p, r, t = precisions[:-1], recalls[:-1], thresholds

    f1s = 2 * (p * r) / (p + r + 1e-8)
    threshold_f1 = float(t[np.argmax(f1s)])

    valid = np.where(r >= target_recall)[0]
    if len(valid) == 0:
        threshold_recall = float(t[0])  # lowest threshold → highest recall
    else:
        # Among candidates meeting recall target, pick highest (most precise)
        threshold_recall = float(t[valid[np.argmax(t[valid])]])

    # KS is a property of the score distribution, not the threshold
    ks_stat, _ = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0])

    return {
        "threshold_default": 0.5,
        "metrics_at_default": _metrics_at_threshold(y_test, y_prob, 0.5, float(ks_stat)),
        "threshold_f1": threshold_f1,
        "metrics_at_f1": _metrics_at_threshold(y_test, y_prob, threshold_f1, float(ks_stat)),
        "threshold_recall": threshold_recall,
        "metrics_at_recall": _metrics_at_threshold(y_test, y_prob, threshold_recall, float(ks_stat)),
    }


# ---------------------------------------------------------------------------
# Decile analysis
# ---------------------------------------------------------------------------


def compute_decile_analysis(
    y_test,
    y_prob,
    save_path: pathlib.Path = None,
) -> pd.DataFrame:
    """Sort customers by predicted churn probability and split into 10 deciles.

    Decile 1 = highest predicted risk, Decile 10 = lowest.
    Reports per-decile and cumulative churn capture rates.
    """
    if save_path is None:
        save_path = REPORTS_DIR / "decile_analysis.csv"
    save_path = pathlib.Path(save_path)

    df = pd.DataFrame({"y_test": np.array(y_test), "y_prob": np.array(y_prob)})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=10, labels=range(1, 11), duplicates="drop")

    total_churners = int(df["y_test"].sum())
    rows = []
    cumulative = 0
    for decile in range(1, 11):
        bucket = df[df["decile"] == decile]
        actual = int(bucket["y_test"].sum())
        cumulative += actual
        rows.append({
            "Decile": decile,
            "Total_Customers": len(bucket),
            "Actual_Churners": actual,
            "Churn_Rate_%": round(actual / len(bucket) * 100, 2) if len(bucket) > 0 else 0.0,
            "Cumulative_Churners": cumulative,
            "Cumulative_Capture_Rate_%": round(cumulative / total_churners * 100, 2) if total_churners > 0 else 0.0,
        })

    decile_df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    decile_df.to_csv(save_path, index=False)
    print(f"Decile analysis saved to {save_path}")
    return decile_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot and save a confusion matrix heatmap."""
    if save_path is None:
        save_path = PLOTS_DIR / "confusion_matrix.png"
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")
    return fig


def plot_roc_curve(
    y_test,
    y_prob,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot and save the ROC curve."""
    if save_path is None:
        save_path = PLOTS_DIR / "roc_curve.png"
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"ROC-AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {save_path}")
    return fig


def plot_feature_importance(
    pipeline,
    feature_names: list,
    save_path: pathlib.Path = None,
    top_n: int = 20,
) -> plt.Figure:
    """Plot and save a horizontal feature importance bar chart."""
    if save_path is None:
        save_path = PLOTS_DIR / "feature_importance.png"
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = clf.feature_importances_

    n = min(len(importances), len(feature_names))
    importances = importances[:n]
    names = feature_names[:n]

    indices = np.argsort(importances)[-top_n:]
    top_names = [names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(top_names, top_vals, color="#1976D2")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {save_path}")
    return fig


def plot_lift_curve(
    decile_df: pd.DataFrame,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot cumulative lift curve vs random baseline."""
    if save_path is None:
        save_path = PLOTS_DIR / "lift_curve.png"
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    deciles = decile_df["Decile"].tolist()
    model_capture = decile_df["Cumulative_Capture_Rate_%"].tolist()
    random_baseline = [d * 10 for d in deciles]  # linear: 10%, 20%, ..., 100%

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deciles, model_capture, marker="o", color="#1976D2", lw=2, label="Model")
    ax.plot(deciles, random_baseline, linestyle="--", color="gray", lw=1, label="Random Baseline")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Cumulative Capture Rate (%)")
    ax.set_title("Cumulative Lift Curve")
    ax.set_xticks(deciles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Lift curve saved to {save_path}")
    return fig
