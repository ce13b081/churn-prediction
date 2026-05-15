"""Model evaluation metrics and matplotlib evaluation plots."""
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Compute classification metrics for a fitted pipeline.

    Returns a dict with:
        accuracy, precision, recall, f1, roc_auc,
        classification_report (str), confusion_matrix (ndarray),
        y_pred (ndarray), y_prob (ndarray)
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
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
        metrics["cv_roc_auc"] = res["best_score"]
        all_metrics[name] = metrics
        print(
            f"  Accuracy={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
    return all_metrics


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
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
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
    """Plot and save a horizontal feature importance bar chart.

    Works with LogisticRegression (|coef|), RandomForest, and XGBoost
    (both use feature_importances_).
    """
    if save_path is None:
        save_path = PLOTS_DIR / "feature_importance.png"
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = clf.feature_importances_

    # Align feature names with importance array length
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
