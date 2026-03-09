"""
metrics.py — Đánh giá hiệu suất mô hình: AUROC, AUPR, F1, Precision, Recall
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    classification_report,
)
from pathlib import Path


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Tính toán tất cả metrics quan trọng.

    Args:
        y_true:    Nhãn thật (0/1)
        y_score:   Xác suất predicted (từ sigmoid)
        threshold: Ngưỡng phân loại binary

    Returns:
        Dict chứa tất cả metrics
    """
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auroc":     float(roc_auc_score(y_true, y_score)),
        "aupr":      float(average_precision_score(y_true, y_score)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }
    return metrics


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    out_path: Path,
):
    """Vẽ ROC và Precision-Recall curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Model Evaluation: {model_name}", fontsize=14, fontweight="bold")

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, lw=2.5, color="#e74c3c", label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=11)

    # Precision-Recall
    ax = axes[1]
    ax.plot(recall, precision, lw=2.5, color="#3498db", label=f"AUPR = {aupr:.4f}")
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="gray", alpha=0.7, label=f"Baseline = {baseline:.3f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#3498db")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
