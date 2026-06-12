"""
Model evaluation — computes ROC, Precision-Recall, confusion matrix
from readings stored in SQLite.

Usage:
    python -m model.evaluate          # prints metrics
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score,
)

DB_PATH = os.getenv("SQLITE_PATH", "data/wildfire.db")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.80))
SAMPLE_LIMIT = 100_000   # cap for speed; 950k rows still gives stable curves


def load_data(limit: int = SAMPLE_LIMIT):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT fire_risk, is_fire_event FROM sensor_readings "
        "ORDER BY RANDOM() LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    y_score = np.array([r[0] for r in rows], dtype=float)
    y_true  = np.array([r[1] for r in rows], dtype=int)
    return y_score, y_true


def compute_all(limit: int = SAMPLE_LIMIT) -> dict:
    y_score, y_true = load_data(limit)

    # --- ROC ---
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    # --- Precision-Recall ---
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    avg_prec = float(average_precision_score(y_true, y_score))

    # --- At operating threshold ---
    y_pred = (y_score >= ALERT_THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision_at_thresh = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_at_thresh    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # --- Downsample curves for JSON transfer (keep ≤ 200 points) ---
    def downsample(x, y, n=200):
        if len(x) <= n:
            return list(x), list(y)
        idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
        return list(x[idx]), list(y[idx])

    roc_fpr, roc_tpr       = downsample(fpr, tpr)
    pr_rec,  pr_prec       = downsample(rec[::-1], prec[::-1])

    # Best threshold (closest to top-left of ROC)
    j = np.argmax(tpr - fpr)
    best_thresh = float(roc_thresh[j])

    return {
        "auc_roc":          roc_auc,
        "avg_precision":    avg_prec,
        "f1":               f1,
        "accuracy":         acc,
        "precision":        precision_at_thresh,
        "recall":           recall_at_thresh,
        "threshold":        ALERT_THRESHOLD,
        "best_threshold":   best_thresh,
        "n_samples":        len(y_true),
        "n_positive":       int(y_true.sum()),
        "n_negative":       int((1 - y_true).sum()),
        "confusion_matrix": cm,         # [[TN,FP],[FN,TP]]
        "roc_curve": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
        },
        "pr_curve": {
            "recall":    pr_rec,
            "precision": pr_prec,
        },
    }


if __name__ == "__main__":
    from loguru import logger
    logger.info("Running evaluation on SQLite data…")
    results = compute_all()
    print(f"\n{'='*40}")
    print(f"  AUC-ROC         : {results['auc_roc']:.6f}")
    print(f"  Avg Precision   : {results['avg_precision']:.6f}")
    print(f"  F1 Score        : {results['f1']:.4f}")
    print(f"  Accuracy        : {results['accuracy']:.4f}")
    print(f"  Precision @{results['threshold']} : {results['precision']:.4f}")
    print(f"  Recall    @{results['threshold']} : {results['recall']:.4f}")
    print(f"  Best Threshold  : {results['best_threshold']:.4f}")
    print(f"  Samples used    : {results['n_samples']:,}")
    print(f"  Positive (fire) : {results['n_positive']:,}")
    print(f"  Negative (safe) : {results['n_negative']:,}")
    cm = results['confusion_matrix']
    print(f"\n  Confusion Matrix (threshold={results['threshold']}):")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"{'='*40}\n")
