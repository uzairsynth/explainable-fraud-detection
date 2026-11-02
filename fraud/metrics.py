"""Metrics that matter for an imbalanced fraud problem.

Accuracy is useless here (predicting "genuine" for everything scores 99.8%), so the
pipeline reports PR-AUC, precision/recall at a chosen threshold, and how much of the
fraud value the model actually stops.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

AMOUNT_BANDS = [(0, 10), (10, 100), (100, 500), (500, np.inf)]


def best_f1_threshold(y_true, scores) -> float:
    """Pick the score cut-off with the best F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, scores, threshold: float, amounts) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)
    pred = scores >= threshold

    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fraud_amount = amounts[y_true].sum()

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fraud_amount_caught_pct": float(amounts[y_true & pred].sum() / fraud_amount) if fraud_amount else 0.0,
        "alerts_per_10k": float(pred.mean() * 10_000),
    }


def slice_report(y_true, scores, threshold: float, amounts) -> list:
    """Recall and precision per transaction-amount band, to check the model is not blind to one band."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    amounts = np.asarray(amounts)
    rows = []
    for low, high in AMOUNT_BANDS:
        mask = (amounts >= low) & (amounts < high)
        frauds = int((y_true & mask).sum())
        alerts = int((pred & mask).sum())
        caught = int((y_true & pred & mask).sum())
        rows.append({
            "band": f"{low:g}-{high:g}" if np.isfinite(high) else f"{low:g}+",
            "transactions": int(mask.sum()),
            "frauds": frauds,
            "recall": caught / frauds if frauds else None,
            "precision": caught / alerts if alerts else None,
        })
    return rows
