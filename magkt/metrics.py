"""Evaluation metrics (micro-averaged over all valid next-step predictions).

  AUC   roc_auc_score(y, p)
  ACC   accuracy at threshold 0.5 (p > 0.5 -> correct)
  RMSE  sqrt(mean((p - y)^2))
  F1    binary F1 of the positive class (correct response) at threshold 0.5
  ECE   10 equal-width bins: sum_b n_b/N * |mean(y)_b - mean(p)_b|

Sanity check: for any binary classifier F1 <= 2*ACC/(1+ACC).
"""
import numpy as np
from sklearn import metrics as skm


def f1_upper_bound(acc):
    return 2.0 * acc / (1.0 + acc)


def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def compute_metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p > 0.5).astype(np.int64)
    acc = skm.accuracy_score(y, pred)
    out = {
        "auc": float(skm.roc_auc_score(y, p)),
        "acc": float(acc),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    if out["f1"] > f1_upper_bound(acc) + 1e-9:
        raise AssertionError("F1 exceeds 2*ACC/(1+ACC); metric computation is inconsistent")
    return out
