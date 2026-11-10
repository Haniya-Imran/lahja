"""Module E support — classification metrics, overall and per language group."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["negative", "neutral", "positive"]
MIN_GROUP = 30  # below this many test examples a group's metrics are reported as unreliable


def evaluate(y_true, y_pred, labels=LABELS) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def evaluate_by_group(df, pred_col: str = "pred", group_col: str = "language") -> dict:
    out = {}
    for g, sub in df.groupby(group_col):
        m = evaluate(sub["label"], sub[pred_col])
        m["reliable"] = len(sub) >= MIN_GROUP
        if not m["reliable"]:
            m["note"] = f"Only {len(sub)} test examples — too few for a reliable estimate."
        present = sorted(set(sub["label"]))
        m["labels_present"] = present
        out[g] = m
    return out


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return round(float(ece), 4)
