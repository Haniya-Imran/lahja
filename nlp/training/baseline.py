"""Simple, honest baseline: character + word TF-IDF with logistic regression.

It sets the bar the fine-tuned Transformer has to beat. Tuned on the validation
split only; the test split is touched once at the end.
Run:  python -m nlp.training.baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data_preparation.io import load_split
from ..evaluation.metrics import evaluate, evaluate_by_group

ROOT = Path(__file__).resolve().parents[2]
DATA, REPORTS = ROOT / "data" / "processed", ROOT / "reports"


def run() -> dict:
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    best = None
    for C in (0.5, 1.0, 2.0, 4.0):
        pipe = Pipeline([
            ("feats", FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ])),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
        ])
        t = time.time()
        pipe.fit(tr["text"], tr["label"])
        f1 = evaluate(va["label"], pipe.predict(va["text"]))["macro_f1"]
        print(f"C={C}: val macro-F1 {f1:.4f} ({time.time() - t:.0f}s)")
        if best is None or f1 > best[0]:
            best = (f1, C, pipe)
    f1, C, pipe = best
    te = te.copy()
    te["pred"] = pipe.predict(te["text"])
    result = {"model": "TF-IDF (char 2-5 + word 1-2) + logistic regression", "C": C, "val_macro_f1": f1,
              "test": evaluate(te["label"], te["pred"]), "test_by_language": evaluate_by_group(te)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baseline_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({"test": {k: r["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                      "by_language": {g: (m["n"], m["macro_f1"]) for g, m in r["test_by_language"].items()}}, indent=1))
