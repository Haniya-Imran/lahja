"""Build the served ensemble: fit the TF-IDF + logistic-regression model, export it to NumPy files,
choose the blend weight on VALIDATION, then score the exact served ensemble on TEST and update the model card.

Run (after nlp.training.export):  python -m nlp.training.export_ensemble
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data_preparation.io import load_split
from ..evaluation.metrics import evaluate, evaluate_by_group, expected_calibration_error
from ..inference.engine import SentimentEngine
from ..inference.ensemble import EnsembleEngine
from ..inference.tfidf_lr import TfidfLR
from .export import error_tags

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negative", "neutral", "positive"]
OUT = ROOT / "models" / "tfidf"
CARD = ROOT / "models" / "sentiment" / "model_card.json"
LOW = 0.55


def main():
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000, dtype=np.float64)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float64)
    pipe = Pipeline([("feats", FeatureUnion([("char", char), ("word", word)])),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))]).fit(tr["text"], tr["label"])
    assert list(pipe.classes_) == LABELS
    c, w = pipe.named_steps["feats"].transformer_list[0][1], pipe.named_steps["feats"].transformer_list[1][1]
    clf = pipe.named_steps["clf"]
    inv = lambda vec: [t for t, _ in sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])]  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps({"labels": LABELS, "char": inv(c), "word": inv(w)}, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "weights.npz", idf_char=c.idf_.astype(np.float64), idf_word=w.idf_.astype(np.float64),
                        coef=clf.coef_.astype(np.float64), intercept=clf.intercept_.astype(np.float64))

    # NumPy re-implementation must match scikit-learn.
    np_model = TfidfLR(OUT)
    sk_te = pipe.predict_proba(te["text"])
    np_te = np_model.predict_proba(te["text"].tolist())
    max_diff = float(np.abs(sk_te - np_te).max())
    assert max_diff < 1e-6, max_diff

    # Blend weight on validation only.
    yva = np.array([LABELS.index(l) for l in va["label"]])
    yte = np.array([LABELS.index(l) for l in te["label"]])
    tr_eng = SentimentEngine(ROOT / "models" / "sentiment", threads=2)
    pT_va = tr_eng.predict_proba(va["text"].tolist())[0]
    pB_va = np_model.predict_proba(va["text"].tolist())
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    scores = {float(g): f1_score(yva, (g * pT_va + (1 - g) * pB_va).argmax(1), average="macro") for g in grid}
    wt = max(scores, key=scores.get)
    # Re-calibrate the blended probabilities with one temperature, also fitted on validation.
    pv = wt * pT_va + (1 - wt) * pB_va
    def nll(T):
        q = np.exp(np.log(pv + 1e-12) / T)
        q /= q.sum(1, keepdims=True)
        return -np.log(q[np.arange(len(yva)), yva] + 1e-12).mean()
    temp = float(min(np.round(np.arange(0.3, 3.0001, 0.01), 2), key=nll))
    cfg = {"display_name": "LingoAI sentiment (fine-tuned Multilingual MiniLM + TF-IDF blend)",
           "weight_transformer": wt, "weight_tfidf": round(1 - wt, 2), "temperature": temp, "low_confidence": LOW,
           "chosen_on": "validation split (grid 0.00-1.00 step 0.05, macro-F1)",
           "val_macro_f1": {"transformer": round(scores[1.0], 4), "tfidf": round(scores[0.0], 4), "blend": round(scores[wt], 4)},
           "tfidf_model": "char 2-5 (char_wb, top 200k) + word 1-2 TF-IDF, logistic regression C=1, class-balanced",
           "numpy_vs_sklearn_max_prob_diff": max_diff}
    (OUT / "ensemble.json").write_text(json.dumps(cfg, indent=2))

    # Score the exact served ensemble on test.
    eng = EnsembleEngine(ROOT / "models" / "sentiment", OUT, threads=2)
    t0 = time.time()
    probs, _ = eng.predict_proba(te["text"].tolist())
    secs = time.time() - t0
    te = te.copy()
    te["pred"] = [LABELS[i] for i in probs.argmax(1)]
    te["conf"] = probs.max(1)
    card = json.loads(CARD.read_text())
    if "transformer_only_test" not in card:
        card["transformer_only_test"] = card["served_model_test"]
        card["transformer_only_by_language"] = card["test_by_language"]
    card["served_model_test"] = evaluate(te["label"], te["pred"])
    card["served_model_test"]["ece"] = expected_calibration_error(probs, yte)
    card["test_by_language"] = evaluate_by_group(te)
    card["throughput_reviews_per_sec_2cpu"] = round(len(te) / secs, 1)
    low = te["conf"] < LOW
    card["low_confidence"] = {"threshold": LOW, "share": round(float(low.mean()), 4),
                              "accuracy_low": round(float((te[low]["pred"] == te[low]["label"]).mean()), 4) if low.any() else None,
                              "accuracy_rest": round(float((te[~low]["pred"] == te[~low]["label"]).mean()), 4)}
    te["tags"] = [error_tags(t, n) for t, n in zip(te["raw_text"], te["n_words"])]
    cats = {}
    for tag in ("negation", "emoji", "very_short", "informal_spelling"):
        sub = te[te["tags"].map(lambda x, tag=tag: tag in x)]
        if len(sub):
            cats[tag] = {"n": int(len(sub)), "accuracy": round(float((sub["pred"] == sub["label"]).mean()), 4)}
    wrong = te[te["pred"] != te["label"]]
    conf = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False)
    card["error_analysis"].update({
        "by_category": cats,
        "top_confusions": [{"true": t, "predicted": p, "count": int(n)} for (t, p), n in conf.head(6).items()],
        "examples": [{"text": r.raw_text[:220], "language": r.language, "true": r.label, "predicted": r.pred,
                      "confidence": round(float(r.conf), 3)} for r in wrong.sample(min(12, len(wrong)), random_state=7).itertuples()],
    })
    card["ensemble"] = cfg
    CARD.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    (ROOT / "reports" / "final_metrics.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in OUT.iterdir()}
    print(json.dumps({"weight": wt, "val": cfg["val_macro_f1"], "test": {k: card["served_model_test"][k] for k in ("accuracy", "macro_f1", "ece")},
                      "by_lang": {g: (m["n"], m["macro_f1"]) for g, m in card["test_by_language"].items()},
                      "low": card["low_confidence"], "max_diff": max_diff, "sizes_mb": sizes, "rev_per_s": card["throughput_reviews_per_sec_2cpu"]}, indent=1))


if __name__ == "__main__":
    main()
