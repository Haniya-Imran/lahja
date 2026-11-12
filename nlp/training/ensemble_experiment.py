"""Experiment: does blending the Transformer with the TF-IDF baseline help (especially Roman Urdu)?

Blend: p = w * p_transformer + (1 - w) * p_baseline, with w chosen on the VALIDATION split only
(one global w, and separately one w per script group). The test split is scored once at the end.
Writes reports/ensemble_experiment.json.
Run:  python -m nlp.training.ensemble_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data_preparation.io import load_split
from ..inference.engine import SentimentEngine

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negative", "neutral", "positive"]
URDU_SCRIPT = {"urdu", "mixed_urdu_english"}


def group(lang: str) -> str:
    return "urdu_script" if lang in URDU_SCRIPT else "latin_script"


def mf1(y, p):
    return f1_score(y, p.argmax(1), average="macro")


def main():
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    pipe = Pipeline([
        ("feats", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ]).fit(tr["text"], tr["label"])
    assert list(pipe.classes_) == LABELS
    eng = SentimentEngine(ROOT / "models" / "sentiment", threads=2)
    y = {}
    P = {}
    for name, df in (("val", va), ("test", te)):
        y[name] = np.array([LABELS.index(l) for l in df["label"]])
        P[name] = {"T": eng.predict_proba(df["text"].tolist())[0], "B": pipe.predict_proba(df["text"])}
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)

    # (a) one global weight
    wg = max(grid, key=lambda w: mf1(y["val"], w * P["val"]["T"] + (1 - w) * P["val"]["B"]))
    # (b) one weight per script group
    gv = np.array([group(l) for l in va["language"]])
    gt = np.array([group(l) for l in te["language"]])
    ws = {}
    for g in ("urdu_script", "latin_script"):
        m = gv == g
        ws[g] = float(max(grid, key=lambda w: mf1(y["val"][m], w * P["val"]["T"][m] + (1 - w) * P["val"]["B"][m])))

    def blend(split, groups, how):
        T, B = P[split]["T"], P[split]["B"]
        if how == "global":
            return wg * T + (1 - wg) * B
        w = np.array([ws[g] for g in groups])[:, None]
        return w * T + (1 - w) * B

    variants = {"transformer": lambda s, g: P[s]["T"], "baseline": lambda s, g: P[s]["B"],
                "blend_global": lambda s, g: blend(s, g, "global"), "blend_per_script": lambda s, g: blend(s, g, "script")}
    out = {"weights": {"global_w_transformer": float(wg), "per_script_w_transformer": ws}, "results": {}}
    for k, f in variants.items():
        pv, pt = f("val", gv), f("test", gt)
        langs = {}
        for lang, idx in te.groupby("language").groups.items():
            ii = te.index.get_indexer(idx)
            if len(ii) >= 30:
                langs[lang] = round(float(mf1(y["test"][ii], pt[ii])), 4)
        out["results"][k] = {"val_macro_f1": round(float(mf1(y["val"], pv)), 4),
                             "test_macro_f1": round(float(mf1(y["test"], pt)), 4),
                             "test_accuracy": round(float((pt.argmax(1) == y["test"]).mean()), 4), "test_by_language": langs}
    (ROOT / "reports" / "ensemble_experiment.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
