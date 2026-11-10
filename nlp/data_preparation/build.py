"""Turn the raw sources into one clean, labelled, de-duplicated corpus with leak-free splits.

Run from the repo root:  python -m nlp.data_preparation.build
Outputs (data/processed/): train.csv, val.csv, test.csv, report.json and models/langid.json
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from ..language_detection.detect import LanguageDetector
from ..preprocessing.normalize import dedupe_key, normalize, words

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT, MODELS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "models"
LABELS = ["negative", "neutral", "positive"]
SEED = 42


# ---- loaders: each returns rows of (text, label, source) ----------------------------------
def _roman_urdu():
    df = pd.read_csv(RAW / "roman_urdu_uci.csv", header=None, usecols=[0, 1], names=["text", "label"],
                     dtype=str, encoding_errors="replace")
    m = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral", "Neative": "negative"}
    return [(t, m.get(str(l).strip()), "roman_urdu_uci") for t, l in df.itertuples(index=False)]


def _daraz():
    df = pd.read_csv(RAW / "daraz_reviews.csv", dtype=str)
    df = df[df["Spam(1) and Not Spam(0)"].astype(str).str.strip() == "0"]
    m = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}
    return [(t, m.get(str(l).strip()), "daraz_reviews") for t, l in zip(df["Reviews"], df["Sentiment"])]


def _urdu_products():
    df = pd.read_csv(RAW / "urdu_products.csv", dtype=str)
    m = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    return [(t, m.get(str(l).strip()), "urdu_products") for t, l in zip(df["sentence"], df["sentiment"])]


def _usc():
    df = pd.read_csv(RAW / "urdu_sentiment_corpus.tsv", sep="\t", dtype=str, quoting=csv.QUOTE_NONE)
    m = {"P": "positive", "N": "negative", "O": "neutral"}
    return [(t, m.get(str(l).strip()), "urdu_sentiment_corpus") for t, l in zip(df["Tweet"], df["Class"])]


def _finegrained():
    rows = []
    m = {"pos": "positive", "neg": "negative", "neu": "neutral"}  # 'mix' and 'nr' have no single sentiment
    for line in (RAW / "english_finegrained.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            lab, txt = line.split("\t", 1)
            if lab in m:
                rows.append((txt, m[lab], "english_finegrained"))
    return rows


def _uci(name):
    rows = []
    for line in (RAW / f"english_uci_{name}.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" in line:
            txt, lab = line.rsplit("\t", 1)
            rows.append((txt, {"1": "positive", "0": "negative"}.get(lab.strip()), f"english_uci_{name}"))
    return rows


LOADERS = [_roman_urdu, _daraz, _urdu_products, _usc, _finegrained] + [lambda n=n: _uci(n) for n in ("amazon", "yelp", "imdb")]
ENGLISH_SOURCES = {"english_finegrained", "english_uci_amazon", "english_uci_yelp", "english_uci_imdb"}


def _split_keys(groups: dict, strata: dict, fracs=(0.8, 0.1, 0.1)):
    """Stratified split over duplicate groups so no text can sit in two splits."""
    rng = random.Random(SEED)
    by_stratum = defaultdict(list)
    for k in groups:
        by_stratum[strata[k]].append(k)
    assign = {}
    for _, keys in sorted(by_stratum.items()):
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, round(n * fracs[2])) if n >= 10 else 0
        n_val = max(1, round(n * fracs[1])) if n >= 10 else 0
        for i, k in enumerate(keys):
            assign[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    report: dict = {"steps": []}
    rows = [r for loader in LOADERS for r in loader()]
    df = pd.DataFrame(rows, columns=["raw_text", "label", "source"])
    report["loaded_by_source"] = df["source"].value_counts().to_dict()
    report["steps"].append({"step": "Loaded", "rows": len(df)})

    n0 = len(df)
    df = df[df["label"].isin(LABELS)]
    report["steps"].append({"step": "Dropped rows with an unknown label", "rows": n0 - len(df)})
    df["text"] = df["raw_text"].map(normalize)
    n0 = len(df)
    df = df[df["text"].map(lambda t: sum(ch.isalpha() for ch in t)) >= 2]  # Python check: pyarrow regex \w is ASCII-only
    report["steps"].append({"step": "Dropped empty / too-short text (<2 letters)", "rows": n0 - len(df)})

    df["key"] = df["text"].map(dedupe_key)
    conflict = df.groupby("key")["label"].nunique()
    bad = set(conflict[conflict > 1].index)
    report["steps"].append({"step": "Dropped texts whose duplicates carry different labels",
                            "rows": int(df["key"].isin(bad).sum()), "groups": len(bad)})
    df = df[~df["key"].isin(bad)]
    n0 = len(df)
    df = df.drop_duplicates("key")
    report["steps"].append({"step": "Removed duplicates (same text after normalisation)", "rows": n0 - len(df)})

    # Split before training the language detector, so the detector never sees test text.
    df["src_lang"] = df["source"].map(lambda s: "english" if s in ENGLISH_SOURCES else
                                      "urdu" if s in ("urdu_products", "urdu_sentiment_corpus") else "roman_urdu")
    strata = dict(zip(df["key"], df["label"] + "|" + df["src_lang"]))
    assign = _split_keys(set(df["key"]), strata)
    df["split"] = df["key"].map(assign)

    tr = df[df["split"] == "train"]
    dict_words = set()
    for line in (RAW / "roman_urdu_dictionary.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            head = line.split(":", 1)[0].strip().lower()
            if head.isascii() and head.isalpha() and len(head) > 2:
                dict_words.add(head)
    lang_model = LanguageDetector.train(
        roman=tr[tr["source"] == "roman_urdu_uci"]["raw_text"].tolist(),
        english=tr[tr["src_lang"] == "english"]["raw_text"].tolist(),
        dictionary_words=dict_words,
    )
    (MODELS / "langid.json").write_text(json.dumps(lang_model, ensure_ascii=False))
    det = LanguageDetector(lang_model)
    d = [det.detect(t) for t in df["raw_text"]]
    df["language"] = [x.language for x in d]
    df["lang_conf"] = [x.confidence for x in d]
    df["n_words"] = df["text"].map(lambda t: len(words(t)))

    cols = ["text", "raw_text", "label", "source", "language", "split", "n_words"]
    for s in ("train", "val", "test"):
        df[df["split"] == s][cols].to_csv(OUT / f"{s}.csv", index=False)

    report["final_rows"] = int(len(df))
    report["by_split"] = df["split"].value_counts().to_dict()
    report["label_by_split"] = {s: df[df["split"] == s]["label"].value_counts().to_dict() for s in ("train", "val", "test")}
    report["by_source_label"] = {s: g["label"].value_counts().to_dict() for s, g in df.groupby("source")}
    report["detected_language"] = df["language"].value_counts().to_dict()
    report["language_by_source"] = {s: g["language"].value_counts().to_dict() for s, g in df.groupby("source")}
    report["test_language_label"] = {l: g["label"].value_counts().to_dict() for l, g in df[df["split"] == "test"].groupby("language")}
    report["words_per_text"] = df["n_words"].describe().round(1).to_dict()
    report["contamination_check"] = {
        "shared_keys_train_test": len(set(df[df.split == "train"]["key"]) & set(df[df.split == "test"]["key"])),
        "shared_keys_train_val": len(set(df[df.split == "train"]["key"]) & set(df[df.split == "val"]["key"])),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
