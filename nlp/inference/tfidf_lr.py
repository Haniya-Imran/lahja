"""TF-IDF + logistic regression scorer in pure NumPy (no scikit-learn needed in production).

It reproduces scikit-learn's TfidfVectorizer (char_wb 2-5 and word 1-2 n-grams, sublinear tf, l2 norm)
and LogisticRegression.predict_proba exactly; nlp/training/export_ensemble.py checks the match.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_WS = re.compile(r"\s\s+")
_TOKEN = re.compile(r"(?u)\b\w+\b")


def char_wb_ngrams(text: str, lo: int = 2, hi: int = 5) -> list[str]:
    text = _WS.sub(" ", text.lower())
    out = []
    for w in text.split():
        w = " " + w + " "
        L = len(w)
        for n in range(lo, hi + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < L:
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:
                break
    return out


def word_ngrams(text: str, lo: int = 1, hi: int = 2) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out = list(toks) if lo == 1 else []
    for n in range(max(lo, 2), min(hi, len(toks)) + 1):
        out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out


class TfidfLR:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.char_index = {t: i for i, t in enumerate(vocab["char"])}
        self.word_index = {t: i for i, t in enumerate(vocab["word"])}
        w = np.load(model_dir / "weights.npz")
        self.idf_char, self.idf_word = w["idf_char"], w["idf_word"]
        n_char = len(self.char_index)
        self.coef_char, self.coef_word = w["coef"][:, :n_char], w["coef"][:, n_char:]
        self.intercept = w["intercept"]
        self.labels = vocab["labels"]

    @staticmethod
    def _part(grams, index, idf, coef):
        c = Counter(g for g in grams if g in index)
        if not c:
            return np.zeros(coef.shape[0])
        ids = np.fromiter((index[g] for g in c), dtype=np.int64, count=len(c))
        vals = np.fromiter((1.0 + math.log(v) for v in c.values()), dtype=np.float64, count=len(c)) * idf[ids]
        vals /= np.sqrt((vals ** 2).sum())
        return coef[:, ids] @ vals

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), len(self.labels)))
        for k, t in enumerate(texts):
            z = (self.intercept + self._part(char_wb_ngrams(t), self.char_index, self.idf_char, self.coef_char)
                 + self._part(word_ngrams(t), self.word_index, self.idf_word, self.coef_word))
            z = z - z.max()
            e = np.exp(z)
            out[k] = e / e.sum()
        return out
