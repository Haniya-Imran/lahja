"""Serving model: blend of the fine-tuned Transformer and the TF-IDF + logistic-regression model.

p = w * p_transformer + (1 - w) * p_tfidf, with w chosen on the validation split
(see nlp/training/export_ensemble.py). Same interface as SentimentEngine, so explanations
(occlusion) and the API work unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import SentimentEngine
from .tfidf_lr import TfidfLR


class EnsembleEngine:
    def __init__(self, transformer_dir: Path, tfidf_dir: Path, threads: int = 0):
        self.transformer = SentimentEngine(transformer_dir, threads=threads)
        self.tfidf = TfidfLR(tfidf_dir)
        self.cfg = json.loads((Path(tfidf_dir) / "ensemble.json").read_text(encoding="utf-8"))
        self.w = float(self.cfg["weight_transformer"])
        assert self.tfidf.labels == self.transformer.labels
        self._meta = {**self.transformer.metadata, "display_name": self.cfg["display_name"], "ensemble": self.cfg}

    @property
    def labels(self) -> list[str]:
        return self.transformer.labels

    @property
    def metadata(self) -> dict:
        return self._meta

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, list[int]]:
        pt, windows = self.transformer.predict_proba(texts, batch_size)
        p = self.w * pt + (1 - self.w) * self.tfidf.predict_proba(texts)
        q = np.exp(np.log(p + 1e-12) / float(self.cfg.get("temperature", 1.0)))
        return q / q.sum(axis=1, keepdims=True), windows

    def predict(self, texts: list[str]) -> list[dict]:
        probs, windows = self.predict_proba(texts)
        low = self.cfg["low_confidence"]
        res = []
        for p, win in zip(probs, windows):
            i = int(p.argmax())
            res.append({"sentiment": self.labels[i], "confidence": round(float(p[i]), 4),
                        "probabilities": {lab: round(float(v), 4) for lab, v in zip(self.labels, p)},
                        "low_confidence": float(p[i]) < low, "windows": win})
        return res
