"""Review pipeline: validate -> detect language/script -> normalise -> Transformer -> result."""
from __future__ import annotations

import time

from nlp.explainability.occlusion import explain as occlusion_explain
from nlp.preprocessing.normalize import normalize

from ..config import settings
from ..errors import ApiError
from . import registry


def _validate(texts: list[str]) -> list[str]:
    if not texts:
        raise ApiError(422, "empty_input", "Send at least one review.")
    if len(texts) > settings.max_batch:
        raise ApiError(413, "batch_too_large", f"Send at most {settings.max_batch} reviews per request.")
    clean = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise ApiError(422, "invalid_review", f"Review {i + 1} is not text.")
        if len(t) > settings.max_chars:
            raise ApiError(413, "review_too_long", f"Review {i + 1} is longer than {settings.max_chars:,} characters.")
        clean.append(t)
    return clean


def analyse(texts: list[str], explain: bool = False) -> dict:
    t0 = time.time()
    texts = _validate(texts)
    engine = registry.get("engine")
    det = registry.get("detector")
    results: list[dict | None] = [None] * len(texts)
    to_model, idx = [], []
    for i, raw in enumerate(texts):
        norm = normalize(raw)
        d = det.detect(raw)
        base = {"index": i, "text": raw, "normalized": norm, "language": d.as_dict()}
        if sum(ch.isalpha() for ch in norm) < 2:
            base.update(sentiment=None, confidence=None, probabilities=None, low_confidence=True,
                        skipped="No words to analyse (empty, numbers or symbols only).")
            results[i] = base
            continue
        results[i] = base
        to_model.append(norm)
        idx.append(i)
    if to_model:
        for i, p in zip(idx, engine.predict(to_model)):
            results[i].update(p)
    if explain and len(texts) == 1 and results[0].get("sentiment"):
        results[0]["explanation"] = occlusion_explain(engine, results[0]["normalized"])
    meta = engine.metadata
    return {"results": results, "model": {"name": meta.get("display_name"), "version": meta.get("version"),
                                           "labels": engine.labels},
            "elapsed_ms": round((time.time() - t0) * 1000)}
