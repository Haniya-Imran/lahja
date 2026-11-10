"""Loads the NLP components once per process and reuses them for every request."""
from __future__ import annotations

import json
import logging
import threading
import time

from nlp.analytics.text_insights import TextInsights
from nlp.inference.engine import SentimentEngine
from nlp.language_detection.detect import LanguageDetector

from ..config import settings
from ..errors import ApiError

log = logging.getLogger("reviews.registry")
_lock = threading.Lock()
_state: dict = {}


def _load():
    t = time.time()
    try:
        engine = SentimentEngine(settings.model_dir, threads=settings.onnx_threads)
    except FileNotFoundError as exc:
        raise ApiError(503, "model_missing", "The fine-tuned model files are not available on this server.") from exc
    det = LanguageDetector(json.loads(settings.langid_path.read_text(encoding="utf-8")))
    ins = TextInsights.load(settings.function_words_path)
    card = json.loads(settings.metrics_path.read_text(encoding="utf-8")) if settings.metrics_path.exists() else {}
    _state.update(engine=engine, detector=det, insights=ins, card=card, load_seconds=round(time.time() - t, 2))
    log.info("NLP components loaded in %.2fs", time.time() - t)


def get(name: str):
    if not _state:
        with _lock:
            if not _state:
                _load()
    return _state[name]


def loaded() -> bool:
    return bool(_state)
