"""Module E — word-level explanations by occlusion (leave-one-word-out).

For each word, the review is re-scored with that word removed. The change in the
predicted class probability is the word's contribution:
  positive value  -> removing the word lowers confidence, so it supported the prediction
  negative value  -> removing the word raises confidence, so it pushed against it

This is a real, model-based measurement (every number comes from re-running the
fine-tuned model), but it is approximate: words interact, and the scores show
association with the model's output, not proof of cause.
"""
from __future__ import annotations

import re

MAX_WORDS = 40
_SPLIT = re.compile(r"\S+")


def explain(engine, text: str, max_words: int = MAX_WORDS) -> dict:
    spans = [(m.start(), m.end()) for m in _SPLIT.finditer(text)]
    if not spans:
        return {"available": False, "reason": "No words to explain."}
    if len(spans) > max_words:
        return {"available": False,
                "reason": f"Explanations are limited to reviews of up to {max_words} words to keep them fast."}
    variants = [text[:s].rstrip() + " " + text[e:].lstrip() for s, e in spans]
    probs, _ = engine.predict_proba([text, *variants])
    base = probs[0]
    k = int(base.argmax())
    tokens = []
    for (s, e), p in zip(spans, probs[1:]):
        tokens.append({"word": text[s:e], "start": s, "end": e, "score": round(float(base[k] - p[k]), 4)})
    top = sorted(tokens, key=lambda t: -abs(t["score"]))[:6]
    return {
        "available": True,
        "method": "Leave-one-word-out occlusion on the fine-tuned model",
        "explains": engine.labels[k],
        "tokens": tokens,
        "top": top,
        "note": ("Each score is how much the confidence for the predicted sentiment drops when that word is removed. "
                 "It is an approximation: it shows what the model reacted to, not proof of why."),
    }
