"""Module F — language-aware word & phrase analytics over analysed reviews.

English-only stopword lists would wipe out meaning in Urdu and Roman Urdu, so
function words are learned from the training corpus instead: frequent words whose
use is spread evenly across the three sentiment classes (high frequency, low
sentiment association) in each language. Those are skipped when counting.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..preprocessing.normalize import normalize, words

LABELS = ("negative", "neutral", "positive")
# Negation changes meaning ("pasand nahi"), so it is never treated as a function word.
NEGATIONS = {"nahi", "nhi", "nahin", "nai", "na", "mat", "not", "no", "never", "nor", "don't", "didn't",
             "isn't", "wasn't", "نہیں", "نہ", "مت", "نہي"}


def build_function_words(df, per_group: int = 150) -> dict:
    """df needs columns text, label, language. Returns {language_group: [words]}."""
    groups = {"urdu": {"urdu", "mixed_urdu_english"}, "latin": {"roman_urdu", "english", "mixed_roman_english"}}
    out = {}
    for g, langs in groups.items():
        sub = df[df["language"].isin(langs)]
        by_label = {lab: Counter() for lab in LABELS}
        for t, lab in zip(sub["text"], sub["label"]):
            by_label[lab].update(set(words(t)))
        tot = Counter()
        for c in by_label.values():
            tot.update(c)
        n_lab = {lab: max(1, (sub["label"] == lab).sum()) for lab in LABELS}
        cand = []
        for w, n in tot.most_common(1500):
            rates = [by_label[lab][w] / n_lab[lab] for lab in LABELS]
            s = sum(rates)
            if s == 0:
                continue
            ps = [r / s for r in rates]
            entropy = -sum(p * math.log(p, 3) for p in ps if p > 0)  # 1.0 = perfectly even across classes
            if entropy >= 0.9 and w not in NEGATIONS:
                cand.append(w)
        out[g] = cand[:per_group]
    return out


class TextInsights:
    def __init__(self, function_words: dict):
        self.stop = set(function_words.get("urdu", [])) | set(function_words.get("latin", []))

    @classmethod
    def load(cls, path: Path) -> "TextInsights":
        return cls(json.loads(Path(path).read_text()))

    def _tokens(self, text: str) -> list[str]:
        return [w for w in words(normalize(text)) if w not in self.stop and len(w) > 1 and not w.isdigit()]

    def analyse(self, items: list[dict], top: int = 15) -> dict:
        """items: [{text, sentiment, language}]"""
        word_by = defaultdict(Counter)
        phrase_by = defaultdict(Counter)
        lengths = []
        for it in items:
            toks = self._tokens(it["text"])
            lengths.append(len(words(normalize(it["text"]))))
            s = it["sentiment"]
            word_by[s].update(set(toks))
            phrase_by[s].update({f"{a} {b}" for a, b in zip(toks, toks[1:])})

        def distinctive(counter_by, label):
            # Words frequent in this sentiment and relatively rare in the others (smoothed log-odds).
            this = counter_by[label]
            other = Counter()
            for lab, c in counter_by.items():
                if lab != label:
                    other.update(c)
            n_this = sum(this.values()) + 1
            n_other = sum(other.values()) + 1
            scored = []
            for w, c in this.items():
                if c < 2:
                    continue
                lo = math.log((c + 0.5) / n_this) - math.log((other.get(w, 0) + 0.5) / n_other)
                scored.append((w, c, lo))
            scored.sort(key=lambda x: (-x[2], -x[1]))
            return [{"term": w, "count": c, "score": round(lo, 3)} for w, c, lo in scored[:top]]

        result = {"words": {}, "phrases": {}}
        for lab in LABELS:
            result["words"][lab] = distinctive(word_by, lab) if word_by[lab] else []
            result["phrases"][lab] = distinctive(phrase_by, lab) if phrase_by[lab] else []
            result.setdefault("frequent", {})[lab] = [
                {"term": w, "count": c} for w, c in word_by[lab].most_common(top)]
        if lengths:
            s = sorted(lengths)
            result["length"] = {"mean": round(sum(s) / len(s), 1), "median": s[len(s) // 2], "min": s[0], "max": s[-1],
                                "buckets": _buckets(s)}
        result["note"] = ("Words are counted once per review. Function words were learned from the training data "
                          "(frequent and evenly spread across sentiments), not taken from an English stopword list.")
        return result


def _buckets(lengths: list[int]) -> list[dict]:
    edges = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10_000)]
    return [{"range": f"{a}-{b}" if b < 10_000 else f"{a}+", "count": sum(a <= x <= b for x in lengths)} for a, b in edges]
