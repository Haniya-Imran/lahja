"""Module B — language & script intelligence.

Two steps:
1. Script: share of Arabic-script vs Latin letters (exact, rule-based).
2. For Latin text, a character n-gram Naive Bayes model (trained on the Roman Urdu and
   English training sources) scores the whole text, and word lexicons built from the same
   corpora count clearly-English and clearly-Roman-Urdu words to spot code-switching.

The detector reports "uncertain" instead of guessing when the evidence is weak.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from ..preprocessing.normalize import is_arabic_char, normalize, words

LABELS = {
    "urdu": "Urdu (Urdu script)",
    "roman_urdu": "Roman Urdu",
    "english": "English",
    "mixed_roman_english": "Mixed Roman Urdu + English",
    "mixed_urdu_english": "Mixed Urdu script + English",
    "uncertain": "Uncertain",
}


@dataclass
class Detection:
    language: str
    label: str
    script: str
    confidence: float
    arabic_share: float
    roman_urdu_words: int
    english_words: int
    note: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _grams(text: str, n_max: int = 4) -> list[str]:
    t = f" {text} "
    return [t[i:i + n] for n in range(1, n_max + 1) for i in range(len(t) - n + 1)]


class LanguageDetector:
    def __init__(self, model: dict):
        self.logp = {c: model["logp"][c] for c in model["classes"]}
        self.unk = model["unk"]
        self.prior = model["prior"]
        self.en = set(model["english_words"])
        self.ru = set(model["roman_urdu_words"])

    # ---- training -----------------------------------------------------------
    @staticmethod
    def train(roman: list[str], english: list[str], dictionary_words: set[str], top_grams: int = 30000) -> dict:
        counts = {"roman_urdu": Counter(), "english": Counter()}
        for cls, texts in (("roman_urdu", roman), ("english", english)):
            for t in texts:
                counts[cls].update(_grams(normalize(t)))
        vocab = [g for g, _ in (counts["roman_urdu"] + counts["english"]).most_common(top_grams)]
        V = len(vocab)
        logp, unk = {}, {}
        for cls, c in counts.items():
            total = sum(c[g] for g in vocab) + V
            logp[cls] = {g: round(math.log((c[g] + 1) / total), 4) for g in vocab}
            unk[cls] = round(math.log(1 / total), 4)
        n = len(roman) + len(english)
        prior = {"roman_urdu": math.log(len(roman) / n), "english": math.log(len(english) / n)}
        # Word lexicons: a word is "clearly" one language if it is frequent there and rare in the other.
        wr, we = Counter(), Counter()
        for t in roman:
            wr.update(words(normalize(t)))
        for t in english:
            we.update(words(normalize(t)))
        tot_r, tot_e = sum(wr.values()), sum(we.values())
        en_words = sorted(w for w, c in we.items() if c >= 2 and len(w) > 1
                          and (c / tot_e) > 4 * (wr.get(w, 0) / tot_r))
        ru_words = sorted({w for w, c in wr.items() if c >= 3 and len(w) > 1
                           and (c / tot_r) > 4 * (we.get(w, 0) / tot_e)} | {w for w in dictionary_words if w not in we})
        return {"classes": ["roman_urdu", "english"], "logp": logp, "unk": unk, "prior": prior,
                "english_words": en_words, "roman_urdu_words": ru_words, "version": 1}

    # ---- inference ----------------------------------------------------------
    def _latin_scores(self, text: str) -> dict[str, float]:
        g = _grams(text)
        s = {}
        for cls in ("roman_urdu", "english"):
            lp, u = self.logp[cls], self.unk[cls]
            s[cls] = self.prior[cls] + sum(lp.get(x, u) for x in g)
        m = max(s.values())
        z = sum(math.exp(v - m) for v in s.values())
        return {k: math.exp(v - m) / z for k, v in s.items()}

    def detect(self, raw: str) -> Detection:
        text = normalize(raw)
        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) < 2:
            return Detection("uncertain", LABELS["uncertain"], "none", 0.0, 0.0, 0, 0,
                             "Too little text to identify a language.")
        ar = sum(is_arabic_char(ch) for ch in letters)
        share = ar / len(letters)
        ws = words(text)
        latin_words = [w for w in ws if all(not is_arabic_char(c) for c in w)]
        en = sum(1 for w in latin_words if w in self.en and w not in self.ru)
        ru = sum(1 for w in latin_words if w in self.ru and w not in self.en)

        if share >= 0.85:
            return Detection("urdu", LABELS["urdu"], "arabic", round(share, 3), round(share, 3), ru, en)
        if share >= 0.15:
            return Detection("mixed_urdu_english", LABELS["mixed_urdu_english"], "mixed", round(min(share, 1 - share) * 2, 3),
                             round(share, 3), ru, en, "Contains both Urdu script and Latin-script words.")
        p = self._latin_scores(" ".join(latin_words) or text)
        top = max(p, key=p.get)
        known = en + ru
        if known >= 4 and min(en, ru) >= 2 and min(en, ru) / known >= 0.25:
            return Detection("mixed_roman_english", LABELS["mixed_roman_english"], "latin", round(min(en, ru) / known * 2, 3),
                             round(share, 3), ru, en, "Clear English and Roman Urdu words in the same text.")
        conf = p[top]
        if conf < 0.75 and known < 2:
            return Detection("uncertain", LABELS["uncertain"], "latin", round(conf, 3), round(share, 3), ru, en,
                             "Latin script, but too few distinctive words to tell Roman Urdu from English.")
        return Detection(top, LABELS[top], "latin", round(conf, 3), round(share, 3), ru, en)


_default: LanguageDetector | None = None


def load(path: Path) -> LanguageDetector:
    global _default
    if _default is None:
        _default = LanguageDetector(json.loads(Path(path).read_text()))
    return _default
