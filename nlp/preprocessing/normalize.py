"""Module C — safe, language-aware text normalisation.

Rules are deliberately light: they remove noise that carries no meaning (URLs,
user handles, decorative diacritics, letters stretched for emphasis beyond two)
without translating or rewriting words. Urdu letters are never removed.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic-script code points used in Urdu text.
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

# Arabic forms commonly typed instead of the Urdu ones (keyboard / OCR variation).
_URDU_LETTER_MAP = str.maketrans({
    "ي": "ی",  # Arabic yeh  -> Farsi/Urdu yeh
    "ى": "ی",  # alef maksura -> Urdu yeh
    "ك": "ک",  # Arabic kaf  -> keheh
    "ۀ": "ۂ",  # heh with yeh above (decomposed form)
})
# Harakat (short-vowel marks) and tatweel: optional in Urdu writing, removed for consistency.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_MENTION = re.compile(r"(?<!\w)@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_ELONGATION = re.compile(r"(\w)\1{2,}")          # "acchaaaa" -> "acchaa"
_PUNCT_RUN = re.compile(r"([!?.،۔])\1{2,}")      # "!!!!!" -> "!!"
_SPACE = re.compile(r"\s+")


def is_arabic_char(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ARABIC_RANGES)


def normalize(text: str) -> str:
    """Normalise one review for the model and for analytics."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.translate(_URDU_LETTER_MAP)
    t = _DIACRITICS.sub("", t)
    t = _URL.sub(" URL ", t)
    t = _MENTION.sub(" @user ", t)
    t = _HASHTAG.sub(r"\1", t)
    # Latin letters are lower-cased (Roman Urdu has no stable capitalisation); Urdu has no case.
    t = "".join(ch.lower() if ch.isascii() else ch for ch in t)
    t = _ELONGATION.sub(r"\1\1", t)
    t = _PUNCT_RUN.sub(r"\1\1", t)
    return _SPACE.sub(" ", t).strip()


_WORD = re.compile(r"[\w؀-ۿ']+", re.UNICODE)


def words(text: str) -> list[str]:
    """Word tokens for analytics and language detection (not the model tokenizer)."""
    return [w for w in _WORD.findall(text) if not w.isdigit()]


def dedupe_key(text: str) -> str:
    """Key for near-duplicate detection: normalised, punctuation and spaces removed."""
    return re.sub(r"[\W_]+", "", normalize(text))
