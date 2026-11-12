"""Unit tests for the NLP pipeline: normalisation, language detection, labels, data hygiene, analytics."""
import json
from pathlib import Path

import pytest

from nlp.analytics.text_insights import NEGATIONS, TextInsights
from nlp.language_detection.detect import load as load_detector
from nlp.preprocessing.normalize import dedupe_key, normalize, words

ROOT = Path(__file__).resolve().parents[2]


# ---------- normalisation ----------
def test_arabic_letter_forms_become_urdu():
    # Arabic yeh (U+064A) and kaf (U+0643) -> Urdu yeh (U+06CC) and keheh (U+06A9)
    assert normalize("يك") == "یک"


def test_diacritics_and_tatweel_removed():
    assert normalize("مـــزہَ") == "مزہ"


def test_urdu_text_kept_and_ascii_lowercased():
    out = normalize("Product بہت اچھا HAI")
    assert "بہت اچھا" in out and "product" in out and "hai" in out


def test_elongation_and_punctuation_runs():
    assert normalize("acchaaaaa!!!!!") == "acchaa!!"


def test_urls_and_mentions_replaced():
    out = normalize("see https://example.com/x @ali great")
    assert "url" in out.split() and "@user" in out and "example.com" not in out


def test_unicode_compatibility_forms():
    assert normalize("ｇｏｏｄ") == "good"  # full-width letters (NFKC)


def test_words_tokeniser_handles_both_scripts():
    assert words(normalize("bohat acha, بہت اچھا!")) == ["bohat", "acha", "بہت", "اچھا"]


def test_dedupe_key_ignores_case_and_punctuation():
    assert dedupe_key("Bohat ACHA!!!") == dedupe_key("bohat acha")


# ---------- language & script detection ----------
@pytest.fixture(scope="module")
def det():
    p = ROOT / "models" / "langid.json"
    if not p.exists():
        pytest.skip("language detector not built")
    return load_detector(p)


@pytest.mark.parametrize("text,lang", [
    ("یہ پروڈکٹ بہت اچھی ہے اور قیمت بھی مناسب ہے", "urdu"),
    ("yeh product bohat acha hai aur qeemat bhi theek hai", "roman_urdu"),
    ("This product is really good and the price is fair", "english"),
    ("یہ phone بہت اچھا ہے but battery weak ہے", "mixed_urdu_english"),
])
def test_detects_language(det, text, lang):
    assert det.detect(text).language == lang


def test_detects_code_switching(det):
    d = det.detect("product quality bohat achi hai but delivery was very late aur packing kharab thi")
    assert d.language == "mixed_roman_english"


def test_symbols_only_is_uncertain(det):
    assert det.detect("!!! ??? 123").language == "uncertain"


def test_detection_dict_has_fields(det):
    d = det.detect("bohat acha").as_dict()
    assert {"language", "label", "script", "confidence", "arabic_share"} <= set(d)


# ---------- labels & data hygiene ----------
def test_label_mapping_finegrained_drops_mixed():
    from nlp.data_preparation import build
    raw = ROOT / "data" / "raw" / "english_finegrained.txt"
    if not raw.exists():
        pytest.skip("raw data not downloaded")
    labels = {lab for _, lab, _ in build._finegrained()}
    assert labels <= {"positive", "negative", "neutral"}


def _splits():
    from nlp.data_preparation.io import load_split
    if not (ROOT / "data" / "processed" / "train.csv").exists():
        pytest.skip("processed data not built")
    return {s: load_split(s) for s in ("train", "val", "test")}


def test_no_train_test_contamination():
    sp = _splits()
    keys = {s: set(df["text"].map(dedupe_key)) for s, df in sp.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["val"]


def test_split_labels_valid_and_every_language_in_test():
    sp = _splits()
    for df in sp.values():
        assert set(df["label"]) <= {"positive", "neutral", "negative"}
        assert (df["text"].str.len() > 0).all()
    assert {"urdu", "roman_urdu", "english"} <= set(sp["test"]["language"])


def test_data_report_contamination_zero():
    p = ROOT / "data" / "processed" / "report.json"
    if not p.exists():
        pytest.skip("no report")
    r = json.loads(p.read_text())
    assert r["contamination_check"]["shared_keys_train_test"] == 0


# ---------- analytics ----------
def test_negations_never_function_words():
    p = ROOT / "models" / "function_words.json"
    if not p.exists():
        pytest.skip("function words not built")
    fw = json.loads(p.read_text())
    assert not (set(fw["urdu"]) | set(fw["latin"])) & NEGATIONS


def test_text_insights_distinctive_terms():
    ti = TextInsights({"urdu": [], "latin": ["hai"]})
    items = [{"text": "bohat zabardast product hai", "sentiment": "positive", "language": "roman_urdu"}] * 5 + \
            [{"text": "bilkul bekaar product hai", "sentiment": "negative", "language": "roman_urdu"}] * 5
    out = ti.analyse(items)
    pos = [w["term"] for w in out["words"]["positive"]]
    neg = [w["term"] for w in out["words"]["negative"]]
    assert "zabardast" in pos and "bekaar" in neg and "hai" not in pos
    assert out["length"]["median"] == 4


# ---------- NumPy TF-IDF must match scikit-learn ----------
@pytest.mark.parametrize("text", ["Bohat   acha product hai!!", "یہ فون بہت اچھا ہے", "a", "Great value, 10/10 would buy"])
def test_ngrams_match_sklearn(text):
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    from nlp.inference.tfidf_lr import char_wb_ngrams, word_ngrams
    c = sk.TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).build_analyzer()
    w = sk.TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").build_analyzer()
    assert char_wb_ngrams(text) == c(text)
    assert word_ngrams(text) == w(text)
