"""API tests against the real served model (skipped if the exported model is missing)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "models" / "sentiment" / "model.onnx").exists():
    pytest.skip("exported model not present", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_test_metrics():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    card = r.json()["card"]
    assert 0 < card["served_model_test"]["macro_f1"] <= 1
    assert "roman_urdu" in card["test_by_language"]
    assert card["data"]["contamination_check"]["shared_keys_train_test"] == 0


@pytest.mark.parametrize("text,lang", [
    ("یہ پروڈکٹ بہت اچھی ہے، میں بہت خوش ہوں", "urdu"),
    ("yeh product bohat acha hai, bilkul paisa wasool", "roman_urdu"),
    ("Excellent quality, I love it and would buy again", "english"),
])
def test_analyze_languages(text, lang):
    r = client.post("/api/analyze", json={"text": text})
    assert r.status_code == 200
    j = r.json()
    assert j["language"]["language"] == lang
    assert j["sentiment"] in {"positive", "neutral", "negative"}
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-3
    assert j["explanation"]["available"] and j["explanation"]["tokens"]


def test_clear_cases_get_expected_sentiment():
    pos = client.post("/api/analyze", json={"text": "Excellent product, works perfectly, very happy", "explain": False}).json()
    neg = client.post("/api/analyze", json={"text": "Terrible, broke after one day, total waste of money", "explain": False}).json()
    assert pos["sentiment"] == "positive" and neg["sentiment"] == "negative"


def test_long_review_uses_windows():
    text = "the product is good and delivery was fine " * 60
    j = client.post("/api/analyze", json={"text": text, "explain": False}).json()
    assert j["windows"] > 1


def test_symbols_only_skipped():
    j = client.post("/api/analyze", json={"text": "!!! 123 ???"}).json()
    assert j["sentiment"] is None and j["skipped"]


@pytest.mark.parametrize("payload,status", [
    ({"text": ""}, 422), ({"text": "   "}, 422), ({"text": "a" * 5001}, 422), ({}, 422),
])
def test_analyze_invalid(payload, status):
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == status
    assert "error" in r.json()


def test_batch():
    r = client.post("/api/analyze/batch", json={"reviews": ["bohat acha", "bekaar", "!!!", "بہت برا تجربہ"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["index"] for x in res] == [0, 1, 2, 3]
    assert res[2]["sentiment"] is None


def test_batch_too_large():
    r = client.post("/api/analyze/batch", json={"reviews": ["ok"] * 65})
    assert r.status_code in (413, 422)


def test_inspect_suggests_review_column():
    csv = "﻿id;rating;comment\n1;5;product bohat acha hai aur delivery fast thi\n2;1;bilkul bekaar cheez, paise zaya\n".encode("utf-8")
    r = client.post("/api/reviews/inspect", files={"file": ("r.csv", csv, "text/csv")})
    assert r.status_code == 200
    j = r.json()
    assert j["suggested_column"] == "comment" and j["delimiter"] == ";" and j["rows"] == 2


def test_inspect_urdu_cp1256_free_utf8():
    csv = "review\nبہت اچھا\nبہت برا\n".encode("utf-8")
    j = client.post("/api/reviews/inspect", files={"file": ("u.csv", csv, "text/csv")}).json()
    assert j["suggested_column"] == "review"


@pytest.mark.parametrize("content,name", [(b"", "e.csv"), (b"\x89PNG\r\n\x1a\n\x00\x00", "x.png")])
def test_inspect_rejects_bad_files(content, name):
    r = client.post("/api/reviews/inspect", files={"file": (name, content, "application/octet-stream")})
    assert r.status_code in (400, 415, 422)


def _items():
    res = client.post("/api/analyze/batch", json={"reviews": ["bohat acha product", "bilkul bekaar", "theek hai",
                                                              "Great value", "بہت برا"]}).json()["results"]
    return [{"text": x["text"], "sentiment": x["sentiment"], "confidence": x["confidence"], "language": x["language"]["language"],
             "low_confidence": x["low_confidence"], "probabilities": x["probabilities"]} for x in res]


def test_analytics_summary():
    j = client.post("/api/analytics/summary", json={"items": _items()}).json()
    assert j["scored"] == 5 and sum(j["sentiment"].values()) == 5
    assert j["language"] and j["business_insights"]


def test_analytics_empty_rejected_and_skipped_only():
    assert client.post("/api/analytics/summary", json={"items": []}).status_code == 422
    j = client.post("/api/analytics/summary", json={"items": [{"text": "!!!", "sentiment": None, "confidence": None,
                    "language": "uncertain", "low_confidence": True, "probabilities": None}]}).json()
    assert j["scored"] == 0 and j["net_sentiment"] is None and j["skipped"] == 1


def test_export_csv_and_html():
    items = _items()
    r = client.post("/api/reports/export", json={"items": items, "format": "csv"})
    assert r.status_code == 200 and r.text.startswith("﻿review,language,sentiment")
    assert len(r.text.strip().splitlines()) == 6
    h = client.post("/api/reports/export", json={"items": items, "format": "html"})
    assert h.status_code == 200 and "<html" in h.text.lower() and "LingoAI" in h.text


def test_export_bad_format():
    assert client.post("/api/reports/export", json={"items": [], "format": "xlsx"}).status_code == 422


def test_served_model_is_blend():
    j = client.get("/api/model-info").json()
    ens = j["metadata"].get("ensemble")
    assert ens and 0 < ens["weight_transformer"] < 1
    assert j["card"]["served_model_test"]["macro_f1"] >= j["card"]["transformer_only_test"]["macro_f1"]


def test_examples():
    j = client.get("/api/examples").json()
    assert len(j["examples"]) >= 6


def test_unknown_route():
    assert client.get("/api/nope").status_code == 404
