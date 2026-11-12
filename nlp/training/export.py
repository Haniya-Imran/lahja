"""Package the fine-tuned checkpoint for serving, and measure every step on held-out data.

Steps (each is checked so quality is never lost silently):
1. Temperature scaling fitted on the validation split (calibrates confidence; argmax unchanged).
2. Vocabulary trimming: keep only the sentence pieces seen in train/val text, every
   single-character piece (so unseen words still segment), and the 30k highest-prior pieces.
   The embedding table shrinks from ~250k rows; agreement with the untrimmed model is reported.
3. ONNX export + dynamic int8 quantisation for fast CPU inference without PyTorch.
4. Final test-set evaluation of the exact served model (overall and per language), error analysis.

Run:  python -m nlp.training.export --ckpt models/checkpoints/final
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negative", "neutral", "positive"]


def _logits(model, tok, texts, max_len, bs=64):
    import torch
    model.eval()
    out = np.zeros((len(texts), 3), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            b = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            b.pop("token_type_ids", None)
            out[idx] = model(**b).logits.float().numpy()
    return out


def fit_temperature(logits, y):
    best_t, best_nll = 1.0, 1e9
    for t in np.arange(0.5, 3.01, 0.05):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return round(best_t, 2)


def trim_vocab(model, tok_json: dict, keep_ids: list[int]):
    """Return (new tokenizer json, model with trimmed embedding) keeping the given piece ids in order."""
    import torch
    old_vocab = tok_json["model"]["vocab"]
    new_tok = copy.deepcopy(tok_json)
    new_tok["model"]["vocab"] = [old_vocab[i] for i in keep_ids]
    remap = {old: new for new, old in enumerate(keep_ids)}
    new_tok["model"]["unk_id"] = remap[tok_json["model"]["unk_id"]]
    for at in new_tok.get("added_tokens", []):
        at["id"] = remap[at["id"]]
    pp = new_tok.get("post_processor") or {}
    for key in ("cls", "sep"):
        if key in pp:
            pp[key][1] = remap[pp[key][1]]
    for st in (pp.get("special_tokens") or {}).values():
        st["ids"] = [remap[i] for i in st["ids"]]
    emb = model.get_input_embeddings()
    new_emb = torch.nn.Embedding(len(keep_ids), emb.weight.shape[1], padding_idx=remap.get(emb.padding_idx))
    new_emb.weight.data = emb.weight.data[torch.tensor(keep_ids)].clone()
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(keep_ids)
    if getattr(model.config, "pad_token_id", None) is not None:
        model.config.pad_token_id = remap[model.config.pad_token_id]
    return new_tok, model


def error_tags(text: str, n_words: int) -> list[str]:
    t = text.lower()
    tags = []
    if re.search(r"\b(nahi|nhi|nahin|na|mat|not|no|never|don't|didn't|isn't|wasn't)\b", t) or re.search(r"نہیں|نہ\b|مت\b", t):
        tags.append("negation")
    if re.search("[\U0001F300-\U0001FAFF☀-➿]", text):
        tags.append("emoji")
    if n_words <= 3:
        tags.append("very_short")
    if re.search(r"(\w)\1\1|\b(bht|bhot|boht|acha|achaa|kya|kia|ni|ny)\b", t):
        tags.append("informal_spelling")
    return tags


def main():
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

    from ..data_preparation.io import load_split
    from ..evaluation.metrics import evaluate, evaluate_by_group, expected_calibration_error
    from ..inference.engine import SentimentEngine

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "checkpoints" / "final"))
    ap.add_argument("--out", default=str(ROOT / "models" / "sentiment"))
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--keep_top", type=int, default=30000)
    ap.add_argument("--base_model", default="microsoft/Multilingual-MiniLM-L12-H384")
    a = ap.parse_args()
    torch.set_num_threads(2)
    ckpt, out = Path(a.ckpt), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).float().eval()
    va, te = load_split("val"), load_split("test")
    yva = np.array([LABELS.index(l) for l in va["label"]])
    yte = np.array([LABELS.index(l) for l in te["label"]])
    card = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # 1. temperature on validation
    lv = _logits(model, tok, va["text"].tolist(), a.max_len)
    T = fit_temperature(lv, yva)
    pv = np.exp(lv / T) / np.exp(lv / T).sum(1, keepdims=True)
    card["calibration"] = {"temperature": T, "val_ece_before": expected_calibration_error(
        np.exp(lv) / np.exp(lv).sum(1, keepdims=True), yva), "val_ece_after": expected_calibration_error(pv, yva)}
    lt = _logits(model, tok, te["text"].tolist(), a.max_len)
    full_pred = lt.argmax(1)
    card["pytorch_fp32_test"] = evaluate(te["label"], [LABELS[i] for i in full_pred])

    # 2. vocabulary trimming (train + val text only; test is never used to decide what to keep)
    tok_json = json.loads(tok.backend_tokenizer.to_str())
    seen = set()
    for split in ("train", "val"):
        for ids in tok(load_split(split)["text"].tolist(), add_special_tokens=False)["input_ids"]:
            seen.update(ids)
    vocab = tok_json["model"]["vocab"]
    special = {i for i, (p, _) in enumerate(vocab) if p in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")}
    single = {i for i, (p, _) in enumerate(vocab) if len(p.replace("▁", "")) <= 1}
    by_score = sorted(range(len(vocab)), key=lambda i: -vocab[i][1])[:a.keep_top]
    keep = sorted(special | seen | single | set(by_score))
    model_vocab = model.get_input_embeddings().weight.shape[0]
    keep = [i for i in keep if i < model_vocab]
    new_tok_json, model = trim_vocab(model, tok_json, keep)
    trimmed_tok = PreTrainedTokenizerFast(tokenizer_object=__import__("tokenizers").Tokenizer.from_str(json.dumps(new_tok_json)),
                                          bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="</s>",
                                          cls_token="<s>", pad_token="<pad>", mask_token="<mask>")
    lt2 = _logits(model, trimmed_tok, te["text"].tolist(), a.max_len)
    card["vocab_trimming"] = {"original_pieces": len(vocab), "kept_pieces": len(keep),
                              "test_prediction_agreement": round(float((lt2.argmax(1) == full_pred).mean()), 4)}

    # 3. ONNX export + int8 quantisation
    fp32 = out / "model_fp32.onnx"
    dummy = trimmed_tok(["review text example"], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    int8 = out / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True)
    fp32.unlink()
    (out / "tokenizer.json").write_text(json.dumps(new_tok_json, ensure_ascii=False))
    ckpt_log = json.loads((ckpt / "training_log.json").read_text()) if (ckpt / "training_log.json").exists() else {}
    meta = {
        "display_name": "LingoAI sentiment (fine-tuned Multilingual MiniLM, int8)",
        "version": "1.0.0", "labels": LABELS, "max_len": a.max_len, "stride": 32, "temperature": T,
        "low_confidence": 0.55, "onnx_file": "model.onnx", "cls_token": "<s>", "sep_token": "</s>", "pad_token": "<pad>",
        "base_model": a.base_model, "base_model_license": "MIT", "tokenizer": "XLM-R SentencePiece (trimmed)",
        "size_mb": round(int8.stat().st_size / 1e6, 1), "created_at": card["created_at"],
        "training": {k: ckpt_log.get(k) for k in ("args", "train_rows", "val_rows", "truncated_train", "class_weights",
                                                   "trainable_params", "best", "history", "minutes_total", "vocab_trim")},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if ckpt_log.get("vocab_trim"):
        card["vocab_trimming"]["before_training"] = ckpt_log["vocab_trim"]

    # 4. evaluate the exact served artifact
    eng = SentimentEngine(out, threads=2)
    t0 = time.time()
    probs, _ = eng.predict_proba(te["text"].tolist())
    secs = time.time() - t0
    te = te.copy()
    te["pred"] = [LABELS[i] for i in probs.argmax(1)]
    te["conf"] = probs.max(1)
    card["served_model_test"] = evaluate(te["label"], te["pred"])
    card["served_model_test"]["ece"] = expected_calibration_error(probs, yte)
    card["served_vs_pytorch_agreement"] = round(float((probs.argmax(1) == full_pred).mean()), 4)
    card["test_by_language"] = evaluate_by_group(te)
    card["throughput_reviews_per_sec_2cpu"] = round(len(te) / secs, 1)
    low = te["conf"] < meta["low_confidence"]
    card["low_confidence"] = {"threshold": meta["low_confidence"], "share": round(float(low.mean()), 4),
                              "accuracy_low": round(float((te[low]["pred"] == te[low]["label"]).mean()), 4) if low.any() else None,
                              "accuracy_rest": round(float((te[~low]["pred"] == te[~low]["label"]).mean()), 4)}
    # error analysis
    te["tags"] = [error_tags(t, n) for t, n in zip(te["raw_text"], te["n_words"])]
    cats = {}
    for tag in ("negation", "emoji", "very_short", "informal_spelling"):
        sub = te[te["tags"].map(lambda x, tag=tag: tag in x)]
        if len(sub):
            cats[tag] = {"n": int(len(sub)), "accuracy": round(float((sub["pred"] == sub["label"]).mean()), 4)}
    wrong = te[te["pred"] != te["label"]]
    confusions = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False)
    card["error_analysis"] = {
        "by_category": cats,
        "top_confusions": [{"true": t, "predicted": p, "count": int(c)} for (t, p), c in confusions.head(6).items()],
        "examples": [{"text": r.raw_text[:220], "language": r.language, "true": r.label, "predicted": r.pred,
                      "confidence": round(float(r.conf), 3)} for r in wrong.sample(min(12, len(wrong)), random_state=7).itertuples()],
        "notes": ["Sarcasm cannot be detected automatically in this data; it is a known weak spot for any sentiment model.",
                  "Many errors are between neutral and a polar class, where human labels also disagree most."],
    }
    baseline = ROOT / "reports" / "baseline_metrics.json"
    if baseline.exists():
        b = json.loads(baseline.read_text())
        card["baseline"] = {"model": b["model"], "test": {k: b["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                            "test_by_language": {g: {"n": m["n"], "macro_f1": m["macro_f1"]} for g, m in b["test_by_language"].items()}}
    comp = ROOT / "reports" / "model_comparison.json"
    if comp.exists():
        card["model_comparison"] = json.loads(comp.read_text())
    card["data"] = json.loads((ROOT / "data" / "processed" / "report.json").read_text())
    from ..data_acquisition.sources import SOURCES
    card["sources"] = [{"key": s.key, "name": s.name, "language": s.language, "homepage": s.homepage, "license": s.license,
                        "citation": s.citation, "labels": s.labels, "notes": list(s.notes)} for s in SOURCES]
    (out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    (ROOT / "reports" / "final_metrics.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(json.dumps({k: card[k] for k in ("calibration", "vocab_trimming", "served_vs_pytorch_agreement",
                                           "throughput_reviews_per_sec_2cpu", "low_confidence")}, indent=1))
    print("TEST", {k: card["served_model_test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")})
    print("BY_LANG", {g: (m["n"], m["macro_f1"]) for g, m in card["test_by_language"].items()}, "size_mb", meta["size_mb"])


if __name__ == "__main__":
    main()
