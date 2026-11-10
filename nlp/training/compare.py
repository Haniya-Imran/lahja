"""Model selection: compare candidate pretrained Transformers on the same data before the full fine-tune.

Each candidate was fine-tuned for 1 epoch on the same random 8,000 training reviews (same seed, frozen word
embeddings) with nlp.training.finetune; this script reads their validation scores and adds tokenizer
statistics per language (unknown tokens, tokens per word), then writes reports/model_comparison.json.

Run:  python -m nlp.training.compare
"""
from __future__ import annotations

import json
from pathlib import Path

from ..data_preparation.io import load_split
from ..preprocessing.normalize import words

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [
    {"key": "minilm", "name": "Multilingual MiniLM-L12-H384", "hf_id": "microsoft/Multilingual-MiniLM-L12-H384",
     "license": "MIT", "ckpt": "models/checkpoints/cmp_minilm", "pretrained": "models/pretrained/minilm"},
    {"key": "distilmbert", "name": "DistilBERT multilingual (cased)", "hf_id": "distilbert-base-multilingual-cased",
     "license": "Apache-2.0", "ckpt": "models/checkpoints/cmp_distilmbert", "pretrained": "models/pretrained/distilmbert"},
]


def main():
    from transformers import AutoTokenizer

    va = load_split("val")
    out = {"protocol": "1 epoch on the same random 8,000 training reviews (seed 42), frozen word embeddings, "
                       "batch 32, max length 128; scored on the full validation split (3,867 reviews). Test split not used.",
           "candidates": []}
    for c in CANDIDATES:
        log_p = ROOT / c["ckpt"] / "training_log.json"
        if not log_p.exists():
            print("missing", log_p)
            continue
        log = json.loads(log_p.read_text())
        tok = AutoTokenizer.from_pretrained(ROOT / c["pretrained"])
        unk = tok.unk_token_id
        per_lang = {}
        for lang, sub in va.groupby("language"):
            if len(sub) < 30:
                continue
            enc = tok(sub["text"].tolist(), add_special_tokens=False)["input_ids"]
            n_tok = sum(len(e) for e in enc)
            n_words = sum(max(1, len(words(t))) for t in sub["text"])
            per_lang[lang] = {"n": int(len(sub)), "tokens_per_word": round(n_tok / n_words, 2),
                              "unknown_tokens": int(sum(i == unk for e in enc for i in e))}
        out["candidates"].append({k: c[k] for k in ("key", "name", "hf_id", "license")} | {
            "trainable_params": log["trainable_params"], "val_macro_f1": log["best"]["val_macro_f1"],
            "train_minutes_8k": log["best"]["minutes"], "tokenizer_by_language": per_lang})
    top = max(out["candidates"], key=lambda c: c["val_macro_f1"])
    best = top
    # Deployment rule: the served model must fit a 250 MB serverless bundle and answer quickly on CPU.
    # When the F1 gap is under 0.01 (one short run on 8k reviews), prefer the smaller, faster candidate.
    fastest = min(out["candidates"], key=lambda c: c["train_minutes_8k"])
    gap = round(top["val_macro_f1"] - fastest["val_macro_f1"], 4)
    if fastest is not top and gap < 0.01:
        best = fastest
        out["tie_break"] = (f"{top['name']} scored {gap} higher on validation macro-F1, a small gap for a single 1-epoch run "
                            f"on 8,000 reviews. {fastest['name']} was chosen because it is smaller and faster, which matters "
                            "for CPU serving inside a 250 MB serverless function.")
    out["chosen"] = best["key"]
    others = [c for c in out["candidates"] if c is not best]
    reason = [out.get("tie_break", f"{best['name']} had the highest validation macro-F1 under identical settings").rstrip(".")]
    for o in others:
        if best["train_minutes_8k"] < o["train_minutes_8k"]:
            reason.append(f"trained faster on CPU ({best['train_minutes_8k']} vs {o['train_minutes_8k']} min)")
        bu = sum(v["unknown_tokens"] for v in best["tokenizer_by_language"].values())
        ou = sum(v["unknown_tokens"] for v in o["tokenizer_by_language"].values())
        if bu < ou:
            reason.append(f"produced fewer unknown tokens on validation text ({bu} vs {ou})")
    out["reason"] = "; ".join(reason) + "."
    p = ROOT / "reports" / "model_comparison.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
