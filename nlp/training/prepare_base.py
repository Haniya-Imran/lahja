"""Shrink the pretrained model's vocabulary *before* fine-tuning so every weight can be trained on a CPU.

Multilingual MiniLM carries a 250k-piece XLM-R vocabulary (96M of its 118M parameters are the word-embedding
table). We keep only: the pieces that occur in train/validation text, every single-character piece (so any
unseen word still splits into known pieces), the special tokens, and the 30k pieces with the highest unigram
prior. The test split is never read. The kept rows are copied unchanged, so the model is mathematically
identical for any text made of kept pieces.

Run:  python -m nlp.training.prepare_base --model models/pretrained/minilm --out models/pretrained/minilm_trimmed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import trim_vocab


def main():
    import tokenizers
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

    from ..data_preparation.io import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_top", type=int, default=30000)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model)
    tok_json = json.loads(tok.backend_tokenizer.to_str())
    vocab = tok_json["model"]["vocab"]
    seen = set()
    for split in ("train", "val"):
        for ids in tok(load_split(split)["text"].tolist(), add_special_tokens=False)["input_ids"]:
            seen.update(ids)
    special = {i for i, (p, _) in enumerate(vocab) if p in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")}
    single = {i for i, (p, _) in enumerate(vocab) if len(p.replace("▁", "")) <= 1}
    by_score = sorted(range(len(vocab)), key=lambda i: -vocab[i][1])[:a.keep_top]
    rows = model.get_input_embeddings().weight.shape[0]
    keep = [i for i in sorted(special | seen | single | set(by_score)) if i < rows]
    before = sum(p.numel() for p in model.parameters())
    new_json, model = trim_vocab(model, tok_json, keep)
    # XLM-R pads with id 1; the MiniLM config says 0 (= <s>), which would freeze the <s> embedding during training.
    import torch
    pad = [p for p, (piece, _) in enumerate(new_json["model"]["vocab"]) if piece == "<pad>"][0]
    emb = model.get_input_embeddings()
    fixed = torch.nn.Embedding(emb.weight.shape[0], emb.weight.shape[1], padding_idx=pad)
    fixed.weight.data = emb.weight.data.clone()
    model.set_input_embeddings(fixed)
    model.config.pad_token_id = pad
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer.from_str(json.dumps(new_json)),
                            bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="</s>",
                            cls_token="<s>", pad_token="<pad>", mask_token="<mask>").save_pretrained(out)
    info = {"original_pieces": len(vocab), "kept_pieces": len(keep), "seen_in_train_val": len(seen),
            "params_before": before, "params_after": sum(p.numel() for p in model.parameters()),
            "rule": "train/val pieces + single characters + special tokens + top-30k unigram prior; test split not used"}
    (out / "vocab_trim.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
