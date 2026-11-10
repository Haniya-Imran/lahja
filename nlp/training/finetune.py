"""Fine-tune a pretrained multilingual Transformer for 3-class review sentiment (CPU-friendly).

Run:  python -m nlp.training.finetune --model microsoft/Multilingual-MiniLM-L12-H384 --epochs 3

- Tokenizer and weights come from the pretrained checkpoint; a new 3-way classification head is trained.
- Model selection uses the validation split only (early stopping on macro-F1). Test is untouched here.
- Reproducible: fixed seeds, config saved next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches_by_length(lengths, batch_size, rng):
    """Group similar lengths into the same batch (less padding = much faster on CPU), shuffle batch order."""
    idx = np.argsort(np.array(lengths) + rng.random(len(lengths)) * 4)
    bs = [idx[i:i + batch_size].tolist() for i in range(0, len(idx), batch_size)]
    rng.shuffle(bs)
    return bs


def main():
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from ..data_preparation.io import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Multilingual-MiniLM-L12-H384")
    ap.add_argument("--tokenizer", default=None, help="Override tokenizer (MiniLM uses the XLM-R tokenizer)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--train_limit", type=int, default=0, help="Use only N training rows (quick comparisons)")
    ap.add_argument("--eval_every", type=int, default=0, help="Evaluate every N steps (0 = once per epoch)")
    ap.add_argument("--out", default=str(ROOT / "models" / "checkpoints" / "run"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--freeze_word_embeddings", action="store_true",
                    help="Keep the multilingual word-embedding table fixed (faster on CPU, preserves cross-lingual alignment)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)
    if a.freeze_word_embeddings:
        emb = model.get_input_embeddings()
        emb.weight.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tr, va = load_split("train"), load_split("val")
    if a.train_limit:
        tr = tr.sample(min(len(tr), a.train_limit), random_state=a.seed)  # random subset keeps language mix

    def enc(df):
        e = tok(df["text"].tolist(), truncation=True, max_length=a.max_len)
        return e["input_ids"], [L2I[l] for l in df["label"]]

    Xtr, ytr = enc(tr)
    Xva, yva = enc(va)
    trunc = sum(len(x) >= a.max_len for x in Xtr)

    # Mild class weighting (inverse frequency, normalised) for the slight label imbalance.
    counts = np.bincount(ytr, minlength=3)
    w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    def collate(ids_list, ys=None):
        L = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        b = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
        return b, (torch.tensor(ys, device=device) if ys is not None else None)

    @torch.no_grad()
    def evaluate():
        model.eval()
        preds, logits_all = [], []
        order = np.argsort([len(x) for x in Xva])
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            b, _ = collate([Xva[j] for j in idx])
            lg = model(**b).logits.float().cpu().numpy()
            logits_all.append((idx, lg))
        full = np.zeros((len(Xva), 3), dtype=np.float32)
        for idx, lg in logits_all:
            full[idx] = lg
        model.train()
        return f1_score(yva, full.argmax(1), average="macro"), full

    steps_per_epoch = math.ceil(len(Xtr) / a.batch)
    total = steps_per_epoch * a.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [{"params": [p for n, p in trainable if not any(x in n for x in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in trainable if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    trim_info = Path(a.model) / "vocab_trim.json"
    log = {"vocab_trim": json.loads(trim_info.read_text()) if trim_info.exists() else None,
           "trainable_params": sum(p.numel() for _, p in trainable), "args": vars(a), "train_rows": len(Xtr), "val_rows": len(Xva), "truncated_train": trunc,
           "class_weights": w.tolist(), "device": device, "history": []}
    best, bad, step, t0 = -1.0, 0, 0, time.time()
    model.train()
    stop = False
    for ep in range(a.epochs):
        run_loss, seen = 0.0, 0
        for bi, idx in enumerate(batches_by_length([len(x) for x in Xtr], a.batch, rng)):
            b, y = collate([Xtr[i] for i in idx], [ytr[i] for i in idx])
            loss = loss_fn(model(**b).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            run_loss += loss.item() * len(idx); seen += len(idx)
            if step % 50 == 0:
                el = time.time() - t0
                print(f"ep {ep + 1} step {step}/{total} loss {run_loss / seen:.4f} "
                      f"{el / 60:.1f} min elapsed, ~{el / step * (total - step) / 60:.0f} min left", flush=True)
            last_in_epoch = bi == steps_per_epoch - 1
            if (a.eval_every and step % a.eval_every == 0) or last_in_epoch:
                f1, _ = evaluate()
                rec = {"epoch": ep + 1, "step": step, "train_loss": round(run_loss / max(seen, 1), 4),
                       "val_macro_f1": round(float(f1), 4), "minutes": round((time.time() - t0) / 60, 1)}
                log["history"].append(rec)
                print("EVAL", rec, flush=True)
                if f1 > best:
                    best, bad = f1, 0
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
                    log["best"] = rec
                else:
                    bad += 1
                    if bad > a.patience:
                        print("Early stopping: validation macro-F1 stopped improving.")
                        stop = True
                        break
                (out / "training_log.json").write_text(json.dumps(log, indent=2))
        if stop:
            break
    log["minutes_total"] = round((time.time() - t0) / 60, 1)
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print("BEST", log.get("best"))


if __name__ == "__main__":
    main()
