"""Module D — Transformer inference engine (ONNX Runtime, CPU).

Loads the fine-tuned model once and reuses it. Uses the model's own tokenizer
(tokenizer.json via the `tokenizers` library), so production does not need PyTorch.

Long reviews are split into overlapping windows; window probabilities are averaged,
weighted by the number of real tokens in each window.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()


@dataclass
class ModelBundle:
    session: object
    tokenizer: object
    labels: list[str]
    max_len: int
    stride: int
    temperature: float
    low_confidence: float
    metadata: dict
    input_names: list[str]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class SentimentEngine:
    def __init__(self, model_dir: Path, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text())
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(model_dir / meta["onnx_file"]), so, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.no_padding()
        tok.no_truncation()
        self.b = ModelBundle(sess, tok, meta["labels"], int(meta["max_len"]), int(meta.get("stride", 32)),
                             float(meta.get("temperature", 1.0)), float(meta.get("low_confidence", 0.55)), meta,
                             [i.name for i in sess.get_inputs()])
        self.cls_id = tok.token_to_id(meta.get("cls_token", "<s>"))
        self.sep_id = tok.token_to_id(meta.get("sep_token", "</s>"))
        self.pad_id = tok.token_to_id(meta.get("pad_token", "<pad>")) or 0

    @property
    def labels(self) -> list[str]:
        return self.b.labels

    @property
    def metadata(self) -> dict:
        return self.b.metadata

    # ---- tokenisation into windows ---------------------------------------------------------
    def _windows(self, text: str) -> list[list[int]]:
        ids = self.b.tokenizer.encode(text, add_special_tokens=False).ids
        body = self.b.max_len - 2
        if len(ids) <= body:
            return [[self.cls_id, *ids, self.sep_id]]
        out, step = [], body - self.b.stride
        for start in range(0, len(ids), step):
            chunk = ids[start:start + body]
            out.append([self.cls_id, *chunk, self.sep_id])
            if start + body >= len(ids):
                break
        return out

    def _run(self, seqs: list[list[int]], batch_size: int = 32) -> np.ndarray:
        logits = []
        for i in range(0, len(seqs), batch_size):
            part = seqs[i:i + batch_size]
            L = max(len(s) for s in part)
            ids = np.full((len(part), L), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(part), L), dtype=np.int64)
            for j, s in enumerate(part):
                ids[j, :len(s)] = s
                mask[j, :len(s)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.b.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            with _LOCK:  # one ONNX run at a time keeps memory predictable on small instances
                logits.append(self.b.session.run(None, feed)[0])
        return np.concatenate(logits, axis=0)

    # ---- public API -----------------------------------------------------------------------
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, list[int]]:
        """Calibrated class probabilities per text, plus the number of windows used."""
        all_seqs, owner, weights = [], [], []
        for k, t in enumerate(texts):
            for w in self._windows(t):
                all_seqs.append(w)
                owner.append(k)
                weights.append(len(w))
        order = np.argsort([len(s) for s in all_seqs])  # sort by length to reduce padding
        logits = np.empty((len(all_seqs), len(self.b.labels)), dtype=np.float32)
        sorted_logits = self._run([all_seqs[i] for i in order], batch_size)
        logits[order] = sorted_logits
        probs = _softmax(logits / self.b.temperature)
        out = np.zeros((len(texts), len(self.b.labels)))
        wsum = np.zeros(len(texts))
        n_windows = [0] * len(texts)
        for p, k, w in zip(probs, owner, weights):
            out[k] += p * w
            wsum[k] += w
            n_windows[k] += 1
        return out / wsum[:, None], n_windows

    def predict(self, texts: list[str]) -> list[dict]:
        probs, windows = self.predict_proba(texts)
        res = []
        for p, w in zip(probs, windows):
            i = int(p.argmax())
            conf = float(p[i])
            res.append({
                "sentiment": self.b.labels[i],
                "confidence": round(conf, 4),
                "probabilities": {lab: round(float(v), 4) for lab, v in zip(self.b.labels, p)},
                "low_confidence": conf < self.b.low_confidence,
                "windows": w,
            })
        return res
