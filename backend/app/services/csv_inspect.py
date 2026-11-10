"""Review Input Studio: validate an uploaded CSV and suggest which column holds the reviews.

The file is read in memory, never executed or stored. The browser keeps the file and
sends reviews for scoring in small batches, so large files never travel in one request.
"""
from __future__ import annotations

import csv
import io

from ..errors import ApiError

_HINTS = ("review", "text", "comment", "feedback", "content", "body", "message", "tweet", "sentence", "opinion")


def inspect(raw: bytes, filename: str, max_mb: float) -> dict:
    if not filename.lower().endswith((".csv", ".txt", ".tsv")):
        raise ApiError(415, "unsupported_type", "Please upload a .csv file.")
    if len(raw) > max_mb * 1024 * 1024:
        raise ApiError(413, "file_too_large", f"The file is larger than {max_mb:g} MB.")
    if raw[:4] == b"PK\x03\x04" or b"\x00" in raw[:2048]:
        raise ApiError(422, "not_csv", "This looks like an Excel or binary file. Save it as CSV (UTF-8) first.")
    for enc in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ApiError(422, "unreadable", "The file's text encoding could not be read. Save it as CSV UTF-8.")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    if len(rows) < 2:
        raise ApiError(422, "no_rows", "The file needs a header row and at least one review.")
    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    body = rows[1:]
    cols = []
    for i, h in enumerate(header):
        vals = [r[i] for r in body if i < len(r)]
        filled = [v for v in vals if v.strip()]
        avg_len = sum(len(v) for v in filled) / max(len(filled), 1)
        letters = sum(sum(ch.isalpha() for ch in v) for v in filled[:500]) / max(sum(len(v) for v in filled[:500]), 1)
        score = (2 if any(k in h.lower() for k in _HINTS) else 0) + min(avg_len / 40, 2) + letters
        cols.append({"name": h, "filled": len(filled), "empty": len(vals) - len(filled), "avg_chars": round(avg_len, 1),
                     "letter_share": round(letters, 2), "score": round(score, 2),
                     "examples": [v[:120] for v in filled[:3]]})
    best = max(cols, key=lambda c: c["score"])
    return {"filename": filename, "rows": len(body), "delimiter": delim, "columns": cols,
            "suggested_column": best["name"], "ragged_rows": sum(len(r) != len(header) for r in body)}
