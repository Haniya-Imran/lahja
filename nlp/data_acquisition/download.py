"""Download every source into data/raw/ and record SHA-256 hashes in data/raw/manifest.json.

Run from the repo root:  python -m nlp.data_acquisition.download
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .sources import ROMAN_URDU_DICTIONARY, SOURCES

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed GitHub URLs
        return r.read()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(force: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for s in SOURCES:
        out = RAW / f"{s.key}{Path(s.archive_member or s.url.split('/')[-1]).suffix or '.txt'}"
        if out.exists() and not force and s.key in manifest:
            if _sha(out.read_bytes()) != manifest[s.key]["sha256"]:
                raise RuntimeError(f"{out.name} does not match its recorded hash; re-download with force=True")
            continue
        blob = _fetch(s.url)
        if s.archive_member:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                member = tar.getmember(s.archive_member)
                if not member.isfile():
                    raise RuntimeError(f"{s.archive_member} is not a regular file")
                blob = tar.extractfile(member).read()
        out.write_bytes(blob)
        prev = manifest.get(s.key, {}).get("sha256")
        if prev and prev != _sha(blob):
            raise RuntimeError(f"{s.key}: content changed since the recorded download")
        manifest[s.key] = {"file": out.name, "url": s.url, "sha256": _sha(blob), "bytes": len(blob),
                           "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    d = RAW / "roman_urdu_dictionary.txt"
    if not d.exists() or force:
        d.write_bytes(_fetch(ROMAN_URDU_DICTIONARY))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    for k, v in download().items():
        print(f"{k:28s} {v['bytes']:>9,d} bytes  sha256 {v['sha256'][:16]}…")
