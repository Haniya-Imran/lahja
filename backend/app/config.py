"""Runtime settings from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _list(name: str, default: str) -> list[str]:
    return [x.strip().rstrip("/") for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(ROOT / "models" / "sentiment")))
    langid_path: Path = Path(os.getenv("LANGID_PATH", str(ROOT / "models" / "langid.json")))
    function_words_path: Path = Path(os.getenv("FUNCTION_WORDS_PATH", str(ROOT / "models" / "function_words.json")))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", str(ROOT / "models" / "sentiment" / "model_card.json")))
    allowed_origins: list[str] = field(default_factory=lambda: _list(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"))
    max_chars: int = int(os.getenv("MAX_REVIEW_CHARS", "5000"))
    max_batch: int = int(os.getenv("MAX_BATCH", "64"))
    max_analytics_items: int = int(os.getenv("MAX_ANALYTICS_ITEMS", "20000"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "4"))
    onnx_threads: int = int(os.getenv("ONNX_THREADS", "0"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
