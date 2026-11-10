"""Vercel entry point: serves the FastAPI backend under /api on the same domain as the frontend."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402,F401
