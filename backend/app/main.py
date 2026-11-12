"""FastAPI entry point for the multilingual review analysis API."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make the `nlp` package importable

from . import errors  # noqa: E402
from .api.routes import router  # noqa: E402
from .config import settings  # noqa: E402

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reviews.api")

app = FastAPI(
    title="LingoAI API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    description=("Review Input Studio -> Language & Script Intelligence -> Text Normalisation -> Transformer Inference -> "
                 "Sentiment Intelligence -> Review Analytics -> Reports. No login; reviews are not stored."),
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["Content-Disposition"])
errors.install(app)


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    if request.url.path not in ("/api/health",):
        log.info("%s %s -> %s %.0fms", request.method, request.url.path, resp.status_code, (time.time() - t) * 1000)
    return resp


app.include_router(router)
