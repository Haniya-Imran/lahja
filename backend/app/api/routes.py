"""HTTP endpoints. All live under /api so the site and API can share one domain."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse, Response

from ..config import settings
from ..errors import ApiError
from ..schemas import AnalyzeIn, BatchIn, ExportIn, ItemsIn
from ..services import analytics, csv_inspect, registry, reports, reviews

router = APIRouter(prefix="/api")


@router.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model_loaded": registry.loaded()}


@router.get("/model-info", tags=["Model"], summary="Model card: base model, training data, metrics, limitations")
def model_info():
    engine = registry.get("engine")
    return {"metadata": engine.metadata, "card": registry.get("card"), "load_seconds": registry.get("load_seconds")}


@router.post("/analyze", tags=["Analyze"], summary="Analyse one review (with word-level explanation)")
def analyze(body: AnalyzeIn):
    if not body.text.strip():
        raise ApiError(422, "empty_review", "Type or paste a review first.")
    out = reviews.analyse([body.text], explain=body.explain)
    return {**out["results"][0], "model": out["model"], "elapsed_ms": out["elapsed_ms"]}


@router.post("/analyze/batch", tags=["Analyze"], summary="Analyse up to 64 reviews in one call")
def analyze_batch(body: BatchIn):
    return reviews.analyse(body.reviews)


@router.post("/reviews/inspect", tags=["Bulk"], summary="Validate a CSV and suggest the review column")
async def inspect(file: UploadFile = File(...)):
    limit = int(settings.max_upload_mb * 1024 * 1024)
    raw = await file.read(limit + 1)
    return csv_inspect.inspect(raw, file.filename or "upload.csv", settings.max_upload_mb)


@router.post("/analytics/summary", tags=["Analytics"], summary="Statistics over results analysed in this session")
def summary(body: ItemsIn):
    return analytics.summary([i.model_dump() for i in body.items])


@router.post("/reports/export", tags=["Reports"], summary="Download results as CSV or a printable HTML report")
def export(body: ExportIn):
    items = [i.model_dump() for i in body.items]
    if body.format == "csv":
        return Response(reports.results_csv(items), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="review-analysis.csv"'})
    meta = registry.get("engine").metadata
    page = reports.html_report(items, {"name": meta.get("display_name"), "version": meta.get("version")})
    return HTMLResponse(page, headers={"Content-Disposition": 'attachment; filename="review-report.html"'})


@router.get("/examples", tags=["Analyze"], summary="Labelled example reviews written for the demo (not from the dataset)")
def examples():
    from ..examples import EXAMPLES
    return {"note": "Examples written for this demo to show each language. They are not customer reviews from the training data.",
            "examples": EXAMPLES}

