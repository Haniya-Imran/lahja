"""Validated request bodies."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    explain: bool = True


class BatchIn(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=64)


class Item(BaseModel):
    text: str = Field(..., max_length=5000)
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    language: Optional[str] = Field(default=None, max_length=40)
    low_confidence: Optional[bool] = None
    probabilities: Optional[dict[str, float]] = None


class ItemsIn(BaseModel):
    items: list[Item] = Field(..., min_length=1, max_length=20000)


class ExportIn(ItemsIn):
    format: Literal["csv", "html"] = "csv"
