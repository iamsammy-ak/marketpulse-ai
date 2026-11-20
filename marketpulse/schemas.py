"""Pydantic models for the model's structured replies.

Free models do not always follow a schema exactly, so validators coerce common variations
("Theme 3", "[12]", "Positive 🙂", objects where strings were expected) instead of rejecting the reply.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_INT = re.compile(r"\d+")
SENTIMENTS = ("positive", "neutral", "negative")


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    match = _INT.search(str(value or ""))
    return int(match.group(0)) if match else default


def _as_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v not in (None, ""))
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    return str(value or "").strip()


class ThemeNote(BaseModel):
    theme: int = Field(description="The theme number exactly as given")
    name: str = Field(description="A short, specific name for the theme (3-6 words)")
    summary: str = Field("", description="Two sentences on what is happening, citing headlines as [n]")
    implication: str = Field("", description="One sentence on why it matters for the companies watched")

    @field_validator("theme", mode="before")
    @classmethod
    def _theme(cls, value: Any) -> int:
        return _as_int(value, -1)

    @field_validator("name", "summary", "implication", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _as_text(value)


class ThemeNotes(BaseModel):
    themes: list[ThemeNote] = []

    @field_validator("themes", mode="before")
    @classmethod
    def _themes(cls, value: Any) -> list:
        if isinstance(value, dict):  # {"0": {...}, "1": {...}}
            return [{"theme": k, **v} if isinstance(v, dict) else {"theme": k, "name": str(v)}
                    for k, v in value.items()]
        return value or []


class HeadlineSentiment(BaseModel):
    ref: int = Field(description="The headline number")
    sentiment: str = Field("neutral", description="positive, neutral or negative for the company it is about")

    @field_validator("ref", mode="before")
    @classmethod
    def _ref(cls, value: Any) -> int:
        return _as_int(value, -1)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _sentiment(cls, value: Any) -> str:
        text = str(value or "").lower()
        for label in ("negative", "positive", "neutral"):
            if label in text:
                return label
        if text.startswith("neg") or text in ("-1", "bad"):
            return "negative"
        if text.startswith("pos") or text in ("1", "+1", "good"):
            return "positive"
        return "neutral"


class SentimentBatch(BaseModel):
    items: list[HeadlineSentiment] = []
