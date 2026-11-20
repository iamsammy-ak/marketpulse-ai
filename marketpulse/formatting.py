"""Human-friendly formatting shared by the UI, the briefing facts and the report."""

from __future__ import annotations

from datetime import date, datetime


def pct(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def change(value: float | None) -> str:
    """Relative change as '+85%' / '−12%'; None means there was no prior activity."""
    if value is None:
        return "new"
    if round(value * 100) == 0:
        return "0%"
    sign = "+" if value > 0 else "−"
    return f"{sign}{abs(value) * 100:.0f}%"


def points(value: float) -> str:
    """Share-of-voice change in percentage points."""
    if round(value * 100, 1) == 0:
        return "0.0 pp"
    sign = "+" if value > 0 else "−"
    return f"{sign}{abs(value) * 100:.1f} pp"


def day(value: date | datetime) -> str:
    return f"{value.day} {value:%b}"


def day_range(start: date, end: date) -> str:
    return f"{day(start)} – {day(end)} {end.year}"


def count(value: int | float) -> str:
    return f"{int(value):,}"
