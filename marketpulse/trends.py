"""Mention trends: daily counts, share of voice, week-on-week momentum and spike detection.

A headline that mentions two competitors counts once for each. Share of voice is a competitor's
share of all competitor mentions in the period, which makes it robust to changes in total feed
volume. Spikes are days whose count is unusually high against that competitor's own trailing
baseline (a z-score), with the spread floored at √mean (Poisson noise) and at one headline so a
quiet competitor going from zero to two headlines is not an alarm.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

TOTAL = "All headlines"


def daily_counts(df: pd.DataFrame, names: list[str], start: date, end: date) -> pd.DataFrame:
    """Headlines per day (rows, every day from start to end) per competitor plus an all-headlines column."""
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(0, index=days, columns=[*names, TOTAL], dtype=int)
    if df.empty:
        return out
    day = pd.to_datetime(df["published"].dt.tz_convert("UTC").dt.date)
    in_range = (day >= days[0]) & (day <= days[-1])
    out[TOTAL] = day[in_range].value_counts().reindex(days, fill_value=0).astype(int)
    exploded = pd.DataFrame({"day": day, "competitors": df["competitors"]})[in_range].explode("competitors")
    exploded = exploded.dropna(subset=["competitors"])
    for name in names:
        counts = exploded.loc[exploded["competitors"] == name, "day"].value_counts()
        out[name] = counts.reindex(days, fill_value=0).astype(int)
    out.index.name = "date"
    return out


def share_of_voice(daily: pd.DataFrame, names: list[str], start: date | None = None,
                   end: date | None = None) -> pd.DataFrame:
    """Mentions and share of all competitor mentions between start and end (inclusive)."""
    window = daily.loc[str(start) if start else None : str(end) if end else None, names]
    mentions = window.sum().astype(int)
    total = int(mentions.sum())
    share = mentions / total if total else mentions * 0.0
    out = pd.DataFrame({"competitor": names, "mentions": mentions.values, "share": share.values.astype(float)})
    return out.sort_values(["mentions", "competitor"], ascending=[False, True]).reset_index(drop=True)


def pct_change(last: float, prior: float) -> float | None:
    """Relative change; None when there is no prior activity (a 'new' signal, not infinite growth)."""
    if prior <= 0:
        return None if last > 0 else 0.0
    return (last - prior) / prior


def momentum(daily: pd.DataFrame, names: list[str], end: date, days: int = 7) -> pd.DataFrame:
    """Last `days` days vs the `days` before them: counts, % change and share-of-voice change (points)."""
    last_start = end - timedelta(days=days - 1)
    prior_end = last_start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=days - 1)
    last = share_of_voice(daily, names, last_start, end).set_index("competitor")
    prior = share_of_voice(daily, names, prior_start, prior_end).set_index("competitor")
    rows = []
    for name in names:
        lm, pm = int(last.loc[name, "mentions"]), int(prior.loc[name, "mentions"])
        rows.append(
            {
                "competitor": name,
                "last": lm,
                "prior": pm,
                "change": pct_change(lm, pm),
                "sov_last": float(last.loc[name, "share"]),
                "sov_prior": float(prior.loc[name, "share"]),
                "sov_change": float(last.loc[name, "share"] - prior.loc[name, "share"]),
            }
        )
    return pd.DataFrame(rows).sort_values("sov_change", ascending=False).reset_index(drop=True)


def detect_spikes(
    daily: pd.DataFrame,
    columns: list[str],
    z_threshold: float = 2.5,
    baseline_days: int = 14,
    min_history: int = 7,
    min_count: int = 3,
) -> pd.DataFrame:
    """Days where count ≥ min_count and (count − trailing mean) / spread ≥ z_threshold.

    The trailing baseline uses the previous `baseline_days` days only (never the day itself), and
    needs at least `min_history` days, so the first week of the window is never flagged.
    """
    rows = []
    for col in columns:
        series = daily[col].astype(float)
        prev = series.shift(1)
        mean = prev.rolling(baseline_days, min_periods=min_history).mean()
        std = prev.rolling(baseline_days, min_periods=min_history).std(ddof=0)
        for day, count in series.items():
            m, s = mean.loc[day], std.loc[day]
            if np.isnan(m) or count < min_count:
                continue
            spread = max(float(s), math.sqrt(float(m)), 1.0)
            z = (count - m) / spread
            if z >= z_threshold:
                rows.append({"date": day.date(), "competitor": col, "count": int(count), "baseline": float(m),
                             "z": float(z)})
    out = pd.DataFrame(rows, columns=["date", "competitor", "count", "baseline", "z"])
    return out.sort_values(["z", "date"], ascending=[False, False]).reset_index(drop=True)


def weekly_share(daily: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Share of voice per 7-day block, counted back from the last day (so the last block is complete)."""
    if daily.empty:
        return pd.DataFrame(columns=["week_end", *names])
    data = daily[names].iloc[::-1].reset_index(drop=True)
    blocks = data.groupby(data.index // 7)
    sums = blocks.sum()
    ends = [daily.index[-1 - 7 * b] for b in sums.index]
    totals = sums.sum(axis=1).replace(0, np.nan)
    share = sums.div(totals, axis=0).fillna(0.0)
    share.insert(0, "week_end", ends)
    full = blocks.size() == 7
    return share[full.values].iloc[::-1].reset_index(drop=True)


def event_matrix(df: pd.DataFrame, names: list[str], event_types: tuple[str, ...]) -> pd.DataFrame:
    """Headlines per competitor (rows) and event type (columns)."""
    out = pd.DataFrame(0, index=names, columns=list(event_types), dtype=int)
    for comps, events in zip(df.get("competitors", []), df.get("events", []), strict=False):
        for c in comps:
            if c in out.index:
                for e in events:
                    out.loc[c, e] += 1
    return out
