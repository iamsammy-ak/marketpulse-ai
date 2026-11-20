"""Glue between the deterministic analysis and the AI steps, shared by the app and the scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .agents import SupportsLLM, describe_themes, judge_sentiment, write_briefing
from .dedupe import deduplicate, to_frame
from .formatting import change, count, day, day_range, pct, points
from .ingest import Dataset
from .tagging import EVENT_TYPES, tag_articles
from .themes import fit_themes, stopwords_for, theme_label
from .trends import daily_counts, detect_spikes, event_matrix, momentum, share_of_voice, weekly_share


@dataclass(frozen=True)
class Params:
    days: int = 30
    dedupe_threshold: float = 0.75
    n_themes: int = 0  # 0 = choose k by silhouette
    spike_z: float = 2.5
    momentum_days: int = 7
    evidence_size: int = 40


@dataclass(frozen=True)
class Computed:
    articles: pd.DataFrame  # deduplicated, tagged headlines, newest first
    raw_count: int  # headlines in the window before near-duplicate merging
    start: date
    end: date
    daily: pd.DataFrame
    weekly: pd.DataFrame
    sov: pd.DataFrame
    momentum: pd.DataFrame
    spikes: pd.DataFrame
    events: pd.DataFrame
    themes: pd.DataFrame
    silhouette: pd.DataFrame
    k_auto: bool
    outlets: pd.DataFrame
    evidence: pd.DataFrame
    capped_windows: int
    failed_requests: int

    @property
    def names(self) -> list[str]:
        return [c for c in self.daily.columns if c != "All headlines"]


# --------------------------------------------------------------------------- compute
def _score(arts: pd.DataFrame, end: date, days: int) -> pd.Series:
    """Evidence priority: reach (outlets carrying the story), tagged event, competitor mention, recency."""
    age = np.array([(end - d).days for d in arts["day"]], dtype=float)
    return (
        np.log1p(arts["copies"].astype(float))
        + 0.6 * arts["events"].map(bool).astype(float)
        + 0.4 * arts["competitors"].map(bool).astype(float)
        + 0.5 * np.clip(1 - age / max(days, 1), 0, 1)
    )


def select_evidence(arts: pd.DataFrame, names: list[str], spikes: pd.DataFrame, size: int, end: date,
                    days: int) -> pd.DataFrame:
    """Pick a numbered evidence list: the headline behind each top spike, the best three per competitor
    and two per theme, then the highest scores overall. Numbered newest first."""
    if arts.empty:
        return arts.assign(ref=pd.Series(dtype=int))
    ranked = arts.assign(_score=_score(arts, end, days)).sort_values(["_score", "published"], ascending=False)
    picked: list[int] = []
    seen: set[int] = set()

    def take(index, n: int) -> None:
        added = 0
        for i in index:
            if added >= n or len(picked) >= size:
                return
            if i not in seen:
                picked.append(i)
                seen.add(i)
                added += 1

    id_to_index = dict(zip(arts["id"], arts.index, strict=False))
    take([id_to_index[i] for i in spikes.head(6).get("id", []) if i in id_to_index], 6)
    for name in names:
        take(ranked.index[ranked["competitors"].map(lambda cs, n=name: n in cs)], 3)
    for theme in sorted(t for t in ranked["theme"].unique() if t >= 0):
        take(ranked.index[ranked["theme"] == theme], 2)
    take(ranked.index, size)
    ev = arts.loc[picked].sort_values("published", ascending=False)
    return ev.assign(ref=range(1, len(ev) + 1))


def theme_table(arts: pd.DataFrame, terms: dict[int, list[str]], end: date, days: int) -> pd.DataFrame:
    rows = []
    clustered = arts[arts["theme"] >= 0]
    last_start = end - timedelta(days=days - 1)
    prior_start = last_start - timedelta(days=days)
    for theme in sorted(clustered["theme"].unique()):
        sub = clustered[clustered["theme"] == theme]
        last = int((sub["day"] >= last_start).sum())
        prior = int(((sub["day"] >= prior_start) & (sub["day"] < last_start)).sum())
        mentions = pd.Series([c for cs in sub["competitors"] for c in cs]).value_counts()
        evts = pd.Series([e for es in sub["events"] for e in es]).value_counts()
        top = sub.sort_values(["copies", "published"], ascending=False).iloc[0]
        rows.append(
            {
                "theme": int(theme),
                "label": theme_label(terms.get(int(theme), [])),
                "terms": terms.get(int(theme), []),
                "size": len(sub),
                "share": len(sub) / max(len(clustered), 1),
                "last": last,
                "prior": prior,
                "change": None if prior == 0 and last > 0 else ((last - prior) / prior if prior else 0.0),
                "competitors": {str(k): int(v) for k, v in mentions.head(4).items()},
                "events": {str(k): int(v) for k, v in evts.head(3).items()},
                "top_title": top["title"],
                "top_link": top["link"],
                "top_id": top["id"],
            }
        )
    return pd.DataFrame(rows)


def analysis_window(ds: Dataset, days: int) -> tuple[date, date]:
    """`days` days ending on the last complete UTC day before the fetch, never before the first headline."""
    end = (ds.fetched_at - timedelta(days=1)).date()
    start = end - timedelta(days=days - 1)
    published = [a.get("published", "") for a in ds.articles if a.get("published")]
    if published:
        first = datetime.fromisoformat(min(published).replace("Z", "+00:00")).date()
        start = max(start, min(first, end))
    return start, end


def compute(ds: Dataset, p: Params) -> Computed:
    start, end = analysis_window(ds, p.days)
    names = ds.watchlist.names

    raw = to_frame(ds.articles)
    raw = raw[(raw["published"].dt.date >= start) & (raw["published"].dt.date <= end)].reset_index(drop=True)
    arts = deduplicate(tag_articles(raw, ds.watchlist.competitors), p.dedupe_threshold)
    arts["day"] = arts["published"].dt.date if len(arts) else pd.Series(dtype=object)

    aliases = [t for c in ds.watchlist.competitors for t in c.terms]
    texts = (arts["title"] + ". " + arts["summary"]).tolist()
    model = fit_themes(texts, stopwords_for(aliases), p.n_themes)
    arts["theme"] = model.labels if len(arts) else pd.Series(dtype=int)

    daily = daily_counts(arts, names, start, end)
    spikes = _annotate_spikes(detect_spikes(daily, names, p.spike_z), arts)
    evidence = select_evidence(arts, names, spikes, p.evidence_size, end, p.days)
    ref_of = dict(zip(evidence["id"], evidence["ref"], strict=False)) if len(evidence) else {}
    arts["ref"] = arts["id"].map(ref_of).fillna(0).astype(int) if len(arts) else pd.Series(dtype=int)
    spikes = spikes.assign(ref=spikes["id"].map(ref_of).fillna(0).astype(int))

    outlets = (
        arts.groupby("source").agg(headlines=("id", "size"), latest=("published", "max")).reset_index()
        .sort_values(["headlines", "latest"], ascending=False).reset_index(drop=True)
        if len(arts) else pd.DataFrame(columns=["source", "headlines", "latest"])
    )
    log = list(ds.fetch_log)
    return Computed(
        articles=arts,
        raw_count=len(raw),
        start=start,
        end=end,
        daily=daily,
        weekly=weekly_share(daily, names),
        sov=share_of_voice(daily, names),
        momentum=momentum(daily, names, end, p.momentum_days),
        spikes=spikes,
        events=event_matrix(arts, names, EVENT_TYPES),
        themes=theme_table(arts, model.terms, end, p.momentum_days) if len(arts) else pd.DataFrame(),
        silhouette=model.silhouette,
        k_auto=model.auto,
        outlets=outlets,
        evidence=evidence,
        capped_windows=sum(1 for r in log if r.get("capped")),
        failed_requests=sum(1 for r in log if r.get("status") == "error"),
    )


def _annotate_spikes(spikes: pd.DataFrame, arts: pd.DataFrame) -> pd.DataFrame:
    """Attach the most widely carried headline of each spike day, so a spike comes with its likely cause."""
    picks = []
    for _, s in spikes.iterrows():
        mask = (arts["day"] == s["date"]) & arts["competitors"].map(lambda cs, n=s["competitor"]: n in cs)
        sub = arts[mask].assign(_event=arts.loc[mask, "events"].map(bool))
        sub = sub.sort_values(["copies", "_event", "published"], ascending=False)
        picks.append(sub.iloc[0] if len(sub) else None)
    return spikes.assign(
        id=[t["id"] if t is not None else "" for t in picks],
        headline=[t["title"] if t is not None else "" for t in picks],
        link=[t["link"] if t is not None else "" for t in picks],
    )


# ----------------------------------------------------------------------- fingerprints
def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(ds: Dataset) -> str:
    """Identifies the watchlist and the exact headlines (not the analysis settings)."""
    items = sorted((a.get("link", ""), a.get("published", "")) for a in ds.articles)
    return _digest({"watchlist": ds.watchlist.to_dict(), "fetched_at": ds.fetched_at.isoformat(), "items": items})


def fingerprint(ds: Dataset, p: Params) -> str:
    """Identifies the data and the settings, i.e. everything the AI inputs depend on."""
    params = {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(p).items()}
    return _digest({"dataset": dataset_fingerprint(ds), "params": params})


# ------------------------------------------------------------------------- AI inputs
def evidence_payload(c: Computed) -> list[dict]:
    return [
        {"ref": int(r["ref"]), "date": r["day"].isoformat(), "source": r["source"], "title": r["title"],
         "competitors": list(r["competitors"]), "events": list(r["events"]), "link": r["link"]}
        for _, r in c.evidence.iterrows()
    ]


def theme_payload(c: Computed, max_headlines: int = 6) -> list[dict]:
    """Per theme: computed stats plus its evidence headlines (and, if needed, other cited-by-number ones)."""
    out = []
    ev = c.evidence
    for _, t in c.themes.iterrows():
        sub = ev[ev["theme"] == t["theme"]].sort_values("published", ascending=False).head(max_headlines)
        out.append(
            {
                "theme": int(t["theme"]),
                "terms": list(t["terms"]),
                "size": int(t["size"]),
                "momentum": change(t["change"]),
                "headlines": [
                    {"ref": int(r["ref"]), "date": r["day"].isoformat(), "source": r["source"], "title": r["title"]}
                    for _, r in sub.iterrows()
                ],
            }
        )
    return out


def briefing_facts(ds: Dataset, p: Params, c: Computed) -> dict:
    arts = c.articles
    facts = {
        "watchlist": ds.watchlist.name,
        "period": f"{day_range(c.start, c.end)} ({p.days} days)",
        "data": f"{count(len(arts))} unique headlines from {count(arts['source'].nunique())} outlets "
        f"({ds.source}{', synthetic' if ds.synthetic else ''})",
        f"share of voice over {p.days} days": {
            r["competitor"]: f"{pct(r['share'])} ({count(r['mentions'])} headlines)" for _, r in c.sov.iterrows()
        },
        f"momentum: last {p.momentum_days} days vs the {p.momentum_days} before": {
            r["competitor"]: f"{change(r['change'])} headlines ({r['last']} vs {r['prior']}); "
            f"share of voice {points(r['sov_change'])}"
            for _, r in c.momentum.iterrows()
        },
        "spikes (unusually high daily coverage)": [
            f"{s['competitor']} on {day(s['date'])}: {s['count']} headlines vs a baseline of {s['baseline']:.1f} "
            f"(z = {s['z']:.1f})" + (f"; most-carried headline [{s['ref']}]" if s["ref"] else "")
            for _, s in c.spikes.head(6).iterrows()
        ]
        or ["none detected"],
        "rule-tagged events per competitor": {
            name: {e: int(v) for e, v in row.items() if v} for name, row in c.events.iterrows()
        },
    }
    return facts


def run_ai(
    llm: SupportsLLM,
    ds: Dataset,
    p: Params,
    c: Computed,
    on_step: Callable[[str], object] | None = None,
) -> dict:
    """Two model calls: theme notes (structured), then the briefing. Returns a JSON-safe dict."""
    step = on_step or (lambda _msg: None)
    themes = theme_payload(c)
    step(f"Theme analyst: naming and summarising {len(themes)} themes…")
    notes = describe_themes(llm, ds.watchlist.name, themes)
    names = {n.theme: n.name for n in notes.themes}
    for t in themes:
        t["name"] = names.get(t["theme"]) or " · ".join(t["terms"]) or f"Theme {t['theme']}"
    step("Briefing writer: drafting the executive briefing from the computed figures…")
    evidence = evidence_payload(c)
    briefing = write_briefing(llm, briefing_facts(ds, p, c), themes, evidence)
    return {
        "fingerprint": fingerprint(ds, p),
        "dataset_fingerprint": dataset_fingerprint(ds),
        "theme_notes": notes.model_dump()["themes"],
        "briefing": briefing,
        "references": [{k: e[k] for k in ("ref", "date", "source", "title", "link")} for e in evidence],
        "model": getattr(llm, "last_model", None),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_sentiment(llm: SupportsLLM, ds: Dataset, p: Params, c: Computed) -> dict:
    """Optional, live only: one model call that labels the tone of every evidence headline."""
    labels = judge_sentiment(llm, evidence_payload(c))
    return {"fingerprint": fingerprint(ds, p), "labels": labels, "model": getattr(llm, "last_model", None)}


def sentiment_table(c: Computed, labels: dict[int, str]) -> pd.DataFrame:
    """Model-judged labels counted per competitor; net = (positive − negative) / labelled headlines."""
    rows = []
    for name in c.names:
        sub = c.evidence[c.evidence["competitors"].map(lambda cs, n=name: n in cs)]
        tones = [labels.get(int(r)) for r in sub["ref"] if labels.get(int(r))]
        pos, neg = tones.count("positive"), tones.count("negative")
        rows.append({"competitor": name, "positive": pos, "neutral": tones.count("neutral"), "negative": neg,
                     "net": (pos - neg) / len(tones) if tones else 0.0})
    return pd.DataFrame(rows)
