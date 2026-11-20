"""Keyless news ingestion: Google News RSS search feeds plus any RSS or Atom URL.

Google News search feeds return at most ~100 items per query, so each query is split into short
date windows (`after:` / `before:` operators). A window that still returns the maximum is flagged
as *capped*: its counts are a lower bound. Raw responses are cached on disk to stay polite.

Only the title, outlet, link, publication time and the feed's short summary are kept, never the
article body.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

from .watchlist import Competitor, Watchlist

GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"
USER_AGENT = "Mozilla/5.0 (compatible; MarketPulse/0.1; RSS reader)"
DEFAULT_CACHE_DIR = Path(".cache") / "feeds"
CACHE_TTL_SECONDS = 6 * 3600
FEED_ITEM_CAP = 100
SUMMARY_MAX_CHARS = 280

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class FeedRequest:
    label: str  # competitor name, topic, or feed host
    kind: str  # "competitor" | "topic" | "rss"
    url: str
    window: str = "latest"


@dataclass(frozen=True)
class Dataset:
    """A set of raw headlines plus where and when they came from."""

    id: str
    watchlist: Watchlist
    articles: tuple[dict, ...]
    fetched_at: datetime
    days: int
    source: str = "Google News RSS"
    synthetic: bool = False
    fetch_log: tuple[dict, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------- URLs
def google_news_url(query: str, start: date | None = None, before: date | None = None) -> str:
    """Google News RSS search URL. `start` is inclusive (`after:`), `before` exclusive (`before:`)."""
    q = query.strip()
    if start is not None:
        q += f" after:{start.isoformat()}"
    if before is not None:
        q += f" before:{before.isoformat()}"
    return f"{GOOGLE_NEWS_SEARCH}?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"


def competitor_query(competitor: Competitor) -> str:
    """Exact-phrase OR query over the name and aliases: `"Nvidia" OR "NVDA"`."""
    return " OR ".join(f'"{term}"' for term in competitor.terms)


def date_windows(end: date, days: int, step: int) -> list[tuple[date, date]]:
    """Consecutive inclusive (first, last) day ranges covering the `days` days up to `end`."""
    days, step = max(1, int(days)), max(1, int(step))
    cur = end - timedelta(days=days - 1)
    out = []
    while cur <= end:
        last = min(cur + timedelta(days=step - 1), end)
        out.append((cur, last))
        cur = last + timedelta(days=1)
    return out


def plan_requests(watchlist: Watchlist, end: date, days: int, step: int) -> list[FeedRequest]:
    """One request per competitor and topic per date window, plus one per extra RSS feed."""
    reqs = []
    queries = [(c.name, "competitor", competitor_query(c)) for c in watchlist.competitors]
    queries += [(t, "topic", t) for t in watchlist.topics]
    for label, kind, query in queries:
        for first, last in date_windows(end, days, step):
            url = google_news_url(query, first, last + timedelta(days=1))
            reqs.append(FeedRequest(label, kind, url, f"{first:%d %b} – {last:%d %b}"))
    for url in watchlist.feeds:
        reqs.append(FeedRequest(urlsplit(url).netloc or url, "rss", url))
    return reqs


def canonical_link(url: str) -> str:
    """Drop tracking parameters and fragments so the same story URL compares equal."""
    parts = urlsplit((url or "").strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


# ----------------------------------------------------------------------------- text
def strip_html(text: str) -> str:
    text = _TAG.sub(" ", html.unescape(text or ""))
    return _SPACE.sub(" ", html.unescape(text).replace("\xa0", " ")).strip()


def split_source_suffix(title: str, source: str) -> str:
    """Google News appends ` - Outlet` to every title; remove it when it matches the outlet."""
    title = strip_html(title)
    if source:
        for sep in (" - ", " | ", " – "):
            suffix = f"{sep}{source}"
            if title.endswith(suffix):
                return title[: -len(suffix)].strip()
    return title


def clean_summary(summary: str, title: str, source: str) -> str:
    """Strip HTML and drop the summary when it only repeats the title and outlet (Google News does this)."""
    text = strip_html(summary)
    rest = text
    for piece in (title, source):
        if piece:
            rest = rest.replace(piece, " ")
    if len(_SPACE.sub(" ", rest).strip(" .-|–")) < 25:
        return ""
    if len(text) > SUMMARY_MAX_CHARS:
        text = text[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _utc(struct) -> datetime | None:
    if not struct:
        return None
    try:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_feed(content: bytes | str, request: FeedRequest) -> list[dict]:
    """Turn RSS/Atom content into headline records. Entries without a title, link or date are skipped."""
    import feedparser

    parsed = feedparser.parse(content)
    feed_title = strip_html(parsed.feed.get("title", "")) if request.kind == "rss" else ""
    records = []
    for entry in parsed.entries:
        link = entry.get("link", "")
        source_info = entry.get("source") or {}
        source = strip_html(source_info.get("title", "")) or feed_title or urlsplit(link).netloc.removeprefix("www.")
        title = split_source_suffix(entry.get("title", ""), source)
        published = _utc(entry.get("published_parsed") or entry.get("updated_parsed"))
        if not title or not link or published is None:
            continue
        records.append(
            {
                "title": title,
                "source": source or "Unknown outlet",
                "link": canonical_link(link),
                "published": published.isoformat().replace("+00:00", "Z"),
                "summary": clean_summary(entry.get("summary", ""), title, source),
                "queries": [request.label],
            }
        )
    return records


# ---------------------------------------------------------------------------- fetch
def fetch_url(url: str, *, cache_dir: Path | None = DEFAULT_CACHE_DIR, ttl: float = CACHE_TTL_SECONDS,
              timeout: float = 20.0) -> tuple[bytes, bool]:
    """GET a feed with a generic User-Agent. Returns (content, served_from_cache)."""
    path = cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.xml" if cache_dir else None
    if path and path.exists() and time.time() - path.stat().st_mtime < ttl:
        return path.read_bytes(), True
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.5"},
        timeout=timeout,
    )
    response.raise_for_status()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return response.content, False


Fetcher = Callable[[str], tuple[bytes, bool]]


def fetch_all(
    reqs: list[FeedRequest],
    fetch: Fetcher = fetch_url,
    *,
    max_workers: int = 4,
    delay: float = 0.0,
    on_progress: Callable[[int, int], object] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fetch every request; one failing feed never stops the others. Returns (records, fetch log)."""

    def one(req: FeedRequest) -> tuple[list[dict], dict]:
        row = {"query": req.label, "kind": req.kind, "window": req.window, "items": 0, "capped": False,
               "status": "ok", "url": req.url}
        try:
            content, cached = fetch(req.url)
            if delay and not cached:
                time.sleep(delay)
            records = parse_feed(content, req)
        except Exception as exc:  # network errors, HTTP errors, malformed XML
            row.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:120]}")
            return [], row
        row.update(items=len(records), capped=req.kind != "rss" and len(records) >= FEED_ITEM_CAP,
                   status="cached" if cached else "ok")
        return records, row

    records: list[dict] = []
    log: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for done, (recs, row) in enumerate(pool.map(one, reqs), start=1):
            records.extend(recs)
            log.append(row)
            if on_progress:
                on_progress(done, len(reqs))
    return merge_exact(records), log


def merge_exact(records: list[dict]) -> list[dict]:
    """Merge records with the same link (the same story returned by several queries)."""
    by_link: dict[str, dict] = {}
    for rec in records:
        key = rec["link"]
        if key not in by_link:
            by_link[key] = {**rec, "queries": list(rec.get("queries", []))}
            continue
        kept = by_link[key]
        for q in rec.get("queries", []):
            if q not in kept["queries"]:
                kept["queries"].append(q)
        if rec["published"] < kept["published"]:
            kept["published"] = rec["published"]
    return sorted(by_link.values(), key=lambda r: r["published"], reverse=True)


def fetch_watchlist(
    watchlist: Watchlist,
    *,
    days: int = 30,
    step: int = 7,
    now: datetime | None = None,
    fetch: Fetcher | None = None,
    max_workers: int = 4,
    delay: float = 0.0,
    on_progress: Callable[[int, int], object] | None = None,
) -> Dataset:
    """Fetch the last `days` days of headlines for a watchlist into a Dataset."""
    now = now or datetime.now(timezone.utc)
    reqs = plan_requests(watchlist, now.date(), days, step)
    records, log = fetch_all(reqs, fetch or fetch_url, max_workers=max_workers, delay=delay, on_progress=on_progress)
    digest = hashlib.sha1(now.isoformat().encode("utf-8")).hexdigest()[:8]
    return Dataset(
        id=f"live-{digest}",
        watchlist=watchlist,
        articles=tuple(records),
        fetched_at=now,
        days=days,
        source="Google News RSS" + (" + custom feeds" if watchlist.feeds else ""),
        fetch_log=tuple(log),
    )
