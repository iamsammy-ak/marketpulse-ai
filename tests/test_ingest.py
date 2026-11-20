"""Feed ingestion: URL building, windowing, parsing and fetching. No network: fetches are faked."""

import zlib
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from marketpulse import ingest
from marketpulse.ingest import (
    FeedRequest,
    canonical_link,
    competitor_query,
    date_windows,
    fetch_url,
    fetch_watchlist,
    google_news_url,
    merge_exact,
    parse_feed,
    plan_requests,
)
from marketpulse.watchlist import Competitor, Watchlist

FIXTURE = (Path(__file__).parent / "fixtures" / "google_news_sample.xml").read_bytes()


def test_google_news_url_and_competitor_query():
    url = google_news_url('"Acme" OR "ACME"', date(2026, 11, 1), date(2026, 11, 3))
    q = parse_qs(urlsplit(url).query)
    assert q["q"] == ['"Acme" OR "ACME" after:2026-11-01 before:2026-11-03']
    assert q["hl"] == ["en-US"] and q["ceid"] == ["US:en"]
    assert competitor_query(Competitor("Lucid", ("Lucid Motors", "lucid"), ("Lucid Capital",))) == (
        '"Lucid" OR "Lucid Motors"'  # duplicate alias dropped, exclusions never searched
    )


def test_date_windows_cover_the_period_without_gaps():
    windows = date_windows(date(2026, 11, 19), days=30, step=7)
    assert windows[0][0] == date(2026, 10, 21) and windows[-1][1] == date(2026, 11, 19)
    for (_, last), (first, _) in zip(windows, windows[1:], strict=False):
        assert (first - last).days == 1
    assert sum((b - a).days + 1 for a, b in windows) == 30
    wl = Watchlist("x", (Competitor("A"), Competitor("B")), ("topic",), ("https://feeds.example/rss",))
    reqs = plan_requests(wl, date(2026, 11, 19), 30, 7)
    assert len(reqs) == 3 * len(windows) + 1
    assert reqs[-1].kind == "rss" and reqs[-1].label == "feeds.example"


def test_parse_feed_strips_html_suffixes_and_skips_undated():
    records = parse_feed(FIXTURE, FeedRequest("Acme", "competitor", "u"))
    assert len(records) == 2  # the undated item is skipped
    first, second = records
    assert first["title"] == "Acme unveils solar roof tiles"
    assert first["source"] == "Daily Planet"
    assert first["summary"] == ""  # Google's summary only repeats title and outlet
    assert first["published"] == "2026-11-16T17:38:07Z"
    assert second["published"] == "2026-11-17T08:00:00Z"  # +0100 converted to UTC
    assert second["link"] == "https://example.com/story?id=7"  # tracking parameter dropped
    assert "battery supply" in second["summary"] and "<" not in second["summary"]
    assert first["queries"] == ["Acme"]


def test_canonical_link_and_exact_merge_keep_earliest_and_all_queries():
    assert canonical_link("https://EXAMPLE.com/a?utm_medium=x&id=1#top") == "https://example.com/a?id=1"
    merged = merge_exact([
        {"link": "l1", "published": "2026-11-02T00:00:00Z", "queries": ["A"]},
        {"link": "l1", "published": "2026-11-01T00:00:00Z", "queries": ["B"]},
        {"link": "l2", "published": "2026-11-03T00:00:00Z", "queries": ["A"]},
    ])
    assert [m["link"] for m in merged] == ["l2", "l1"]  # newest first
    assert merged[1]["queries"] == ["A", "B"] and merged[1]["published"].startswith("2026-11-01")


def test_fetch_url_caches_on_disk(tmp_path, monkeypatch):
    calls = []

    class Response:
        content = FIXTURE

        def raise_for_status(self):
            pass

    def fake_get(url, headers, timeout):
        calls.append(headers["User-Agent"])
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    assert fetch_url("https://feeds.example/a", cache_dir=tmp_path) == (FIXTURE, False)
    assert fetch_url("https://feeds.example/a", cache_dir=tmp_path) == (FIXTURE, True)
    assert len(calls) == 1 and "MarketPulse" in calls[0]


def test_fetch_watchlist_logs_capped_windows_and_survives_failures():
    item = "<item><title>Acme story {i} - Outlet</title><link>https://n.example/{w}/{i}</link>" \
           "<pubDate>Mon, 16 Nov 2026 10:00:00 GMT</pubDate><source url='x'>Outlet</source></item>"

    def fake_fetch(url):
        if "Broken" in url:
            raise ConnectionError("offline")
        n = 100 if "Acme" in url else 3
        body = "".join(item.format(i=i, w=zlib.crc32(url.encode())) for i in range(n))
        return f"<rss version='2.0'><channel><title>t</title>{body}</channel></rss>".encode(), False

    wl = Watchlist("w", (Competitor("Acme"), Competitor("Broken")), ("solar",))
    ds = fetch_watchlist(wl, days=14, step=7, now=datetime(2026, 11, 20, 8, tzinfo=timezone.utc), fetch=fake_fetch)
    log = list(ds.fetch_log)
    assert len(log) == 3 * 2
    assert all(r["capped"] for r in log if r["query"] == "Acme")
    assert all(r["status"] == "error" for r in log if r["query"] == "Broken")
    assert not any(r["capped"] for r in log if r["query"] == "solar")
    assert len(ds.articles) == 2 * 100 + 2 * 3
    assert ingest.FEED_ITEM_CAP == 100
