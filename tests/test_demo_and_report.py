from datetime import timezone

from marketpulse.demo import ANALYSES_DIR, load_saved_analysis, load_snapshots
from marketpulse.pipeline import Params, compute, fingerprint
from marketpulse.report import build_report
from marketpulse.synthetic import generate_dataset

STORED_FIELDS = {"title", "source", "link", "published", "summary", "queries"}


def test_bundled_snapshots_are_valid_and_store_metadata_only():
    snapshots = load_snapshots()
    assert len(snapshots) >= 2
    for ds in snapshots:
        assert ds.fetched_at.tzinfo == timezone.utc and ds.days == 30
        assert len(ds.watchlist.competitors) >= 4 and ds.watchlist.topics
        assert len(ds.articles) > 500 and not ds.synthetic
        sample = ds.articles[:: max(1, len(ds.articles) // 50)]
        for article in sample:
            assert set(article) == STORED_FIELDS  # never article bodies
            assert article["link"].startswith("http") and article["published"].endswith("Z")
            assert len(article["summary"]) <= 300
        assert ds.fetch_log and all(row["status"] in {"ok", "cached", "error"} for row in ds.fetch_log)


def test_saved_analyses_belong_to_a_snapshot():
    fingerprints = {fingerprint(ds, Params()) for ds in load_snapshots()}
    for path in ANALYSES_DIR.glob("*.json"):
        assert path.stem in fingerprints, f"{path.name} does not belong to any snapshot (stale fixture?)"
        assert {"briefing", "theme_notes", "references", "model"} <= set(load_saved_analysis(path.stem))
    assert load_saved_analysis("0000000000000000") is None


def test_report_without_ai_has_every_computed_section():
    ds = generate_dataset()
    c = compute(ds, Params())
    report = build_report(ds, Params(), c, None)
    for heading in ("Share of voice and momentum", "## Spikes", "## Themes", "Rule-tagged events", "## References"):
        assert heading in report
    assert "synthetic" in report and "computed figures only" in report
    assert c.sov.iloc[0]["competitor"] in report
    assert report.count("<https://example.com/synthetic/") == len(c.evidence)
    analysis = {"model": "fake/model", "briefing": "## Key moves\n- Brightcell agreed to buy Gridwise [1].",
                "theme_notes": [{"theme": 0, "name": "Named theme", "summary": "S."}]}
    with_ai = build_report(ds, Params(), c, analysis)
    assert "Executive briefing (AI-written)" in with_ai and "Named theme" in with_ai and "1 of 1" in with_ai
