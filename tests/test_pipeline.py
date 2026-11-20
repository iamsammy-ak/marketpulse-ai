"""End-to-end deterministic analysis on the seeded synthetic dataset, whose signals are known."""

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from marketpulse.pipeline import Params, analysis_window, compute, dataset_fingerprint, fingerprint
from marketpulse.synthetic import SPIKE_DAY, generate_dataset


@pytest.fixture(scope="module")
def synthetic():
    ds = generate_dataset()
    return ds, compute(ds, Params())


def test_synthetic_generator_is_seeded_and_labelled():
    a, b = generate_dataset(), generate_dataset()
    assert a.articles == b.articles and a.synthetic
    assert generate_dataset(seed=8).articles != a.articles
    assert all(r["link"].startswith("https://example.com/synthetic/") for r in a.articles)


def test_pipeline_recovers_the_planted_signals(synthetic):
    ds, c = synthetic
    first_day = analysis_window(ds, 30)[0]
    planted = date.fromordinal(first_day.toordinal() + SPIKE_DAY)
    top = c.spikes.iloc[0]
    assert top["date"] == planted and top["competitor"] in {"Gridwise", "Brightcell"}
    assert "Gridwise" in top["headline"] and top["ref"] > 0  # the spike's headline is citable
    assert c.momentum.iloc[0]["competitor"] == "Volt Harbor"  # the planted riser leads share-of-voice gains
    assert c.raw_count > len(c.articles)  # wire copies were merged
    wire = c.articles[c.articles["outlets"].map(lambda o: any(x.endswith("Wire") for x in o))]
    assert wire["copies"].max() >= 10
    assert c.sov["share"].sum() == pytest.approx(1.0)
    assert set(c.events.columns) >= {"Launch", "M&A", "Layoffs"}
    assert 4 <= len(c.themes) <= 12 and c.themes["size"].sum() <= len(c.articles)


def test_evidence_list_is_numbered_and_covers_every_competitor(synthetic):
    ds, c = synthetic
    ev = c.evidence
    assert ev["ref"].tolist() == list(range(1, len(ev) + 1)) and len(ev) <= Params().evidence_size
    assert ev["published"].is_monotonic_decreasing
    for name in ds.watchlist.names:
        assert ev["competitors"].map(lambda cs, n=name: n in cs).any(), name
    assert set(c.articles.loc[c.articles["ref"] > 0, "id"]) == set(ev["id"])


def test_window_ends_on_the_last_complete_day_and_respects_data():
    ds = generate_dataset()
    start, end = analysis_window(ds, 30)
    assert end == date(2026, 11, 19)  # fetched 20 Nov 06:00 UTC, so 20 Nov is incomplete
    assert (end - start).days == 29
    late = replace(ds, articles=tuple(a for a in ds.articles if a["published"] >= "2026-11-10"))
    assert analysis_window(late, 30)[0] == date(2026, 11, 10)


def test_fingerprints_track_data_and_settings():
    ds = generate_dataset()
    base = fingerprint(ds, Params())
    assert base == fingerprint(ds, Params(dedupe_threshold=0.75000000001))
    assert base != fingerprint(ds, Params(spike_z=3.0))
    assert dataset_fingerprint(ds) == dataset_fingerprint(replace(ds, fetch_log=({"x": 1},)))
    later = replace(ds, fetched_at=datetime(2026, 11, 21, tzinfo=timezone.utc))
    assert dataset_fingerprint(later) != dataset_fingerprint(ds)
