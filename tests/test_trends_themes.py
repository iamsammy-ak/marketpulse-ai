from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from marketpulse.themes import fit_themes, stopwords_for
from marketpulse.trends import TOTAL, daily_counts, detect_spikes, momentum, pct_change, share_of_voice, weekly_share

START = date(2026, 11, 1)


def _articles(rows):
    """rows: (day offset, [competitors])"""
    return pd.DataFrame({
        "published": pd.to_datetime([f"{START + timedelta(days=d)}T12:00:00Z" for d, _ in rows], utc=True),
        "competitors": [cs for _, cs in rows],
    })


def _series(counts, name="A"):
    days = pd.date_range(START, periods=len(counts), freq="D")
    return pd.DataFrame({name: counts}, index=days)


def test_daily_counts_fill_missing_days_and_count_each_mention():
    df = _articles([(0, ["A"]), (0, ["A", "B"]), (2, ["B"]), (2, []), (9, ["A"])])  # day 9 is outside
    daily = daily_counts(df, ["A", "B"], START, START + timedelta(days=3))
    assert list(daily.index.date) == [START + timedelta(days=i) for i in range(4)]
    assert daily["A"].tolist() == [2, 0, 0, 0]
    assert daily["B"].tolist() == [1, 0, 1, 0]
    assert daily[TOTAL].tolist() == [2, 0, 2, 0]  # headlines, not mentions


def test_share_of_voice_sums_to_one_and_survives_silence():
    daily = pd.DataFrame({"A": [3, 1], "B": [1, 0], TOTAL: [4, 1]}, index=pd.date_range(START, periods=2))
    sov = share_of_voice(daily, ["A", "B"])
    assert sov["competitor"].tolist() == ["A", "B"]
    assert sov["share"].sum() == pytest.approx(1.0)
    assert sov.loc[0, "share"] == pytest.approx(0.8)
    quiet = share_of_voice(daily * 0, ["A", "B"])
    assert quiet["share"].tolist() == [0.0, 0.0]
    longer = pd.DataFrame({"A": [1] * 10, "B": [0] * 7 + [1] * 3}, index=pd.date_range(START, periods=10))
    weekly = weekly_share(longer, ["A", "B"])  # only complete 7-day blocks, counted back from the last day
    assert len(weekly) == 1 and weekly.loc[0, "week_end"] == pd.Timestamp(START + timedelta(days=9))
    assert weekly.loc[0, "A"] + weekly.loc[0, "B"] == pytest.approx(1.0)


def test_momentum_compares_last_week_with_the_week_before():
    a = [1] * 7 + [3] * 7
    b = [2] * 7 + [2] * 7
    daily = pd.DataFrame({"A": a, "B": b}, index=pd.date_range(START, periods=14))
    mom = momentum(daily, ["A", "B"], START + timedelta(days=13), 7).set_index("competitor")
    assert (mom.loc["A", "last"], mom.loc["A", "prior"]) == (21, 7)
    assert mom.loc["A", "change"] == pytest.approx(2.0)
    assert mom.loc["A", "sov_change"] == pytest.approx(21 / 35 - 7 / 21)
    assert mom.index[0] == "A"  # sorted by share-of-voice gain
    assert pct_change(5, 0) is None and pct_change(0, 0) == 0.0 and pct_change(1, 4) == pytest.approx(-0.75)


def test_spike_detection_flags_bursts_not_noise():
    rng = np.random.default_rng(1)
    counts = rng.poisson(4, 30).tolist()
    counts[24] = 20
    spikes = detect_spikes(_series(counts), ["A"], z_threshold=3.0)
    assert spikes["date"].tolist() == [START + timedelta(days=24)]
    assert spikes.loc[0, "baseline"] < 6 and spikes.loc[0, "z"] > 3
    assert detect_spikes(_series([4] * 30), ["A"]).empty  # flat series
    assert detect_spikes(_series([0] * 29 + [2]), ["A"]).empty  # below the minimum count
    assert detect_spikes(_series([30] + [0] * 29), ["A"]).empty  # no baseline yet on day one


TOPICS = {
    "battery": "battery cells factory lithium supply plant output",
    "drone": "drone delivery flights airspace pilot program suburbs",
    "union": "union workers strike wages contract vote warehouse",
}


def _corpus(n_per_topic=24, seed=3):
    rng = np.random.default_rng(seed)
    texts, truth = [], []
    for topic, words in TOPICS.items():
        vocab = words.split()
        for _ in range(n_per_topic):
            texts.append("Acme " + " ".join(rng.choice(vocab, size=5, replace=True)))
            truth.append(topic)
    return texts, truth


def test_themes_recover_planted_topics_and_name_them():
    texts, truth = _corpus()
    model = fit_themes(texts, stopwords_for(["Acme"]), n_themes=3)
    assert model.k == 3 and not model.auto
    for topic in TOPICS:  # every planted topic lands (almost) entirely in one theme
        labels = model.labels[[i for i, t in enumerate(truth) if t == topic]]
        assert np.bincount(labels).max() / len(labels) >= 0.9
    all_terms = " ".join(t for terms in model.terms.values() for t in terms)
    assert "acme" not in all_terms  # watchlist names never label a theme
    auto = fit_themes(texts, stopwords_for(["Acme"]))
    assert auto.auto and auto.k == int(auto.silhouette.loc[auto.silhouette["score"].idxmax(), "k"])
    again = fit_themes(texts, stopwords_for(["Acme"]))
    assert (again.labels == auto.labels).all()  # seeded, so reproducible


def test_themes_degrade_gracefully_on_tiny_or_empty_input():
    small = fit_themes(["Acme wins", "Acme loses"], set())
    assert small.k == 1 and small.labels.tolist() == [0, 0]
    assert fit_themes([], set()).labels.size == 0
