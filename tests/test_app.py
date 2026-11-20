"""End-to-end UI tests: run the real Streamlit script headlessly (no browser, no network, no model calls)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from marketpulse import ingest
from marketpulse.demo import load_snapshots

TABS = ["📰 Feed", "🏁 Competitors", "🧩 Themes", "📈 Trends", "🧠 Briefing", "🔗 Sources"]


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")


def _app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.secrets["OPENROUTER_API_KEY"] = ""  # never pick up a key from the developer's machine
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_renders_metrics_tabs_and_ai_setup_step():
    at = _app()
    assert [m.label for m in at.metric][:6] == ["Headlines", "Outlets", "Leader", "Top riser", "Themes", "Spikes"]
    assert [t.label for t in at.tabs] == TABS
    infos = " ".join(i.value for i in at.info)
    assert "AI analysis runs once an OpenRouter key is configured" in infos
    assert "not available" not in infos
    assert any("Snapshot of Google News RSS taken" in c.value for c in at.caption)


def test_every_sample_and_the_custom_watchlist_render():
    at = _app()
    for ds in load_snapshots():
        at.selectbox(key="sample_choice").set_value(ds.watchlist.name).run()
        assert not at.exception, (ds.watchlist.name, at.exception)
        assert _metric(at, "Leader") in ds.watchlist.names
    at.selectbox(key="sample_choice").set_value("Custom watchlist").run()
    assert not at.exception and not at.metric
    assert any("No headlines yet" in i.value for i in at.info)


def test_settings_and_feed_filters_update_the_analysis():
    at = _app()
    at.select_slider(key="n_themes").set_value(5).run()
    assert _metric(at, "Themes") == "5"
    spikes_before = int(_metric(at, "Spikes"))
    at.slider(key="spike_z").set_value(4.0).run()
    assert int(_metric(at, "Spikes")) <= spikes_before
    at.multiselect(key="f_comp").set_value([at.multiselect(key="f_comp").options[0]])
    at.text_input(key="f_q").set_value("zzzz-no-such-word").run()
    assert not at.exception
    assert any(c.value.startswith("0 matching headlines") for c in at.caption)


def test_custom_watchlist_fetches_and_analyses_without_network(monkeypatch):
    now = datetime.now(timezone.utc)
    rng = np.random.default_rng(5)
    words = ["drone", "pilot", "launch", "union", "strike", "wages", "battery", "factory", "recall", "funding"]

    def fake_fetch(url):
        items = []
        for i in range(30):
            who = "Acme" if i % 3 else "Globex"
            title = f"{who} {' '.join(rng.choice(words, 4))} update {i}"
            when = (now - timedelta(days=int(rng.integers(1, 20)))).strftime("%a, %d %b %Y %H:%M:%S GMT")
            items.append(f"<item><title>{title} - Outlet {i % 4}</title><link>https://n.example/{abs(hash(url))}/{i}"
                         f"</link><pubDate>{when}</pubDate><source url='x'>Outlet {i % 4}</source></item>")
        return f"<rss version='2.0'><channel><title>t</title>{''.join(items)}</channel></rss>".encode(), False

    monkeypatch.setattr(ingest, "fetch_url", fake_fetch)
    at = _app()
    at.selectbox(key="sample_choice").set_value("Custom watchlist").run()
    at.text_area(key="wl_competitors").set_value("Acme\nGlobex").run()
    next(b for b in at.button if b.label.startswith("📥")).click().run()
    assert not at.exception, at.exception
    assert at.radio(key="source").value == "Live Google News"
    assert int(_metric(at, "Headlines").replace(",", "")) > 50
    assert _metric(at, "Leader") == "Acme"
