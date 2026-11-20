"""MarketPulse AI: monitor competitor and industry news with computed trends and a cited AI briefing."""

from __future__ import annotations

import html
import re
from dataclasses import replace

import pandas as pd
import streamlit as st

from marketpulse import charts
from marketpulse.citations import check_citations, link_citations
from marketpulse.config import load_settings
from marketpulse.demo import load_saved_analysis, load_snapshots
from marketpulse.formatting import change, count, day, day_range, pct, points
from marketpulse.ingest import Dataset, date_windows, fetch_watchlist
from marketpulse.llm import LLMClient, LLMError
from marketpulse.pipeline import (
    Computed,
    Params,
    compute,
    dataset_fingerprint,
    fingerprint,
    run_ai,
    run_sentiment,
    sentiment_table,
)
from marketpulse.report import build_report
from marketpulse.tagging import EVENT_TYPES, rules_table
from marketpulse.ui import (
    card,
    hero,
    inject_css,
    link,
    muted,
    nowrap,
    pill,
    ref,
    share_bar,
    signed,
    table,
    tags,
)
from marketpulse.watchlist import Watchlist, format_competitors, parse_competitors, parse_lines

st.set_page_config(page_title="MarketPulse AI", page_icon="📡", layout="wide")
inject_css()


@st.cache_resource(show_spinner=False)
def _snapshots() -> list[Dataset]:
    return load_snapshots()


@st.cache_data(show_spinner="Analysing headlines…", max_entries=16)
def analyse(key: str, _ds: Dataset, _p: Params) -> Computed:
    """`key` is the fingerprint of the data and settings, so the cache never mixes datasets."""
    return compute(_ds, _p)


SNAPSHOTS = _snapshots()
SAMPLE_NAMES = [s.watchlist.name for s in SNAPSHOTS]
CUSTOM = "Custom watchlist"
SNAPSHOT, LIVE = "Bundled snapshot", "Live Google News"
THEME_OPTIONS = ["Auto", *range(4, 13)]
PLOT_CONFIG = {"displayModeBar": False}
ss = st.session_state


# ----------------------------------------------------------------------------- state
def _load_choice() -> None:
    choice = ss["sample_choice"]
    if choice == CUSTOM:
        wl = Watchlist(name="My watchlist")
        ss["source"] = LIVE
    else:
        wl = SNAPSHOTS[SAMPLE_NAMES.index(choice)].watchlist
        ss["source"] = SNAPSHOT
    ss["wl_name"] = wl.name
    ss["wl_competitors"] = format_competitors(wl.competitors)
    ss["wl_topics"] = "\n".join(wl.topics)
    ss["wl_feeds"] = "\n".join(wl.feeds)


if "sample_choice" not in ss:
    ss["sample_choice"] = SAMPLE_NAMES[0] if SAMPLE_NAMES else CUSTOM
    ss.update(days=30, dedupe=0.75, n_themes="Auto", spike_z=2.5, step=7)
    _load_choice()
if "pending_source" in ss:  # set by the fetch button, applied before the radio is drawn
    ss["source"] = ss.pop("pending_source")


def current_watchlist() -> Watchlist:
    return Watchlist(
        name=ss.wl_name.strip() or "Untitled watchlist",
        competitors=parse_competitors(ss.wl_competitors),
        topics=parse_lines(ss.wl_topics),
        feeds=parse_lines(ss.wl_feeds),
    )


def search_key(wl: Watchlist) -> tuple:
    """What a fetch depends on: renaming or adding exclusions does not need a new fetch."""
    return tuple((c.name, c.aliases) for c in wl.competitors), wl.topics, wl.feeds


def cite_html(text: str, links: dict[int, str]) -> str:
    """Escape model text and turn valid [n] citations into links."""
    escaped = html.escape(text or "")

    def repl(match: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", match.group(1))]
        if nums and all(n in links for n in nums):
            return " ".join(f'<a href="{html.escape(links[n], quote=True)}" target="_blank">[{n}]</a>' for n in nums)
        return match.group(0)

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", repl, escaped)


def ai_setup_note(outputs: list[str]) -> None:
    """Demo mode: the AI step is a setup step, so say how to enable it and what it will add."""
    st.info(
        "**AI analysis runs once an OpenRouter key is configured:** add `OPENROUTER_API_KEY` (see README → "
        "Configuration), then click **✨ Write AI briefing**.\n\nFor this watchlist it will produce:\n"
        + "\n".join(f"- {item}" for item in outputs)
    )


# --------------------------------------------------------------------------- sidebar
settings = load_settings()
is_custom = ss.sample_choice == CUSTOM
with st.sidebar:
    st.markdown("## 📡 MarketPulse AI")
    if settings.live:
        fallbacks = len(settings.models) - 1
        st.success(f"**Live AI** via OpenRouter\n\n`{settings.models[0]}`\n\n+{fallbacks} fallback models")
    else:
        st.info(
            "**Demo mode**: every chart and table is computed from the headlines.\n\n"
            "Add `OPENROUTER_API_KEY` to enable the AI briefing (README → Configuration)."
        )
    st.selectbox("Watchlist", SAMPLE_NAMES + [CUSTOM], key="sample_choice", on_change=_load_choice)
    snap_date = SNAPSHOTS[SAMPLE_NAMES.index(ss.sample_choice)].fetched_at if not is_custom else None
    st.radio(
        "Headlines from",
        [SNAPSHOT, LIVE],
        key="source",
        disabled=is_custom,
        captions=[f"Saved {snap_date:%d %b %Y}" if snap_date else "Not available for a custom list",
                  "Fetch now: keyless, cached 6 h"],
    )

    st.markdown("### Analysis settings")
    st.select_slider("Look-back window (days)", options=[14, 21, 30], key="days")
    st.slider("Near-duplicate threshold", 0.5, 0.95, step=0.05, key="dedupe",
              help="TF-IDF cosine similarity above which two headlines that mention the same competitors and "
                   "appeared within 3 days count as one story.")
    st.select_slider("Themes", options=THEME_OPTIONS, key="n_themes",
                     help="Auto picks the number of themes (4-12) with the best silhouette score.")
    st.slider("Spike threshold (z-score)", 2.0, 4.0, step=0.5, key="spike_z",
              help="How unusual a day must be against the competitor's previous 14 days.")
    with st.expander("Live fetch"):
        st.select_slider("Days per query window", options=[2, 3, 7], key="step",
                         help="Google News returns at most 100 items per query. Shorter windows return more "
                              "headlines but send more requests.")
        wl_preview = current_watchlist()
        n_queries = len(wl_preview.competitors) + len(wl_preview.topics)
        n_requests = n_queries * len(date_windows(pd.Timestamp.now("UTC").date(), ss.days, ss.step)) + len(
            wl_preview.feeds)
        st.caption(f"{n_requests} requests per fetch, two at a time, cached for 6 hours.")

# ------------------------------------------------------------------------------ main
hero(
    "MarketPulse AI",
    "Market intelligence from keyless news feeds: share of voice, momentum, spikes and emerging themes "
    "computed from the headlines, plus an executive briefing in which every claim cites its source.",
    ["Google News RSS", "Share of voice", "Spike detection", "Theme clustering", "Cited AI briefing"],
)

with st.expander("🎯 Watchlist", expanded=is_custom):
    st.text_input("Name", key="wl_name")
    left, right = st.columns([1.35, 1])
    left.text_area(
        "Competitors, one per line: `Name: alias, alias, -excluded phrase`",
        key="wl_competitors",
        height=158,
        placeholder="DoorDash\nUber Eats\nInstacart: Maplebear",
        help="Names match whole words and must start with a capital; short or all-capital aliases match exactly. "
             "A phrase after a minus is ignored when matching (e.g. -Lucid Capital).",
    )
    right.text_area("Industry topics, one search per line", key="wl_topics", height=72,
                    placeholder="food delivery")
    right.text_area("Extra RSS or Atom feeds (URLs)", key="wl_feeds", height=68,
                    placeholder="https://example.com/feed.xml")

wl = current_watchlist()
has_queries = bool(wl.competitors or wl.topics or wl.feeds)

btn_fetch, btn_ai, note_col = st.columns([1, 1, 2.2], vertical_alignment="center")
if btn_fetch.button("📥 Fetch latest headlines", width="stretch", disabled=not has_queries):
    with st.status("Fetching Google News RSS feeds…", expanded=True) as status:
        bar = st.progress(0.0)
        live_ds = fetch_watchlist(
            wl, days=ss.days, step=ss.step, max_workers=2,
            on_progress=lambda done, total: bar.progress(done / total, text=f"{done} of {total} feeds"),
        )
        ok = sum(1 for r in live_ds.fetch_log if r["status"] != "error")
        status.update(label=f"Fetched {len(live_ds.articles):,} headlines from {ok} of {len(live_ds.fetch_log)} feeds",
                      state="complete", expanded=False)
    ss["live_ds"], ss["live_key"], ss["pending_source"] = live_ds, search_key(wl), LIVE
    st.rerun()

ds: Dataset | None = None
if ss.source == SNAPSHOT and not is_custom:
    ds = replace(SNAPSHOTS[SAMPLE_NAMES.index(ss.sample_choice)], watchlist=wl)  # edits re-tag saved headlines
elif ss.get("live_ds") is not None:
    ds = replace(ss["live_ds"], watchlist=wl)

if ds is None:
    btn_ai.button("✨ Write AI briefing", type="primary", disabled=True, width="stretch")
    note_col.caption("Fetch headlines first. Fetching uses public RSS feeds and needs no API key.")
    st.info("**No headlines yet.** Add competitors (and optionally topics or RSS feeds) in the watchlist above, then "
            "press **Fetch latest headlines**. Or pick a sample watchlist in the sidebar to explore a saved snapshot.")
    st.stop()

p = Params(days=ss.days, dedupe_threshold=ss.dedupe, n_themes=0 if ss.n_themes == "Auto" else int(ss.n_themes),
           spike_z=ss.spike_z)
fp, dfp = fingerprint(ds, p), dataset_fingerprint(ds)
c = analyse(fp, ds, p)
arts = c.articles
names = c.names
colors = charts.colors_for(names)

# ------------------------------------------------------------------------ AI analysis
can_run = settings.live and len(arts) > 0 and bool(names)
if btn_ai.button("✨ Write AI briefing", type="primary", disabled=not can_run, width="stretch"):
    with st.status("Running the AI analysts…", expanded=True) as status:
        try:
            llm = LLMClient(settings, app_name="MarketPulse AI")
            result = run_ai(llm, ds, p, c, on_step=status.write)
            ss["analysis"] = result
            status.update(label=f"Briefing ready · {result['model']}", state="complete", expanded=False)
        except LLMError as exc:
            status.update(label="The AI briefing failed", state="error")
            st.error(str(exc))

analysis, freshness = None, None
current = ss.get("analysis")
if current and current["fingerprint"] == fp:
    analysis, freshness = current, "fresh"
elif (saved := load_saved_analysis(fp)) is not None:
    analysis, freshness = saved, "saved"
elif current and current["dataset_fingerprint"] == dfp:
    analysis, freshness = current, "settings_changed"

if freshness == "saved":
    note_col.caption(f"Showing the saved AI analysis for this sample (`{analysis['model']}`, "
                     f"{analysis['generated_at'][:10]}). No model calls used.")
elif freshness == "fresh":
    note_col.caption(f"AI briefing by `{analysis['model']}`. Repeat runs are served from the cache.")
elif freshness == "settings_changed":
    note_col.caption("⚠️ Settings changed after the AI briefing: figures are current, re-run to refresh the text.")
elif settings.live:
    note_col.caption("Two calls to free OpenRouter models, usually 30-90 seconds. Results are cached.")
else:
    note_col.caption("Demo mode: every figure below is computed. The AI briefing runs once an OpenRouter key is "
                     "configured.")

theme_notes = {n["theme"]: n for n in analysis.get("theme_notes", [])} if freshness in ("fresh", "saved") else {}
theme_names = {k: v["name"] for k, v in theme_notes.items() if v.get("name")}
ref_links = {int(r["ref"]): r["link"] for r in analysis.get("references", [])} if analysis else {}

if ds.synthetic:
    st.caption(f"🧪 **Synthetic snapshot**: fictional companies from a seeded generator · window "
               f"{day_range(c.start, c.end)}")
elif ss.source == SNAPSHOT:
    st.caption(f"📦 **Snapshot of Google News RSS taken {ds.fetched_at:%d %b %Y, %H:%M} UTC**, not live · "
               f"window {day_range(c.start, c.end)} (complete days) · press Fetch for today's headlines")
else:
    stale = ss.get("live_key") != search_key(wl)
    st.caption(f"🛰️ **Live fetch at {ds.fetched_at:%d %b %Y, %H:%M} UTC** · window {day_range(c.start, c.end)}"
               + (" · ⚠️ the watchlist's searches changed since this fetch: fetch again" if stale else ""))

if arts.empty or not names:
    st.warning("No headlines in this window match the watchlist. Check the names, widen the window or fetch again.")
    st.stop()

# ------------------------------------------------------------------------- headline
leader = c.sov.iloc[0]
riser = c.momentum.iloc[0]
kpis = st.columns(6)
kpis[0].metric("Headlines", count(len(arts)), f"{c.raw_count - len(arts):,} merged", delta_color="off",
               delta_arrow="off", help=f"Unique headlines in the window after merging near-duplicates "
               f"({c.raw_count:,} before merging).")
top10 = c.outlets["headlines"].head(10).sum() / max(len(arts), 1)
kpis[1].metric("Outlets", count(arts["source"].nunique()), f"top 10: {pct(top10)}", delta_color="off",
               delta_arrow="off", help="Distinct outlets; the delta is the share of headlines from the ten largest.")
kpis[2].metric("Leader", charts.short(leader["competitor"], 12), f"{pct(leader['share'])} share",
               delta_color="off", delta_arrow="off", help="Largest share of competitor mentions in the window.")
kpis[3].metric("Top riser", charts.short(riser["competitor"], 12) if riser["sov_change"] > 0 else "—",
               points(riser["sov_change"]) if riser["sov_change"] > 0 else None,
               help=f"Largest share-of-voice gain: last {p.momentum_days} days vs the {p.momentum_days} before.")
kpis[4].metric("Themes", str(len(c.themes)), "silhouette" if c.k_auto else "fixed", delta_color="off",
               delta_arrow="off", help="Clusters of related headlines found with TF-IDF and k-means.")
kpis[5].metric("Spikes", str(len(c.spikes)), f"z ≥ {p.spike_z:g}", delta_color="off", delta_arrow="off",
               help="Days with unusually high coverage of a competitor against its own trailing baseline.")
summary = f"{leader['competitor']} leads share of voice ({pct(leader['share'])})"
if riser["sov_change"] > 0:
    summary += (f"; {riser['competitor']} gained the most in the last {p.momentum_days} days "
                f"({points(riser['sov_change'])})")
if len(c.spikes):
    top_spike = c.spikes.iloc[0]
    summary += f"; biggest spike: {top_spike['competitor']} on {day(top_spike['date'])}"
pill(summary)
st.write("")

tabs = st.tabs(["📰 Feed", "🏁 Competitors", "🧩 Themes", "📈 Trends", "🧠 Briefing", "🔗 Sources"])


def theme_title(t: int, label: str) -> str:
    return f"{t + 1}. {theme_names.get(t, label)}"


def headline_cell(r: pd.Series) -> str:
    extra = f" · carried by {len(r['outlets'])} outlets" if len(r["outlets"]) > 1 else ""
    return f"{link(r['title'], r['link'])} {ref(int(r['ref']))}<br>{muted(r['source'] + extra)}"


# ------------------------------------------------------------------------------ feed
with tabs[0]:
    theme_opts = ["All themes"] + [theme_title(int(t), lab) for t, lab in zip(c.themes["theme"], c.themes["label"],
                                                                                 strict=True)]
    if ss.get("f_theme") not in theme_opts:
        ss["f_theme"] = "All themes"
    ss["f_comp"] = [x for x in ss.get("f_comp", []) if x in names]
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.3, 1.3])
    sel_comp = f1.multiselect("Competitors", names, key="f_comp")
    sel_event = f2.multiselect("Event type (rule-based)", list(EVENT_TYPES), key="f_event")
    sel_theme = f3.selectbox("Theme", theme_opts, key="f_theme")
    query = f4.text_input("Search headlines", key="f_q", placeholder="e.g. drone")
    view = arts
    if sel_comp:
        view = view[view["competitors"].map(lambda cs: any(x in cs for x in sel_comp))]
    if sel_event:
        view = view[view["events"].map(lambda es: any(e in es for e in sel_event))]
    if sel_theme != "All themes":
        view = view[view["theme"] == theme_opts.index(sel_theme) - 1]
    if query.strip():
        view = view[view["title"].str.contains(re.escape(query.strip()), case=False)]
    shown = view.head(50)
    st.caption(f"{len(view):,} matching headlines, newest first" + (" · showing the first 50" if len(view) > 50 else "")
               + ". Numbers in red are citation references used by the briefing.")
    theme_label = dict(zip(c.themes["theme"], c.themes["label"], strict=True)) if len(c.themes) else {}
    rows = [
        [
            nowrap(day(r["day"])),
            headline_cell(r),
            tags(list(r["competitors"]), colors),
            tags(list(r["events"])),
            muted(charts.short(theme_title(int(r["theme"]), theme_label.get(r["theme"], "")), 30))
            if r["theme"] >= 0 else muted("—"),
        ]
        for _, r in shown.iterrows()
    ]
    table(["Date", "Headline", "Competitors", "Events", "Theme"], rows, html_cols=(0, 1, 2, 3, 4))
    export = view[["day", "title", "source", "link", "competitors", "events", "theme", "ref"]].assign(
        competitors=view["competitors"].map("; ".join), events=view["events"].map("; ".join),
        theme=view["theme"] + 1)
    st.download_button("Download matching headlines (CSV)", export.to_csv(index=False),
                       file_name=f"{wl.name}-headlines.csv", mime="text/csv")

# ----------------------------------------------------------------------- competitors
with tabs[1]:
    left, right = st.columns(2)
    left.plotly_chart(charts.sov_bar(c.sov, colors, p.days), width="stretch", config=PLOT_CONFIG)
    right.plotly_chart(charts.momentum_bar(c.momentum, p.momentum_days), width="stretch", config=PLOT_CONFIG)
    mom = c.momentum.set_index("competitor")
    rows = []
    for _, r in c.sov.iterrows():
        name = r["competitor"]
        m = mom.loc[name]
        top_events = [e for e, v in c.events.loc[name].sort_values(ascending=False).head(2).items() if v]
        rows.append([
            tags([name], colors), f"{r['mentions']:,}", share_bar(r["share"]), int(m["last"]), int(m["prior"]),
            signed(change(m["change"]), m["change"]), signed(points(m["sov_change"]), m["sov_change"]),
            int((c.spikes["competitor"] == name).sum()), tags(top_events),
        ])
    table(["Competitor", "Headlines", "Share of voice", f"Last {p.momentum_days}d", f"Prior {p.momentum_days}d",
           "Change", "Share Δ", "Spikes", "Top events"], rows, html_cols=(0, 2, 5, 6, 8))
    st.caption("Share of voice = a competitor's share of all competitor mentions, so it is not distorted by the feed "
               "returning fewer items for older dates. Raw changes are shown for context.")

    st.markdown("##### Event mix by competitor")
    st.caption("Rule-based: keyword rules (listed under 🔗 Sources → Methods) tag each headline; one headline can "
               "carry several events.")
    st.plotly_chart(charts.event_heatmap(c.events), width="stretch", config=PLOT_CONFIG)

    st.markdown("##### Most-carried stories per competitor")
    rows = []
    for name in names:
        sub = arts[arts["competitors"].map(lambda cs, n=name: n in cs)]
        for _, r in sub.sort_values(["copies", "published"], ascending=False).head(2).iterrows():
            rows.append([tags([name], colors), headline_cell(r), nowrap(day(r["day"]))])
    table(["Competitor", "Headline", "Date"], rows, html_cols=(0, 1, 2))

    st.markdown("##### Headline tone")
    sent = ss.get("sentiment")
    if sent and sent["fingerprint"] == fp:
        st.caption(f"Model-judged by `{sent['model']}`: labels for the {len(sent['labels'])} evidence headlines. "
                   "Counts and the net score are computed in Python.")
        st.plotly_chart(charts.sentiment_bars(sentiment_table(c, sent["labels"])), width="stretch", config=PLOT_CONFIG)
    elif settings.live:
        if st.button("🙂 Judge headline tone (one model call)"):
            with st.spinner("Labelling headline tone…"):
                try:
                    ss["sentiment"] = run_sentiment(LLMClient(settings, app_name="MarketPulse AI"), ds, p, c)
                    st.rerun()
                except LLMError as exc:
                    st.error(str(exc))
    else:
        st.caption("Model-judged tone runs once an OpenRouter key is configured: a button here then labels each "
                   "evidence headline as positive, neutral or negative for its company, and the counts per "
                   "competitor are computed in Python.")

# ---------------------------------------------------------------------------- themes
with tabs[2]:
    if c.themes.empty or len(c.themes) < 2:
        st.info("Too few headlines with distinctive words to form themes. Widen the window or add topics.")
    else:
        left, right = st.columns([1.35, 1])
        left.plotly_chart(charts.theme_bars(c.themes, theme_names), width="stretch", config=PLOT_CONFIG)
        if len(c.silhouette):
            right.plotly_chart(charts.silhouette_chart(c.silhouette, len(c.themes), c.k_auto), width="stretch",
                               config=PLOT_CONFIG)
        right.caption("Headlines → TF-IDF (company names and boilerplate removed) → 8-dimension LSA → k-means for "
                      "k = 4-12; the highest cosine silhouette wins. Labels are each theme's most distinctive terms.")
        st.markdown("##### Which competitors each theme involves")
        st.plotly_chart(charts.theme_competitor_heatmap(arts, c.themes, names, theme_names), width="stretch",
                        config=PLOT_CONFIG)
        if not theme_notes and not settings.live:
            ai_setup_note([
                f"a short name for each of the {len(c.themes)} themes (the titles below are the computed top terms)",
                "a two-sentence summary per theme and a line on why it matters, citing headlines as [n]",
            ])
        elif not theme_notes:
            st.info("Press **Write AI briefing** to add a short name, a summary and an implication to each theme.")
        cols = st.columns(2)
        for i, (_, t) in enumerate(c.themes.iterrows()):
            if i and i % 2 == 0:  # a new row per pair keeps the cards aligned
                cols = st.columns(2)
            tid = int(t["theme"])
            note = theme_notes.get(tid, {})
            sub = arts[arts["theme"] == tid].sort_values(["copies", "published"], ascending=False).head(3)
            body = ""
            if note.get("summary"):
                body += f"<p>{cite_html(note['summary'], ref_links)}</p>"
            if note.get("implication"):
                body += f"<p><b>Why it matters:</b> {cite_html(note['implication'], ref_links)}</p>"
            body += f"<div>{tags(list(t['competitors']), colors)}</div>"
            body += "<ul>" + "".join(f"<li>{link(r['title'], r['link'])} {muted('· ' + r['source'])}</li>"
                                     for _, r in sub.iterrows()) + "</ul>"
            meta = (f"{t['size']:,} headlines · {pct(t['share'])} of clustered · {change(t['change'])} week on week"
                    + (f" · terms: {html.escape(', '.join(t['terms']))}" if note else ""))
            with cols[i % 2]:
                card(theme_title(tid, t["label"]), meta, body)

# ---------------------------------------------------------------------------- trends
with tabs[3]:
    st.plotly_chart(charts.daily_lines(c.daily, names, colors, c.spikes), width="stretch", config=PLOT_CONFIG)
    left, right = st.columns(2)
    if len(c.weekly):
        left.plotly_chart(charts.weekly_share_area(c.weekly, names, colors), width="stretch", config=PLOT_CONFIG)
    right.plotly_chart(charts.event_volume(arts, c.start, c.end), width="stretch", config=PLOT_CONFIG)
    st.markdown("##### Spikes")
    if c.spikes.empty:
        st.caption(f"No day reached z ≥ {p.spike_z:g}. Lower the threshold in the sidebar to see smaller bursts.")
    else:
        rows = [
            [nowrap(day(s["date"])), tags([s["competitor"]], colors), s["count"], f"{s['baseline']:.1f}",
             f'<span class="mp-num">{s["z"]:.1f}</span>',
             (f"{link(s['headline'], s['link'])} {ref(int(s['ref']))}" if s["headline"] else "—")]
            for _, s in c.spikes.iterrows()
        ]
        table(["Date", "Competitor", "Headlines", "Baseline", "z", "Most-carried headline that day"], rows,
              html_cols=(0, 1, 4, 5))
    st.caption("z = (headlines that day − mean of the previous 14 days) ÷ spread, with the spread floored at √mean "
               "and at 1 so quiet competitors do not trigger alarms. Google News returns fewer items for older "
               "dates, so raw counts drift upward toward the present; compare competitors by share of voice.")

# -------------------------------------------------------------------------- briefing
with tabs[4]:
    briefing = analysis.get("briefing") if analysis else None
    if briefing:
        check = check_citations(briefing, len(ref_links))
        if freshness == "settings_changed":
            st.warning("This briefing was written before you changed the settings. Its citations still point to "
                       "the headlines it was given; re-run to refresh it.")
        m = st.columns(4)
        m[0].metric("Sentences checked", len(check.claims))
        m[1].metric("Cite a source", f"{len(check.claims) - len(check.uncited)}")
        m[2].metric("Coverage", pct(check.coverage))
        m[3].metric("Invalid refs", len(check.invalid_refs))
        verdict = ("Citation check passed: every sentence cites a listed headline" if check.passed else
                   f"Citation check: {len(check.uncited)} uncited sentence(s), "
                   f"{len(check.invalid_refs)} invalid reference(s)")
        pill(verdict, good=check.passed)
        st.write("")
        with st.container(border=True):
            st.markdown(link_citations(re.sub(r"^#{1,3} ", "#### ", briefing, flags=re.M), ref_links))
        if check.uncited or check.invalid_refs:
            st.markdown("##### Flagged by the citation check")
            flagged = [[s.section, s.text, "no citation" if not s.refs else
                        "unknown " + ", ".join(f"[{n}]" for n in s.invalid)]
                       for s in check.sentences if (not s.refs and s in check.claims) or s.invalid]
            table(["Section", "Sentence", "Problem"], flagged)
    elif settings.live:
        st.info("Press **Write AI briefing** for an executive briefing (Key moves, Competitor watch, Emerging "
                "themes, Risks and opportunities, What to watch). Every sentence must cite the numbered evidence "
                "list, and a deterministic check verifies the citations.")
    else:
        ai_setup_note([
            "an executive briefing in five sections: Key moves, Competitor watch, Emerging themes, Risks and "
            "opportunities, What to watch",
            f"citations to the {len(c.evidence)} numbered headlines in 🔗 Sources, written from the computed figures "
            "above (the model does not calculate any number)",
            "a deterministic citation check: every [n] must exist and sentences without a citation are flagged",
        ])
        st.caption("The report below already includes every computed figure and the numbered references.")
    report_md = build_report(ds, p, c, analysis if freshness in ("fresh", "saved") else None)
    st.download_button("⬇️ Download briefing report (Markdown)", report_md,
                       file_name=f"{wl.name}-marketpulse-briefing.md", mime="text/markdown", type="primary")

# --------------------------------------------------------------------------- sources
with tabs[5]:
    st.markdown("##### Evidence list")
    st.caption(f"The {len(c.evidence)} headlines the briefing may cite as [n], chosen deterministically: the "
               "headline behind each top spike, the best three per competitor and two per theme, then the widest "
               "reach. Links go to the publisher via Google News.")
    rows = [[ref(int(r["ref"])), nowrap(day(r["day"])), f"{link(r['title'], r['link'])}<br>{muted(r['source'])}",
             tags(list(r["competitors"]), colors), muted(r["event_hits"]) if r["event_hits"] else muted("—")]
            for _, r in c.evidence.iterrows()]
    table(["Ref", "Date", "Headline", "Competitors", "Event rule matched"], rows, html_cols=(0, 1, 2, 3, 4))

    left, right = st.columns([1.2, 1])
    left.plotly_chart(charts.outlets_bar(c.outlets), width="stretch", config=PLOT_CONFIG)
    log = pd.DataFrame(list(ds.fetch_log))
    with right:
        st.markdown("##### Provenance")
        lines = [
            f"- **Source:** {ds.source}" + (" (synthetic data, fictional companies)" if ds.synthetic else ""),
            f"- **Fetched:** {ds.fetched_at:%d %b %Y, %H:%M} UTC; analysis window {day_range(c.start, c.end)}",
            f"- **Headlines:** {c.raw_count:,} in the window, {c.raw_count - len(arts):,} near-duplicates merged, "
            f"{len(arts):,} unique from {arts['source'].nunique():,} outlets",
            f"- **Mentions:** {(arts['competitors'].map(len) > 0).mean():.0%} of headlines name a competitor; "
            f"{(arts['events'].map(len) > 0).mean():.0%} carry a rule-tagged event",
        ]
        if len(log):
            lines.append(f"- **Requests:** {len(log)} feed requests; {c.capped_windows} windows hit the 100-item cap "
                         f"(counts there are lower bounds); {c.failed_requests} failed")
        st.markdown("\n".join(lines))
        st.caption("Only the title, outlet, link, date and the feed's short summary are stored; article text is "
                   "never fetched.")
    if len(log):
        with st.expander("Fetch log by query"):
            per_query = log.groupby(["query", "kind"], sort=False).agg(
                windows=("window", "size"), headlines=("items", "sum"), capped=("capped", "sum"),
                failed=("status", lambda s: int((s == "error").sum()))).reset_index()
            table(["Query", "Type", "Windows", "Headlines", "Capped", "Failed"],
                  per_query[["query", "kind", "windows", "headlines", "capped", "failed"]].values.tolist())
    with st.expander("Methods"):
        st.markdown(
            "**Event types are rule-based.** A headline gets an event type when its title or summary contains one of "
            "these keywords (`*` = any ending, `$#` = an amount of money, `#` = a number):"
        )
        table(["Event", "Keywords"], rules_table())
        st.markdown(
            "**Competitor mentions** match the name or an alias as a whole word, ignoring case except that the first "
            "letter must be a capital; aliases of three letters or fewer, or written in capitals, match exactly.  \n"
            "**Near-duplicates**: TF-IDF cosine ≥ the threshold, same competitors, within 3 days; the earliest "
            "headline is kept.  \n"
            "**Momentum** compares the last 7 days with the 7 before; **spikes** use a 14-day trailing z-score."
        )

st.divider()
st.caption(
    "MarketPulse AI is a monitoring aid, not investment advice. Figures are computed from headline metadata; "
    "event tags are keyword rules and AI-written text can be wrong, so check the linked sources."
)
