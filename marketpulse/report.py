"""Downloadable Markdown briefing: computed figures, themes, spikes and (when available) the AI briefing
with its citation check and numbered references."""

from __future__ import annotations

from datetime import date

from .citations import check_citations
from .formatting import change, count, day, day_range, pct, points
from .ingest import Dataset
from .pipeline import Computed, Params


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def build_report(ds: Dataset, p: Params, c: Computed, analysis: dict | None) -> str:
    arts = c.articles
    notes = {n["theme"]: n for n in (analysis or {}).get("theme_notes", [])}
    lines = [
        f"# {ds.watchlist.name}: MarketPulse briefing",
        "",
        f"_Generated {date.today().isoformat()} · {day_range(c.start, c.end)} · {ds.source}"
        + (" (synthetic, fictional companies)" if ds.synthetic else f", fetched {ds.fetched_at:%Y-%m-%d %H:%M} UTC")
        + (f" · AI text by `{analysis['model']}`" if analysis else " · computed figures only, no AI text")
        + "_",
        "",
        f"**{count(len(arts))} unique headlines** from {count(arts['source'].nunique())} outlets "
        f"({count(c.raw_count - len(arts))} near-duplicates merged).",
        "",
        "## Share of voice and momentum",
        "",
        f"| Competitor | Headlines | Share of voice | Last {p.momentum_days} days | Prior {p.momentum_days} | "
        "Change | Share change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    mom = c.momentum.set_index("competitor")
    for _, r in c.sov.iterrows():
        m = mom.loc[r["competitor"]]
        lines.append(
            f"| {r['competitor']} | {r['mentions']:,} | {pct(r['share'])} | {m['last']} | {m['prior']} | "
            f"{change(m['change'])} | {points(m['sov_change'])} |"
        )
    lines += ["", "## Spikes", ""]
    if c.spikes.empty:
        lines.append(f"No day reached a z-score of {p.spike_z:g}.")
    else:
        lines += ["| Date | Competitor | Headlines | Baseline | z | Most-carried headline |",
                  "|---|---|---:|---:|---:|---|"]
        for _, s in c.spikes.iterrows():
            lines.append(f"| {day(s['date'])} | {s['competitor']} | {s['count']} | {s['baseline']:.1f} | "
                         f"{s['z']:.1f} | {_cell(s['headline'])} |")
    lines += ["", "## Themes", "", "| # | Theme | Headlines | Week on week | Distinguishing terms |",
              "|---:|---|---:|---:|---|"]
    for _, t in c.themes.iterrows():
        name = notes.get(int(t["theme"]), {}).get("name") or t["label"]
        lines.append(f"| {int(t['theme']) + 1} | {_cell(name)} | {t['size']:,} | {change(t['change'])} | "
                     f"{_cell(', '.join(t['terms']))} |")
    for _, t in c.themes.iterrows():
        note = notes.get(int(t["theme"]))
        if note and note.get("summary"):
            lines += ["", f"**{int(t['theme']) + 1}. {note['name']}**: {note['summary']} {note.get('implication', '')}"]
    lines += ["", "## Rule-tagged events", "", "| Competitor | " + " | ".join(c.events.columns) + " |",
              "|---|" + "---:|" * len(c.events.columns)]
    for name, row in c.events.iterrows():
        lines.append(f"| {name} | " + " | ".join(str(int(v)) for v in row.values) + " |")

    if analysis and analysis.get("briefing"):
        check = check_citations(analysis["briefing"], len(c.evidence))
        lines += [
            "",
            "## Executive briefing (AI-written)",
            "",
            analysis["briefing"].strip(),
            "",
            f"_Citation check: {len(check.claims) - len(check.uncited)} of {len(check.claims)} sentences cite a "
            f"source; invalid references: {', '.join(map(str, check.invalid_refs)) or 'none'}._",
        ]
    lines += ["", "## References", ""]
    for _, r in c.evidence.iterrows():
        lines.append(f"{r['ref']}. {_cell(r['title'])}. {r['source']}, {day(r['day'])}. <{r['link']}>")
    lines += [
        "",
        "---",
        "_Figures are computed from headline metadata (title, outlet, date), not full articles. Google News "
        "returns at most 100 items per query window and fewer items for older dates, so counts are samples. "
        "Event tags are keyword rules; AI-written text can be wrong and should be checked against the sources._",
    ]
    return "\n".join(lines)
