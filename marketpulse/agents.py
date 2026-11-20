"""The AI steps. The model never computes a number: it names and summarises themes from headlines,
writes the executive briefing from computed facts, and (optionally, live only) judges headline tone.
Every claim must cite the numbered evidence list as [n]; `citations.py` checks that afterwards."""

from __future__ import annotations

import json
from typing import Protocol

from .llm import clean_text
from .schemas import SentimentBatch, ThemeNotes


class SupportsLLM(Protocol):
    def complete_json(self, system: str, user: str, schema: type, **kwargs): ...
    def chat(self, system: str, user: str, **kwargs) -> str: ...


THEMES_SYSTEM = """You are a market-intelligence analyst. You name and summarise clusters of news headlines
for an executive audience. Use only the headlines provided and add no outside facts.
Cite headlines by their number in square brackets, for example [4] or [4, 9]."""

BRIEFING_SYSTEM = """You write competitive-intelligence briefings for executives.
Use Markdown with exactly these sections: ## Key moves, ## Competitor watch, ## Emerging themes,
## Risks and opportunities, ## What to watch.
Write short bullet points. Every bullet must cite at least one headline from the evidence list as [n].
Use only the evidence and the computed figures provided. Write figures exactly as given; never invent
numbers, events, quotes or sources, and do not speculate beyond what the headlines say.
300-450 words. Return the briefing only, without a title line."""

SENTIMENT_SYSTEM = """You label the tone of news headlines for the company each headline is mainly about.
positive = good news for that company (growth, wins, launches well received); negative = bad news
(losses, lawsuits, recalls, cuts, setbacks); neutral = factual or mixed. Judge the headline text only."""


def evidence_lines(evidence: list[dict]) -> str:
    """'[n] YYYY-MM-DD · Outlet · Title (competitors; events)' lines for the prompt."""
    lines = []
    for item in evidence:
        tags = []
        if item.get("competitors"):
            tags.append("mentions " + ", ".join(item["competitors"]))
        if item.get("events"):
            tags.append("rule-tagged " + ", ".join(item["events"]))
        suffix = f" ({'; '.join(tags)})" if tags else ""
        lines.append(f"[{item['ref']}] {item['date']} · {item['source']} · {item['title']}{suffix}")
    return "\n".join(lines)


def describe_themes(llm: SupportsLLM, watchlist_name: str, themes: list[dict]) -> ThemeNotes:
    blocks = []
    for t in themes:
        heads = "\n".join(f"  [{h['ref']}] {h['date']} · {h['source']} · {h['title']}" for h in t["headlines"])
        blocks.append(
            f"Theme {t['theme']}: {t['size']} headlines, {t['momentum']} week on week; "
            f"distinguishing terms: {', '.join(t['terms']) or 'none'}\n{heads}"
        )
    user = (
        f"Watchlist: {watchlist_name}\n\nThemes found by clustering recent headlines:\n\n"
        + "\n\n".join(blocks)
        + "\n\nFor every theme return: theme (its number), name (3-6 words, specific), summary (two sentences "
        "on what is happening, citing headlines as [n]) and implication (one sentence on why it matters for "
        "the companies on the watchlist, citing [n])."
    )
    return llm.complete_json(THEMES_SYSTEM, user, ThemeNotes, max_tokens=4000)


def write_briefing(llm: SupportsLLM, facts: dict, themes: list[dict], evidence: list[dict]) -> str:
    theme_text = "\n".join(
        f"- Theme {t['theme']} “{t['name']}”: {t['size']} headlines, {t['momentum']} week on week" for t in themes
    )
    user = (
        "Computed figures (authoritative, use exactly as written):\n"
        + json.dumps(facts, indent=2, ensure_ascii=False)
        + "\n\nThemes:\n"
        + theme_text
        + "\n\nEvidence list (cite as [n]):\n"
        + evidence_lines(evidence)
    )
    return clean_text(llm.chat(BRIEFING_SYSTEM, user, temperature=0.3, max_tokens=2200))


def judge_sentiment(llm: SupportsLLM, evidence: list[dict]) -> dict[int, str]:
    user = (
        "Label every headline below. Reply with items: a list of {ref, sentiment}.\n\n" + evidence_lines(evidence)
    )
    batch = llm.complete_json(SENTIMENT_SYSTEM, user, SentimentBatch, max_tokens=3000)
    valid = {item["ref"] for item in evidence}
    return {s.ref: s.sentiment for s in batch.items if s.ref in valid}
