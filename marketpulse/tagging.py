"""Transparent, rule-based tagging: which competitors a headline mentions and what kind of event it reports.

Competitors are matched on whole words in the title and summary, ignoring case except that the
first letter must be a capital (a proper-noun signal, so the adjective "lucid" is not Lucid).
Aliases of three characters or fewer, or written in capitals ("BYD", "NVDA"), must match exactly.
Phrases listed as exclusions ("Lucid Capital") are blanked out before matching.

Event types come from keyword rules, listed in `EVENT_RULES` and shown in the app, so a reader can
always see why a headline was tagged. Keyword syntax: words match whole words, case-insensitively;
a trailing `*` matches any ending ("launch*" → launches, launched); `$#` matches a money amount
("$2.5 billion"); `#` matches a number; an all-capital keyword ("IPO") matches case-sensitively.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from .watchlist import Competitor

EVENT_RULES: dict[str, tuple[str, ...]] = {
    "Launch": (
        "launch*", "unveil*", "introduces", "introduced", "debut*", "rolls out", "rolled out", "rolling out",
        "now available", "goes on sale", "new model", "releases",
    ),
    "Funding": (
        "raises $#", "raised $#", "raising $#", "funding round", "funding", "series a", "series b", "series c",
        "series d", "valuation", "IPO", "venture capital", "invest* $#",
    ),
    "M&A": (
        "acquir*", "acquisition*", "merger*", "merge with", "merges with", "takeover*", "buyout*", "agrees to buy",
        "offers to buy", "deal to buy", "plans to buy", "bid to buy", "offer* $# for", "bid* for", "divest*",
        "spin off", "spinoff",
    ),
    "Partnership": (
        "partner*", "teams up", "team up", "collaborat*", "alliance", "joint venture", "signs deal",
        "signs agreement", "inks deal", "supply deal", "supply agreement", "multi-year deal",
    ),
    "Layoffs": (
        "layoff*", "lay off", "lays off", "laid off", "laying off", "job cut*", "cut* jobs", "cut* # jobs",
        "cuts # workers", "workforce reduction", "redundanc*", "restructur*", "hiring freeze",
    ),
    "Regulation": (
        "regulat*", "antitrust", "tariff*", "export control*", "export curb*", "export ban*", "export rule*",
        "ban", "bans", "banned", "sanction*", "FTC", "NHTSA", "European Commission", "lawmaker*", "subsid*",
        "probe*", "investigation*", "recall*", "Commerce Department", "Justice Department", "White House",
    ),
    "Earnings": (
        "earnings", "quarterly result*", "results", "Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter",
        "third quarter", "fourth quarter", "revenue*", "profit*", "guidance", "forecast*", "beats estimates",
        "misses estimates", "quarterly deliveries", "record deliveries", "deliveries rise", "deliveries fall",
        "deliveries drop", "deliveries jump", "sales rise", "sales fall", "sales jump", "sales drop",
    ),
    "Legal": (
        "lawsuit*", "sue", "sues", "sued", "suing", "court", "courts", "judge*", "settle*", "verdict*",
        "patent*", "class action", "litigation", "fined", "fines", "penalty", "penalties", "jury", "DOJ",
    ),
}
EVENT_TYPES = tuple(EVENT_RULES)


def keyword_regex(keyword: str) -> str:
    """Translate one rule keyword into a whole-word regular expression."""
    parts = []
    for word in keyword.split():
        if word == "$#":
            parts.append(r"\$\s?\d[\d.,]*(?:\s?(?:billion|million|bn|mn|[bmk]))?")
        elif word == "#":
            parts.append(r"\d[\d.,]*")
        elif word.endswith("*"):
            parts.append(re.escape(word[:-1]) + r"\w*")
        else:
            parts.append(re.escape(word))
    head = r"(?<!\w)" if keyword[0].isalnum() else ""
    return head + r"[\s-]+".join(parts) + r"(?!\w)"


@lru_cache(maxsize=1)
def _event_patterns() -> dict[str, list[re.Pattern]]:
    """One alternation per event for case-insensitive keywords, one for all-capital keywords."""
    out: dict[str, list[re.Pattern]] = {}
    for event, keywords in EVENT_RULES.items():
        loose = [keyword_regex(k) for k in keywords if not k.isupper()]
        exact = [keyword_regex(k) for k in keywords if k.isupper()]
        out[event] = [re.compile("|".join(loose), re.IGNORECASE)] if loose else []
        if exact:
            out[event].append(re.compile("|".join(exact)))
    return out


def classify_events(text: str) -> dict[str, str]:
    """{event type: the words that triggered it} for every event whose rules match."""
    hits: dict[str, str] = {}
    for event, patterns in _event_patterns().items():
        found = [m for p in patterns if (m := p.search(text or ""))]
        if found:
            hits[event] = min(found, key=lambda m: m.start()).group(0)
    return hits


def _alias_pattern(term: str) -> tuple[re.Pattern, bool]:
    """(pattern, needs_capital). Short or all-capital aliases match exactly."""
    exact = len(term) <= 3 or term.isupper()
    body = r"\s+".join(re.escape(w) for w in term.split())
    pattern = re.compile(rf"(?<!\w){body}(?!\w)", 0 if exact else re.IGNORECASE)
    return pattern, not exact and term[:1].isupper()


class EntityMatcher:
    """Finds which watchlist competitors a piece of text mentions, by name or alias."""

    def __init__(self, competitors: tuple[Competitor, ...] | list[Competitor]) -> None:
        self.rules = [
            (
                c.name,
                [_alias_pattern(t) for t in c.terms],
                [re.compile(rf"(?<!\w){re.escape(x)}(?!\w)", re.IGNORECASE) for x in c.exclude],
            )
            for c in competitors
        ]

    @staticmethod
    def _first(text: str, pattern: re.Pattern, needs_capital: bool) -> int | None:
        for match in pattern.finditer(text):
            if not needs_capital or match.group(0)[:1].isupper():
                return match.start()
        return None

    def find(self, text: str) -> list[str]:
        found = []
        for name, patterns, excludes in self.rules:
            subject = text or ""
            for ex in excludes:
                subject = ex.sub(lambda m: " " * len(m.group(0)), subject)
            positions = [pos for p, cap in patterns if (pos := self._first(subject, p, cap)) is not None]
            if positions:
                found.append((min(positions), name))
        return [name for _, name in sorted(found)]


def tag_articles(df: pd.DataFrame, competitors: tuple[Competitor, ...] | list[Competitor]) -> pd.DataFrame:
    """Add `competitors` (in order of first mention), `events` and `event_hits` columns."""
    matcher = EntityMatcher(competitors)
    texts = (df["title"] + ". " + df["summary"].fillna("")).tolist() if len(df) else []
    events = [classify_events(t) for t in texts]
    return df.assign(
        competitors=[matcher.find(t) for t in texts],
        events=[list(e) for e in events],
        event_hits=[", ".join(f"{k}: “{v}”" for k, v in e.items()) for e in events],
    )


def rules_table() -> list[list[str]]:
    """Rows for the methods table: event type and its keywords."""
    return [[event, ", ".join(keywords)] for event, keywords in EVENT_RULES.items()]
