"""Deterministic citation check for the AI briefing.

The briefing must cite the numbered evidence list as [n]. This module splits the Markdown into
sentences, extracts every citation (`[3]`, `[3, 7]`, `[3][7]`, `[3-5]`), verifies that each number
exists in the evidence list, and flags factual sentences that cite nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE = re.compile(r"\[(\d+(?:\s*(?:,|–|-)\s*\d+)*)\]")
_SENT_END = re.compile(r"[.!?](?:\s*\[[\d\s,–-]+\])*(?=\s+[\"“(\[A-Z0-9]|\s*$)")
_ABBREV = {"inc", "corp", "co", "ltd", "vs", "st", "mr", "ms", "dr", "no", "jan", "feb", "mar", "apr", "jun", "jul",
           "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "approx", "est"}
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
MIN_WORDS = 4  # shorter fragments ("None this week.") are not treated as claims


def parse_refs(group: str) -> list[int]:
    """'3, 5-7' → [3, 5, 6, 7]. Ranges longer than 20 are kept as their endpoints only."""
    refs: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        bounds = re.split(r"\s*[–-]\s*", part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            lo, hi = int(bounds[0]), int(bounds[1])
            refs.extend(range(lo, hi + 1) if 0 <= hi - lo <= 20 else [lo, hi])
        elif part.isdigit():
            refs.append(int(part))
    return refs


def citations_in(text: str) -> list[int]:
    return [n for group in _CITE.findall(text) for n in parse_refs(group)]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping trailing citations with their sentence and not splitting
    after abbreviations such as 'U.S.' or 'Inc.'."""
    out, start = [], 0
    for match in _SENT_END.finditer(text):
        before = text[start : match.start()].split()
        last_word = before[-1].lower().strip("([\"“") if before else ""
        if re.fullmatch(r"(?:[a-z]\.)*[a-z]", last_word) or last_word in _ABBREV:
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            out.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


@dataclass
class SentenceCheck:
    section: str
    text: str
    refs: list[int]
    invalid: list[int]


@dataclass
class CitationReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    n_sources: int = 0

    @property
    def claims(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if len(s.text.split()) >= MIN_WORDS]

    @property
    def uncited(self) -> list[SentenceCheck]:
        return [s for s in self.claims if not s.refs]

    @property
    def invalid_refs(self) -> list[int]:
        return sorted({n for s in self.sentences for n in s.invalid})

    @property
    def cited_refs(self) -> list[int]:
        return sorted({n for s in self.sentences for n in s.refs if n not in s.invalid})

    @property
    def coverage(self) -> float:
        claims = self.claims
        return (len(claims) - len(self.uncited)) / len(claims) if claims else 0.0

    @property
    def passed(self) -> bool:
        return bool(self.claims) and not self.invalid_refs and not self.uncited


def check_citations(markdown: str, n_sources: int) -> CitationReport:
    """Check every sentence of a Markdown briefing against an evidence list numbered 1..n_sources."""
    report = CitationReport(n_sources=n_sources)
    section = ""
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        line = _BULLET.sub("", line).replace("**", "")
        for sentence in split_sentences(line):
            refs = citations_in(sentence)
            invalid = sorted({n for n in refs if not 1 <= n <= n_sources})
            report.sentences.append(SentenceCheck(section, sentence, refs, invalid))
    return report


def link_citations(markdown: str, links: dict[int, str]) -> str:
    """Turn each valid [n] into a Markdown link to the source; unknown numbers are left as text."""

    def repl(match: re.Match) -> str:
        refs = parse_refs(match.group(1))
        if not refs or any(n not in links for n in refs):
            return match.group(0)
        return " ".join(f"[[{n}]]({links[n]})" for n in refs)

    return _CITE.sub(repl, markdown or "")
