"""What to monitor: competitors (with aliases), industry topics and extra RSS feeds.

The app edits a watchlist as plain text, one competitor per line; a leading minus marks a phrase
that must not count as a mention (it is blanked out before matching):

    Nvidia: NVDA
    Lucid: Lucid Motors, Lucid Group, -Lucid Capital
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Competitor:
    name: str
    aliases: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def terms(self) -> tuple[str, ...]:
        """The name followed by every alias, without duplicates (case-insensitive)."""
        seen: set[str] = set()
        out = []
        for term in (self.name, *self.aliases):
            if term and term.lower() not in seen:
                seen.add(term.lower())
                out.append(term)
        return tuple(out)


@dataclass(frozen=True)
class Watchlist:
    name: str
    competitors: tuple[Competitor, ...] = ()
    topics: tuple[str, ...] = ()
    feeds: tuple[str, ...] = field(default_factory=tuple)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.competitors]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "competitors": [
                {"name": c.name, "aliases": list(c.aliases), **({"exclude": list(c.exclude)} if c.exclude else {})}
                for c in self.competitors
            ],
            "topics": list(self.topics),
            "feeds": list(self.feeds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Watchlist:
        return cls(
            name=str(data.get("name", "")).strip() or "Untitled watchlist",
            competitors=tuple(
                Competitor(str(c["name"]).strip(), _clean(c.get("aliases", [])), _clean(c.get("exclude", [])))
                for c in data.get("competitors", [])
                if str(c.get("name", "")).strip()
            ),
            topics=tuple(t.strip() for t in data.get("topics", []) if t.strip()),
            feeds=tuple(f.strip() for f in data.get("feeds", []) if f.strip()),
        )


def _clean(items) -> tuple[str, ...]:
    return tuple(str(i).strip() for i in items or [] if str(i).strip())


def parse_competitors(text: str) -> tuple[Competitor, ...]:
    """Parse `Name: alias, alias, -excluded phrase` lines. Blank lines and duplicate names are ignored."""
    out: list[Competitor] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        name, _, rest = line.partition(":")
        name = name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        items = [a.strip() for a in rest.split(",") if a.strip()]
        aliases = tuple(a for a in items if not a.startswith("-"))
        exclude = tuple(a[1:].strip() for a in items if a.startswith("-") and a[1:].strip())
        out.append(Competitor(name, aliases, exclude))
    return tuple(out)


def format_competitors(competitors: tuple[Competitor, ...] | list[Competitor]) -> str:
    lines = []
    for c in competitors:
        items = [*c.aliases, *(f"-{x}" for x in c.exclude)]
        lines.append(f"{c.name}: {', '.join(items)}" if items else c.name)
    return "\n".join(lines)


def parse_lines(text: str) -> tuple[str, ...]:
    """One entry per line, stripped, de-duplicated, order kept."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line and line not in out:
            out.append(line)
    return tuple(out)
