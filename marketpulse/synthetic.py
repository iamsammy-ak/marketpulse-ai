"""Seeded synthetic snapshot with fictional companies, used only when live feeds cannot be fetched.

Everything here is invented, and datasets built from it are labelled synthetic in the app.
Assumptions, chosen to exercise every part of the analysis:

- Industry: home batteries. Five fictional companies with base rates of 6, 4, 3, 2 and 1.5
  headlines a day (Poisson), plus 3 untagged industry headlines a day.
- Volt Harbor's daily rate rises linearly to 2.5× over the last 10 days (a momentum signal).
- On day 22 of 30, Brightcell agrees to acquire Gridwise: 12 distinct headlines that day plus 10
  syndicated copies (a spike that survives near-duplicate merging).
- Headlines are drawn from templates for launches, funding, M&A, partnerships, layoffs,
  regulation, earnings and legal news, so rule-based event tags have something to find.
- 15% of headlines are re-published by a second fictional outlet with a light rewording
  (near-duplicates for the dedupe step).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

from .ingest import Dataset
from .watchlist import Competitor, Watchlist

COMPANIES = [
    ("Volt Harbor", ("VoltHarbor",), 6.0),
    ("Brightcell", (), 4.0),
    ("Ampleo", (), 3.0),
    ("Gridwise", (), 2.0),
    ("Kestrel Energy", ("Kestrel",), 1.5),
]
OUTLETS = ["Grid Ledger", "Battery Weekly", "Power Desk", "Home Energy Review", "Charge Times", "The Cell Report"]
PRODUCTS = ["HomeCell 2", "PowerWall X", "Nova 10", "Stack Pro", "Solar Pair"]
PLACES = ["Texas", "Germany", "California", "Australia", "Japan", "Spain"]
TEMPLATES = [
    "{c} launches {p} home battery with longer warranty",
    "{c} unveils {p} for small apartments",
    "{c} raises $#m to expand battery production",
    "{c} partners with installers in {place}",
    "{c} signs supply agreement for lithium iron phosphate cells",
    "{c} cuts # jobs in restructuring push",
    "Regulators open probe into {c} battery fire reports",
    "{c} third quarter revenue beats estimates on strong demand",
    "{c} sued over warranty claims in {place}",
    "{c} expands dealer network in {place}",
    "Customers in {place} report long wait times for {c} installs",
    "{c} adds smart tariff switching to its app",
]
INDUSTRY = [
    "Home battery prices fall as cell costs drop",
    "Utilities test paying households for stored solar power",
    "New safety standard for residential batteries takes effect",
    "Installers report record demand for home batteries in {place}",
    "Why heat pumps and home batteries are being sold together",
]
QUALIFIERS = ["", "as demand climbs", "ahead of winter", "after a slow summer", "in a push for share",
              "for the first time", "despite supply delays", "as rivals cut prices", "following customer feedback",
              "to meet utility rules", "with new financing options", "in its biggest market"]
REWORDS = [("launches", "rolls out"), ("unveils", "introduces"), ("raises", "secures"), ("cuts", "will cut"),
           ("partners with", "teams up with")]
SPIKE_DAY = 21  # zero-based day index of the injected acquisition story
SPIKE_TITLES = [
    "Brightcell agrees to acquire Gridwise in $#m deal",
    "What Brightcell's Gridwise acquisition means for installers",
    "Brightcell shares jump after Gridwise takeover announcement",
    "Gridwise customers ask about warranties after Brightcell deal",
    "Analysts split on the price Brightcell paid for Gridwise",
    "Brightcell chief explains why it bought Gridwise",
    "Gridwise staff face uncertainty as Brightcell takeover closes",
    "Regulators expected to review Brightcell and Gridwise merger",
    "Brightcell plus Gridwise: the new number two in home batteries",
    "Inside the talks that led Brightcell to Gridwise",
    "Brightcell to keep the Gridwise brand for two years",
    "Investors cheer Brightcell move for Gridwise software team",
]


def _fill(template: str, rng: np.random.Generator, company: str) -> str:
    text = template.replace("$#m", f"${int(rng.integers(20, 400))} million")
    text = text.replace("#", str(int(rng.integers(40, 600))))
    return text.format(c=company, p=PRODUCTS[int(rng.integers(len(PRODUCTS)))],
                       place=PLACES[int(rng.integers(len(PLACES)))])


def _record(title: str, outlet: str, when: datetime, query: str) -> dict:
    slug = hashlib.sha1(f"{title}|{outlet}|{when.isoformat()}".encode()).hexdigest()[:10]
    return {
        "title": title,
        "source": outlet,
        "link": f"https://example.com/synthetic/{slug}",
        "published": when.isoformat().replace("+00:00", "Z"),
        "summary": "",
        "queries": [query],
    }


def generate_dataset(seed: int = 7, days: int = 30,
                     fetched_at: datetime = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)) -> Dataset:
    rng = np.random.default_rng(seed)
    first_day = (fetched_at - timedelta(days=days)).replace(hour=0, minute=0, second=0)
    records: list[dict] = []
    for d in range(days):
        day_start = first_day + timedelta(days=d)
        for name, _aliases, base in COMPANIES:
            rate = base * (1 + 1.5 * max(0, d - (days - 10)) / 10) if name == "Volt Harbor" else base
            for _ in range(int(rng.poisson(rate))):
                when = day_start + timedelta(minutes=int(rng.integers(0, 24 * 60)))
                qualifier = QUALIFIERS[int(rng.integers(len(QUALIFIERS)))]
                title = _fill(TEMPLATES[int(rng.integers(len(TEMPLATES)))], rng, name)
                title += f" {qualifier}" if qualifier else ""
                outlet = OUTLETS[int(rng.integers(len(OUTLETS)))]
                records.append(_record(title, outlet, when, name))
                if rng.random() < 0.15:  # syndicated copy with light rewording
                    copy = title
                    for a, b in REWORDS:
                        copy = copy.replace(a, b)
                    other = OUTLETS[(OUTLETS.index(outlet) + 1) % len(OUTLETS)]
                    records.append(_record(copy, other, when + timedelta(hours=3), name))
        for _ in range(int(rng.poisson(3))):
            when = day_start + timedelta(minutes=int(rng.integers(0, 24 * 60)))
            title = _fill(INDUSTRY[int(rng.integers(len(INDUSTRY)))], rng, "")
            records.append(_record(title, OUTLETS[int(rng.integers(len(OUTLETS)))], when, "home batteries"))
    spike_start = first_day + timedelta(days=SPIKE_DAY, hours=8)
    for i, template in enumerate(SPIKE_TITLES):
        records.append(_record(_fill(template, rng, ""), OUTLETS[i % len(OUTLETS)],
                               spike_start + timedelta(minutes=37 * i), "Brightcell"))
    original = records[-len(SPIKE_TITLES)]  # the announcement headline
    for i in range(10):  # wire copies of the announcement carried by other outlets
        records.append(_record(original["title"], f"{OUTLETS[i % len(OUTLETS)]} Wire",
                               spike_start + timedelta(hours=1, minutes=11 * i), "Brightcell"))
    watchlist = Watchlist(
        name="Home batteries (synthetic)",
        competitors=tuple(Competitor(n, a) for n, a, _ in COMPANIES),
        topics=("home batteries",),
    )
    records.sort(key=lambda r: r["published"], reverse=True)
    return Dataset(id="synthetic-home-batteries", watchlist=watchlist, articles=tuple(records), fetched_at=fetched_at,
                   days=days, source="Synthetic generator (fictional companies)", synthetic=True)
