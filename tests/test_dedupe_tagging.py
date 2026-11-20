import pandas as pd

from marketpulse.dedupe import deduplicate, near_duplicate_groups, normalise_title, to_frame
from marketpulse.tagging import EntityMatcher, classify_events, keyword_regex, tag_articles
from marketpulse.watchlist import Competitor, format_competitors, parse_competitors


def _frame(rows):
    return to_frame({"title": t, "source": s, "link": f"https://x.example/{i}", "published": p, "queries": ["q"]}
                    for i, (t, s, p) in enumerate(rows))


def test_normalise_title_unifies_quotes_case_and_punctuation():
    assert normalise_title("Nvidia’s  CEO: ‘We’re Sold Out’ — Report!") == "nvidias ceo were sold out report"
    assert normalise_title("") == ""


def test_near_duplicates_merge_into_earliest_with_outlet_count():
    df = _frame([
        ("DoorDash offers $17 billion for Instacart in grocery push", "Outlet A", "2026-11-02T10:00:00Z"),
        ("DoorDash offers $17 billion for Instacart in big grocery push", "Outlet B", "2026-11-02T08:00:00Z"),
        ("DoorDash offers $17 billion for Instacart in grocery push", "Outlet C", "2026-11-03T09:00:00Z"),
        ("Drone deliveries expand to Texas suburbs", "Outlet A", "2026-11-02T09:00:00Z"),
    ])
    out = deduplicate(tag_articles(df, [Competitor("DoorDash"), Competitor("Instacart")]), threshold=0.7)
    assert len(out) == 2
    deal = out[out["title"].str.contains("DoorDash")].iloc[0]
    assert deal["source"] == "Outlet B"  # the earliest copy is kept
    assert deal["copies"] == 3 and sorted(deal["outlets"]) == ["Outlet A", "Outlet B", "Outlet C"]


def test_templated_headlines_about_different_companies_are_not_merged():
    df = _frame([
        ("Why Nvidia stock is falling today after the chip export news", "Outlet A", "2026-11-02T10:00:00Z"),
        ("Why AMD stock is falling today after the chip export news", "Outlet A", "2026-11-02T11:00:00Z"),
    ])
    tagged = tag_articles(df, [Competitor("Nvidia"), Competitor("AMD")])
    assert len(deduplicate(tagged, threshold=0.7)) == 2
    assert len(deduplicate(df, threshold=0.7)) == 1  # without the competitor check they would collapse


def test_dedupe_time_gap_and_edge_cases():
    titles = ["Acme recalls 4,400 scooters", "Acme recalls 4,400 scooters"]
    assert near_duplicate_groups(titles, 0.8, times=[0, 10 * 86400]) == [0, 1]  # weeks apart: new story
    assert near_duplicate_groups(titles, 0.8, times=[0, 3600]) == [0, 0]
    assert near_duplicate_groups([], 0.8) == []
    assert near_duplicate_groups(["the", "a"], 0.8) == [0, 1]  # nothing but stop words
    assert deduplicate(to_frame([]), 0.8).empty


def test_entity_matching_rules_and_watchlist_text():
    m = EntityMatcher(parse_competitors(
        "Intel\nGM: General Motors\nLucid: Lucid Motors, -Lucid Capital\nUber Eats\nNvidia: NVDA"))
    assert m.find("Intel's foundry wins a customer") == ["Intel"]
    assert m.find("Artificial Intelligence chips sell out") == []  # whole words only
    assert m.find("gm crops and GM trucks") == ["GM"]  # short alias: exact case
    assert m.find("A lucid plan from regulators") == []  # names need a capital first letter
    assert m.find("Lucid Capital cuts its price target") == []  # excluded phrase
    assert m.find("Lucid Capital says Lucid Motors will grow") == ["Lucid"]
    assert m.find("NVDA slips while UBER EATS expands") == ["Nvidia", "Uber Eats"]
    assert m.find("General Motors and Intel team up") == ["GM", "Intel"]  # order of first mention
    text = "Lucid: Lucid Motors, -Lucid Capital\nBYD"
    comps = parse_competitors(text + "\nbyd: duplicate\n\n")
    assert [c.name for c in comps] == ["Lucid", "BYD"]
    assert comps[0].aliases == ("Lucid Motors",) and comps[0].exclude == ("Lucid Capital",)
    assert format_competitors(comps) == text


EVENT_CASES = [
    ("Acme unveils its new scooter in Paris", {"Launch"}),
    ("Acme raises $40 million in Series B funding", {"Funding"}),
    ("Acme agrees to acquire Globex", {"M&A"}),
    ("Is Acme a stock to buy right now?", set()),  # not an acquisition
    ("Acme cuts 1,000 jobs in restructuring", {"Layoffs"}),
    ("FTC opens antitrust probe into Acme", {"Regulation"}),
    ("Acme third-quarter revenue beats estimates", {"Earnings"}),
    ("Acme sued over patent claims", {"Legal"}),
    ("Acme fine-tunes its app", set()),  # "fine" is not "fined"
    ("Acme teams up with Globex on charging", {"Partnership"}),
]


def test_event_rules_and_their_explanations():
    for headline, expected in EVENT_CASES:
        assert set(classify_events(headline)) == expected, headline
    hits = classify_events("Acme launches a scooter after it raised $2.5 billion")
    assert hits == {"Launch": "launches", "Funding": "raised $2.5 billion"}
    assert classify_events("ipo season") == {}  # all-capital keywords match case-sensitively
    assert keyword_regex("cut* # jobs").startswith(r"(?<!\w)cut\w*")
    tagged = tag_articles(pd.DataFrame({"title": ["Acme unveils a bike"], "summary": [""]}), [Competitor("Acme")])
    assert tagged.loc[0, "competitors"] == ["Acme"] and tagged.loc[0, "event_hits"] == "Launch: “unveils”"
