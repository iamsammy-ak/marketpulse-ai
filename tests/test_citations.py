from marketpulse.citations import check_citations, citations_in, link_citations, parse_refs, split_sentences

BRIEFING = """## Key moves
- DoorDash offered $17 billion for Instacart [3]. The U.S. Justice Department will review the deal [5, 7].
- Uber Eats replaced Grubhub as Amazon's partner.

## What to watch
- A ruling is expected soon [12].
- None.
"""


def test_parse_refs_and_sentence_splitting():
    assert parse_refs("3") == [3]
    assert parse_refs("3, 7") == [3, 7]
    assert parse_refs("4-6") == [4, 5, 6]
    assert parse_refs("1–3, 9") == [1, 2, 3, 9]
    assert citations_in("A [1][2] and [3, 4].") == [1, 2, 3, 4]
    text = "Shares rose. [2] The U.S. regulator opened a probe [4]. Acme Inc. said it will comply."
    assert split_sentences(text) == ["Shares rose. [2]", "The U.S. regulator opened a probe [4].",
                                     "Acme Inc. said it will comply."]


def test_check_flags_uncited_sentences_and_unknown_references():
    report = check_citations(BRIEFING, n_sources=10)
    assert [s.text for s in report.uncited] == ["Uber Eats replaced Grubhub as Amazon's partner."]
    assert report.invalid_refs == [12]
    assert report.cited_refs == [3, 5, 7]
    assert len(report.claims) == 4  # headings and the 1-word "None." are not claims
    assert report.uncited[0].section == "Key moves"
    assert not report.passed and report.coverage == 0.75


def test_check_passes_when_everything_is_cited():
    report = check_citations("## Key moves\n- Acme bought Globex for cash [1].\n- Rivals reacted quickly [2].", 2)
    assert report.passed and report.coverage == 1.0
    assert not check_citations("", 5).passed


def test_link_citations_only_links_known_numbers():
    out = link_citations("Deal agreed [1, 2]; ruling due [9].", {1: "https://a.example", 2: "https://b.example"})
    assert out == "Deal agreed [[1]](https://a.example) [[2]](https://b.example); ruling due [9]."
