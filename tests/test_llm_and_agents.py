"""LLM parsing and agent tests. No network: a fake client stands in for OpenRouter."""

import json

import pytest

from marketpulse.agents import describe_themes, evidence_lines, judge_sentiment
from marketpulse.citations import check_citations
from marketpulse.config import Settings
from marketpulse.llm import LLMClient, LLMError, clean_text, extract_json
from marketpulse.pipeline import Params, compute, fingerprint, run_ai, run_sentiment, sentiment_table
from marketpulse.schemas import HeadlineSentiment, ThemeNotes
from marketpulse.synthetic import generate_dataset

THEMES_REPLY = {
    "themes": [
        {"theme": "Theme 0", "name": "Installer partnerships", "summary": "Firms sign installers [1].",
         "implication": {"text": "Distribution matters [2]."}},
        {"theme": 1, "name": ["Apartment", "launches"], "summary": "New small units [3]."},
    ]
}
BRIEFING = ("<think>plan</think>## Key moves\n- Brightcell agreed to buy Gridwise [1].\n"
            "- Rivals have stayed quiet so far.")


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.last_model = "fake/model"
        self.prompts = []

    def chat(self, system, user, **kwargs):
        self.prompts.append(user)
        return self.replies.pop(0)

    def complete_json(self, system, user, schema, **kwargs):
        return schema.model_validate(extract_json(self.chat(system, user)))


@pytest.fixture(scope="module")
def synthetic():
    ds = generate_dataset()
    return ds, compute(ds, Params())


def test_extract_json_and_clean_text():
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<think>{"no": 1}</think>Sure! {"a": 3} done') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("[1, 2]")
    assert clean_text("```markdown\n## Key moves\n```") == "## Key moves"


def test_schema_coercion_absorbs_free_model_quirks():
    notes = ThemeNotes.model_validate(THEMES_REPLY)
    assert [n.theme for n in notes.themes] == [0, 1]
    assert notes.themes[0].implication == "Distribution matters [2]."
    assert notes.themes[1].name == "Apartment launches"
    as_dict = ThemeNotes.model_validate({"themes": {"2": {"name": "Recalls"}, "3": "Pricing"}})
    assert [(n.theme, n.name) for n in as_dict.themes] == [(2, "Recalls"), (3, "Pricing")]
    labels = [HeadlineSentiment.model_validate({"ref": r, "sentiment": s}).sentiment
              for r, s in [("[4]", "Positive 🙂"), (5, "NEG"), (6, "mixed")]]
    assert labels == ["positive", "negative", "neutral"]
    assert HeadlineSentiment.model_validate({"ref": "[4]"}).ref == 4


def test_run_ai_uses_computed_facts_and_returns_serialisable_analysis(synthetic):
    ds, c = synthetic
    fake = FakeLLM([json.dumps(THEMES_REPLY), BRIEFING])
    steps = []
    result = run_ai(fake, ds, Params(), c, on_step=steps.append)
    json.dumps(result)
    assert result["fingerprint"] == fingerprint(ds, Params()) and len(steps) == 2
    assert result["briefing"].startswith("## Key moves")
    assert len(result["references"]) == len(c.evidence) and result["references"][0]["ref"] == 1
    briefing_prompt = fake.prompts[1]
    assert "Installer partnerships" in briefing_prompt  # theme names flow into the briefing
    assert c.sov.iloc[0]["competitor"] in briefing_prompt and "share of voice" in briefing_prompt
    check = check_citations(result["briefing"], len(result["references"]))
    assert [s.text for s in check.uncited] == ["Rivals have stayed quiet so far."]


def test_theme_and_sentiment_agents(synthetic):
    _, c = synthetic
    fake = FakeLLM([json.dumps(THEMES_REPLY)])
    notes = describe_themes(fake, "Home batteries", [{"theme": 0, "terms": ["x"], "size": 3, "momentum": "+5%",
                                                      "headlines": [{"ref": 1, "date": "2026-11-01",
                                                                     "source": "S", "title": "T"}]}])
    assert notes.themes[0].name == "Installer partnerships" and "[1] 2026-11-01 · S · T" in fake.prompts[0]
    evidence = [{"ref": 1, "date": "d", "source": "s", "title": "t", "competitors": ["A"], "events": []},
                {"ref": 2, "date": "d", "source": "s", "title": "u", "competitors": [], "events": ["Launch"]}]
    assert "(mentions A)" in evidence_lines(evidence) and "(rule-tagged Launch)" in evidence_lines(evidence)
    reply = {"items": [{"ref": 1, "sentiment": "positive"}, {"ref": 99, "sentiment": "negative"}]}
    assert judge_sentiment(FakeLLM([json.dumps(reply)]), evidence) == {1: "positive"}  # unknown ref dropped
    labels = {int(r): "negative" for r in c.evidence["ref"]}
    table = sentiment_table(c, labels)
    assert (table["net"] <= 0).all() and table["negative"].sum() >= len(table)
    result = run_sentiment(FakeLLM([json.dumps({"items": []})]), generate_dataset(), Params(), c)
    assert result["labels"] == {}


def test_client_requires_key_and_caches(tmp_path, monkeypatch):
    with pytest.raises(LLMError):
        LLMClient(Settings(api_key=None))
    client = LLMClient(Settings(api_key="test", models=("fake/model",)), cache_dir=tmp_path)
    calls = []

    def fake_call(*args, **kw):
        calls.append(1)
        return "hello", "fake/model"

    monkeypatch.setattr(client, "_call_with_fallback", fake_call)
    assert client.chat("s", "u") == client.chat("s", "u") == "hello"
    assert len(calls) == 1


def test_complete_json_repairs_once(tmp_path, monkeypatch):
    client = LLMClient(Settings(api_key="test", models=("fake/model",)), cache_dir=tmp_path)
    replies = iter(["not json at all", json.dumps(THEMES_REPLY)])
    monkeypatch.setattr(client, "_call_with_fallback", lambda *args, **kw: (next(replies), "fake/model"))
    assert client.complete_json("sys", "user", ThemeNotes).themes[0].name == "Installer partnerships"
    assert len(list(tmp_path.glob("*.json"))) == 1  # only the valid repaired answer stays cached
