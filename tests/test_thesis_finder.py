from thesis_finder.budget import Budget
from thesis_finder.cli import build_loop
from thesis_finder.config import Config
from thesis_finder.dedup import Deduper
from thesis_finder.embeddings import LocalEmbedder
from thesis_finder.loop import LoopState, assignments, stop_reason
from thesis_finder.models import Judgement, TopicCandidate
from thesis_finder.prompts import avoid_block
from thesis_finder.store import TopicStore
from thesis_finder.strategies import STRATEGIES

PROVIDERS = ["claude", "gemini", "openai"]
ALL_PAIRS = {(p, s.id) for p in PROVIDERS for s in STRATEGIES}


def topic(title, rq="How does it work?", whats_new="applies it to Austrian SMEs"):
    return TopicCandidate(title=title, research_question=rq, rationale="r", method="m",
                          data_sources=["d"], source_urls=["https://x.org"],
                          closest_work="Doe (2023)", whats_new=whats_new)


def judgement(score, passes):
    return Judgement(novelty=score, feasibility=score, data_access=score, demand=score, fit=score,
                     passes=passes, reason="because")


def test_rotation_covers_every_pair_in_one_rotation():
    seen = set()
    for r in range(len(STRATEGIES)):
        jobs = assignments(r, PROVIDERS)
        assert len({s for _, s in jobs}) == len(PROVIDERS)  # different strategy per provider
        seen.update(jobs)
    assert seen == ALL_PAIRS


def test_dedup_catches_exact_and_similar_titles_including_invalid():
    store = TopicStore(":memory:")
    deduper = Deduper(store, LocalEmbedder(), duplicate_threshold=0.8, borderline_threshold=0.6)
    first = topic("Generative AI adoption in Austrian SMEs", "What drives generative AI adoption in Austrian SMEs?")
    check = deduper.check(first)
    assert check.duplicate_of is None
    store.add_topic(first, status="invalid", reason="already published", strategy_id=0, found_by="claude",
                    judgements={"claude": judgement(2, False)}, embedding=check.embedding)

    assert deduper.check(topic("The generative AI adoption in Austrian SMEs")).duplicate_of is not None
    similar = topic("Adoption of generative AI in Austrian SMEs", "What drives adoption of generative AI in SMEs in Austria?")
    assert deduper.check(similar).duplicate_of is not None
    assert deduper.check(topic("Process mining for ESG reporting", "Can process mining audit ESG data?")).duplicate_of is None


def test_avoid_block_lists_invalid_topics_with_reason():
    store = TopicStore(":memory:")
    store.add_topic(topic("Topic A"), status="valid", reason="good", strategy_id=0, found_by="claude",
                    judgements={}, embedding=None)
    store.add_topic(topic("Topic B"), status="invalid", reason="no data available", strategy_id=1,
                    found_by="openai", judgements={}, embedding=None)
    text = avoid_block(store.all_topics("valid"), store.all_topics("invalid"))
    assert "Topic A" in text
    assert "Topic B (rejected: no data available)" in text


def test_stop_rules():
    cfg = Config()
    budget = Budget(20)

    state = LoopState(new_ratios=[0.0, 0.0, 0.0], valid_in_store=100)
    assert stop_reason(state, cfg, PROVIDERS, budget) is None  # coverage not complete yet

    state.covered = set(ALL_PAIRS)
    assert "target" in stop_reason(state, cfg, PROVIDERS, budget)

    state.valid_in_store = 5
    assert "saturated" in stop_reason(state, cfg, PROVIDERS, budget)

    state.new_ratios = [0.0, 0.5, 0.05]
    assert stop_reason(state, cfg, PROVIDERS, budget) is None

    budget.charge("claude", "claude-opus-5-5", input_tokens=0, output_tokens=2_000_000)  # $40
    assert "budget" in stop_reason(state, cfg, PROVIDERS, budget)


def test_budget_charges_tokens_and_searches():
    budget = Budget(10)
    cost = budget.charge("openai", "gpt-6.1-sol", input_tokens=1_000_000, output_tokens=100_000, searches=10)
    assert abs(cost - (2.0 + 1.0 + 0.1)) < 1e-9


def test_demo_loop_runs_until_saturation(tmp_path):
    cfg = Config(store_path=str(tmp_path / "demo.db"))
    loop = build_loop(cfg, demo=True)
    reason = loop.run()
    assert "saturated" in reason or "target" in reason
    store = loop.store
    titles = [t.title for t in store.all_topics()]
    assert len(titles) == len(set(titles))  # nothing stored twice
    assert store.count() > 0
    assert any(t.times_seen > 1 for t in store.all_topics())  # rediscoveries were counted, not re-stored


def test_budget_counts_cache_tokens_and_logs(tmp_path):
    from thesis_finder.budget import summarize_log

    log = tmp_path / "costs.csv"
    budget = Budget(10, log_path=str(log))
    cost = budget.charge("claude", "claude-sonnet-5-5", input_tokens=0, output_tokens=0,
                         cache_write_tokens=1_000_000, cache_read_tokens=1_000_000, purpose="search")
    assert abs(cost - (2.0 * 1.25 + 2.0 * 0.10)) < 1e-9
    text = summarize_log(str(log))
    assert "claude" in text and "search" in text and "Total: $2.70" in text


def test_no_new_round_when_it_would_overshoot_budget():
    cfg = Config()
    budget = Budget(5)
    budget.charge("claude", "claude-sonnet-5-5", output_tokens=300_000)  # $3 spent, $2 left
    state = LoopState(round_costs=[3.0])
    assert "nearly used" in stop_reason(state, cfg, PROVIDERS, budget)
    state.round_costs = [1.0]
    assert stop_reason(state, cfg, PROVIDERS, budget) is None


def _store_with_valid(n):
    store = TopicStore(":memory:")
    for i in range(n):
        store.add_topic(topic(f"Valid topic number {i} with even" + ("x" if i % 2 else "")),
                        status="valid", reason="ok", strategy_id=0, found_by="claude",
                        judgements={"claude": judgement(4, True)}, embedding=None)
    return store


def test_review_checks_top_picks_fails_back_to_invalid_and_parks():
    from thesis_finder.providers.fake import FakeCoordinator
    from thesis_finder.review import ReviewStep

    cfg = Config()
    cfg.review.top_k = 3
    cfg.review.park_after = 2
    store = _store_with_valid(6)
    step = ReviewStep(cfg, store, Budget(10), FakeCoordinator(), find_related=lambda t, q: [])

    out = step.run()
    assert len(out.checked) + len(out.failed) == 3
    assert store.count("checked") == len(out.checked)
    for t in store.all_topics("invalid"):
        assert t.reason.startswith("failed literature check")
        assert t.lit_review["key_references"]
    assert store.count("valid") == 3 and not out.parked  # passed over once

    out2 = step.run()  # the remaining 3 are now the top picks
    assert len(out2.checked) + len(out2.failed) == 3
    assert store.count("valid") == 0


def test_unpicked_topics_get_parked():
    from thesis_finder.models import Ranking, RankedPick
    from thesis_finder.providers.fake import FakeCoordinator
    from thesis_finder.review import ReviewStep

    class AlwaysFirst(FakeCoordinator):
        def rank(self, prompt):
            return Ranking(picks=[RankedPick(id=1, reason="x")])

    cfg = Config()
    cfg.review.top_k = 1
    cfg.review.park_after = 1
    store = _store_with_valid(3)
    out = ReviewStep(cfg, store, Budget(10), AlwaysFirst(), find_related=lambda t, q: []).run()
    assert len(out.parked) == 2
    assert store.count("parked") == 2


def test_low_novelty_from_any_judge_fails():
    from thesis_finder.validator import Validator

    class J:
        def __init__(self, name, j):
            self.name, self.j = name, j

        def judge(self, prompt):
            return self.j

    weak = Judgement(novelty=2, feasibility=5, data_access=5, demand=5, fit=5, passes=True, reason="done before")
    judges = [J("a", judgement(5, True)), J("b", weak), J("c", judgement(5, True))]
    v = Validator(judges, Config().profile, 2, 4.0, min_novelty=3, find_related=lambda t, q: [])
    verdict = v.validate(topic("Some topic"))
    assert not verdict.valid and "not novel enough" in verdict.reason


def test_store_migrates_old_database(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""CREATE TABLE topics (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
        norm_title TEXT NOT NULL, research_question TEXT NOT NULL, rationale TEXT, method TEXT, data_sources TEXT,
        source_urls TEXT, strategy_id INTEGER, found_by TEXT, status TEXT NOT NULL, reason TEXT, judgements TEXT,
        mean_score REAL, embedding BLOB, times_seen INTEGER NOT NULL DEFAULT 1, first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL);
        INSERT INTO topics (title, norm_title, research_question, status, first_seen, last_seen)
        VALUES ('Old', 'old', 'Q?', 'valid', 'x', 'x');""")
    conn.commit()
    conn.close()
    store = TopicStore(str(path))
    assert store.all_topics()[0].passed_over == 0


def test_long_report_is_split_when_extraction_fails(tmp_path):
    from thesis_finder.models import CandidateList
    from thesis_finder.providers.fake import FakeCoordinator

    class Fussy(FakeCoordinator):
        def extract(self, prompt):
            if prompt.count("TOPIC:") > 4:
                raise RuntimeError("cut off")
            return super().extract(prompt)

    cfg = Config(store_path=str(tmp_path / "t.db"))
    loop = build_loop(cfg, demo=True)
    loop.coordinator = Fussy()
    report = "\n".join(f"TOPIC: Topic {i} | Question {i}?" for i in range(6))
    from thesis_finder.strategies import get
    topics = loop._extract(report, get(0))
    assert len(topics) == 6


def test_truncated_claude_json_raises_provider_error_and_is_charged():
    import json
    import anthropic
    import httpx2
    from thesis_finder.config import ClaudeSearchConfig, ModelConfig
    from thesis_finder.providers.base import ProviderError
    from thesis_finder.providers.claude import ClaudeProvider

    def handler(req):
        body = json.loads(req.content)
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": '{"topics": [{"title": "cut'}],
            "stop_reason": "max_tokens", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 16000}})

    p = ClaudeProvider.__new__(ClaudeProvider)
    p.client = anthropic.Anthropic(api_key="x", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    p.models, p.budget, p.search = ModelConfig(), Budget(10), ClaudeSearchConfig()
    try:
        p.extract("report " * 100)
        raise AssertionError("expected ProviderError")
    except ProviderError:
        pass
    assert p.budget.spent_usd > 0


class CountingJudge:
    def __init__(self, name, j):
        self.name, self.j, self.calls = name, j, 0

    def judge(self, prompt):
        self.calls += 1
        return self.j


def test_topic_without_a_stated_addition_fails_without_judging():
    from thesis_finder.validator import Validator

    j = CountingJudge("a", judgement(5, True))
    v = Validator([j], Config().profile, 1, 3.5, find_related=lambda t, q: [])
    verdict = v.validate(topic("Some topic", whats_new=""))
    assert not verdict.valid and "no concrete contribution" in verdict.reason
    assert j.calls == 0


def test_third_judge_only_breaks_ties():
    from thesis_finder.validator import Validator

    judges = [CountingJudge("a", judgement(2, False)), CountingJudge("b", judgement(3, False)),
              CountingJudge("c", judgement(5, True))]
    v = Validator(judges, Config().profile, 2, 3.5, min_novelty=1, find_related=lambda t, q: [])
    assert not v.validate(topic("Some topic")).valid
    assert judges[2].calls == 0

    judges = [CountingJudge("a", judgement(4, True)), CountingJudge("b", judgement(3, False)),
              CountingJudge("c", judgement(5, True))]
    v = Validator(judges, Config().profile, 2, 3.5, min_novelty=1, find_related=lambda t, q: [])
    assert v.validate(topic("Some topic")).valid
    assert judges[2].calls == 1  # split decision: the third judge breaks the tie

    judges = [CountingJudge("a", judgement(4, True)), CountingJudge("b", judgement(4, True)),
              CountingJudge("c", judgement(1, False))]
    v = Validator(judges, Config().profile, 2, 3.5, min_novelty=1, find_related=lambda t, q: [])
    assert v.validate(topic("Some topic")).valid
    assert judges[2].calls == 0  # the first two agree, so the third is not paid


def test_prior_art_rejects_topic_that_is_already_published(tmp_path):
    from thesis_finder.prior_art import PriorArtChecker
    from thesis_finder.sources.openalex import Work

    published = Work(title="Generative AI adoption in Austrian SMEs", year=2024, cited_by=12, doi=None,
                     abstract="What drives generative AI adoption in Austrian SMEs")
    other = Work(title="Blockchain in shipping", year=2023, cited_by=3, doi=None, abstract="Ports and ledgers")
    cfg = Config(store_path=str(tmp_path / "t.db"))
    loop = build_loop(cfg, demo=True)
    loop.prior_art = PriorArtChecker(LocalEmbedder(), [lambda q: [other, published]], cutoff=0.8)
    judge = CountingJudge("a", judgement(5, True))
    loop.validator.judges = [judge]

    cand = topic("Generative AI adoption in Austrian SMEs", "What drives generative AI adoption in Austrian SMEs?")
    verdict = loop._validate(cand, None)
    assert not verdict.valid and verdict.reason.startswith("already done: Generative AI adoption")
    assert judge.calls == 0

    fresh = topic("Process mining for ESG reporting", "Can process mining make ESG reporting auditable?")
    assert loop._validate(fresh, None).valid
    assert judge.calls == 1


def test_fresh_window_start():
    from datetime import date
    from thesis_finder.prompts import fresh_since

    assert fresh_since(18, date(2026, 10, 26)) == "April 2025"
    assert fresh_since(12, date(2026, 1, 5)) == "January 2025"


def test_flex_halves_token_cost_but_not_search_fees():
    budget = Budget(10)
    cost = budget.charge("openai", "gpt-6.1-sol", input_tokens=1_000_000, output_tokens=100_000, searches=10,
                         token_factor=0.5)
    assert abs(cost - (1.5 + 0.1)) < 1e-9


def test_openai_falls_back_to_standard_when_flex_has_no_capacity():
    import httpx
    import openai
    from thesis_finder.config import ModelConfig
    from thesis_finder.providers.openai_provider import OpenAIProvider

    calls = []

    def method(**kwargs):
        calls.append(kwargs.get("service_tier"))
        if kwargs.get("service_tier") == "flex":
            resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
            raise openai.RateLimitError("Resource unavailable", response=resp, body=None)
        return "ok"

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.models, p.budget = ModelConfig(), Budget(10)
    assert p._create(method, True, model="m") == "ok"
    assert calls == ["flex", None]


def test_openalex_query_is_short_keywords_and_retries_on_400(monkeypatch):
    import httpx
    from thesis_finder.sources import openalex

    long = ("Cloud Exit-Readiness Analyser Based on Infrastructure-as-Code How can an artefact be designed "
            "that derives an organisation's cloud lock-in and switching readiness?")
    q = openalex.keyword_query(long)
    assert len(q.split()) <= 10 and "?" not in q and "'" not in q
    assert q.startswith("cloud exit readiness analyser infrastructure code")

    sent = []

    def fake_get(url, params, timeout):
        sent.append(params["search"])
        req = httpx.Request("GET", url)
        if len(sent) == 1:
            return httpx.Response(400, request=req)
        return httpx.Response(200, request=req, json={"results": [{"display_name": "Paper", "publication_year": 2025}]})

    monkeypatch.setattr(openalex.httpx, "get", fake_get)
    works = openalex.search_works(long, limit=3)
    assert [w.title for w in works] == ["Paper"]
    assert len(sent) == 2 and len(sent[1].split()) == 4


def test_openai_reviewer_parses_ranking_and_review():
    from thesis_finder.config import ModelConfig
    from thesis_finder.models import LitReview, Ranking, RankedPick
    from thesis_finder.providers.openai_provider import OpenAIProvider

    class Resp:
        def __init__(self, parsed):
            self.output_parsed, self.usage, self.service_tier = parsed, None, "default"

    seen = []

    class Responses:
        def parse(self, **kwargs):
            seen.append(kwargs)
            if kwargs["text_format"] is Ranking:
                return Resp(Ranking(picks=[RankedPick(id=2, reason="best")]))
            return Resp(LitReview(already_answered=False, gap_is_real=True, enough_literature=True,
                                  data_reachable=True, passes=True, summary="s", key_references=["Doe 2025"],
                                  reason="ok"))

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.models, p.budget, p.flex_search, p.flex_judge = ModelConfig(), Budget(10), False, False
    p.client = type("C", (), {"responses": Responses()})()
    assert p.rank("rank these").picks[0].id == 2
    assert p.assess_review("assess").passes
    assert seen[0]["reasoning"] == {"effort": "high"} and seen[0]["model"] == "gpt-6.1-sol"
