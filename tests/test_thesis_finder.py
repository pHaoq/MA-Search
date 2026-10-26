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


def topic(title, rq="How does it work?"):
    return TopicCandidate(title=title, research_question=rq, rationale="r", method="m",
                          data_sources=["d"], source_urls=["https://x.org"])


def judgement(score, passes):
    return Judgement(novelty=score, feasibility=score, data_access=score, demand=score, fit=score,
                     passes=passes, reason="because")


def test_rotation_covers_every_pair_in_six_rounds():
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
