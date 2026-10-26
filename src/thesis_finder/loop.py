"""The topic-mining loop: plan, deep-search with every provider, extract, dedup, validate, store, repeat."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Set, Tuple

from . import prompts, strategies
from .budget import Budget
from .config import Config
from .dedup import Deduper
from .models import CandidateList, ResearchResult
from .providers.base import Provider
from .sources.openalex import Work, search_works
from .store import TopicStore
from .validator import Validator

log = logging.getLogger(__name__)


class Coordinator(Protocol):
    def plan(self, prompt: str) -> str: ...
    def extract(self, prompt: str) -> CandidateList: ...
    def same_topic(self, prompt: str) -> bool: ...


@dataclass
class LoopState:
    rounds_done: int = 0
    new_ratios: List[float] = field(default_factory=list)  # one per counted round
    covered: Set[Tuple[str, int]] = field(default_factory=set)  # (provider, strategy) pairs that ran
    valid_in_store: int = 0
    new_this_run: int = 0
    valid_this_run: int = 0
    round_costs: List[float] = field(default_factory=list)


def stop_reason(state: LoopState, cfg: Config, providers: List[str], budget: Budget) -> Optional[str]:
    """The stop rules. Budget and max rounds always apply; target and saturation
    only count once every provider has tried every strategy."""
    loop = cfg.loop
    if budget.exhausted:
        return f"budget cap reached (${budget.spent_usd:.2f} of ${budget.limit_usd:.2f})"
    if state.round_costs:
        # Don't start a round that would likely overshoot the cap.
        expected = max(state.round_costs[-3:])
        if expected > budget.remaining:
            return (f"budget nearly used (${budget.spent_usd:.2f} of ${budget.limit_usd:.2f}; "
                    f"a round costs about ${expected:.2f})")
    if state.rounds_done >= loop.max_rounds:
        return f"max rounds reached ({loop.max_rounds})"
    needed = {(p, s.id) for p in providers for s in strategies.STRATEGIES}
    if not needed <= state.covered:
        return None
    if state.valid_in_store >= loop.target_valid:
        return f"target reached ({state.valid_in_store} valid topics stored)"
    recent = state.new_ratios[-loop.saturation_rounds:]
    if len(recent) == loop.saturation_rounds and all(r < loop.saturation_new_ratio for r in recent):
        pct = int(loop.saturation_new_ratio * 100)
        return f"saturated (under {pct}% new topics for {loop.saturation_rounds} rounds in a row)"
    return None


def assignments(round_index: int, providers: List[str]) -> List[Tuple[str, int]]:
    """Rotate strategies so each provider gets a different one each round and every
    provider covers all strategies within len(STRATEGIES) rounds."""
    n = len(strategies.STRATEGIES)
    step = max(1, n // max(1, len(providers)))
    return [(p, (round_index + i * step) % n) for i, p in enumerate(providers)]


class TopicLoop:
    def __init__(
        self,
        cfg: Config,
        store: TopicStore,
        budget: Budget,
        searchers: List[Provider],
        coordinator: Coordinator,
        deduper: Deduper,
        validator: Validator,
        seed_search=search_works,
    ):
        self.cfg = cfg
        self.store = store
        self.budget = budget
        self.searchers = {p.name: p for p in searchers}
        self.coordinator = coordinator
        self.deduper = deduper
        self.validator = validator
        self.seed_search = seed_search

    def run(self, max_rounds: Optional[int] = None) -> str:
        if max_rounds is not None:
            self.cfg.loop.max_rounds = max_rounds
        run_id = self.store.start_run()
        state = LoopState(valid_in_store=self.store.count("valid"))
        names = list(self.searchers)
        reason = None
        try:
            while True:
                reason = stop_reason(state, self.cfg, names, self.budget)
                if reason:
                    break
                before = self.budget.spent_usd
                self._round(run_id, state, names)
                state.round_costs.append(self.budget.spent_usd - before)
                state.rounds_done += 1
        except KeyboardInterrupt:
            reason = "stopped by you (Ctrl+C)"
        self.store.end_run(run_id, reason, self.budget.spent_usd)
        log.info("Stopped: %s. New topics this run: %d (%d valid). Spent $%.2f.",
                 reason, state.new_this_run, state.valid_this_run, self.budget.spent_usd)
        log.info("Spend by provider and job: %s", self.budget.summary())
        return reason

    def _round(self, run_id: int, state: LoopState, names: List[str]) -> None:
        jobs = assignments(state.rounds_done, names)
        log.info("Round %d: %s", state.rounds_done + 1,
                 ", ".join(f"{p} -> {strategies.get(s).name}" for p, s in jobs))

        avoid = prompts.avoid_block(self.store.all_topics("valid"), self.store.all_topics("invalid"))
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(lambda job: self._search(job[0], job[1], avoid), jobs))

        found_total = new_total = 0
        any_success = False
        for result in results:
            if result.error:
                log.warning("%s failed on strategy %d: %s", result.provider, result.strategy_id, result.error)
                continue
            any_success = True
            state.covered.add((result.provider, result.strategy_id))
            found, new, valid = self._process(result)
            found_total += found
            new_total += new
            state.new_this_run += new
            state.valid_this_run += valid
            self.store.record_round(run_id, state.rounds_done + 1, result.provider, result.strategy_id, found, new, valid)
            if self.budget.exhausted:
                break

        state.valid_in_store = self.store.count("valid")
        if any_success:
            ratio = new_total / found_total if found_total else 0.0
            state.new_ratios.append(ratio)
            log.info("Round %d: %d topics found, %d new (%.0f%%). Store: %d valid, %d invalid. Spent $%.2f so far.",
                     state.rounds_done + 1, found_total, new_total, ratio * 100,
                     state.valid_in_store, self.store.count("invalid"), self.budget.spent_usd)

    def _search(self, provider: str, strategy_id: int, avoid: str) -> ResearchResult:
        strategy = strategies.get(strategy_id)
        try:
            seeds = self._seed_papers(strategy)
            n = self.cfg.loop.topics_per_report
            if self.cfg.loop.planner == "claude":
                brief = self.coordinator.plan(
                    prompts.planner_prompt(self.cfg.profile, strategy, provider, avoid, seeds, n))
            else:
                brief = prompts.template_brief(self.cfg.profile, strategy, avoid, seeds, n)
            return self.searchers[provider].deep_search(brief, prompts.deep_search_system(), strategy_id)
        except Exception as exc:
            return ResearchResult(provider=provider, strategy_id=strategy_id, text="", error=str(exc))

    def _seed_papers(self, strategy: strategies.Strategy) -> List[Work]:
        if not strategy.use_openalex:
            return []
        profile = self.cfg.profile
        query = " ".join([profile.field, *profile.interests[:3], strategy.openalex_hint]).strip()
        sort = "cited_by_count:desc" if strategy.id == 2 else None
        try:
            return self.seed_search(query, limit=8, sort=sort, from_year=2019)
        except Exception as exc:
            log.warning("OpenAlex seed search failed: %s", exc)
            return []

    def _process(self, result: ResearchResult) -> Tuple[int, int, int]:
        strategy = strategies.get(result.strategy_id)
        candidates = self.coordinator.extract(prompts.extract_prompt(result.text, strategy)).topics
        new = valid = 0
        for cand in candidates:
            if not cand.source_urls:
                cand.source_urls = result.sources[:5]
            check = self.deduper.check(cand)
            if check.duplicate_of:
                self.store.mark_seen(check.duplicate_of.id)
                log.debug("Duplicate (%.2f): %s ~ %s", check.similarity, cand.title, check.duplicate_of.title)
                continue
            verdict = self.validator.validate(cand)
            self.store.add_topic(
                cand,
                status="valid" if verdict.valid else "invalid",
                reason=verdict.reason,
                strategy_id=result.strategy_id,
                found_by=result.provider,
                judgements=verdict.judgements,
                embedding=check.embedding,
            )
            new += 1
            valid += int(verdict.valid)
            log.info("  %s %s", "VALID  " if verdict.valid else "invalid", cand.title)
            if self.budget.exhausted:
                break
        return len(candidates), new, valid
