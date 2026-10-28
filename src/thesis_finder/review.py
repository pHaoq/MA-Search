"""The review step: promotes the best valid topics to checked.

1. The strongest model ranks all valid topics against each other and picks the top k.
2. Each pick gets a literature review on a cheaper model.
3. Pass: status checked. Fail: back to invalid, with the review's reason.
   Valid topics that were ranked but not picked stay valid; after `park_after`
   reviews without being picked they are parked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Protocol

from . import prompts
from .budget import Budget
from .config import Config
from .models import LitReview, Ranking, ResearchResult
from .sources.openalex import Work, search_works
from .store import TopicStore

log = logging.getLogger(__name__)


class Reviewer(Protocol):
    def rank(self, prompt: str) -> Ranking: ...
    def literature_review(self, brief: str, system: str) -> ResearchResult: ...
    def assess_review(self, prompt: str) -> LitReview: ...


@dataclass
class ReviewOutcome:
    checked: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    parked: List[str] = field(default_factory=list)
    cost_usd: float = 0.0


def _related(title: str, question: str) -> List[Work]:
    try:
        return search_works(f"{title} {question}", limit=10, from_year=2018)
    except Exception as exc:
        # The literature review still runs its own web search; OpenAlex is only a head start.
        log.warning("OpenAlex search failed (%s); the review continues with web search only.", exc)
        return []


class ReviewStep:
    def __init__(self, cfg: Config, store: TopicStore, budget: Budget, reviewer: Reviewer,
                 find_related: Callable[[str, str], List[Work]] = _related):
        self.cfg = cfg
        self.store = store
        self.budget = budget
        self.reviewer = reviewer
        self.find_related = find_related

    def run(self) -> ReviewOutcome:
        rc = self.cfg.review
        outcome = ReviewOutcome()
        start = self.budget.spent_usd
        pool = self.store.ranked("valid", rc.max_candidates)
        if not pool:
            log.info("Review: no valid topics to review.")
            return outcome

        if len(pool) <= rc.top_k:
            picks = [t.id for t in pool]
        else:
            log.info("Review: ranking %d valid topics...", len(pool))
            ranking = self.reviewer.rank(prompts.rank_prompt(self.cfg.profile, pool, rc.top_k))
            known = {t.id for t in pool}
            picks = []
            for pick in ranking.picks:
                if pick.id in known and pick.id not in picks:
                    picks.append(pick.id)
            picks = picks[: rc.top_k]
        log.info("Review: %d valid topics, reviewing the top %d.", len(pool), len(picks))

        for n, topic_id in enumerate(picks, 1):
            if self.budget.exhausted:
                log.info("Review: budget cap reached, the rest stay valid for the next review.")
                break
            topic = self.store.get(topic_id)
            log.info("  [%d/%d] Literature review: %s (takes a few minutes)", n, len(picks), topic.title)
            try:
                report = self.reviewer.literature_review(
                    prompts.literature_brief(self.cfg.profile, topic, self.find_related(topic.title, topic.research_question)),
                    prompts.deep_search_system(),
                )
                verdict = self.reviewer.assess_review(prompts.assess_review_prompt(topic, report.text))
            except Exception as exc:
                log.warning("Review of '%s' failed, it stays valid: %s", topic.title, exc)
                continue
            if verdict.passes:
                self.store.set_review(topic.id, "checked", verdict.reason, verdict.model_dump_json())
                outcome.checked.append(topic.title)
                log.info("  CHECKED  %s", topic.title)
            else:
                self.store.set_review(topic.id, "invalid", f"failed literature check: {verdict.reason}",
                                      verdict.model_dump_json())
                outcome.failed.append(topic.title)
                log.info("  invalid  %s (%s)", topic.title, verdict.reason)

        for topic in pool:
            if topic.id not in picks and self.store.pass_over(topic.id, rc.park_after):
                outcome.parked.append(topic.title)
        outcome.cost_usd = self.budget.spent_usd - start
        log.info("Review done: %d checked, %d back to invalid, %d parked. Cost $%.2f.",
                 len(outcome.checked), len(outcome.failed), len(outcome.parked), outcome.cost_usd)
        return outcome
