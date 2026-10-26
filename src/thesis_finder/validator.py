"""A topic is valid when enough judges from different providers pass it."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import prompts
from .models import Judgement, Profile, TopicCandidate
from .providers.base import Provider
from .sources.openalex import Work, related_work

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    valid: bool
    reason: str
    judgements: Dict[str, Judgement]


class Validator:
    def __init__(
        self,
        judges: List[Provider],
        profile: Profile,
        min_pass_votes: int,
        min_mean_score: float,
        min_novelty: int = 1,
        find_related: Callable[[str, str], List[Work]] = related_work,
    ):
        self.judges = judges
        self.profile = profile
        self.min_pass_votes = min(min_pass_votes, len(judges))
        self.min_mean_score = min_mean_score
        self.min_novelty = min_novelty
        self.find_related = find_related

    def validate(self, topic: TopicCandidate, related: Optional[List[Work]] = None) -> Verdict:
        if not topic.whats_new.strip():
            return Verdict(False, "no concrete contribution: it does not say what it adds beyond existing work", {})
        if related is None:
            related = self.find_related(topic.title, topic.research_question)
        prompt = prompts.judge_prompt(self.profile, topic, related, self.profile.timeframe_months)

        # Ask the first judges; ask the rest only if the verdict is still open. Most topics fail,
        # and when the first judges already sink one, the remaining calls are money for nothing.
        first = max(1, self.min_pass_votes)
        judgements = self._ask(self.judges[:first], prompt)
        if self.judges[first:] and not self._decided(judgements):
            judgements.update(self._ask(self.judges[first:], prompt))

        if not judgements:
            return Verdict(False, "no judge could score it", judgements)

        votes = sum(1 for j in judgements.values() if j.passes)
        mean = sum(j.mean for j in judgements.values()) / len(judgements)
        needed = min(self.min_pass_votes, len(judgements))
        low_novelty = [j for j in judgements.values() if j.novelty < self.min_novelty]
        if low_novelty:
            return Verdict(False, f"not novel enough: {low_novelty[0].reason}", judgements)
        if votes >= needed and mean >= self.min_mean_score:
            best = max(judgements.values(), key=lambda j: j.mean)
            return Verdict(True, best.reason, judgements)

        failing = [j for j in judgements.values() if not j.passes] or list(judgements.values())
        worst = min(failing, key=lambda j: j.mean)
        return Verdict(False, worst.reason, judgements)

    def _decided(self, judgements: Dict[str, Judgement]) -> bool:
        """True when the remaining judges cannot make the topic valid."""
        if any(j.novelty < self.min_novelty for j in judgements.values()):
            return True
        fails = sum(1 for j in judgements.values() if not j.passes)
        return len(self.judges) - fails < self.min_pass_votes

    def _ask(self, judges: List[Provider], prompt: str) -> Dict[str, Judgement]:
        judgements: Dict[str, Judgement] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(judges))) as pool:
            futures = {pool.submit(j.judge, prompt): j.name for j in judges}
            for future, name in futures.items():
                try:
                    judgements[name] = future.result()
                except Exception as exc:  # one failing judge should not stop the loop
                    log.warning("Judge %s failed: %s", name, exc)
        return judgements
