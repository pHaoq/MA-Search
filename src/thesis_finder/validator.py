"""A topic is valid when enough judges from different providers pass it."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List

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

    def validate(self, topic: TopicCandidate) -> Verdict:
        related = self.find_related(topic.title, topic.research_question)
        prompt = prompts.judge_prompt(self.profile, topic, related, self.profile.timeframe_months)

        judgements: Dict[str, Judgement] = {}
        with ThreadPoolExecutor(max_workers=len(self.judges)) as pool:
            futures = {pool.submit(j.judge, prompt): j.name for j in self.judges}
            for future, name in futures.items():
                try:
                    judgements[name] = future.result()
                except Exception as exc:  # one failing judge should not stop the loop
                    log.warning("Judge %s failed: %s", name, exc)

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
