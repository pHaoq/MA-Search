"""Offline stand-ins for the real providers. `thesis-finder run --demo` and the tests use them,
so the loop can be tried without API keys or cost."""

from __future__ import annotations

import random
import re

from ..models import CandidateList, Judgement, LitReview, Ranking, RankedPick, ResearchResult, TopicCandidate
from .base import Provider

# A small, finite pool so a demo run saturates the way a real one eventually does.
TOPIC_POOL = [
    ("Generative AI adoption in Austrian SMEs", "What drives and blocks generative AI adoption in Austrian SMEs?"),
    ("Explainable AI and trust in credit decisions", "How do explanations change trust in AI credit decisions?"),
    ("Low-code platforms and IT governance", "How do companies govern citizen development on low-code platforms?"),
    ("Digital product passports in manufacturing", "What data-sharing barriers block digital product passports?"),
    ("ERP cloud migration in the public sector", "Which factors explain failed ERP cloud migrations in public agencies?"),
    ("Algorithmic management in gig work", "How do gig workers respond to algorithmic management in DACH?"),
    ("Data spaces and Gaia-X participation", "Why do firms join or avoid European data spaces?"),
    ("Process mining for ESG reporting", "Can process mining make ESG reporting auditable?"),
    ("Cybersecurity awareness training effectiveness", "Which training designs reduce phishing susceptibility?"),
    ("AI copilots and software developer productivity", "How do AI copilots change developer productivity and skills?"),
    ("Platform ecosystems in agriculture", "How do digital platforms reshape value capture in agriculture?"),
    ("Robotic process automation after deployment", "Why do RPA bots get abandoned after deployment?"),
]


class FakeProvider(Provider):
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.rng = random.Random(f"{name}-{seed}")

    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        picks = self.rng.sample(TOPIC_POOL, 3)
        lines = [f"TOPIC: {title} | {rq}" for title, rq in picks]
        return ResearchResult(provider=self.name, strategy_id=strategy_id, text="\n".join(lines),
                              sources=["https://example.org/source"])

    def judge(self, prompt: str) -> Judgement:
        # Deterministic per topic, so all fake judges tend to agree.
        title = re.search(r"Proposed topic: (.*)", prompt)
        score = 2 + (sum(map(ord, title.group(1) if title else "")) % 4)  # 2..5
        return Judgement(novelty=score, feasibility=score, data_access=score, demand=score, fit=score,
                         passes=score >= 4, reason="fake judgement")


class FakeCoordinator:
    """Plays Claude's coordinator jobs: plan, extract, same_topic."""

    def plan(self, prompt: str) -> str:
        return prompt

    def extract(self, prompt: str) -> CandidateList:
        topics = []
        for title, rq in re.findall(r"TOPIC: (.*?) \| (.*)", prompt):
            topics.append(TopicCandidate(title=title, research_question=rq, rationale="demo", method="survey",
                                         data_sources=["demo"], source_urls=["https://example.org/source"],
                                         closest_work="Doe (2023) Demo paper",
                                         whats_new="moves the question to Austrian SMEs"))
        return CandidateList(topics=topics)

    def same_topic(self, prompt: str) -> bool:
        return False

    def rank(self, prompt: str) -> Ranking:
        ids = [int(i) for i in re.findall(r"\[id (\d+)\]", prompt)]
        k = int(re.search(r"pick the (\d+) best", prompt).group(1))
        return Ranking(picks=[RankedPick(id=i, reason="demo") for i in ids[:k]])

    def literature_review(self, brief: str, system: str) -> ResearchResult:
        title = re.search(r"Topic: (.*)", brief).group(1)
        return ResearchResult(provider="claude", strategy_id=-1, text=f"Review of {title}")

    def assess_review(self, prompt: str) -> LitReview:
        # Deterministic: about half pass.
        title = re.search(r"Topic: (.*)", prompt).group(1)
        ok = len(title) % 2 == 0
        return LitReview(already_answered=not ok, gap_is_real=ok, enough_literature=True, data_reachable=True,
                         passes=ok, summary="demo review", key_references=["Doe (2024) Demo paper"],
                         reason="gap confirmed" if ok else "already answered by prior work")
