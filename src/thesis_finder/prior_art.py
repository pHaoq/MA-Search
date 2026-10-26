"""Prior-art check: before any judge is paid, look for existing papers that already answer the topic.

OpenAlex and Semantic Scholar (both free) return the closest papers; each is compared to the topic
by embedding similarity. A topic too close to an existing paper is rejected without an LLM call.
The papers found are handed to the judges as the novelty context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .embeddings import Embedder, topic_text
from .models import TopicCandidate
from .sources.openalex import Work

log = logging.getLogger(__name__)

Search = Callable[[str], List[Work]]


@dataclass
class PriorArtResult:
    papers: List[Work] = field(default_factory=list)  # closest first
    closest: Optional[Work] = None
    similarity: float = 0.0

    def already_done(self, cutoff: float) -> bool:
        return self.closest is not None and self.similarity >= cutoff


def paper_text(work: Work) -> str:
    return f"{work.title}. {work.abstract[:400]}"


class PriorArtChecker:
    def __init__(self, embedder: Embedder, searches: List[Search], cutoff: float, keep: int = 8):
        self.embedder = embedder
        self.searches = searches
        self.cutoff = cutoff
        self.keep = keep

    def check(self, topic: TopicCandidate, topic_embedding: Optional[np.ndarray] = None) -> PriorArtResult:
        query = f"{topic.title} {topic.research_question}"
        papers: List[Work] = []
        seen = set()
        for search in self.searches:
            try:
                found = search(query)
            except Exception as exc:  # a free API being down should not stop the loop
                log.debug("Prior-art search failed: %s", exc)
                continue
            for work in found:
                key = (work.doi or work.title).lower()
                if work.title and key not in seen:
                    seen.add(key)
                    papers.append(work)
        if not papers:
            return PriorArtResult()

        if topic_embedding is None:
            topic_embedding = self.embedder.embed([topic_text(topic.title, topic.research_question)])[0]
        vectors = self.embedder.embed([paper_text(w) for w in papers])
        sims = vectors @ topic_embedding
        order = np.argsort(-sims)
        ranked = [papers[i] for i in order]
        best = int(order[0])
        result = PriorArtResult(papers=ranked[: self.keep], closest=papers[best], similarity=float(sims[best]))
        log.debug("  prior art %.2f for %r: %s", result.similarity, topic.title, result.closest.title)
        return result
