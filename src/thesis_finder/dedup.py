"""Decide whether a candidate topic is already in the store (valid or invalid)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .embeddings import Embedder, topic_text
from .models import TopicCandidate
from .store import StoredTopic, TopicStore

# (candidate, stored) -> True if they are the same topic. Used for borderline similarity.
SameTopicJudge = Callable[[TopicCandidate, StoredTopic], bool]


@dataclass
class DedupResult:
    duplicate_of: Optional[StoredTopic]
    similarity: float
    embedding: np.ndarray


class Deduper:
    def __init__(
        self,
        store: TopicStore,
        embedder: Embedder,
        duplicate_threshold: float,
        borderline_threshold: float,
        same_topic: Optional[SameTopicJudge] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.duplicate_threshold = duplicate_threshold
        self.borderline_threshold = borderline_threshold
        self.same_topic = same_topic

    def check(self, candidate: TopicCandidate) -> DedupResult:
        vector = self.embedder.embed([topic_text(candidate.title, candidate.research_question)])[0]

        exact = self.store.find_by_title(candidate.title)
        if exact:
            return DedupResult(exact, 1.0, vector)

        ids, matrix = self.store.embeddings()
        if matrix is None or matrix.shape[1] != vector.shape[0]:
            return DedupResult(None, 0.0, vector)

        sims = matrix @ vector
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        match = self.store.get(ids[best])

        if best_sim >= self.duplicate_threshold:
            return DedupResult(match, best_sim, vector)
        if best_sim >= self.borderline_threshold and self.same_topic and self.same_topic(candidate, match):
            return DedupResult(match, best_sim, vector)
        return DedupResult(None, best_sim, vector)
