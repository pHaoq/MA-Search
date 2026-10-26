"""What every model provider must offer the loop."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List

from ..models import Judgement, ResearchResult

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    name: str

    @abstractmethod
    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        """Run one deep web search on the brief and return the cited report."""

    @abstractmethod
    def judge(self, prompt: str) -> Judgement:
        """Score one topic."""


def urls_in(text: str) -> List[str]:
    seen: List[str] = []
    for url in URL_RE.findall(text):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.append(url)
    return seen
