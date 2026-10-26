"""OpenAI: GPT with the web search tool in background mode for search, structured output for judging.
With flex processing on, calls run at half the token price and fall back to standard when flex
has no capacity."""

from __future__ import annotations

import logging
import time

import openai
from openai import OpenAI

from ..budget import FLEX_FACTOR, Budget
from ..config import ModelConfig
from ..models import Judgement, ResearchResult
from .base import Provider, ProviderError, urls_in

log = logging.getLogger(__name__)

POLL_SECONDS = 10
MAX_WAIT_SECONDS = 30 * 60
RUNNING = {"queued", "in_progress"}


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, models: ModelConfig, budget: Budget, flex_search: bool = False,
                 flex_judge: bool = False):
        # Flex requests can sit in a queue, so allow 15 minutes before the SDK times out.
        self.client = OpenAI(timeout=900.0)
        self.models = models
        self.budget = budget
        self.flex_search = flex_search
        self.flex_judge = flex_judge

    def _create(self, method, flex: bool, **kwargs):
        """Call with flex if enabled; on no capacity (429) or a model without flex, retry at standard."""
        if flex:
            try:
                return method(service_tier="flex", **kwargs)
            except openai.APIStatusError as exc:
                if exc.status_code not in (400, 429):
                    raise
                log.info("OpenAI flex unavailable (%s); using standard processing.", exc.status_code)
        return method(**kwargs)

    def _charge(self, model: str, response, searches: int, purpose: str) -> None:
        usage = response.usage
        if not usage:
            return
        details = getattr(usage, "input_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        factor = FLEX_FACTOR if getattr(response, "service_tier", None) == "flex" else 1.0
        self.budget.charge(self.name, model, usage.input_tokens - cached, usage.output_tokens, searches,
                           cache_read_tokens=cached, purpose=purpose, token_factor=factor)

    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        model = self.models.openai_research
        response = self._create(
            self.client.responses.create,
            self.flex_search,
            model=model,
            instructions=system,
            input=brief,
            tools=[{"type": "web_search", "search_context_size": "high"}],
            reasoning={"effort": "high"},
            background=True,
        )
        waited = 0
        while response.status in RUNNING:
            if waited >= MAX_WAIT_SECONDS:
                raise ProviderError(f"OpenAI deep search timed out after {waited}s")
            time.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            response = self.client.responses.retrieve(response.id)

        searches = sum(1 for item in response.output if item.type == "web_search_call")
        self._charge(model, response, searches, "search")
        if response.status != "completed":
            raise ProviderError(f"OpenAI deep search ended with status {response.status}")

        text = response.output_text
        sources = []
        for item in response.output:
            if item.type != "message":
                continue
            for part in item.content:
                for ann in getattr(part, "annotations", None) or []:
                    if ann.type == "url_citation" and ann.url not in sources:
                        sources.append(ann.url)
        for url in urls_in(text):
            if url not in sources:
                sources.append(url)
        return ResearchResult(provider=self.name, strategy_id=strategy_id, text=text, sources=sources)

    def judge(self, prompt: str) -> Judgement:
        model = self.models.openai_judge
        response = self._create(self.client.responses.parse, self.flex_judge, model=model, input=prompt,
                                text_format=Judgement)
        self._charge(model, response, 0, "judge")
        if response.output_parsed is None:
            raise ProviderError("OpenAI returned no judgement")
        return response.output_parsed
