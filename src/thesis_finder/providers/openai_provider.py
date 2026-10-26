"""OpenAI: GPT with the web search tool in background mode for search, structured output for judging."""

from __future__ import annotations

import time

from openai import OpenAI

from ..budget import Budget
from ..config import ModelConfig
from ..models import Judgement, ResearchResult
from .base import Provider, ProviderError, urls_in

POLL_SECONDS = 10
MAX_WAIT_SECONDS = 30 * 60
RUNNING = {"queued", "in_progress"}


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, models: ModelConfig, budget: Budget):
        self.client = OpenAI()
        self.models = models
        self.budget = budget

    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        model = self.models.openai_research
        response = self.client.responses.create(
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
        if response.usage:
            self.budget.charge(self.name, model, response.usage.input_tokens, response.usage.output_tokens, searches)
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
        response = self.client.responses.parse(model=model, input=prompt, text_format=Judgement)
        if response.usage:
            self.budget.charge(self.name, model, response.usage.input_tokens, response.usage.output_tokens)
        if response.output_parsed is None:
            raise ProviderError("OpenAI returned no judgement")
        return response.output_parsed
