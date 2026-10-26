"""Claude: deep search with the web search and web fetch tools, plus the coordinator jobs
(planning, extracting topics, duplicate checks)."""

from __future__ import annotations

import logging
from typing import List, Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from ..budget import Budget
from ..config import ClaudeSearchConfig, ModelConfig
from ..models import CandidateList, Judgement, LitReview, Ranking, ResearchBrief, ResearchResult, SameTopic
from .base import Provider, ProviderError, urls_in

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# Server-side fallback on a refusal; routes by refusal category, no model list to maintain.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_CONTINUATIONS = 5


def _log_tool_call(block) -> None:
    args = block.input if isinstance(block.input, dict) else {}
    if block.name == "web_search":
        log.info("      searching: %s", args.get("query", ""))
    elif block.name == "web_fetch":
        log.info("      reading:   %s", args.get("url", ""))


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, models: ModelConfig, budget: Budget, search: ClaudeSearchConfig | None = None):
        self.client = anthropic.Anthropic()
        self.models = models
        self.budget = budget
        self.search = search or ClaudeSearchConfig()

    def deep_search(self, brief: str, system: str, strategy_id: int, *, model: str | None = None,
                    purpose: str = "search") -> ResearchResult:
        model = model or self.models.claude_research
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": self.search.max_searches},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": self.search.max_fetches,
             "max_content_tokens": self.search.max_fetch_tokens},
        ]
        messages: List[dict] = [{"role": "user", "content": brief}]
        content: list = []
        for _ in range(MAX_CONTINUATIONS + 1):
            # Streamed so each search and fetch shows up as it happens; a review can take minutes.
            with self.client.beta.messages.stream(
                model=model,
                max_tokens=16000,
                system=system,
                tools=tools,
                messages=messages,
                output_config={"effort": self.search.effort},
                fallbacks="default",
                betas=[FALLBACK_BETA],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_stop" and event.content_block.type == "server_tool_use":
                        _log_tool_call(event.content_block)
                response = stream.get_final_message()
            self._charge(model, response.usage, purpose)
            content.extend(response.content)
            if response.stop_reason == "refusal":
                raise ProviderError("Claude declined this research brief")
            if response.stop_reason != "pause_turn":
                break
            # A long server-tool turn paused; send it back so Claude continues where it stopped.
            messages = [
                {"role": "user", "content": brief},
                {"role": "assistant", "content": response.content},
            ]

        text_parts, sources = [], []
        for block in content:
            if block.type != "text":
                continue
            text_parts.append(block.text)
            for citation in block.citations or []:
                url = getattr(citation, "url", None)
                if url and url not in sources:
                    sources.append(url)
        text = "".join(text_parts)
        for url in urls_in(text):
            if url not in sources:
                sources.append(url)
        return ResearchResult(provider=self.name, strategy_id=strategy_id, text=text, sources=sources)

    def judge(self, prompt: str) -> Judgement:
        return self._parse(self.models.claude_judge, prompt, Judgement, fallback=True, purpose="judge")

    # Coordinator jobs

    def plan(self, prompt: str) -> str:
        return self._parse(self.models.claude_planner, prompt, ResearchBrief, fallback=True, purpose="plan").prompt

    def extract(self, prompt: str) -> CandidateList:
        return self._parse(self.models.claude_fast, prompt, CandidateList, fallback=False, max_tokens=16000,
                           purpose="extract", effort="low")

    def same_topic(self, prompt: str) -> bool:
        return self._parse(self.models.claude_fast, prompt, SameTopic, fallback=False, max_tokens=256,
                           purpose="dedup").same

    # Review step

    def rank(self, prompt: str) -> Ranking:
        """The strongest model ranks valid topics against each other."""
        return self._parse(self.models.claude_ranker, prompt, Ranking, fallback=True, max_tokens=16000,
                           purpose="rank")

    def literature_review(self, brief: str, system: str) -> ResearchResult:
        return self.deep_search(brief, system, -1, model=self.models.claude_reviewer, purpose="review")

    def assess_review(self, prompt: str) -> LitReview:
        return self._parse(self.models.claude_reviewer, prompt, LitReview, fallback=True, purpose="review")

    def _parse(self, model: str, prompt: str, schema: Type[T], *, fallback: bool, max_tokens: int = 8000,
               purpose: str = "", effort: str | None = None) -> T:
        kwargs = {}
        if fallback:
            kwargs = {"fallbacks": "default", "betas": [FALLBACK_BETA]}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        try:
            response = self.client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **kwargs,
            )
        except ValidationError as exc:
            # Usually the answer hit max_tokens and the JSON was cut off. The SDK raises before
            # we see the usage, so charge an estimate to keep the budget honest.
            self.budget.charge(self.name, model, len(prompt) // 4, max_tokens, purpose=purpose)
            raise ProviderError(f"Claude's {schema.__name__} was cut off or malformed") from exc
        self._charge(model, response.usage, purpose)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise ProviderError(f"Claude returned no {schema.__name__} (stop_reason={response.stop_reason})")
        return response.parsed_output

    def _charge(self, model: str, usage, purpose: str) -> None:
        searches = 0
        if getattr(usage, "server_tool_use", None):
            searches = usage.server_tool_use.web_search_requests or 0
        self.budget.charge(
            self.name, model, usage.input_tokens, usage.output_tokens, searches,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            purpose=purpose,
        )
