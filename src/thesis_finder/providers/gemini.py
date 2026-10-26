"""Gemini: the Deep Research agent (Interactions API) for search, Gemini 4 Argon for judging."""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from ..budget import FLEX_FACTOR, Budget
from ..config import ModelConfig
from ..models import Judgement, ResearchResult
from .base import Provider, ProviderError, urls_in

POLL_SECONDS = 15
MAX_WAIT_SECONDS = 30 * 60
DONE = {"completed", "failed", "cancelled", "incomplete", "budget_exceeded"}

log = logging.getLogger(__name__)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, models: ModelConfig, budget: Budget, flex_search: bool = False,
                 flex_judge: bool = False):
        # Flex requests can sit in a queue; allow 15 minutes (the value is in milliseconds).
        self.client = genai.Client(http_options=types.HttpOptions(timeout=900_000))
        self.models = models
        self.budget = budget
        self.flex_search = flex_search
        self.flex_judge = flex_judge

    def _with_flex(self, use_flex: bool, call, flex_kwargs: dict, plain_kwargs: dict):
        """Try flex (half price) first; if it is refused or has no capacity, run at standard."""
        if use_flex:
            try:
                return call(**flex_kwargs), True
            except Exception as exc:  # 429/503 no capacity, or flex not offered for this model
                log.info("Gemini flex unavailable (%s); using standard processing.", exc)
        return call(**plain_kwargs), False

    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        agent = self.models.gemini_agent
        kwargs = {"agent": agent, "input": f"{system}\n\n{brief}", "background": True}
        interaction, flex = self._with_flex(self.flex_search, self.client.interactions.create,
                                            {**kwargs, "service_tier": "flex"}, kwargs)
        waited = 0
        while interaction.status not in DONE:
            if waited >= MAX_WAIT_SECONDS:
                raise ProviderError(f"Gemini Deep Research timed out after {waited}s")
            time.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            interaction = self.client.interactions.get(interaction.id)

        usage = interaction.usage
        tier = str(getattr(interaction, "service_tier", None) or ("flex" if flex else "")).lower()
        if usage:
            self.budget.charge(
                self.name,
                agent,
                usage.total_input_tokens or 0,
                (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0),
                purpose="search",
                token_factor=FLEX_FACTOR if tier.endswith("flex") else 1.0,
            )
        if interaction.status != "completed":
            raise ProviderError(f"Gemini Deep Research ended with status {interaction.status}")
        text = interaction.output_text or ""
        return ResearchResult(provider=self.name, strategy_id=strategy_id, text=text, sources=urls_in(text))

    def judge(self, prompt: str) -> Judgement:
        model = self.models.gemini_judge
        def config(**extra):
            return types.GenerateContentConfig(response_mime_type="application/json", response_schema=Judgement,
                                               **extra)

        response, flex = self._with_flex(
            self.flex_judge,
            self.client.models.generate_content,
            {"model": model, "contents": prompt, "config": config(service_tier="flex")},
            {"model": model, "contents": prompt, "config": config()},
        )
        meta = response.usage_metadata
        if meta:
            self.budget.charge(
                self.name,
                model,
                meta.prompt_token_count or 0,
                (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
                purpose="judge",
                token_factor=FLEX_FACTOR if flex else 1.0,
            )
        if isinstance(response.parsed, Judgement):
            return response.parsed
        if response.text:
            return Judgement.model_validate_json(response.text)
        raise ProviderError("Gemini returned no judgement")
