"""Gemini: the Deep Research agent (Interactions API) for search, Gemini 4 Argon for judging."""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from ..budget import Budget
from ..config import ModelConfig
from ..models import Judgement, ResearchResult
from .base import Provider, ProviderError, urls_in

POLL_SECONDS = 15
MAX_WAIT_SECONDS = 30 * 60
DONE = {"completed", "failed", "cancelled", "incomplete", "budget_exceeded"}


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, models: ModelConfig, budget: Budget):
        self.client = genai.Client()
        self.models = models
        self.budget = budget

    def deep_search(self, brief: str, system: str, strategy_id: int) -> ResearchResult:
        agent = self.models.gemini_agent
        interaction = self.client.interactions.create(
            agent=agent,
            input=f"{system}\n\n{brief}",
            background=True,
        )
        waited = 0
        while interaction.status not in DONE:
            if waited >= MAX_WAIT_SECONDS:
                raise ProviderError(f"Gemini Deep Research timed out after {waited}s")
            time.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            interaction = self.client.interactions.get(interaction.id)

        usage = interaction.usage
        if usage:
            self.budget.charge(
                self.name,
                agent,
                usage.total_input_tokens or 0,
                (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0),
            )
        if interaction.status != "completed":
            raise ProviderError(f"Gemini Deep Research ended with status {interaction.status}")
        text = interaction.output_text or ""
        return ResearchResult(provider=self.name, strategy_id=strategy_id, text=text, sources=urls_in(text))

    def judge(self, prompt: str) -> Judgement:
        model = self.models.gemini_judge
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Judgement,
            ),
        )
        meta = response.usage_metadata
        if meta:
            self.budget.charge(
                self.name,
                model,
                meta.prompt_token_count or 0,
                (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
            )
        if isinstance(response.parsed, Judgement):
            return response.parsed
        if response.text:
            return Judgement.model_validate_json(response.text)
        raise ProviderError("Gemini returned no judgement")
