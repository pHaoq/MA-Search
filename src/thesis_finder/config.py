"""Load config.yaml (falls back to config.example.yaml defaults)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from .models import Profile


class ModelConfig(BaseModel):
    claude_research: str = "claude-sonnet-5-5"
    claude_planner: str = "claude-opus-5-5"
    claude_judge: str = "claude-sonnet-5-5"
    claude_fast: str = "claude-haiku-5-5"
    claude_ranker: str = "claude-fable-5-1"  # the strongest model; ranks valid topics in the review
    claude_reviewer: str = "claude-sonnet-5-5"  # literature review of the top picks
    gemini_agent: str = "deep-research-preview-04-2026"
    gemini_judge: str = "gemini-4-argon"
    openai_research: str = "gpt-6.1-sol"
    openai_judge: str = "gpt-6.1-sol"


class EmbeddingConfig(BaseModel):
    # "openai", "gemini" or "local" (free, word-overlap only; fine for testing)
    provider: str = "openai"
    model: str = "text-embedding-3-small"


class ClaudeSearchConfig(BaseModel):
    """Limits for one Claude deep search. Every search result and fetched page is added to
    the context and re-read on each step, so these numbers drive most of Claude's cost."""

    max_searches: int = 8
    max_fetches: int = 4
    max_fetch_tokens: int = 15000  # cap on the size of one fetched page
    effort: str = "medium"


class LoopConfig(BaseModel):
    budget_usd: float = 5.0
    target_valid: int = 40
    saturation_rounds: int = 3
    saturation_new_ratio: float = 0.10
    max_rounds: int = 60
    topics_per_report: int = 5
    # "template": build research briefs from a fixed template (free).
    # "claude": let Claude write each brief (one extra Claude call per search).
    planner: str = "template"


class DedupConfig(BaseModel):
    duplicate_threshold: float = 0.88  # cosine similarity at or above: same topic
    borderline_threshold: float = 0.78  # between the two: ask Claude Haiku


class ValidationConfig(BaseModel):
    min_pass_votes: int = 2
    min_mean_score: float = 4.0
    min_novelty: int = 3  # every judge must give at least this novelty score


class ReviewConfig(BaseModel):
    """The step that promotes valid topics to checked."""

    every_rounds: int = 6  # one full rotation: every provider has tried every strategy
    at_end_of_run: bool = True
    top_k: int = 5
    park_after: int = 3  # a valid topic passed over in this many reviews is parked
    max_candidates: int = 60  # most valid topics handed to the ranker at once


class Config(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    providers: List[str] = Field(default_factory=lambda: ["claude", "gemini", "openai"])
    models: ModelConfig = Field(default_factory=ModelConfig)
    claude_search: ClaudeSearchConfig = Field(default_factory=ClaudeSearchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    store_path: str = "data/topics.db"
    cost_log_path: str = "data/costs.csv"
    # Override or add prices: {"model-id": [input_usd_per_1m, output_usd_per_1m]}
    prices: Dict[str, List[float]] = Field(default_factory=dict)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def load_config(path: Optional[str] = None) -> Config:
    load_dotenv()
    candidates = [Path(path)] if path else [Path("config.yaml"), Path("config.example.yaml")]
    for candidate in candidates:
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text()) or {}
            return Config.model_validate(data)
    if path:
        raise FileNotFoundError(path)
    return Config()
