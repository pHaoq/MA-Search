"""Data shapes shared by every agent."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """Who the thesis is for. Every prompt includes this."""

    field: str = "Information Systems"
    degree: str = "Master"
    country: str = "Austria"
    university: str = ""
    interests: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=lambda: ["English", "German"])
    timeframe_months: int = 6
    notes: str = ""

    def as_text(self) -> str:
        lines = [
            f"Field: {self.field} ({self.degree} thesis)",
            f"Country: {self.country}" + (f", university: {self.university}" if self.university else ""),
            f"Interests: {', '.join(self.interests) or 'open'}",
            f"Methods I can use: {', '.join(self.methods) or 'open'}",
            f"Languages: {', '.join(self.languages)}",
            f"Time for the thesis: about {self.timeframe_months} months, one person",
        ]
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class TopicCandidate(BaseModel):
    """A thesis topic as extracted from a deep-search report."""

    title: str = Field(description="Short topic title, at most 15 words")
    research_question: str = Field(description="One concrete research question")
    rationale: str = Field(description="Why this is open and relevant, citing what the sources say")
    method: str = Field(description="A realistic method for a master thesis")
    data_sources: List[str] = Field(description="Where the data would come from")
    source_urls: List[str] = Field(description="URLs from the report that support this topic")


class CandidateList(BaseModel):
    topics: List[TopicCandidate]


class ResearchBrief(BaseModel):
    """What the planner hands to one deep-search agent."""

    prompt: str = Field(description="The full research brief for the deep-search agent")


class Judgement(BaseModel):
    """One model's verdict on one topic. Scores are 1 (poor) to 5 (excellent)."""

    novelty: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    data_access: int = Field(ge=1, le=5)
    demand: int = Field(ge=1, le=5, description="Evidence the research community or industry wants this answered")
    fit: int = Field(ge=1, le=5, description="Fit with the student's profile")
    passes: bool
    reason: str = Field(description="One or two sentences; if it fails, the main reason")

    @property
    def mean(self) -> float:
        return (self.novelty + self.feasibility + self.data_access + self.demand + self.fit) / 5


class SameTopic(BaseModel):
    same: bool


class ResearchResult(BaseModel):
    provider: str
    strategy_id: int
    text: str
    sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None
