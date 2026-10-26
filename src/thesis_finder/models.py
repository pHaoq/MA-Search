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


class RankedPick(BaseModel):
    id: int = Field(description="The topic id from the list")
    reason: str = Field(description="Why it ranks this high, one sentence")


class Ranking(BaseModel):
    picks: List[RankedPick] = Field(description="The best topics, best first")


class LitReview(BaseModel):
    """Outcome of the literature review of one top-ranked topic."""

    already_answered: bool = Field(description="The research question is already answered by existing work")
    gap_is_real: bool = Field(description="The literature confirms an open gap this thesis would fill")
    enough_literature: bool = Field(description="There is enough prior work to build a thesis on")
    data_reachable: bool = Field(description="A master student can realistically get the needed data")
    passes: bool
    summary: str = Field(description="What the literature says about this topic, 3 to 6 sentences")
    key_references: List[str] = Field(description="The most relevant papers: authors, year, title, URL or DOI")
    reason: str = Field(description="Main reason it passes or fails, one sentence")


class SameTopic(BaseModel):
    same: bool


class ResearchResult(BaseModel):
    provider: str
    strategy_id: int  # -1 for a literature review
    text: str
    sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None
