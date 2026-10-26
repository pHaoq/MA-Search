"""All prompt text in one place, so it is easy to tune."""

from __future__ import annotations

from typing import List

from .models import Profile, TopicCandidate
from .sources.openalex import Work
from .store import StoredTopic
from .strategies import Strategy


def avoid_block(valid: List[StoredTopic], invalid: List[StoredTopic], limit: int = 30) -> str:
    """Known topics, so the agents look elsewhere. Invalid ones carry their reason:
    the exact topic is off the table, but a neighbour that fixes the reason is welcome."""
    lines = []
    if valid:
        lines.append("Already collected (do not propose these again):")
        lines += [f"- {t.title}" for t in valid[-limit:]]
    if invalid:
        lines.append("")
        lines.append(
            "Already rejected, with the reason. Do not propose these again. A related topic is fine "
            "only if it clearly fixes the reason it was rejected:"
        )
        lines += [f"- {t.title} (rejected: {t.reason})" for t in invalid[-limit:]]
    return "\n".join(lines) if lines else "No topics collected yet."


def planner_prompt(
    profile: Profile,
    strategy: Strategy,
    provider: str,
    avoid: str,
    seed_papers: List[Work],
    topics_per_report: int,
) -> str:
    seeds = "\n".join(w.short() for w in seed_papers) or "(none)"
    return f"""You plan research for a student looking for a master thesis topic (a theme, not a job).

Student profile:
{profile.as_text()}

Strategy for this round: {strategy.name}
Goal: {strategy.goal}
Where to look: {strategy.where_to_look}

Starting papers from OpenAlex (may be empty):
{seeds}

{avoid}

Write a research brief for a deep-research agent ({provider}). The brief must:
1. Restate the goal and name concrete sources to check, current year first.
2. Ask for {topics_per_report} distinct candidate thesis topics that fit the profile and are NOT in the lists above.
3. For each topic ask for: a title, one research question, why it is open (with source URLs), a feasible method, and where the data would come from.
4. Ask the agent to cite a URL for every claim.
Return only the brief."""


def template_brief(
    profile: Profile,
    strategy: Strategy,
    avoid: str,
    seed_papers: List[Work],
    topics_per_report: int,
) -> str:
    """A research brief built without an LLM call (planner: template)."""
    seeds = "\n".join(w.short(200) for w in seed_papers)
    seed_part = f"\nStarting papers from OpenAlex:\n{seeds}\n" if seeds else ""
    return f"""Find candidate master thesis topics for this student.

Student profile:
{profile.as_text()}

Strategy: {strategy.name}
Goal: {strategy.goal}
Where to look: {strategy.where_to_look}
{seed_part}
{avoid}

Deliver {topics_per_report} distinct topics that fit the profile and are not in the lists above.
For each: a title, one research question, why it is open (with source URLs), a feasible method,
and where the data would come from. Cite a URL for every claim. Keep the report concise."""


def deep_search_system() -> str:
    return (
        "You are a research assistant helping a student find a master thesis topic. "
        "Search the web thoroughly, read primary sources, and cite a URL for every claim. "
        "Prefer sources from the last two years. Be concrete; skip generic topics."
    )


def extract_prompt(report: str, strategy: Strategy) -> str:
    return f"""Below is a research report written for the strategy "{strategy.name}".
Extract every distinct candidate master thesis topic it proposes or clearly implies.
Keep the report's source URLs with the topic they support. Do not invent topics or URLs.

REPORT:
{report}"""


def judge_prompt(profile: Profile, topic: TopicCandidate, related: List[Work], timeframe: int) -> str:
    papers = "\n".join(w.short(250) for w in related) or "(no close papers found)"
    return f"""You review a proposed master thesis topic. Be strict but fair.

Student profile:
{profile.as_text()}

Proposed topic: {topic.title}
Research question: {topic.research_question}
Why it is open: {topic.rationale}
Method: {topic.method}
Data: {', '.join(topic.data_sources)}
Sources: {', '.join(topic.source_urls) or 'none given'}

Closest existing papers (OpenAlex search):
{papers}

Score 1 (poor) to 5 (excellent):
- novelty: not already answered by the papers above or well-known work
- feasibility: one person can finish it in about {timeframe} months at master level
- data_access: the student can realistically get the data
- demand: there is evidence the research community or industry wants this answered
- fit: matches the student's field, interests and methods
Set passes=true only if the topic is a good master thesis for this student overall.
In reason, give the main strength, or the main reason it fails."""


def same_topic_prompt(a: TopicCandidate, b: StoredTopic) -> str:
    return f"""Are these two master thesis topics essentially the same topic (same question, same context)?

A: {a.title}. {a.research_question}
B: {b.title}. {b.research_question}"""
