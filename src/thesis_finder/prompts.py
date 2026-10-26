"""All prompt text in one place, so it is easy to tune."""

from __future__ import annotations

from datetime import date
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


def fresh_since(months: int, today: date | None = None) -> str:
    """The start of the freshness window, e.g. 'April 2025' for 18 months before October 2026."""
    today = today or date.today()
    index = today.year * 12 + today.month - 1 - months
    return date(index // 12, index % 12 + 1, 1).strftime("%B %Y")


ORIGINALITY = """What makes a topic original enough:
- It must not repeat a published study. Name the closest existing paper or project for each topic
  and say exactly what the thesis adds: a new context (country, industry, company size), a new
  method, new data, or a new population. "Nobody has studied X in general" is not enough.
- Prefer gaps that opened recently (new regulation, technology, market change) over old,
  well-studied questions.
- Do not take a paper's title or research question and rephrase it."""


def planner_prompt(
    profile: Profile,
    strategy: Strategy,
    provider: str,
    avoid: str,
    seed_papers: List[Work],
    topics_per_report: int,
    since: str = "",
) -> str:
    seeds = "\n".join(w.short() for w in seed_papers) or "(none)"
    fresh = f" Only use calls, agendas, projects and job ads from {since} or later." if since else ""
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
1. Restate the goal and name concrete sources to check, current year first.{fresh}
2. Ask for {topics_per_report} distinct candidate thesis topics that fit the profile and are NOT in the lists above.
3. For each topic ask for: a title, one research question, why it is open (with source URLs), the closest
   existing paper and what the thesis adds beyond it, a feasible method, and where the data would come from.
4. Ask the agent to cite a URL for every claim.
5. Include these rules:
{ORIGINALITY}
Return only the brief."""


def template_brief(
    profile: Profile,
    strategy: Strategy,
    avoid: str,
    seed_papers: List[Work],
    topics_per_report: int,
    since: str = "",
) -> str:
    """A research brief built without an LLM call (planner: template)."""
    seeds = "\n".join(w.short(200) for w in seed_papers)
    seed_part = f"\nRecent papers from OpenAlex (a starting point; find what they leave open):\n{seeds}\n" if seeds else ""
    fresh = (f"\nOnly use calls for papers, agendas, funded projects and job ads from {since} or later; "
             "older ones are likely answered by now.") if since else ""
    return f"""Find candidate master thesis topics for this student.

Student profile:
{profile.as_text()}

Strategy: {strategy.name}
Goal: {strategy.goal}
Where to look: {strategy.where_to_look}{fresh}
{seed_part}
{avoid}

{ORIGINALITY}

Deliver {topics_per_report} distinct topics that fit the profile and are not in the lists above.
For each: a title, one research question, why it is open (with source URLs), the closest existing
paper or project (authors, year, title) and what this thesis adds beyond it, a feasible method,
and where the data would come from. Cite a URL for every claim. Keep the report concise."""


def deep_search_system() -> str:
    return (
        "You are a research assistant helping a student find a master thesis topic. "
        "Search the web thoroughly, read primary sources, and cite a URL for every claim. "
        "Prefer sources from the last 18 months. Be concrete; skip generic topics and topics a published "
        "paper already answers."
    )


def extract_prompt(report: str, strategy: Strategy, max_topics: int = 8) -> str:
    return f"""Below is a research report written for the strategy "{strategy.name}".
Extract the distinct candidate master thesis topics it proposes, at most {max_topics}, the strongest first.
Keep every field short: rationale at most two sentences, method and each data source one short phrase,
at most 3 source URLs per topic, taken from the report. Do not invent topics or URLs.
closest_work: the closest existing paper or project the report names for the topic (short citation).
whats_new: what the thesis adds beyond that work, in one sentence. Leave closest_work and whats_new
empty if the report does not say; do not make them up.

REPORT:
{report}"""


def judge_prompt(profile: Profile, topic: TopicCandidate, related: List[Work], timeframe: int) -> str:
    papers = "\n".join(w.short(250) for w in related) or "(no close papers found)"
    return f"""You review a proposed master thesis topic. Be strict: most proposals should fail.

Student profile:
{profile.as_text()}

Proposed topic: {topic.title}
Research question: {topic.research_question}
Why it is open: {topic.rationale}
Method: {topic.method}
Data: {', '.join(topic.data_sources)}
Sources: {', '.join(topic.source_urls) or 'none given'}
Closest existing work, as named by the proposer: {topic.closest_work or 'not named'}
What it adds beyond that: {topic.whats_new or 'not stated'}

Closest existing papers (OpenAlex and Semantic Scholar search, closest first):
{papers}

Score 1 (poor) to 5 (excellent):
- novelty: the stated addition is concrete and real, and the question is not already answered by
  the papers above or well-known work. A rephrased version of an existing paper scores 1 or 2.
- feasibility: one person can finish it in about {timeframe} months at master level
- data_access: the student can realistically get the data
- demand: there is evidence the research community or industry wants this answered
- fit: matches the student's field, interests and methods
Use 4 or 5 only for clearly strong work; a typical proposal is a 3.
Set passes=true only if this would be a top-10% master thesis topic for this student, one a
demanding supervisor would accept as it stands. Vague, generic or already well-studied topics fail.
In reason, give the main strength, or the main reason it fails."""


def same_topic_prompt(a: TopicCandidate, b: StoredTopic) -> str:
    return f"""Are these two master thesis topics essentially the same topic (same question, same context)?

A: {a.title}. {a.research_question}
B: {b.title}. {b.research_question}"""


def rank_prompt(profile: Profile, topics: List[StoredTopic], top_k: int) -> str:
    listing = "\n\n".join(
        f"[id {t.id}] {t.title}\n  Question: {t.research_question}\n  Why open: {t.rationale}\n"
        f"  Adds beyond {t.closest_work or 'existing work'}: {t.whats_new or 'not stated'}\n"
        f"  Method: {t.method}\n  Judges' mean score: {t.mean_score:.1f}, rediscovered {t.times_seen}x"
        for t in topics
    )
    return f"""You are a demanding thesis supervisor. Below are {len(topics)} candidate master thesis topics
that passed a first screening. Compare them against each other and pick the {top_k} best for this student.

Student profile:
{profile.as_text()}

Judge on: originality (a concrete addition beyond existing work, not a rephrased paper);
a clear, answerable research question; a real and current gap; feasibility for one
person in about {profile.timeframe_months} months; access to data; relevance to the field; fit with the student.
Prefer specific topics over broad ones. Do not pick two topics that are near-duplicates.

Topics:
{listing}

Return exactly {top_k} picks (or fewer if fewer are given), best first, using the ids above."""


def literature_brief(profile: Profile, topic: StoredTopic, related: List[Work]) -> str:
    papers = "\n".join(w.short(250) for w in related) or "(none)"
    return f"""Do a focused literature review for this proposed master thesis in {profile.field}.

Topic: {topic.title}
Research question: {topic.research_question}
Claimed gap: {topic.rationale}
Closest work named by the proposer: {topic.closest_work or 'not named'}
Claimed addition: {topic.whats_new or 'not stated'}
Planned method and data: {topic.method}; {', '.join(topic.data_sources)}

Papers found by an OpenAlex search (a starting point, not complete):
{papers}

Find the 10 to 20 most relevant papers, prioritising peer-reviewed work from the last six years.
Then answer, citing papers for each point:
1. Is this research question already answered? By whom? Does the claimed addition really go beyond them?
2. Is the claimed gap real and still open?
3. Is there enough prior work (theory, methods) to build a master thesis on?
4. Can a master student realistically get the data?
List the key references with authors, year, title and a URL or DOI."""


def assess_review_prompt(topic: StoredTopic, report: str) -> str:
    return f"""Here is a literature review of a proposed master thesis topic.

Topic: {topic.title}
Research question: {topic.research_question}

REVIEW:
{report}

Turn it into a verdict. passes=true only if the question is NOT already answered, the gap is real,
there is enough literature to build on, and the data is reachable. Take key_references from the
review only; do not invent any."""
