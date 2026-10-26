"""The seven ways of finding a thesis topic. Each deep-search agent gets one per round."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Strategy:
    id: int
    name: str
    goal: str
    where_to_look: str
    # Whether to give the agent a list of papers from OpenAlex as a starting point
    use_openalex: bool = False
    openalex_hint: str = ""


STRATEGIES: List[Strategy] = [
    Strategy(
        id=0,
        name="Conference calls for papers",
        goal=(
            "Find the questions the research community wants answered right now, as stated in "
            "current and upcoming conference calls for papers, track descriptions and workshop calls."
        ),
        where_to_look=(
            "Call for papers, track and workshop pages of ECIS, ICIS, WI (Wirtschaftsinformatik), "
            "HICSS, AMCIS, PACIS and their workshops for the current and next year."
        ),
    ),
    Strategy(
        id=1,
        name="Research agendas and editorials",
        goal=(
            "Find open problems that journal editors and authors explicitly name as important: "
            "special issue calls, editorials, 'research agenda' and 'future research' papers."
        ),
        where_to_look=(
            "Special issue calls and editorials in MIS Quarterly, Information Systems Research, JAIS, "
            "EJIS, JMIS, Business & Information Systems Engineering (BISE), Electronic Markets; "
            "papers titled 'research agenda'."
        ),
        use_openalex=True,
        openalex_hint="research agenda",
    ),
    Strategy(
        id=2,
        name="Replicate or transfer a study",
        goal=(
            "Pick well-cited studies and propose repeating them in a new context: another industry, "
            "country (for example Austria or the DACH region), company size, or technology."
        ),
        where_to_look="Highly cited papers in the field, their stated limitations and contexts studied.",
        use_openalex=True,
        openalex_hint="",
    ),
    Strategy(
        id=3,
        name="Find contradictions",
        goal=(
            "Find questions where published studies reach conflicting results. A thesis that explains "
            "why the results differ is a solid contribution."
        ),
        where_to_look="Meta-analyses, literature reviews, and papers that report mixed or opposite findings.",
        use_openalex=True,
        openalex_hint="mixed results literature review",
    ),
    Strategy(
        id=4,
        name="Funded project databases",
        goal=(
            "Find currently funded research projects and pick a sub-aspect they do not fully cover."
        ),
        where_to_look=(
            "CORDIS (EU Horizon Europe projects, cordis.europa.eu), the FFG project database (Austria), "
            "DFG GEPRIS (Germany, gepris.dfg.de)."
        ),
    ),
    Strategy(
        id=5,
        name="Industry job postings",
        goal=(
            "Find recurring skills, tools or problems in current job ads that show what companies "
            "struggle with, and turn one into a research question."
        ),
        where_to_look=(
            "Job boards in Austria and Germany (karriere.at, AMS, Bundesagentur fuer Arbeit, StepStone, "
            "LinkedIn public listings) and industry reports on skill gaps."
        ),
    ),
    Strategy(
        id=6,
        name="Combine gaps and contexts",
        goal=(
            "Create original topics by combining things that have not been combined yet: a gap named in a "
            "recent paper with a contradiction from another stream, a known finding with a new context "
            "(Austria or DACH, SMEs, the public sector, one specific industry), or two of the student's "
            "interests. Check that the combination has not been published, and say why it matters now."
        ),
        where_to_look=(
            "Recent 'future research' sections and literature reviews (last two years), Austrian and EU "
            "reports on SMEs, industry and the public sector (WKO, Statistik Austria, Eurostat), and the "
            "student's interests and seed papers."
        ),
        use_openalex=True,
        openalex_hint="future research directions",
    ),
]


def get(strategy_id: int) -> Strategy:
    return STRATEGIES[strategy_id]
