"""OpenAlex: free index of 250M+ scholarly works. Used for seeding searches and the novelty check."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

API = "https://api.openalex.org/works"


@dataclass
class Work:
    title: str
    year: Optional[int]
    cited_by: int
    doi: Optional[str]
    abstract: str

    def short(self, max_abstract: int = 400) -> str:
        abstract = self.abstract[:max_abstract] + ("..." if len(self.abstract) > max_abstract else "")
        return f"- {self.title} ({self.year}, cited {self.cited_by}) {self.doi or ''}\n  {abstract}"


def _abstract(inverted: Optional[dict]) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, poses in inverted.items() for pos in poses]
    return " ".join(word for _, word in sorted(positions))


def search_works(
    query: str,
    *,
    limit: int = 10,
    sort: Optional[str] = None,
    from_year: Optional[int] = None,
    timeout: float = 30.0,
) -> List[Work]:
    params = {"search": query, "per-page": str(limit)}
    if sort:
        params["sort"] = sort
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = httpx.get(API, params=params, timeout=timeout)
    resp.raise_for_status()
    works = []
    for item in resp.json().get("results", []):
        works.append(
            Work(
                title=item.get("display_name") or "",
                year=item.get("publication_year"),
                cited_by=item.get("cited_by_count") or 0,
                doi=item.get("doi"),
                abstract=_abstract(item.get("abstract_inverted_index")),
            )
        )
    return works


def related_work(title: str, research_question: str, limit: int = 5) -> List[Work]:
    """Closest existing papers to a proposed topic, for the novelty check."""
    try:
        return search_works(f"{title} {research_question}"[:300], limit=limit)
    except httpx.HTTPError:
        return []
