"""OpenAlex: free index of 250M+ scholarly works. Used for seeding searches and the novelty check."""

from __future__ import annotations

import os
import re
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


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "based", "by", "can", "do", "does", "for", "from", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "rather", "than", "that", "the", "their", "them", "this", "to",
    "what", "when", "which", "while", "who", "why", "with", "within", "without", "an", "be", "being", "designed",
}


def keyword_query(text: str, max_words: int = 10) -> str:
    """Turn a long title or question into a short keyword query. OpenAlex answers 400 to long,
    punctuated queries (apostrophes, question marks, and words it reads as AND/OR/NOT)."""
    words, seen = [], set()
    for word in re.findall(r"[A-Za-z0-9]+", text):
        low = word.lower()
        if low in STOPWORDS or len(low) < 3 or low in seen:
            continue
        seen.add(low)
        words.append(low)
        if len(words) >= max_words:
            break
    return " ".join(words)


def search_works(
    query: str,
    *,
    limit: int = 10,
    sort: Optional[str] = None,
    from_year: Optional[int] = None,
    timeout: float = 30.0,
) -> List[Work]:
    params = {"search": keyword_query(query), "per-page": str(limit)}
    if sort:
        params["sort"] = sort
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = httpx.get(API, params=params, timeout=timeout)
    if resp.status_code == 400 and len(params["search"].split()) > 4:
        # Still refused: retry with the first few keywords (usually from the title).
        params["search"] = " ".join(params["search"].split()[:4])
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
        return search_works(f"{title} {research_question}", limit=limit)
    except httpx.HTTPError:
        return []
