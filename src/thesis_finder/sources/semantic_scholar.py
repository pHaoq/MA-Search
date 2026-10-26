"""Semantic Scholar: free paper search, a second index next to OpenAlex for the prior-art check.
Works without a key but is rate-limited; set SEMANTIC_SCHOLAR_API_KEY for steadier access."""

from __future__ import annotations

import os
from typing import List

import httpx

from .openalex import Work

API = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_papers(query: str, *, limit: int = 10, timeout: float = 30.0) -> List[Work]:
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    params = {"query": query[:300], "limit": str(limit), "fields": "title,year,citationCount,externalIds,abstract"}
    resp = httpx.get(API, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    works = []
    for item in resp.json().get("data") or []:
        doi = (item.get("externalIds") or {}).get("DOI")
        works.append(
            Work(
                title=item.get("title") or "",
                year=item.get("year"),
                cited_by=item.get("citationCount") or 0,
                doi=f"https://doi.org/{doi}" if doi else None,
                abstract=item.get("abstract") or "",
            )
        )
    return works
