"""Write the stored topics as a Markdown shortlist."""

from __future__ import annotations

from datetime import date

from . import strategies
from .store import TopicStore


def markdown_report(store: TopicStore, limit: int = 20) -> str:
    valid = store.ranked_valid(limit)
    lines = [
        f"# Thesis topic shortlist ({date.today().isoformat()})",
        "",
        f"{store.count('valid')} valid and {store.count('invalid')} rejected topics in the store. "
        f"Top {len(valid)} by mean judge score, then by how often the agents rediscovered them.",
        "",
    ]
    for rank, t in enumerate(valid, 1):
        score = f"{t.mean_score:.1f}" if t.mean_score is not None else "n/a"
        lines += [
            f"## {rank}. {t.title}",
            "",
            f"**Research question:** {t.research_question}",
            "",
            f"- Score {score} / 5, seen {t.times_seen}x, found by {t.found_by} via "
            f"{strategies.get(t.strategy_id).name}",
            f"- Method: {t.method}",
            f"- Data: {', '.join(t.data_sources)}",
            f"- Why it is open: {t.rationale}",
            f"- Judges: {t.reason}",
        ]
        if t.source_urls:
            lines.append("- Sources: " + ", ".join(t.source_urls[:5]))
        lines.append("")
    return "\n".join(lines)
