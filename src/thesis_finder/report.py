"""Write the stored topics as a Markdown shortlist: checked topics first, then the best valid ones."""

from __future__ import annotations

from datetime import date
from typing import List

from . import strategies
from .store import StoredTopic, TopicStore


def _topic_lines(rank: int, t: StoredTopic) -> List[str]:
    score = f"{t.mean_score:.1f}" if t.mean_score is not None else "n/a"
    lines = [
        f"### {rank}. {t.title}",
        "",
        f"**Research question:** {t.research_question}",
        "",
        f"- Score {score} / 5, seen {t.times_seen}x, found by {t.found_by} via "
        f"{strategies.get(t.strategy_id).name}",
        f"- Method: {t.method}",
        f"- Data: {', '.join(t.data_sources)}",
        f"- Why it is open: {t.rationale}",
    ]
    if t.whats_new:
        lines.append(f"- Adds beyond {t.closest_work or 'existing work'}: {t.whats_new}")
    if t.lit_review:
        lines.append(f"- Literature review: {t.lit_review.get('summary', '')}")
        refs = t.lit_review.get("key_references") or []
        if refs:
            lines.append("- Key references:")
            lines += [f"    - {r}" for r in refs[:10]]
    else:
        lines.append(f"- Judges: {t.reason}")
    if t.source_urls:
        lines.append("- Sources: " + ", ".join(t.source_urls[:5]))
    lines.append("")
    return lines


def markdown_report(store: TopicStore, limit: int = 20) -> str:
    checked = store.ranked("checked")
    valid = store.ranked("valid", limit)
    lines = [
        f"# Thesis topic shortlist ({date.today().isoformat()})",
        "",
        f"{store.count('checked')} checked, {store.count('valid')} valid, {store.count('parked')} parked and "
        f"{store.count('invalid')} invalid topics in the store.",
        "",
        "## Checked topics",
        "",
        "These passed the first screening, were ranked into the top picks by the strongest model, "
        "and passed a literature review.",
        "",
    ]
    if not checked:
        lines += ["None yet. Run more rounds, or `thesis-finder review`.", ""]
    for rank, t in enumerate(checked, 1):
        lines += _topic_lines(rank, t)
    lines += ["## Best valid topics (not yet reviewed)", ""]
    for rank, t in enumerate(valid, 1):
        lines += _topic_lines(rank, t)
    return "\n".join(lines)
