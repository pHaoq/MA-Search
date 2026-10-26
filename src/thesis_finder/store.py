"""SQLite topic store: every topic ever seen, valid or invalid, plus run history."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import Judgement, TopicCandidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    norm_title TEXT NOT NULL,
    research_question TEXT NOT NULL,
    rationale TEXT,
    method TEXT,
    data_sources TEXT,          -- JSON list
    source_urls TEXT,           -- JSON list
    strategy_id INTEGER,
    found_by TEXT,
    status TEXT NOT NULL,       -- 'valid' or 'invalid'
    reason TEXT,
    judgements TEXT,            -- JSON {provider: Judgement}
    mean_score REAL,
    embedding BLOB,
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topics_norm ON topics(norm_title);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT NOT NULL,
    ended TEXT,
    stop_reason TEXT,
    spent_usd REAL
);
CREATE TABLE IF NOT EXISTS rounds (
    run_id INTEGER NOT NULL,
    round INTEGER NOT NULL,
    provider TEXT NOT NULL,
    strategy_id INTEGER NOT NULL,
    found INTEGER NOT NULL,
    new INTEGER NOT NULL,
    valid INTEGER NOT NULL
);
"""


def normalize_title(title: str) -> str:
    words = re.findall(r"[a-z0-9]+", title.lower())
    stop = {"a", "an", "the", "of", "in", "on", "for", "and", "to", "with", "how", "what", "why"}
    return " ".join(w for w in words if w not in stop)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StoredTopic:
    id: int
    title: str
    research_question: str
    status: str
    reason: str
    strategy_id: int
    found_by: str
    mean_score: Optional[float]
    times_seen: int
    rationale: str = ""
    method: str = ""
    data_sources: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class TopicStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # topics

    def add_topic(
        self,
        candidate: TopicCandidate,
        *,
        status: str,
        reason: str,
        strategy_id: int,
        found_by: str,
        judgements: dict[str, Judgement],
        embedding: Optional[np.ndarray],
    ) -> int:
        scores = [j.mean for j in judgements.values()]
        mean = sum(scores) / len(scores) if scores else None
        now = _now()
        with self._lock, self.conn:
            cur = self.conn.execute(
                """INSERT INTO topics (title, norm_title, research_question, rationale, method,
                   data_sources, source_urls, strategy_id, found_by, status, reason, judgements,
                   mean_score, embedding, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.title,
                    normalize_title(candidate.title),
                    candidate.research_question,
                    candidate.rationale,
                    candidate.method,
                    json.dumps(candidate.data_sources),
                    json.dumps(candidate.source_urls),
                    strategy_id,
                    found_by,
                    status,
                    reason,
                    json.dumps({k: v.model_dump() for k, v in judgements.items()}),
                    mean,
                    embedding.astype(np.float32).tobytes() if embedding is not None else None,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def mark_seen(self, topic_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE topics SET times_seen = times_seen + 1, last_seen = ? WHERE id = ?",
                (_now(), topic_id),
            )

    def find_by_title(self, title: str) -> Optional[StoredTopic]:
        row = self.conn.execute(
            "SELECT * FROM topics WHERE norm_title = ? LIMIT 1", (normalize_title(title),)
        ).fetchone()
        return self._row(row) if row else None

    def all_topics(self, status: Optional[str] = None) -> List[StoredTopic]:
        if status:
            rows = self.conn.execute("SELECT * FROM topics WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self.conn.execute("SELECT * FROM topics ORDER BY id")
        return [self._row(r) for r in rows]

    def ranked_valid(self, limit: Optional[int] = None) -> List[StoredTopic]:
        sql = "SELECT * FROM topics WHERE status = 'valid' ORDER BY mean_score DESC, times_seen DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row(r) for r in self.conn.execute(sql)]

    def count(self, status: Optional[str] = None) -> int:
        if status:
            return self.conn.execute("SELECT COUNT(*) FROM topics WHERE status = ?", (status,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]

    def embeddings(self) -> tuple[List[int], Optional[np.ndarray]]:
        rows = self.conn.execute("SELECT id, embedding FROM topics WHERE embedding IS NOT NULL").fetchall()
        if not rows:
            return [], None
        ids = [r["id"] for r in rows]
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return ids, matrix

    def get(self, topic_id: int) -> StoredTopic:
        return self._row(self.conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone())

    # runs and rounds

    def start_run(self) -> int:
        with self._lock, self.conn:
            return int(self.conn.execute("INSERT INTO runs (started) VALUES (?)", (_now(),)).lastrowid)

    def end_run(self, run_id: int, stop_reason: str, spent_usd: float) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE runs SET ended = ?, stop_reason = ?, spent_usd = ? WHERE id = ?",
                (_now(), stop_reason, spent_usd, run_id),
            )

    def record_round(self, run_id: int, rnd: int, provider: str, strategy_id: int, found: int, new: int, valid: int) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, rnd, provider, strategy_id, found, new, valid),
            )

    def runs(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()

    def _row(self, r: sqlite3.Row) -> StoredTopic:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        return StoredTopic(
            id=r["id"],
            title=r["title"],
            research_question=r["research_question"],
            status=r["status"],
            reason=r["reason"] or "",
            strategy_id=r["strategy_id"],
            found_by=r["found_by"] or "",
            mean_score=r["mean_score"],
            times_seen=r["times_seen"],
            rationale=r["rationale"] or "",
            method=r["method"] or "",
            data_sources=json.loads(r["data_sources"] or "[]"),
            source_urls=json.loads(r["source_urls"] or "[]"),
            embedding=emb,
        )
