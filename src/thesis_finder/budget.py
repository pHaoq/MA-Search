"""Tracks what a run has spent, so the loop can stop at the budget cap.

Prices are USD as of October 2026. Check the providers' pricing pages and
override them in config.yaml under `prices:` when they change.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# model id -> (input $/1M tokens, output $/1M tokens)
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-4-argon": (2.00, 10.00),  # rises to 4 / 20 on 2026-12-01
    "gemini-3.8-flash": (0.75, 3.75),
    # The Deep Research agent bills the tokens its model uses; no flat rate is
    # published, so this is an estimate. Check your Google AI Studio usage.
    "deep-research-preview-04-2026": (2.00, 12.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6-luna": (0.10, 0.50),
    "text-embedding-3-small": (0.02, 0.0),
    "gemini-embedding-001": (0.15, 0.0),
}

# Per-search fees, USD per call
SEARCH_FEES: Dict[str, float] = {
    "claude": 10.0 / 1000,
    "openai": 10.0 / 1000,
    "gemini": 14.0 / 1000,
}

# Prompt-cache tokens, as a multiple of the input price
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.10

LOG_FIELDS = ["time", "provider", "model", "purpose", "input_tokens", "output_tokens",
              "cache_write_tokens", "cache_read_tokens", "searches", "usd"]


class Budget:
    def __init__(self, limit_usd: float, overrides: Dict[str, List[float]] | None = None,
                 log_path: Optional[str] = None):
        self.limit_usd = limit_usd
        self.prices = dict(DEFAULT_PRICES)
        for model, pair in (overrides or {}).items():
            self.prices[model] = (float(pair[0]), float(pair[1]))
        self.spent_usd = 0.0
        self.by_provider: Dict[str, float] = {}
        self.by_purpose: Dict[str, float] = {}
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()

    def charge(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        searches: int = 0,
        *,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        purpose: str = "",
    ) -> float:
        in_price, out_price = self.prices.get(model, (0.0, 0.0))
        cost = (
            input_tokens * in_price
            + cache_write_tokens * in_price * CACHE_WRITE_FACTOR
            + cache_read_tokens * in_price * CACHE_READ_FACTOR
            + output_tokens * out_price
        ) / 1e6
        cost += searches * SEARCH_FEES.get(provider, 0.0)
        with self._lock:
            self.spent_usd += cost
            self.by_provider[provider] = self.by_provider.get(provider, 0.0) + cost
            key = f"{provider} {purpose}".strip()
            self.by_purpose[key] = self.by_purpose.get(key, 0.0) + cost
            self._log([datetime.now(timezone.utc).isoformat(timespec="seconds"), provider, model, purpose,
                       input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, searches, f"{cost:.5f}"])
        return cost

    def _log(self, row: list) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(LOG_FIELDS)
            writer.writerow(row)

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.limit_usd

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def summary(self) -> str:
        parts = [f"{k}: ${v:.2f}" for k, v in sorted(self.by_purpose.items(), key=lambda kv: -kv[1])]
        return ", ".join(parts) or "nothing spent"


def summarize_log(path: str) -> str:
    """Totals from the cost log, by provider and purpose, across all runs."""
    p = Path(path)
    if not p.exists():
        return f"No cost log yet at {path}."
    totals: Dict[Tuple[str, str, str], List[float]] = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["provider"], row["purpose"], row["model"])
            t = totals.setdefault(key, [0, 0, 0, 0, 0.0])
            t[0] += 1
            t[1] += int(row["input_tokens"]) + int(row["cache_write_tokens"]) + int(row["cache_read_tokens"])
            t[2] += int(row["output_tokens"])
            t[3] += int(row["searches"])
            t[4] += float(row["usd"])
    lines = [f"{'provider':8} {'purpose':12} {'model':30} {'calls':>5} {'in tok':>10} {'out tok':>9} {'search':>6} {'usd':>8}"]
    for (prov, purpose, model), t in sorted(totals.items(), key=lambda kv: -kv[1][4]):
        lines.append(f"{prov:8} {purpose:12} {model:30} {t[0]:5d} {t[1]:10d} {t[2]:9d} {t[3]:6d} {t[4]:8.2f}")
    lines.append(f"Total: ${sum(t[4] for t in totals.values()):.2f}")
    return "\n".join(lines)
