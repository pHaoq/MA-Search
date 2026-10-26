"""Tracks what a run has spent, so the loop can stop at the budget cap.

Prices are USD as of October 2026. Check the providers' pricing pages and
override them in config.yaml under `prices:` when they change.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

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


class Budget:
    def __init__(self, limit_usd: float, overrides: Dict[str, List[float]] | None = None):
        self.limit_usd = limit_usd
        self.prices = dict(DEFAULT_PRICES)
        for model, pair in (overrides or {}).items():
            self.prices[model] = (float(pair[0]), float(pair[1]))
        self.spent_usd = 0.0
        self.by_provider: Dict[str, float] = {}
        self._lock = threading.Lock()

    def charge(self, provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0, searches: int = 0) -> float:
        in_price, out_price = self.prices.get(model, (0.0, 0.0))
        cost = input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
        cost += searches * SEARCH_FEES.get(provider, 0.0)
        with self._lock:
            self.spent_usd += cost
            self.by_provider[provider] = self.by_provider.get(provider, 0.0) + cost
        return cost

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.limit_usd

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)
