"""Turn topic text into vectors so near-duplicates can be found by meaning."""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol

import numpy as np

from .budget import Budget


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalEmbedder:
    """Free hashed bag-of-words vectors. Only catches topics that share words;
    good for tests and dry runs, use an API embedder for real runs."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]{3,}", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _unit(out)


class OpenAIEmbedder:
    def __init__(self, model: str, budget: Optional[Budget] = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.budget = budget

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        if self.budget:
            self.budget.charge("openai", self.model, input_tokens=resp.usage.prompt_tokens)
        return _unit(np.array([d.embedding for d in resp.data], dtype=np.float32))


class GeminiEmbedder:
    def __init__(self, model: str, budget: Optional[Budget] = None):
        from google import genai

        self.client = genai.Client()
        self.model = model
        self.budget = budget

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.models.embed_content(model=self.model, contents=texts)
        if self.budget:
            # The embed response carries no token count; estimate ~4 characters per token.
            self.budget.charge("gemini", self.model, input_tokens=sum(len(t) for t in texts) // 4)
        return _unit(np.array([e.values for e in resp.embeddings], dtype=np.float32))


def make_embedder(provider: str, model: str, budget: Optional[Budget] = None) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model, budget)
    if provider == "gemini":
        return GeminiEmbedder(model, budget)
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")


def topic_text(title: str, research_question: str) -> str:
    return f"{title}. {research_question}"
