"""
Engram Working Memory
---------------------
Fixed token budget. When exceeded, evict the chunk with the lowest
retention score (NOT the oldest — that's MemGPT's FIFO approach).

The last 2 chunks (most recent exchange) are always protected from eviction
so the model never loses track of the immediate conversation turn.
"""

import time
import numpy as np
from core.scorer import Scorer
from embeddings.embedder import Embedder


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)   # ~4 chars/token approximation


class WorkingMemory:
    def __init__(self, max_tokens: int, scorer: Scorer, embedder: Embedder):
        self.max_tokens = max_tokens
        self.scorer = scorer
        self.embedder = embedder
        self.chunks: list[dict] = []

    # ------------------------------------------------------------------ public

    def add(self, role: str, content: str):
        emb = self.embedder.embed(content)
        self.chunks.append({
            "role":         role,
            "content":      content,
            "embedding":    emb,
            "timestamp":    time.time(),
            "access_count": 0,
            "tokens":       _approx_tokens(content),
        })

    def evict_one(self, query_vec: np.ndarray | None = None) -> dict | None:
        """Remove and return the lowest-scored chunk. Returns None if nothing evictable."""
        if len(self.chunks) <= 2:
            return None   # protect last turn

        candidates = self.chunks[:-2]   # never evict the 2 most recent
        now = time.time()

        scored = [
            (i, self.scorer.score(c, query_vec, now))
            for i, c in enumerate(self.chunks)
            if c in candidates
        ]
        if not scored:
            return None

        worst_idx = min(scored, key=lambda x: x[1])[0]
        evicted = self.chunks.pop(worst_idx)
        return evicted

    def is_over_budget(self) -> bool:
        return self.total_tokens() > self.max_tokens

    def total_tokens(self) -> int:
        return sum(c["tokens"] for c in self.chunks)

    def to_messages(self) -> list[dict]:
        for c in self.chunks:
            c["access_count"] += 1
        return [{"role": c["role"], "content": c["content"]} for c in self.chunks]

    def status(self) -> str:
        return f"{self.total_tokens()}/{self.max_tokens} tokens | {len(self.chunks)} chunks in context"
