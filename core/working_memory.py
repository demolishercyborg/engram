import time
import numpy as np
from core.scorer import Scorer
from embeddings.embedder import Embedder


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class WorkingMemory:
    def __init__(self, max_tokens: int, scorer: Scorer, embedder: Embedder):
        self.max_tokens = max_tokens
        self.scorer = scorer
        self.embedder = embedder
        self.chunks: list[dict] = []

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
        """Remove and return the lowest-scored chunk.
        Index 0 (system prompt) and the last 2 chunks (current turn) are protected.
        """
        # need at least: system(0), some middle chunks, last 2 turns
        if len(self.chunks) <= 3:
            return None

        # evictable = everything except index 0 and the last 2
        now = time.time()
        scored = [
            (i, self.scorer.score(self.chunks[i], query_vec, now))
            for i in range(1, len(self.chunks) - 2)
        ]
        if not scored:
            return None

        worst_idx = min(scored, key=lambda x: x[1])[0]
        return self.chunks.pop(worst_idx)

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
