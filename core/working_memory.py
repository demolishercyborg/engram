from __future__ import annotations
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

    def add(self, role: str, content: str, **extra):
        # Embed a synthetic string when content is empty (e.g. an assistant
        # turn that is purely a tool call) so the chunk still has a usable
        # vector for scoring/eviction.
        embed_text = content or extra.get("_embed_hint", "")
        emb = self.embedder.embed(embed_text)
        chunk = {
            "role":         role,
            "content":      content,
            "embedding":    emb,
            "timestamp":    time.time(),
            "access_count": 0,
            "tokens":       _approx_tokens(content),
        }
        for k, v in extra.items():
            if k.startswith("_"):
                continue
            chunk[k] = v
        self.chunks.append(chunk)

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

        worst_idx, worst_score = min(scored, key=lambda x: x[1])
        evicted = self.chunks.pop(worst_idx)
        if cb := getattr(self, "_viz_on_evict", None):
            cb(evicted, worst_score)
        return evicted

    def is_over_budget(self) -> bool:
        return self.total_tokens() > self.max_tokens

    def total_tokens(self) -> int:
        return sum(c["tokens"] for c in self.chunks)

    def to_messages(self) -> list[dict]:
        msgs = []
        for c in self.chunks:
            c["access_count"] += 1
            m = {"role": c["role"], "content": c["content"]}
            if "tool_calls" in c:
                m["tool_calls"] = c["tool_calls"]
            if "tool_call_id" in c:
                m["tool_call_id"] = c["tool_call_id"]
            if "name" in c:
                m["name"] = c["name"]
            msgs.append(m)
        return msgs

    def status(self) -> str:
        return f"{self.total_tokens()}/{self.max_tokens} tokens | {len(self.chunks)} chunks in context"
