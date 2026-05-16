"""
Engram Scorer
-------------
Every chunk in working memory gets a retention score on every eviction cycle.
Lowest score gets evicted. Formula:

    score = w_sim  * semantic_similarity(chunk, current_query)
          + w_rec  * recency(chunk.timestamp)
          + w_freq * access_frequency(chunk.access_count)

Weights are tunable. Defaults favour semantic relevance.

This is the core improvement over vanilla MemGPT (which uses pure FIFO).
A semantically relevant 20-turn-old chunk survives over an irrelevant new one.
"""

import time
import math
import numpy as np
from embeddings.embedder import Embedder


W_SIM  = 0.60   # semantic similarity to current query
W_REC  = 0.30   # recency
W_FREQ = 0.10   # historical access frequency


class Scorer:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def score(self, chunk: dict, query_vec: np.ndarray | None, now: float) -> float:
        # --- similarity ---
        if query_vec is not None and chunk.get("embedding") is not None:
            sim = self.embedder.similarity(query_vec, chunk["embedding"])
        else:
            sim = 0.5   # neutral when no query context

        # --- recency: exponential decay over minutes ---
        age_minutes = (now - chunk.get("timestamp", now)) / 60.0
        rec = math.exp(-age_minutes / 30.0)   # half-life ~30 min

        # --- access frequency: log-scaled, capped at 1 ---
        freq = min(1.0, math.log1p(chunk.get("access_count", 0)) / math.log1p(20))

        return W_SIM * sim + W_REC * rec + W_FREQ * freq
