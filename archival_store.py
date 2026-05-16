"""
Engram Archival Store
---------------------
SQLite-backed long-term storage for evicted chunks.
Retrieval uses cosine similarity over stored embeddings.

Failure mode tracking: every chunk that enters archival is logged.
If a chunk is never recalled before session end, it is flagged as
a potential information loss — surfaced in the eval report.
"""

import sqlite3
import json
import time
import numpy as np
from embeddings.embedder import Embedder

DB_PATH = "db/engram.db"


class ArchivalStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_schema()

    # ------------------------------------------------------------------ schema

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                role         TEXT,
                content      TEXT,
                embedding    TEXT,
                timestamp    REAL,
                evicted_at   REAL,
                recalled     INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS loss_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id  INTEGER,
                content   TEXT,
                reason    TEXT,
                logged_at REAL
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ write

    def insert(self, chunk: dict):
        emb = chunk.get("embedding")
        if emb is None:
            emb = self.embedder.embed(chunk["content"])
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()

        self.conn.execute(
            """INSERT INTO chunks
               (role, content, embedding, timestamp, evicted_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                chunk["role"],
                chunk["content"],
                json.dumps(emb),
                chunk.get("timestamp", time.time()),
                time.time(),
                chunk.get("access_count", 0),
            )
        )
        self.conn.commit()
        preview = chunk["content"][:70] + ("..." if len(chunk["content"]) > 70 else "")
        print(f"  [Archival ↓] evicted → '{preview}'")

    # ------------------------------------------------------------------ read

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = self.embedder.embed(query)
        rows = self.conn.execute(
            "SELECT id, role, content, embedding, timestamp, access_count FROM chunks"
        ).fetchall()

        if not rows:
            return []

        now = time.time()
        scored = []
        for row in rows:
            id_, role, content, emb_json, ts, ac = row
            emb = np.array(json.loads(emb_json))
            sim = self.embedder.similarity(q_vec, emb)
            scored.append((sim, {"id": id_, "role": role, "content": content}))

        scored.sort(reverse=True)
        results = [c for _, c in scored[:top_k]]

        # mark as recalled
        for r in results:
            self.conn.execute(
                "UPDATE chunks SET recalled = 1, access_count = access_count + 1 WHERE id = ?",
                (r["id"],)
            )
        self.conn.commit()
        return results

    # --------------------------------------------------------- failure tracking

    def log_potential_loss(self, chunk_id: int, content: str, reason: str):
        """Call this when a chunk is evicted that might be unrecoverable."""
        self.conn.execute(
            "INSERT INTO loss_log (chunk_id, content, reason, logged_at) VALUES (?, ?, ?, ?)",
            (chunk_id, content, reason, time.time())
        )
        self.conn.commit()

    def unrecalled_chunks(self) -> list[dict]:
        """Return all archival chunks never recalled — the failure mode surface."""
        rows = self.conn.execute(
            "SELECT id, role, content, evicted_at FROM chunks WHERE recalled = 0"
        ).fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "evicted_at": r[3]} for r in rows]

    # ------------------------------------------------------------------ stats

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def recall_rate(self) -> float:
        total = self.count()
        if total == 0:
            return 1.0
        recalled = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE recalled = 1").fetchone()[0]
        return recalled / total
