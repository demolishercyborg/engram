import os
import sqlite3
import json
import time
import numpy as np
from embeddings.embedder import Embedder

# absolute path so it works regardless of cwd
_DB_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")
_DB_PATH = os.path.join(_DB_DIR, "engram.db")


class ArchivalStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        os.makedirs(_DB_DIR, exist_ok=True)
        self.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                role         TEXT,
                content      TEXT,
                embedding    BLOB,
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

    def insert(self, chunk: dict):
        emb = chunk.get("embedding")
        if emb is None:
            emb = self.embedder.embed(chunk["content"])
        if isinstance(emb, np.ndarray):
            emb_blob = emb.astype(np.float32).tobytes()
        else:
            emb_blob = np.array(emb, dtype=np.float32).tobytes()

        self.conn.execute(
            """INSERT INTO chunks
               (role, content, embedding, timestamp, evicted_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                chunk["role"],
                chunk["content"],
                emb_blob,
                chunk.get("timestamp", time.time()),
                time.time(),
                chunk.get("access_count", 0),
            )
        )
        self.conn.commit()
        preview = chunk["content"][:70] + ("..." if len(chunk["content"]) > 70 else "")
        print(f"  [Archival ↓] evicted → '{preview}'")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = self.embedder.embed(query)
        rows = self.conn.execute(
            "SELECT id, role, content, embedding FROM chunks"
        ).fetchall()

        if not rows:
            return []

        scored = []
        for id_, role, content, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            # handle dimension mismatch from old JSON-encoded rows
            if emb.shape != q_vec.shape:
                continue
            sim = self.embedder.similarity(q_vec, emb)
            scored.append((sim, {"id": id_, "role": role, "content": content}))

        scored.sort(reverse=True)
        results = [c for _, c in scored[:top_k]]

        for r in results:
            self.conn.execute(
                "UPDATE chunks SET recalled = 1, access_count = access_count + 1 WHERE id = ?",
                (r["id"],)
            )
        self.conn.commit()
        return results

    def unrecalled_chunks(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, role, content, evicted_at FROM chunks WHERE recalled = 0"
        ).fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "evicted_at": r[3]} for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def recall_rate(self) -> float:
        total = self.count()
        if total == 0:
            return 1.0
        recalled = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE recalled = 1").fetchone()[0]
        return recalled / total
