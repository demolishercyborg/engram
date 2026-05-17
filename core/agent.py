from __future__ import annotations
"""
Engram Agent — orchestrates working memory, archival store, and the LLM.
The LLM is an active participant: it decides when to call memory_search()
or archival_save() via tool calls. Eviction and scoring happen beneath it.
"""

import json
import numpy as np
from embeddings.embedder import Embedder
from core.scorer import Scorer
from core.working_memory import WorkingMemory
from core.archival_store import ArchivalStore


SYSTEM_PROMPT = """You are Engram, a helpful assistant with a tiered memory system.

You have access to two tools: `memory_search` and `archival_save`.

Rules:
1. Only call a tool if you genuinely need it. For most questions you can answer directly without searching memory.
2. Search memory at most ONCE per turn. If the result is empty, answer immediately from your own knowledge — do NOT simulate, mention, or attempt further tool calls.
3. Never write tool calls or "Awaiting results..." in your text. Tool calls must be issued as actual function calls, not written out in your response.
4. If you don't know something, say so plainly and briefly.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search archival memory for context that may have been evicted. Use whenever you need earlier information not visible in your current context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "top_k": {"type": "integer", "description": "Max results (default 3)", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "archival_save",
            "description": "Explicitly save a fact or summary to archival memory so it is never lost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to save"}
                },
                "required": ["content"]
            }
        }
    }
]


class EngramAgent:
    def __init__(self, max_tokens: int = 2048, viz=None):
        self.embedder = Embedder()
        self.scorer   = Scorer(self.embedder)
        self.working  = WorkingMemory(max_tokens, self.scorer, self.embedder)
        self.archival = ArchivalStore(self.embedder)
        self.viz      = viz
        self._pending_recalls: list[dict] = []

        # system prompt is always at index 0 — protected from eviction
        self.working.add("system", SYSTEM_PROMPT)

    # ------------------------------------------------------------------ public

    def chat(self, user_input: str, llm_fn) -> str:
        query_vec = self.embedder.embed(user_input)
        if self.viz:
            self.viz.set_query(query_vec)
        self._add_and_evict("user", user_input, query_vec)

        called_tools: set[str] = set()  # deduplicate: (name, query) pairs

        for _ in range(6):
            messages   = self.working.to_messages()
            response   = llm_fn(messages, TOOLS)
            message    = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                content = message.get("content", "")
                self._add_and_evict("assistant", content, query_vec)
                return content

            # Record the assistant's tool-call turn so the conversation
            # threading stays coherent. Without this, the next iteration
            # sees tool results with no originating assistant message,
            # which breaks chat templates and causes the model to repeat
            # the same tool call.
            asst_content = message.get("content", "") or ""
            self._add_and_evict(
                "assistant",
                asst_content,
                query_vec,
                tool_calls=tool_calls,
                _embed_hint=" ".join(
                    f"{tc.get('function', {}).get('name', '')} "
                    f"{json.dumps(tc.get('function', {}).get('arguments', {}), default=str)}"
                    for tc in tool_calls
                ),
            )

            made_new_call = False
            for tc in tool_calls:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                dedup_key = f"{name}:{args.get('query', args.get('content', ''))}"
                if dedup_key in called_tools:
                    continue
                called_tools.add(dedup_key)
                made_new_call = True

                result = self._handle_tool(name, args, query_vec)
                if self.viz:
                    self.viz.log_tool(name, args, str(result))
                else:
                    print(f"  [Tool] {name}({args}) → {str(result)[:100]}")

                tool_call_id = tc.get("id") or f"call_{name}"
                self._add_and_evict(
                    "tool",
                    f"[Tool result — {name}]: {result}",
                    query_vec,
                    tool_call_id=tool_call_id,
                    name=name,
                )

                # Now that the tool result is in place, re-insert any
                # recalled archival chunks. Doing this AFTER the tool
                # result preserves the assistant→tool adjacency that
                # chat templates expect.
                if self._pending_recalls:
                    for r in self._pending_recalls:
                        self._add_and_evict(
                            "system", f"[Recalled]: {r['content']}", query_vec
                        )
                    self._pending_recalls = []

            # All tool calls this round were duplicates — force a response
            if not made_new_call:
                messages = self.working.to_messages()
                response = llm_fn(messages, None)
                content  = response.get("message", {}).get("content", "")
                self._add_and_evict("assistant", content, query_vec)
                return content

        return "[Engram: max tool rounds reached]"

    def status(self) -> str:
        return (
            f"{self.working.status()} | "
            f"Archival: {self.archival.count()} chunks | "
            f"Recall rate: {self.archival.recall_rate():.0%}"
        )

    # ------------------------------------------------------------------ private

    def _add_and_evict(self, role: str, content: str, query_vec: np.ndarray | None = None, **extra):
        self.working.add(role, content, **extra)
        while self.working.is_over_budget():
            evicted = self.working.evict_one(query_vec)
            if evicted is None:
                break
            self.archival.insert(evicted)

    def _handle_tool(self, name: str, args: dict, query_vec: np.ndarray) -> str:
        if name == "memory_search":
            query   = args.get("query", "")
            top_k   = args.get("top_k", 3)
            results = self.archival.search(query, top_k)
            if not results:
                return "No relevant memories found."
            # Stash for the caller to re-insert AFTER the tool result is
            # appended; this keeps the assistant→tool message pair
            # adjacent for the chat template.
            self._pending_recalls = list(results)
            body = "\n".join(f"- {r['content']}" for r in results)
            return f"Recalled {len(results)} chunks:\n{body}"

        if name == "archival_save":
            content = args.get("content", "")
            self.archival.insert({"role": "system", "content": content})
            return f"Saved: '{content[:60]}'"

        return f"Unknown tool: {name}"
