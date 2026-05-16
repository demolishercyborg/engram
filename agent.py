"""
Engram Agent
------------
Orchestrates working memory, archival store, and the LLM.
The LLM is an ACTIVE participant — it decides when to call
memory_search() or archival_save() via tool calls.
Eviction and scoring happen transparently underneath.
"""

import json
import numpy as np
from embeddings.embedder import Embedder
from core.scorer import Scorer
from core.working_memory import WorkingMemory
from core.archival_store import ArchivalStore


SYSTEM_PROMPT = """You are Engram, an assistant with a tiered memory system.

Your in-context window is LIMITED. Important information may be evicted to archival memory automatically. When you suspect you're missing earlier context, call `memory_search` with a relevant query to retrieve it.

You can also explicitly save critical facts using `archival_save` before they are lost.

Always check memory before saying "I don't recall" or "you didn't mention."
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
    def __init__(self, max_tokens: int = 2048):
        self.embedder  = Embedder()
        self.scorer    = Scorer(self.embedder)
        self.working   = WorkingMemory(max_tokens, self.scorer, self.embedder)
        self.archival  = ArchivalStore(self.embedder)

        # seed system prompt (protected — never evicted, scored highest)
        self.working.add("system", SYSTEM_PROMPT)

    # ------------------------------------------------------------------ public

    def chat(self, user_input: str, llm_fn) -> str:
        """
        llm_fn: callable(messages, tools) -> response dict from Ollama
        Returns the assistant's final text response.
        """
        query_vec = self.embedder.embed(user_input)
        self._add_and_evict("user", user_input, query_vec)

        # agentic loop — model may call tools multiple times
        for _ in range(6):
            messages  = self.working.to_messages()
            response  = llm_fn(messages, TOOLS)
            message   = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                content = message.get("content", "")
                self._add_and_evict("assistant", content, query_vec)
                return content

            for tc in tool_calls:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                result = self._handle_tool(name, args, query_vec)
                print(f"  [Tool] {name}({args}) → {str(result)[:100]}")
                self._add_and_evict("system", f"[Tool result — {name}]: {result}", query_vec)

        return "[Engram: max tool rounds reached]"

    def status(self) -> str:
        return (
            f"{self.working.status()} | "
            f"Archival: {self.archival.count()} chunks | "
            f"Recall rate: {self.archival.recall_rate():.0%}"
        )

    # ------------------------------------------------------------------ private

    def _add_and_evict(self, role: str, content: str, query_vec: np.ndarray | None = None):
        self.working.add(role, content)
        while self.working.is_over_budget():
            evicted = self.working.evict_one(query_vec)
            if evicted is None:
                break
            self.archival.insert(evicted)

    def _handle_tool(self, name: str, args: dict, query_vec: np.ndarray) -> str:
        if name == "memory_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 3)
            results = self.archival.search(query, top_k)
            if not results:
                return "No relevant memories found."
            # inject recalled chunks back into working memory
            for r in results:
                self._add_and_evict("system", f"[Recalled]: {r['content']}", query_vec)
            return f"Recalled {len(results)} chunks."

        if name == "archival_save":
            content = args.get("content", "")
            self.archival.insert({"role": "system", "content": content})
            return f"Saved: '{content[:60]}'"

        return f"Unknown tool: {name}"
