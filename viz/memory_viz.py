"""
Engram Memory Visualizer
------------------------
Real-time CLI view of working memory, evictions, recalls, and tool calls.
Inspired by Claude Code's structured terminal output style.
"""

import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

_ROLE_STYLE = {
    "system":    "dim white",
    "user":      "bold cyan",
    "assistant": "bold green",
}

_ROLE_LABEL = {
    "system":    "sys",
    "user":      "usr",
    "assistant": "ast",
}


def _bar(score: float, width: int = 8) -> str:
    n = round(score * width)
    return "█" * n + "░" * (width - n)


def _preview(text: str, max_len: int = 52) -> str:
    text = text.replace("\n", " ").strip()
    return (text[:max_len] + "…") if len(text) > max_len else text


class MemoryViz:
    def __init__(self):
        self._agent = None
        self._last_query_vec = None

    # ------------------------------------------------------------------ attach

    def attach(self, agent):
        """Wire callbacks into the agent's sub-components."""
        self._agent = agent
        agent.working._viz_on_evict  = self._on_evict
        agent.archival._viz_on_recall = self._on_recall

    def set_query(self, query_vec):
        self._last_query_vec = query_vec

    # ------------------------------------------------------------------ events

    def _on_evict(self, chunk: dict, score: float):
        t = Text()
        t.append("  ↓ evict  ", style="bold red")
        t.append(f"[{_ROLE_LABEL.get(chunk['role'], chunk['role'])}] ", style="dim red")
        t.append(f'"{_preview(chunk["content"], 48)}"', style="red")
        t.append(f"   score {score:.2f}", style="dim red")
        console.print(t)

    def _on_recall(self, chunks: list[dict], query: str):
        for chunk in chunks:
            t = Text()
            t.append("  ↑ recall ", style="bold cyan")
            t.append(f"[{_ROLE_LABEL.get(chunk['role'], chunk['role'])}] ", style="dim cyan")
            t.append(f'"{_preview(chunk["content"], 40)}"', style="cyan")
            t.append(f"   ← {_preview(query, 22)!r}", style="dim cyan")
            console.print(t)

    def log_tool(self, name: str, args: dict, result: str):
        t = Text()
        t.append("  ⚡ tool   ", style="bold yellow")
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        t.append(f"{name}(", style="yellow")
        t.append(args_str, style="bright_yellow")
        t.append(")", style="yellow")
        t.append(f"   → {_preview(result, 55)}", style="dim yellow")
        console.print(t)

    # ------------------------------------------------------------------ render

    def render(self):
        """Print the full working memory state. Call before each user turn."""
        agent  = self._agent
        chunks = agent.working.chunks
        scorer = agent.working.scorer
        now    = time.time()
        qv     = self._last_query_vec

        # token budget bar
        used   = agent.working.total_tokens()
        budget = agent.working.max_tokens
        pct    = used / budget
        bw     = 24
        filled = round(pct * bw)
        bar    = "█" * filled + "░" * (bw - filled)
        bar_style = "green" if pct < 0.70 else ("yellow" if pct < 0.90 else "red")

        header = Text()
        header.append(f"{used}", style="bold")
        header.append(f"/{budget} tokens  ", style="dim")
        header.append(f"[{bar}]  ", style=bar_style)
        header.append(f"{len(chunks)} chunks  ·  ", style="dim")
        header.append(f"archival {agent.archival.count()}  ·  ", style="dim")
        recall_rate = agent.archival.recall_rate()
        recall_style = "green" if recall_rate >= 0.8 else ("yellow" if recall_rate >= 0.5 else "red")
        header.append(f"recall ", style="dim")
        header.append(f"{recall_rate:.0%}", style=recall_style)

        # chunks table
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold dim",
            pad_edge=False,
            show_edge=False,
            padding=(0, 1),
        )
        table.add_column("Score", width=15, no_wrap=True)
        table.add_column("Role", width=4,  no_wrap=True)
        table.add_column("Content", no_wrap=True)
        table.add_column("Tok", width=4, justify="right", no_wrap=True)

        for i, chunk in enumerate(chunks):
            role  = chunk["role"]
            style = _ROLE_STYLE.get(role, "")
            label = _ROLE_LABEL.get(role, role[:3])

            if i == 0:
                score_cell = Text("── pinned ──", style="dim")
            else:
                s = scorer.score(chunk, qv, now)
                score_cell = Text()
                score_cell.append(_bar(s) + "  ", style=style)
                score_cell.append(f"{s:.2f}", style="dim")

            table.add_row(
                score_cell,
                Text(label, style=style),
                Text(_preview(chunk["content"]), style=style),
                Text(str(chunk["tokens"]), style="dim"),
            )

        console.print()
        console.print(Rule("[bold]Working Memory[/bold]", style="bright_black"))
        console.print(header)
        console.print(table)
        console.print(Rule(style="bright_black"))
        console.print()
