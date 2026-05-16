"""
Engram Evaluator
----------------
Implements the two improvements from the paper critique:

1. STRONGER EVALUATION
   - Token efficiency: working tokens used vs naive full-context baseline
   - Recall accuracy: % of probed facts correctly retrieved from archival
   - Compared against a full-context (no eviction) baseline

2. EXPLICIT FAILURE MODE ACKNOWLEDGMENT
   - Tracks every chunk evicted but never recalled
   - Flags potential information losses with content preview
   - Reports "loss rate" = 1 - recall_rate

Usage:
    eval = Evaluator(agent)
    eval.run_probe_set(probes, llm_fn)
    eval.report()
"""

import time
from dataclasses import dataclass, field


@dataclass
class ProbeResult:
    question: str
    expected_keyword: str
    answer: str
    hit: bool
    latency_ms: float


@dataclass
class EvalReport:
    probe_results: list[ProbeResult] = field(default_factory=list)
    total_tokens_used: int = 0
    baseline_tokens: int = 0
    archival_count: int = 0
    recall_rate: float = 0.0
    unrecalled_chunks: list[dict] = field(default_factory=list)

    def print(self):
        hits = sum(1 for p in self.probe_results if p.hit)
        total = len(self.probe_results)
        acc = hits / total if total else 0

        print("\n" + "="*60)
        print("  ENGRAM EVALUATION REPORT")
        print("="*60)

        print(f"\n── Recall Accuracy ──────────────────────────")
        for p in self.probe_results:
            icon = "✓" if p.hit else "✗"
            print(f"  {icon}  Q: {p.question[:50]}")
            print(f"       Expected: '{p.expected_keyword}' | Got: '{p.answer[:60]}'")
            print(f"       Latency: {p.latency_ms:.0f}ms")
        print(f"\n  Accuracy: {hits}/{total} ({acc:.0%})")

        print(f"\n── Token Efficiency ─────────────────────────")
        print(f"  Working memory used : {self.total_tokens_used} tokens")
        print(f"  Full-context baseline: {self.baseline_tokens} tokens")
        savings = max(0, self.baseline_tokens - self.total_tokens_used)
        pct = savings / self.baseline_tokens if self.baseline_tokens else 0
        print(f"  Token savings        : {savings} ({pct:.0%})")

        print(f"\n── Archival Stats ───────────────────────────")
        print(f"  Chunks in archival   : {self.archival_count}")
        print(f"  Recall rate          : {self.recall_rate:.0%}")
        loss_rate = 1 - self.recall_rate
        print(f"  Loss rate            : {loss_rate:.0%}")

        print(f"\n── Failure Mode: Unrecalled Chunks ──────────")
        if not self.unrecalled_chunks:
            print("  None — all evicted chunks were recalled ✓")
        else:
            print(f"  WARNING: {len(self.unrecalled_chunks)} chunk(s) evicted and NEVER recalled.")
            print("  These represent potential information loss:\n")
            for c in self.unrecalled_chunks[:5]:
                preview = c["content"][:80] + ("..." if len(c["content"]) > 80 else "")
                print(f"    • [{c['role']}] {preview}")
            if len(self.unrecalled_chunks) > 5:
                print(f"    ... and {len(self.unrecalled_chunks) - 5} more.")
            print()
            print("  Mitigation: lower MAX_TOKENS budget, increase top_k on recall,")
            print("  or use archival_save for critical facts before they are evicted.")

        print("="*60 + "\n")


class Evaluator:
    def __init__(self, agent):
        self.agent = agent

    def run_probe_set(self, probes: list[dict], llm_fn) -> EvalReport:
        """
        probes: list of {"question": str, "keyword": str}
        keyword must appear in the agent's answer for a hit.
        """
        report = EvalReport()

        for probe in probes:
            t0 = time.time()
            answer = self.agent.chat(probe["question"], llm_fn)
            ms = (time.time() - t0) * 1000
            hit = probe["keyword"].lower() in answer.lower()
            report.probe_results.append(ProbeResult(
                question=probe["question"],
                expected_keyword=probe["keyword"],
                answer=answer,
                hit=hit,
                latency_ms=ms
            ))

        report.total_tokens_used  = self.agent.working.total_tokens()
        report.baseline_tokens    = self._estimate_baseline()
        report.archival_count     = self.agent.archival.count()
        report.recall_rate        = self.agent.archival.recall_rate()
        report.unrecalled_chunks  = self.agent.archival.unrecalled_chunks()

        return report

    def _estimate_baseline(self) -> int:
        """
        Baseline = all chunks ever seen (working + archival) with no eviction.
        Approximation: working tokens + sum of archival content lengths / 4.
        """
        rows = self.agent.archival.conn.execute(
            "SELECT content FROM chunks"
        ).fetchall()
        archival_tokens = sum(max(1, len(r[0]) // 4) for r in rows)
        return self.agent.working.total_tokens() + archival_tokens
