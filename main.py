"""
Engram — local LLM with scored tiered memory, powered by Nemotron-Nano-9B-v2
-----------------------------------------------------------------------------
Usage:
    python main.py              # interactive chat
    python main.py --eval       # run built-in eval suite then chat
    python main.py --budget 512 --eval
"""

import os
import sys
import argparse

# ensure repo root is on the path so all subpackages resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import snapshot_download
from core.agent import EngramAgent
from eval.evaluator import Evaluator
from llm.nemotron_client import make_llm_fn
from viz.memory_viz import MemoryViz


# ── config ────────────────────────────────────────────────────────────────────
NEMOTRON_REPO = "Qwen/Qwen3.6-27B-FP8"
MAX_TOKENS    = 2048


# ── eval data ─────────────────────────────────────────────────────────────────
SEED_FACTS = [
    ("user",      "My name is Alex and I am a backend engineer at Stripe."),
    ("assistant", "Nice to meet you Alex!"),
    ("user",      "I mostly work in Go and Python. My team owns the payments API."),
    ("assistant", "Got it — Go and Python, payments API team at Stripe."),
    ("user",      "I have a recurring bug in our webhook dispatcher that drops events under load."),
    ("assistant", "That sounds like a race condition or queue backpressure issue."),
    ("user",      "On another note, my dog's name is Biscuit."),
    ("assistant", "Haha, great name."),
    ("user",      "I prefer tabs over spaces, strongly."),
    ("user",      "Anyway, let me ask you about something totally unrelated — what is the capital of France?"),
    ("assistant", "Paris."),
    ("user",      "What's the time complexity of merge sort?"),
    ("assistant", "O(n log n) in all cases."),
    ("user",      "Explain briefly how TCP handshake works."),
    ("assistant", "SYN → SYN-ACK → ACK — three-way handshake establishes a connection."),
    ("user",      "What's 2 to the power of 10?"),
    ("assistant", "1024."),
]

EVAL_PROBES = [
    {"question": "What is my name?",                     "keyword": "Alex"},
    {"question": "Where do I work?",                     "keyword": "Stripe"},
    {"question": "What programming languages do I use?", "keyword": "Go"},
    {"question": "What bug am I dealing with?",          "keyword": "webhook"},
    {"question": "What is my dog's name?",               "keyword": "Biscuit"},
]


def run_chat(agent: EngramAgent, llm_fn):
    viz = agent.viz
    print(f"\n{'='*60}")
    print("  Engram — tiered memory LLM (Nemotron-Nano-9B-v2)")
    print(f"  Budget: {agent.working.max_tokens} tokens | type 'status' or 'quit'")
    if viz:
        print("  Memory visualization ON")
    print(f"{'='*60}\n")

    while True:
        if viz:
            viz.render()

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "status":
            print(f"\n  {agent.status()}\n")
            continue

        print()
        response = agent.chat(user_input, llm_fn)
        print(f"\nEngram: {response}")
        if not viz:
            print(f"  ↳ {agent.status()}\n")


def run_eval(agent: EngramAgent, llm_fn):
    print("\n[Eval] Seeding facts into memory...")
    for role, content in SEED_FACTS:
        agent.working.add(role, content)
        qv = agent.embedder.embed(content)
        while agent.working.is_over_budget():
            ev = agent.working.evict_one(qv)
            if ev:
                agent.archival.insert(ev)
            else:
                break

    print(f"[Eval] {len(SEED_FACTS)} facts seeded. Status: {agent.status()}")
    print("[Eval] Running probes...\n")

    evaluator = Evaluator(agent)
    report    = evaluator.run_probe_set(EVAL_PROBES, llm_fn)
    report.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=MAX_TOKENS,
                        help="Working memory token budget")
    parser.add_argument("--eval",   action="store_true",
                        help="Run eval suite before chat")
    parser.add_argument("--viz",    action="store_true",
                        help="Show live memory visualization")
    args = parser.parse_args()

    viz = None
    if args.viz:
        viz = MemoryViz()

    print(f"[Engram] Downloading/verifying {NEMOTRON_REPO} ...")
    model_path = snapshot_download(repo_id=NEMOTRON_REPO)
    print(f"[Engram] Model at: {model_path}")

    llm_fn = make_llm_fn(model_path)
    agent  = EngramAgent(max_tokens=args.budget, viz=viz)

    if viz:
        viz.attach(agent)

    if args.eval:
        run_eval(agent, llm_fn)

    run_chat(agent, llm_fn)


if __name__ == "__main__":
    main()
