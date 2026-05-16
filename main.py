"""
Engram — local LLM with scored tiered memory
---------------------------------------------
Usage:
    python main.py              # interactive chat
    python main.py --eval       # run built-in eval suite then chat
    python main.py --model qwen2.5:32b   # specify model
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("db", exist_ok=True)

from core.agent import EngramAgent
from eval.evaluator import Evaluator
from llm.ollama_client import make_llm_fn


# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "llama3.2"     # change to your ollama model
MAX_TOKENS     = 2048           # working memory budget (lower = more aggressive eviction)


# ── built-in eval probes ──────────────────────────────────────────────────────
#   These are injected early in the conversation, then probed after many turns
#   to test whether Engram correctly recalls evicted information.
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
    {"question": "What is my name?",                        "keyword": "Alex"},
    {"question": "Where do I work?",                        "keyword": "Stripe"},
    {"question": "What programming languages do I use?",    "keyword": "Go"},
    {"question": "What bug am I dealing with?",             "keyword": "webhook"},
    {"question": "What is my dog's name?",                  "keyword": "Biscuit"},
]


def run_chat(agent: EngramAgent, llm_fn):
    print(f"\n{'='*60}")
    print("  Engram — tiered memory LLM")
    print(f"  Budget: {agent.working.max_tokens} tokens | type 'status' or 'quit'")
    print(f"{'='*60}\n")

    while True:
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

        response = agent.chat(user_input, llm_fn)
        print(f"\nEngram: {response}")
        print(f"  ↳ {agent.status()}\n")


def run_eval(agent: EngramAgent, llm_fn):
    print("\n[Eval] Seeding facts into memory...")
    for role, content in SEED_FACTS:
        agent.working.add(role, content)
        # trigger eviction manually since we're bypassing agent.chat
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
    report = evaluator.run_probe_set(EVAL_PROBES, llm_fn)
    report.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--budget", type=int, default=MAX_TOKENS, help="Working memory token budget")
    parser.add_argument("--eval",   action="store_true", help="Run eval suite before chat")
    args = parser.parse_args()

    llm_fn = make_llm_fn(args.model)
    agent  = EngramAgent(max_tokens=args.budget)

    if args.eval:
        run_eval(agent, llm_fn)

    run_chat(agent, llm_fn)


if __name__ == "__main__":
    main()
