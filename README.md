# Engram

A local LLM with scored tiered memory, powered by **NVIDIA Nemotron-Nano-9B-v2**. Runs entirely on-device — no API keys, no cloud.

Instead of crashing when context gets too long, Engram automatically scores every chunk in the context window by **semantic relevance + recency + access frequency**, evicts the least important to a SQLite archival store, and recalls them on demand when the model needs them again.

---

## How it differs from MemGPT

| | MemGPT | Engram |
|---|---|---|
| Model | Any LLM | Nemotron-Nano-9B-v2 (local) |
| Eviction strategy | FIFO (oldest out) | Scored (least relevant out) |
| Retrieval trigger | LLM tool call | LLM tool call |
| Retrieval method | Vector similarity | Vector similarity |
| Failure tracking | None | Explicit — unrecalled chunks surfaced |
| Evaluation | Domain-specific | Token efficiency + recall accuracy vs full-context baseline |
| Runs locally | Partial | Fully local (HuggingFace + SQLite) |

The core insight: a semantically relevant 20-turn-old chunk should survive eviction over an irrelevant recent one. MemGPT doesn't do this. Engram does.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  main.py                    │
│            CLI + eval harness               │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│               core/agent.py                 │
│   Agentic loop — LLM calls memory tools     │
└──────┬────────────────────────┬─────────────┘
       │                        │
┌──────▼──────────┐   ┌─────────▼─────────────┐
│ working_memory  │   │    archival_store      │
│ token-budgeted  │◄──│    SQLite + vectors    │
│ scored eviction │   │    failure tracking    │
└──────┬──────────┘   └────────────────────────┘
       │
┌──────▼──────────┐
│   scorer.py     │
│ sim + rec + freq│
└─────────────────┘
```

---

## Scoring Algorithm

Every chunk in working memory gets a retention score on each eviction cycle. The lowest-scored chunk is evicted.

```
score = 0.60 × semantic_similarity(chunk, current_query)
      + 0.30 × recency(chunk.timestamp)          # exp decay, 30-min half-life
      + 0.10 × access_frequency(chunk.access_count)
```

Weights are tunable in `core/scorer.py`.

---

## File Structure

```
engram/
├── main.py                  # CLI entrypoint, --eval flag
├── requirements.txt
├── core/
│   ├── agent.py             # orchestrates working memory + archival + LLM
│   ├── scorer.py            # retention scoring algorithm
│   ├── working_memory.py    # token-budgeted context with scored eviction
│   └── archival_store.py    # SQLite store + failure mode tracking
├── embeddings/
│   └── embedder.py          # sentence-transformers (fallback: bag-of-chars)
├── eval/
│   └── evaluator.py         # recall accuracy, token savings, loss report
├── llm/
│   └── nemotron_client.py   # Nemotron local inference via HuggingFace
└── db/
    └── engram.db            # auto-created on first run
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch transformers accelerate huggingface_hub sentence-transformers numpy
```

### 2. Run

On first run, Engram downloads **nvidia/NVIDIA-Nemotron-Nano-9B-v2** to your HuggingFace cache (~18GB). Every subsequent run loads from disk — no network required.

```bash
# Interactive chat
python main.py

# Run eval suite then drop into chat
python main.py --eval

# Stress-test eviction with a tight budget
python main.py --budget 512 --eval
```

### Hardware

Tested on **NVIDIA GX-10** (Grace-Blackwell). Requires a CUDA-capable GPU with ~18GB VRAM (bfloat16). CPU inference works but is slow.

---

## Evaluation

Run with `--eval` to get a full report:

```
════════════════════════════════════════════════════════════
  ENGRAM EVALUATION REPORT
════════════════════════════════════════════════════════════

── Recall Accuracy ──────────────────────────
  ✓  Q: What is my name?
       Expected: 'Alex' | Got: 'Your name is Alex.'
  ✓  Q: Where do I work?
       Expected: 'Stripe' | Got: 'You work at Stripe.'
  ✗  Q: What bug am I dealing with?
       Expected: 'webhook' | Got: 'I don't have that in context...'

  Accuracy: 4/5 (80%)

── Token Efficiency ─────────────────────────
  Working memory used  : 312 tokens
  Full-context baseline: 1840 tokens
  Token savings        : 1528 (83%)

── Failure Mode: Unrecalled Chunks ──────────
  WARNING: 1 chunk(s) evicted and NEVER recalled.
  • [user] I have a recurring bug in our webhook dispatcher...

  Mitigation: lower MAX_TOKENS budget, increase top_k on recall,
  or use archival_save for critical facts before eviction.
════════════════════════════════════════════════════════════
```

The failure mode report surfaces every chunk evicted but never recalled — an explicit acknowledgment of the core weakness in any memory system: **if the model fails to save something and never searches for it, it is gone.**

---

## CLI Commands

| Command | Description |
|---|---|
| `status` | Show working memory usage, archival count, recall rate |
| `quit` | Exit |

---

## Configuration

```python
# main.py
MAX_TOKENS = 2048   # working memory budget (lower = more aggressive eviction)

# core/scorer.py
W_SIM  = 0.60   # weight: semantic similarity
W_REC  = 0.30   # weight: recency
W_FREQ = 0.10   # weight: access frequency
```

---

## LLM Tools

The model has two tools it can call at any time:

**`memory_search(query, top_k)`** — searches archival store via vector similarity, injects results back into working memory.

**`archival_save(content)`** — explicitly saves a fact to archival before it gets evicted.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (~18GB VRAM for bfloat16)
- `pip install torch transformers accelerate huggingface_hub sentence-transformers numpy`
