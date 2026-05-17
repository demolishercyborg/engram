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
│        CLI + eval harness + --viz           │
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
┌──────▼──────────┐   ┌────────────────────────┐
│   scorer.py     │   │   viz/memory_viz.py    │
│ sim + rec + freq│   │  live terminal display │
└─────────────────┘   └────────────────────────┘
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
├── main.py                  # CLI entrypoint, --eval, --viz flags
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
├── viz/
│   └── memory_viz.py        # live CLI memory visualizer
└── db/
    └── engram.db            # auto-created on first run
```

---

## Setup

### 1. Install dependencies

```bash
pip3 install ninja packaging
pip3 install -r requirements.txt --no-build-isolation
```

### 2. Run

On first run, Engram downloads **nvidia/NVIDIA-Nemotron-Nano-9B-v2** to your HuggingFace cache (~18GB). Every subsequent run loads from disk — no network required.

```bash
# Interactive chat
python main.py

# Live memory visualization (recommended for demos)
python main.py --viz

# Run eval suite then drop into chat
python main.py --eval

# Stress-test eviction with visualization
python main.py --viz --budget 512 --eval
```

### Hardware

Tested on **NVIDIA GX-10** (Grace-Blackwell). Requires a CUDA-capable GPU with ~18GB VRAM (bfloat16). CPU inference works but is slow.

---

## Memory Visualizer

Run with `--viz` to see a live view of working memory, evictions, and recalls before every turn:

```
─── Working Memory ─────────────────────────────────────────────────────────
66/2048 tokens  [██░░░░░░░░░░░░░░░░░░░░░░]  5 chunks · archival 3 · recall 67%
Score            Role  Content                                            Tok
────────────────────────────────────────────────────────────────────────────
── pinned ──     sys   You are Engram, an assistant with a tiered mem…    14
████░░░░  0.50   usr   My name is Alex and I am a backend engineer…       13
█████░░░  0.58   ast   Nice to meet you Alex!                              5
█████░░░  0.64   usr   I have a recurring bug in our webhook dispatc…     19
█████░░░  0.61   ast   That sounds like a race condition or queue…        15
────────────────────────────────────────────────────────────────────────────

  ↓ evict  [usr] "I have a recurring bug in our webhook dispatcher"   score 0.28
  ↑ recall [usr] "I have a recurring bug in our webhook dispatcher"   ← 'webhook bug'
  ⚡ tool   memory_search(query='webhook bug', top_k=3)   → Recalled 1 chunks.
```

- Score bars show each chunk's retention score relative to the current query
- System prompt shows `── pinned ──` — it is never evicted
- Token bar turns yellow above 70% capacity, red above 90%
- Evictions, recalls, and tool calls print inline as they fire

---

## Evaluation

Run with `--eval` to get a full report. The eval suite is a deliberately small, deterministic harness designed to answer one question: **does the tiered memory actually preserve the facts a user mentioned earlier, even after eviction?**

### What the suite does

1. **Seeds** 17 short conversational turns into working memory (`SEED_FACTS` in `main.py`). These look like a real chat — the user introduces themselves, mentions where they work, languages they use, an ongoing bug, their dog's name, plus a handful of unrelated trivia questions to add noise.
2. After every seed turn, `WorkingMemory.is_over_budget()` is checked. If the budget is exceeded, the lowest-scored chunk is evicted to the SQLite archival store. With the default `--budget 2048` the seed fits in working memory; with `--budget 256` you force aggressive eviction and the model has to call `memory_search` to recover facts.
3. **Probes** 5 questions (`EVAL_PROBES`) that target facts buried in the seed: name, employer, languages, current bug, dog's name. Each probe goes through the normal `agent.chat()` loop, so the model decides on its own whether to call `memory_search` or answer directly.

### What gets measured

| Section | How it's computed | What it tells you |
|---|---|---|
| **Recall Accuracy** | Case-insensitive substring match of the expected keyword in the model's answer (`evaluator.py:80`) | Did the agent recover the fact at all? |
| **Latency** | Wall-clock time per probe | Cost of memory recall vs. direct answer |
| **Working memory used** | `WorkingMemory.total_tokens()` after probes finish | What the model was actually paying for in context |
| **Full-context baseline** | `working_memory + sum(len(c)//4 for c in archival)` | What a non-tiered system would have shipped to the LLM every turn |
| **Token savings** | `baseline − working_memory_used` | The whole point: how much context the tiering saved |
| **Recall rate** | `recalled_chunks / total_archival_chunks` | Of everything that got evicted, how much did the model later ask back? |
| **Unrecalled chunks** | Archival rows with `recalled = 0` | **The failure mode** — facts the model lost forever because it never searched for them |

The failure-mode report is the part most memory systems hide. It surfaces every chunk evicted but never recalled — an explicit acknowledgment of the core weakness in any tiered memory: **if the model fails to save something and never searches for it, it is gone.**

### Caveat on the default budget

At `--budget 2048` the 17 seed facts fit comfortably (~334 tokens) so nothing gets evicted and the model answers from context directly. To actually exercise the archival/recall path end-to-end, drop the budget:

```bash
python main.py --eval --budget 256        # forces eviction + tool calls
python main.py --eval --budget 256 --viz  # plus live visualization of every eviction/recall
```

---

## CLI Flags

| Flag | Description |
|---|---|
| `--viz` | Enable live memory visualization |
| `--eval` | Run recall eval suite before chat |
| `--budget N` | Set working memory token budget (default: 2048) |

## CLI Commands (during chat)

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

**`memory_search(query, top_k)`** — searches archival store via vector similarity, returns the recalled content as a `tool` message and re-injects the chunks into working memory so the next turn keeps them in context.

**`archival_save(content)`** — explicitly saves a fact to archival before it gets evicted.

### Conversation threading

Tool calls are recorded in working memory using the standard OpenAI-style three-message pattern:

```
assistant  { content: "...", tool_calls: [{ id, function: { name, arguments } }] }
tool       { content: "<result>", tool_call_id: <id>, name: <tool_name> }
assistant  { content: "<final answer>" }
```

This matters because `tokenizer.apply_chat_template` (the path Nemotron-Nano-9B-v2 uses) expects this exact threading. Earlier iterations recorded only the tool *results* (as bare `system` messages) and dropped the assistant's tool-call turn on the floor — the model then saw orphan tool results with no originating request, which broke the template and caused it to re-issue the same tool call in a loop. The agent now writes the assistant turn (with `tool_calls` metadata) before processing results, and any chunks pulled back from archival are re-inserted *after* the matching `tool` message so the assistant→tool pair stays adjacent.

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (~18GB VRAM for bfloat16)
- `pip install torch transformers accelerate huggingface_hub sentence-transformers numpy rich`
