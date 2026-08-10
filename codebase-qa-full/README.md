# Codebase Q&A Engine — Final Phase

> Point at any GitHub repo. Ask a question in plain English. Get a precise, cited answer — with a full breakdown of exactly what it cost and how long every stage took.

This is the **final phase** of the project, built on top of Phase 1 (AST chunking + ONNX embeddings) and Phase 2 (hybrid search + Redis caching). This phase adds everything that turns "a RAG demo" into "a system with measured, defensible engineering decisions": query rewriting, reranking, token budgeting, call-graph-aware retrieval, incremental ingestion, and a retrieval evaluation framework.

> **LLM provider: Groq, not Anthropic.** Every text-generation call (HyDE rewriting, context compression, the final answer) runs on Groq's `llama-3.1-8b-instant` instead of Claude. Groq has a genuine no-credit-card free tier (~30 requests/min, ~6,000 tokens/min, 14,400 requests/day) and is dramatically faster (~560 tokens/sec on Groq's custom LPU hardware) — you'll notice the `llm_generation` stage in the cost trace below is the FASTEST stage in the pipeline, not the slowest. Get a free key at [console.groq.com/keys](https://console.groq.com/keys). The only embedding/reranking models (ONNX + CrossEncoder) are unaffected — those were always local and free regardless of LLM provider.

---

## What's new in this phase

| Capability | File | What it solves |
|---|---|---|
| **HyDE query rewriting + intent detection** | `app/query/rewriter.py` | Bridges the gap between English questions and code-shaped embeddings; classifies intent in the same call |
| **Call graph (`calls` → `called_by`)** | `app/engine/call_graph.py` | Turns "find a function" into "understand a flow" by walking callers/callees |
| **Graph-aware retrieval** | `app/query/retriever.py` | Automatically expands results outward along the call graph for flow/usage/debug questions |
| **Cross-encoder reranking** | `app/engine/reranker.py` | Narrows 20 candidates to the true top 5, locally, for $0 |
| **Token budget manager** | `app/engine/token_budget.py` | Compresses chunks in parallel, enforces a hard context-token cap before the LLM call |
| **Per-stage cost/latency tracer** | `app/engine/cost_tracker.py` | Every response shows exactly what each pipeline stage cost and took |
| **SSE streaming** | `app/query/pipeline.py`, `app/routers/query.py` | First token in ~300ms instead of waiting for the full answer |
| **Incremental ingest** | `app/ingest/incremental.py` | git diff + content hashing — only re-embed what actually changed |
| **Evaluation framework** | `app/eval/` | Auto-generated golden dataset, Recall@5, Recall@10, MRR, latency percentiles, per-file quality scores |

---

## Full architecture

```
                         POST /repos {github_url}
                                 │
                                 ▼
                    clone → walk .py files → AST chunk
                                 │
                    build_called_by()  ◄── NEW: inverts calls into called_by
                    across ALL chunks, once, before embedding
                                 │
              check embedding cache (Redis) → embed only misses (ONNX)
                                 │
                    upsert to Qdrant + build BM25 index
                                 │
                                 ▼
                       [repo ready for queries]


                    POST /repos/{id}/ask  or  GET /repos/{id}/stream
                                 │
                                 ▼
                ┌────────────────────────────────────┐
                │ 1. L1 cache check (Redis)           │  ~1ms on hit, full pipeline skipped
                └────────────────┬───────────────────┘
                                 │ MISS
                ┌────────────────▼───────────────────┐
                │ 2. HyDE rewrite + intent detection  │  one Groq call, ~60ms
                │    (app/query/rewriter.py)          │
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │ 3. Embed HyDE snippet (L2 cache)    │  ~25ms, or ~1ms cached
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │ 4. Vector search + BM25 in parallel │  ~25ms
                │    → RRF fusion → top 20            │
                │    → IF intent needs it: graph       │
                │      expand via calls/called_by      │  (app/query/retriever.py)
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │ 5. Cross-encoder rerank → top 5      │  ~80ms, $0
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │ 6. Parallel compression + token      │  ~200ms (parallel, not 1000ms)
                │    budget enforcement (max 500 tok)  │
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │ 7. Final LLM call — Groq (llama-3.1-8b-instant), streamed  │  ~300ms to first token
                └────────────────┬───────────────────┘
                                 │
                       cache result + return trace


                    POST /repos/{id}/sync   (incremental re-ingest)
                                 │
                                 ▼
                git pull → git diff(last_commit, HEAD) → changed files only
                                 │
                re-chunk changed files → compare content_hash vs stored
                                 │
                embed ONLY chunks whose body actually changed
                                 │
                rebuild call graph (whole repo, in-memory, cheap)
                                 │
                rebuild BM25 index (whole repo, in-memory, cheap)


                    POST /eval/run?repo_id=...
                                 │
                                 ▼
                auto-generate golden Q&A from docstrings
                                 │
                run hybrid retrieval for each question, record rank found
                                 │
                compute Recall@5, Recall@10, MRR, latency P50/P95/P99,
                worst-performing files (sorted by recall, ascending)
```

---

## Project structure (final phase)

```
codebase-qa-phase4/
├── app/
│   ├── main.py                   # loads embedder, reranker, qdrant, redis at startup
│   ├── config.py
│   │
│   ├── models/
│   │   └── chunk.py               # CodeChunk — now includes called_by
│   │
│   ├── schemas/
│   │   └── api.py
│   │
│   ├── engine/
│   │   ├── ast_chunker.py         # unchanged from Phase 1 — extracts calls
│   │   ├── embedder.py            # unchanged from Phase 1 — ONNX, $0/query
│   │   ├── vectordb.py            # + retrieve_by_ids, scroll_repo_chunks
│   │   ├── bm25.py                # unchanged from Phase 2
│   │   ├── fusion.py              # unchanged from Phase 2 — RRF
│   │   ├── call_graph.py          # NEW — build_called_by, expand_by_graph
│   │   ├── reranker.py            # NEW — CrossEncoder, local, $0
│   │   ├── token_budget.py        # NEW — counting, compression, budget cap
│   │   └── cost_tracker.py        # NEW — StageTimer, RequestTrace
│   │
│   ├── cache/
│   │   └── redis_cache.py         # unchanged from Phase 2 — two-layer cache
│   │
│   ├── query/                     # NEW module
│   │   ├── rewriter.py            # HyDE + intent detection (one call)
│   │   ├── retriever.py           # hybrid search + optional graph expand
│   │   └── pipeline.py            # full orchestrator: run_query + stream_query
│   │
│   ├── ingest/
│   │   ├── cloner.py              # unchanged from Phase 1
│   │   ├── pipeline.py            # + call graph build, + commit hash tracking
│   │   └── incremental.py         # NEW — git diff + content hash re-ingest
│   │
│   ├── eval/                      # NEW module
│   │   ├── golden_dataset.py      # auto Q&A generation from docstrings
│   │   ├── metrics.py             # Recall@K, MRR, latency percentiles
│   │   └── runner.py              # runs the benchmark end to end
│   │
│   └── routers/
│       ├── repos.py               # + POST /repos/{id}/sync
│       ├── search.py               # unchanged — Phase 2 baseline, kept for comparison
│       ├── query.py                # NEW — /ask and /stream (the final pipeline)
│       ├── stats.py                # unchanged
│       └── eval.py                 # NEW — POST /eval/run
│
├── tests/
│   ├── test_chunker.py
│   ├── test_hybrid_search.py
│   ├── test_call_graph.py          # NEW
│   └── test_eval_metrics.py        # NEW
│
├── docker-compose.yml               # Qdrant + Redis — no new infra this phase
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Using the API

### Ingest a repo (same as Phase 1/2)
```bash
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/encode/httpx", "branch": "master"}'
```

### Ask with the full pipeline
```bash
curl -X POST "http://localhost:8000/repos/a1b2c3d4/ask?question=how+does+connection+pooling+work&top_k=5"
```
```json
{
  "question": "how does connection pooling work",
  "answer": "Connection pooling is implemented in the ConnectionPool class (httpx/_transports/default.py)...",
  "rewritten_query": "class ConnectionPool:\n    def acquire(self):\n        ...",
  "intent": "understand_flow",
  "sources": [
    {"name": "ConnectionPool", "type": "class", "file": "httpx/_transports/default.py", "line_start": 45, "line_end": 89, "score": 7.21}
  ],
  "cache_hit": false,
  "trace": {
    "total_latency_ms": 410.8,
    "total_cost_usd": 0.000125,
    "cache_hits": [],
    "stages": [
      {"stage": "query_cache_l1", "latency_ms": 0.4, "cost_usd": 0.0, "cache_hit": false},
      {"stage": "hyde_rewrite", "latency_ms": 31.4, "cost_usd": 0.000016},
      {"stage": "hybrid_retrieval_graph_expanded", "latency_ms": 41.7, "cost_usd": 0.0},
      {"stage": "reranker", "latency_ms": 79.3, "cost_usd": 0.0},
      {"stage": "context_compression", "latency_ms": 142.3, "cost_usd": 0.000065},
      {"stage": "llm_generation", "latency_ms": 96.7, "cost_usd": 0.000044}
    ]
  }
}
```

Notice `intent: "understand_flow"` triggered `hybrid_retrieval_graph_expanded` — the call graph walk happened automatically because the question asked "how does X work," not "find function X."

### Stream the same question
```bash
curl -N "http://localhost:8000/repos/a1b2c3d4/stream?question=how+does+connection+pooling+work"
```
Emits `sources` immediately, then `token` events as the answer is generated, then `done` with the full trace.

### Incremental sync after a code change
```bash
curl -X POST http://localhost:8000/repos/a1b2c3d4/sync
```
```json
{
  "status": "done", "mode": "incremental",
  "changed_files": 3, "embedded_chunks": 11,
  "skipped_chunks": 847, "total_chunks": 858,
  "new_commit": "a3f9c21..."
}
```
847 chunks were untouched and skipped entirely — only 11 actually needed re-embedding.

### Run the evaluation benchmark
```bash
curl -X POST "http://localhost:8000/eval/run?repo_id=a1b2c3d4&max_questions=100"
```
```json
{
  "repo_id": "a1b2c3d4", "questions_run": 87,
  "recall_at_5": 0.839, "recall_at_10": 0.908, "mrr": 0.701,
  "latency_ms": {"p50": 38.2, "p95": 71.4, "p99": 95.0},
  "worst_files": [
    {"file": "httpx/_legacy/old_client.py", "recall": 0.33, "question_count": 3}
  ]
}
```

---

## Design decisions and the reasoning behind each

**Why fold intent detection into the same call as HyDE instead of a separate classification call?**
A separate call would cost another ~150ms and another paid round trip for a single classification token. Since we already pay for one Groq call to generate the HyDE snippet, asking it to also return `intent` in the same JSON response is effectively free — zero extra latency, zero extra cost.

**Why is the call graph "approximately right" rather than fully accurate?**
`calls` is extracted from bare AST `Call` nodes — `self._get_user(x)` is recorded as `"_get_user"`, not as a fully-qualified symbol. Inverting this means two unrelated classes that both define a method called `save()` get treated as the same callee. True symbol resolution needs a real type checker (Jedi, an LSP server, or a full compiler frontend) — out of scope here. For RAG retrieval, an approximately-right call graph still meaningfully improves flow-style answers; it doesn't need compiler-grade precision to be useful, and the code says so honestly rather than overclaiming.

**Why does graph expansion only trigger for certain intents, not every query?**
`find_function` queries ("where is X defined") are usually well-served by one precise chunk — expanding the graph would dilute the context with tangentially related code and cost more tokens for no benefit. `understand_flow`, `find_usage`, and `debug` genuinely need the surrounding context. Gating the expensive operation (a full-repo scroll + BFS) behind intent detection means you only pay for it when it actually helps.

**Why benchmark hybrid retrieval alone in the eval framework, skipping HyDE/rerank/LLM?**
Evaluating the final LLM-generated answer's correctness is a much harder, fuzzier problem requiring an LLM-as-judge or human review. Recall@K and MRR on retrieval are objective, fast, and cheap to compute repeatedly — and they isolate the part of the pipeline that's most likely to silently regress (a chunking or indexing change). If the right chunk never reaches the LLM, no amount of prompt engineering fixes the answer; this is the metric that catches that failure mode early.

**Why content-hash comparison instead of just trusting git diff's file-level change?**
A file showing up in `git diff` doesn't mean every function in it changed — maybe only a docstring elsewhere in the file changed, or a comment was added. Content-hashing at the CHUNK level (not the file level) means we only pay the ONNX embedding cost for functions whose actual body changed, even within a file that technically shows up as "modified."

**Why is the eval framework's golden dataset auto-generated instead of hand-labeled?**
Hand-labeling a benchmark is slow enough that most projects skip evaluation entirely — which is why most RAG systems can only claim "it seems to work." Every chunk with a docstring is a self-labeling question: "What does `{name}` do?" with the chunk itself as ground truth. This isn't a substitute for a curated production eval set, but it requires zero manual effort and degrades honestly — a repo with poor docstring coverage gets an honest "not enough data" message rather than a misleading score.

---

## What to try once it's running

1. Ask the same question via `/search` (Phase 2 baseline) and `/ask` (final pipeline). Compare the `sources` returned and the latency — this is your evidence for what HyDE + reranking actually changed.
2. Ask a "how does X work end-to-end" question and inspect the trace's stage name — confirm it says `hybrid_retrieval_graph_expanded`, then check whether the returned sources include both an entry point and its immediate callees/callers.
3. Make a small code change in your test repo, commit it, hit `/sync`, and compare `embedded_chunks` vs `skipped_chunks` — this is the proof that incremental ingest is doing real work.
4. Run `/eval/run` once right after ingest, note the `recall_at_5`. Deliberately make chunking worse (e.g. temporarily strip docstrings from a few functions) and run it again — watch the number move. This is what makes "we measured it" a true statement instead of a slogan.
