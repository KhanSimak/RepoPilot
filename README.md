# RepoPilot — Autonomous Codebase Intelligence & Coding Agent

**RepoPilot** is an end-to-end codebase intelligence and autonomous coding system that can ingest a GitHub repository, understand its structure, answer questions with precise code citations, investigate bugs, generate code changes, test those changes, and iteratively fix failures before producing a reviewable patch.

It combines **AST-based code understanding, hybrid retrieval, call-graph reasoning, reranking, caching, incremental indexing, persistent repository memory, automated testing, evaluation, and an iterative LangGraph coding agent**.

The goal is not simply to build another "chat with your code" RAG application.

RepoPilot is designed to move from:

> **Understand the repository → investigate the problem → generate a change → test the change → fix failures → produce a reviewable patch.**

---

# 🚀 What It Does

Point RepoPilot at a GitHub repository:

```text
                    GitHub Repository
                           │
                           ▼
                      AST Parsing
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             Code Chunks       Call Graph
                  │                 │
                  └────────┬────────┘
                           ▼
                    Repository Index
                     │            │
                     ▼            ▼
                   Qdrant        BM25
                     │            │
                     └─────┬──────┘
                           ▼
                    Hybrid Retrieval
                           │
                           ▼
                       RRF Fusion
                           │
                           ▼
                    Graph Expansion
                           │
                           ▼
                      Reranking
                           │
                           ▼
                  Context Compression
                           │
                           ▼
                    LLM / Agent
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Grounded Answer             Coding Agent
                                         │
                                         ▼
                                   Generate Patch
                                         │
                                         ▼
                                   Git Worktree
                                         │
                                         ▼
                                   Run Tests
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                           PASS                  FAIL
                              │                     │
                              ▼                     ▼
                       Reviewable Patch       Investigate
                                                    │
                                                    ▼
                                               Fix Code
                                                    │
                                                    ▼
                                               Run Tests
                                                    │
                                                    └───►
```

---

# ✨ Core Capabilities

## 1. Codebase Q&A

Users can ask natural-language questions about an entire repository:

```text
How does connection pooling work?

Where is authentication implemented?

What calls the create_user function?

Why does this request timeout?

Trace the request from the API endpoint to the database.

Where is this configuration value used?
```

RepoPilot retrieves the relevant code and produces an answer grounded in the repository.

Responses can include:

* file paths
* line ranges
* symbols
* retrieved source context
* retrieval scores
* pipeline latency
* cache information

---

# 2. Hybrid Code Retrieval

RepoPilot combines multiple retrieval strategies:

```text
                 User Question
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Vector Search          BM25 Search
             │                   │
             └─────────┬─────────┘
                       ▼
                  RRF Fusion
                       │
                       ▼
                Candidate Chunks
                       │
                       ▼
                 Reranker
                       │
                       ▼
                 Top Context
```

The retrieval layer uses:

* AST-aware chunking
* dense vector retrieval
* BM25 lexical search
* Reciprocal Rank Fusion
* reranking
* call-graph expansion

### Why both vector and BM25?

Vector retrieval is useful for semantic questions:

```text
"How does authentication happen?"
```

BM25 is particularly useful for exact code concepts:

```text
ConnectionPool
create_user
JWT_SECRET
/api/login
TimeoutError
```

Combining both makes retrieval more robust across different question types.

---

# 3. AST-Based Code Understanding

Instead of blindly splitting source code by character count, RepoPilot parses Python code using the AST.

This allows it to preserve logical structures such as:

```text
Class
 ├── method
 ├── method
 └── method
```

rather than producing arbitrary text fragments.

AST extraction also identifies function calls that can later be used to build the repository's call graph.

---

# 4. Call-Graph-Aware Retrieval

RepoPilot builds relationships between functions and methods.

For example:

```text
POST /users
     │
     ▼
create_user()
     │
     ▼
validate_user()
     │
     ▼
get_existing_user()
     │
     ▼
database.query()
```

This becomes particularly useful for questions such as:

```text
How does user creation work end-to-end?

Why does this API endpoint fail?

What functions depend on this function?

Where is this function being called?
```

For flow-oriented queries, the retriever can expand outward through callers and callees.

### Intent-gated graph expansion

Not every question needs graph traversal.

For example:

```text
"Where is create_user defined?"
```

usually requires one precise chunk.

Whereas:

```text
"How does create_user work end-to-end?"
```

benefits from retrieving related functions.

RepoPilot therefore uses query intent to decide when graph expansion is useful.

---

# 5. Autonomous Coding Agent

The Q&A system is extended into an iterative coding agent using **LangGraph**.

The agent does not immediately generate code when given a bug report.

Instead, it follows an investigation loop:

```text
                  User Request
                       │
                       ▼
                  Understand
                   Problem
                       │
                       ▼
                 Retrieve Code
                       │
                       ▼
                Inspect Call Graph
                       │
                       ▼
               Retrieve History
                       │
                       ▼
              Gather Evidence
                       │
                       ▼
              Evidence Sufficient?
                  /          \
                No            Yes
                │              │
                ▼              ▼
          Investigate       Generate
             Again           Patch
                               │
                               ▼
                         Apply in Git
                          Worktree
                               │
                               ▼
                          Run Tests
                               │
                    ┌──────────┴──────────┐
                    │                     │
                   PASS                  FAIL
                    │                     │
                    ▼                     ▼
              Inspect Result        Analyze Failure
                    │                     │
                    ▼                     ▼
             Reviewable Patch        Modify Code
                                          │
                                          ▼
                                      Run Tests
                                          │
                                          └──────►
```

The important difference is that **code generation is only one step in the process**.

The agent must validate what it generated.

---

# 🧪 6. Agent-Driven Testing & Self-Correction

A generated patch is not considered successful simply because the LLM produced syntactically valid code.

RepoPilot can apply the proposed changes inside an isolated Git worktree and run the repository's tests.

The workflow is:

```text
Generate Code
     │
     ▼
Apply Patch
     │
     ▼
Run Test Suite
     │
     ▼
┌───────────────┐
│ Test Results  │
└───────┬───────┘
        │
   ┌────┴────┐
   │         │
 PASS       FAIL
   │         │
   ▼         ▼
Continue   Inspect
             │
             ▼
        Error / Failure
             │
             ▼
       Retrieve Relevant
            Code
             │
             ▼
       Modify Generated
             Code
             │
             ▼
          Re-test
             │
             └──────────►
```

This means the agent can encounter:

```text
pytest
  ↓
3 tests failed
  ↓
Agent reads failures
  ↓
Finds affected implementation
  ↓
Updates patch
  ↓
pytest
  ↓
1 test failed
  ↓
Agent investigates again
  ↓
Updates patch
  ↓
pytest
  ↓
All tests passed
```

The important property is that **test failures become evidence for the next reasoning iteration**.

The agent does not simply generate one answer and stop.

---

# 🔐 7. Isolated Code Changes

Generated code is never directly committed to the main repository.

The coding agent works inside an isolated Git worktree:

```text
Main Repository
      │
      ├──────────────► Agent Worktree
      │                       │
      │                       ▼
      │                  Apply Patch
      │                       │
      │                       ▼
      │                   Run Tests
      │                       │
      │                       ▼
      │                  Agent Iterates
      │                       │
      │                       ▼
      │                 Final Diff
      │
      └──────────────────────────────►
                         Human Review
```

The final output is a reviewable diff.

The human remains responsible for deciding whether the change should actually be committed or merged.

---

# 🧠 8. Persistent Repository Memory

One of the major extensions beyond ordinary codebase RAG is **repository-scoped persistent memory**.

The agent can remember relevant information from previous investigations and coding sessions.

For example:

```text
Repository: payment-service

Previous Investigation:
"Payment requests were timing out because the HTTP connection
pool was exhausted."

Previous Change:
"Increased connection pool timeout and added retry handling."

Previous Test Result:
"Added regression test for connection pool exhaustion."

Current Request:
"Payment requests are timing out again."
```

Instead of starting from zero, the agent can retrieve the relevant previous investigation.

---

## Memory is not the source of truth

Persistent memory is deliberately treated as **historical context**, not authoritative repository state.

The hierarchy is:

```text
Current Repository Code
        │
        ▼
Current Runtime / Test Evidence
        │
        ▼
Current Investigation
        │
        ▼
Relevant Historical Memory
```

If memory says:

```text
"function X uses Redis"
```

but the current repository shows:

```text
function X uses PostgreSQL
```

the current repository wins.

This prevents stale historical information from overriding the actual code.

---

# 🎯 Repository-Scoped Memory

Memory is scoped to a repository rather than being a generic global conversation history.

Example:

```text
Repository A
 ├── Investigation 1
 ├── Investigation 2
 └── Previous changes

Repository B
 ├── Investigation 1
 └── Previous changes
```

When working on Repository A, the agent retrieves only relevant historical context from Repository A.

This prevents unrelated project history from contaminating the investigation.

---

# 🔎 Selective Memory Retrieval

The agent does not inject every previous conversation into the prompt.

Instead:

```text
New Request
     │
     ▼
Memory Retrieval
     │
     ▼
Relevant Previous Runs
     │
     ▼
Top 1–3 Memories
     │
     ▼
Current Investigation
```

This keeps the context small while still allowing the agent to benefit from previous work.

The current repository remains the source of truth.

---

# 🔄 Complete Coding-Agent Loop

Putting retrieval, memory, reasoning, code generation, and testing together:

```text
                    USER REQUEST
                         │
                         ▼
                  Query / Intent
                     Analysis
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Repository Retrieval          Memory Retrieval
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Evidence Gathering
                         │
                         ▼
                  Call Graph Analysis
                         │
                         ▼
                 Evidence Sufficient?
                    /           \
                  No             Yes
                  │               │
                  ▼               ▼
             Investigate       Generate
                Again            Patch
                                  │
                                  ▼
                           Isolated Worktree
                                  │
                                  ▼
                              Run Tests
                                  │
                         ┌────────┴────────┐
                         │                 │
                       PASS              FAIL
                         │                 │
                         ▼                 ▼
                   Inspect Diff       Analyze Failure
                         │                 │
                         │                 ▼
                         │            Retrieve Code
                         │                 │
                         │                 ▼
                         │            Fix Patch
                         │                 │
                         │                 ▼
                         │             Run Tests
                         │                 │
                         │                 └──────►
                         ▼
                  Final Reviewable
                       Change
```

This is the core of the autonomous coding portion of RepoPilot.

---

# ⚡ 9. Retrieval Pipeline

The production query pipeline is:

```text
User Question
     │
     ▼
L1 Redis Cache
     │
     ├── HIT ───────────────► Cached Response
     │
     ▼ MISS
HyDE Query Rewriting
+
Intent Detection
     │
     ▼
Embedding
     │
     ▼
┌───────────────┬───────────────┐
│ Vector Search │   BM25 Search │
└───────┬───────┴───────┬───────┘
        │               │
        └───────┬───────┘
                ▼
           RRF Fusion
                │
                ▼
         Top 20 Candidates
                │
                ▼
        Intent-Gated Graph
             Expansion
                │
                ▼
       Cross-Encoder / API
             Reranking
                │
                ▼
             Top 5
                │
                ▼
       Context Compression
                │
                ▼
           Token Budget
                │
                ▼
              LLM
                │
                ▼
          Grounded Answer
```

---

# 🗃️ 10. Incremental Repository Ingestion

RepoPilot avoids reprocessing an entire repository whenever a small change occurs.

The synchronization pipeline is:

```text
git diff
   │
   ▼
Changed Files
   │
   ▼
AST Re-chunk
   │
   ▼
Content Hash Comparison
   │
   ├── Unchanged → Skip
   │
   └── Changed → Re-embed
```

For example:

```json
{
  "status": "done",
  "mode": "incremental",
  "changed_files": 3,
  "embedded_chunks": 11,
  "skipped_chunks": 847,
  "total_chunks": 858
}
```

Only the chunks whose actual content changed need to be re-embedded.

---

# 📊 11. Retrieval Evaluation

The project includes an evaluation framework instead of relying purely on manual testing.

It measures:

* Recall@5
* Recall@10
* MRR
* P50 latency
* P95 latency
* P99 latency
* Per-file retrieval quality

Example:

```json
{
  "questions_run": 87,
  "recall_at_5": 0.839,
  "recall_at_10": 0.908,
  "mrr": 0.701,
  "latency_ms": {
    "p50": 38.2,
    "p95": 71.4,
    "p99": 95.0
  }
}
```

The retrieval benchmark isolates retrieval quality from the much harder problem of judging free-form LLM answers.

---

# 🧩 12. Architecture

```text
app/
├── main.py
├── config.py
│
├── models/
│   └── chunk.py
│
├── schemas/
│   └── api.py
│
├── engine/
│   ├── ast_chunker.py
│   ├── embedder.py
│   ├── vectordb.py
│   ├── bm25.py
│   ├── fusion.py
│   ├── call_graph.py
│   ├── reranker.py
│   ├── token_budget.py
│   └── cost_tracker.py
│
├── cache/
│   └── redis_cache.py
│
├── memory/
│   └── repository_memory.py
│
├── agent/
│   ├── graph.py
│   ├── investigator.py
│   ├── coder.py
│   ├── tester.py
│   └── worktree.py
│
├── query/
│   ├── rewriter.py
│   ├── retriever.py
│   └── pipeline.py
│
├── ingest/
│   ├── cloner.py
│   ├── pipeline.py
│   └── incremental.py
│
├── eval/
│   ├── golden_dataset.py
│   ├── metrics.py
│   └── runner.py
│
└── routers/
    ├── repos.py
    ├── search.py
    ├── query.py
    ├── agent.py
    ├── stats.py
    └── eval.py
```

---

# 🛠️ Tech Stack

| Category              | Technology                         |
| --------------------- | ---------------------------------- |
| Language              | Python                             |
| API                   | FastAPI                            |
| Agent orchestration   | LangGraph                          |
| Vector database       | Qdrant                             |
| Cache / memory        | Redis                              |
| LLM inference         | Groq / OpenRouter                  |
| Embeddings            | ONNX Runtime                       |
| Lexical retrieval     | BM25                               |
| Reranking             | Cohere Rerank                      |
| Code understanding    | Python AST                         |
| Retrieval fusion      | Reciprocal Rank Fusion             |
| Repository operations | GitPython                          |
| Code isolation        | Git Worktrees                      |
| Streaming             | Server-Sent Events                 |
| Evaluation            | Recall@K, MRR, latency percentiles |
| Deployment            | Docker                             |

---

# ⚙️ Engineering Decisions

## Why AST chunking?

Traditional RAG systems often split code by character count.

That can break logical structures across chunk boundaries.

AST-based chunking instead preserves functions, classes, and other meaningful code structures.

---

## Why BM25 + Vector Search?

Vector retrieval handles semantic similarity.

BM25 handles exact code vocabulary.

Together they cover:

```text
Semantic questions
       +
Exact symbols
       +
Error messages
       +
Configuration keys
       +
API routes
```

---

## Why RRF?

Vector search and BM25 produce independent rankings.

Reciprocal Rank Fusion combines those rankings without requiring their raw scores to be directly comparable.

---

## Why reranking?

Initial retrieval optimizes for recall.

Reranking improves precision by evaluating:

```text
Question ↔ Candidate Code
```

and selecting the strongest candidates for the final context.

---

## Why graph expansion only for certain queries?

Graph traversal can add useful context but can also introduce irrelevant code.

Therefore it is enabled for intents such as:

```text
understand_flow
find_usage
debug
```

while precise lookup queries can avoid unnecessary expansion.

---

## Why content hashes?

A modified file does not necessarily mean every function inside that file changed.

Chunk-level content hashes allow the system to determine which actual code units changed.

---

# 🧠 Why Persistent Memory?

Normal RAG answers questions using the current repository.

RepoPilot's coding agent additionally needs to understand **what happened previously**.

For example:

```text
Day 1:
Agent investigates timeout.

Day 2:
Agent adds a regression test.

Day 10:
Another timeout appears.

Day 10 agent:
"Have we seen this failure before?"
        │
        ▼
Retrieve relevant repository memory
        │
        ▼
Use it as historical evidence
        │
        ▼
Verify against current code
        │
        ▼
Continue investigation
```

This makes the agent less repetitive across multiple coding sessions while maintaining the current repository as the authority.

---

# 🧪 Why Testing Is Part of the Agent

A coding LLM can produce code that:

* looks reasonable
* compiles
* passes a superficial inspection
* but breaks existing behavior

RepoPilot therefore treats test execution as another source of evidence.

```text
LLM says:
"This patch should fix the bug."

        ↓

Agent does NOT assume it is correct.

        ↓

Run tests.

        ↓

Tests fail.

        ↓

Failure becomes new evidence.

        ↓

Agent investigates and modifies patch.

        ↓

Run tests again.
```

This creates a closed-loop coding process rather than a one-shot code-generation system.

---

# 🔐 Safety Model

The coding agent follows several safety principles:

### Current repository is authoritative

Historical memory cannot override current source code.

### Evidence before modification

The agent investigates before generating a patch.

### Isolated execution

Changes are applied inside a separate Git worktree.

### Tests before completion

A patch is not treated as successful simply because it was generated.

### Human review before commit

The final diff remains reviewable before it enters the main repository.

---

# 🌊 Streaming

The API supports Server-Sent Events.

Instead of waiting for the entire response:

```text
Request
  │
  ▼
Sources
  │
  ▼
Token
  │
  ▼
Token
  │
  ▼
Token
  │
  ▼
Done + Trace
```

This allows clients to display the answer progressively.

---

# 🔌 API

## Ingest a repository

```bash
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"github_url":"https://github.com/encode/httpx","branch":"master"}'
```

---

## Ask a question

```bash
curl -X POST \
"http://localhost:8000/repos/a1b2c3d4/ask?question=how+does+connection+pooling+work&top_k=5"
```

Example:

```json
{
  "question": "how does connection pooling work",
  "answer": "Connection pooling is implemented in the ConnectionPool class...",
  "rewritten_query": "class ConnectionPool...",
  "intent": "understand_flow",
  "sources": [
    {
      "name": "ConnectionPool",
      "type": "class",
      "file": "httpx/_transports/default.py",
      "line_start": 45,
      "line_end": 89,
      "score": 7.21
    }
  ]
}
```

---

## Stream an answer

```bash
curl -N \
"http://localhost:8000/repos/a1b2c3d4/stream?question=how+does+connection+pooling+work"
```

---

## Synchronize a repository

```bash
curl -X POST \
http://localhost:8000/repos/a1b2c3d4/sync
```

---

## Run evaluation

```bash
curl -X POST \
"http://localhost:8000/eval/run?repo_id=a1b2c3d4&max_questions=100"
```

---

# 🐳 Running Locally

Start Qdrant and Redis:

```bash
docker compose up -d
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

---

# 📁 Project Evolution

RepoPilot evolved through several stages:

```text
Phase 1
AST Code Chunking
      │
      ▼
Phase 2
Vector + BM25 Hybrid Retrieval
      │
      ▼
Phase 3
Caching + Query Optimization
      │
      ▼
Phase 4
HyDE + Reranking + Call Graph
      │
      ▼
Phase 5
Incremental Ingestion + Evaluation
      │
      ▼
Phase 6
LangGraph Autonomous Coding Agent
      │
      ▼
Phase 7
Persistent Repository Memory
      │
      ▼
Phase 8
Automated Testing + Self-Correction
```

The architecture therefore evolved from a basic code search/RAG system into an agent capable of investigating and modifying a repository.

---

# 🎯 Design Philosophy

RepoPilot follows several principles.

### 1. Retrieval before generation

If the relevant code is not retrieved, the LLM should not be expected to know it.

### 2. Evidence before action

The coding agent investigates before proposing a modification.

### 3. Tests are evidence

A test failure is not simply an error — it is information the agent can use to continue its investigation.

### 4. Memory is contextual, not authoritative

Historical information helps the agent, but current repository state always wins.

### 5. Human review before commit

The agent can investigate, generate, test, and refine changes, but the final change remains reviewable before being committed.

### 6. Measure instead of assuming

Retrieval quality and latency are benchmarked explicitly.

### 7. Optimize where it matters

Caching, incremental ingestion, graph expansion, and reranking are used to improve real pipeline behavior rather than simply adding components for complexity.

---

# 🔮 Future Directions

Potential extensions include:

* multi-language AST support
* compiler/LSP-backed symbol resolution
* stronger test-aware patch validation
* automated test generation
* dependency graph reasoning
* security-aware code review
* stronger agent verification loops
* production observability integration
* richer long-term repository memory
* end-to-end coding-agent evaluation benchmarks

---

# 📌 Project Goal

RepoPilot aims to bridge the gap between **codebase RAG** and **autonomous software engineering**.

The system combines:

```text
Code Understanding
        +
Hybrid Retrieval
        +
Call-Graph Reasoning
        +
Persistent Repository Memory
        +
Agentic Investigation
        +
Code Generation
        +
Automated Testing
        +
Self-Correction
        +
Safe Code Isolation
        +
Human Review
```

The resulting workflow is:

> **Ask about the code → investigate the repository → use relevant historical context → gather evidence → generate a change → run tests → analyze failures → fix the change → retest → return a reviewable patch.**

That is the central idea behind RepoPilot: **an AI coding agent that does not stop at generating code, but investigates, validates, and iterates against the actual repository.**

---

## License

MIT
