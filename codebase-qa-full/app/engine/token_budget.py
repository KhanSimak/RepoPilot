"""
token_budget.py — counting tokens before you pay for them

THE PROBLEM:
  Naive RAG sends every retrieved chunk to the LLM in full. Five 100-line
  functions is easily 2,000 tokens of context for a question that might
  only need 8 relevant lines from each. You pay for all 2,000 every time.

THE FIX, IN TWO PARTS:
  1. Compress each chunk to just the lines relevant to THIS question,
     using a cheap Groq call per chunk, run in PARALLEL (asyncio.gather)
     so 5 chunks compress in ~200ms total instead of ~1000ms sequential.
  2. After compression, still enforce a hard token cap with tiktoken.
     If we're somehow still over budget, drop the lowest-ranked chunks
     rather than silently sending an oversized, expensive prompt.

WHY THIS STILL MATTERS EVEN THOUGH GROQ IS NEAR-FREE:
  llama-3.1-8b-instant is $0.05/$0.08 per 1M tokens — compression calls
  cost fractions of a cent regardless. The budget cap exists for a
  second reason beyond cost: it also caps LATENCY (fewer tokens for the
  model to read and generate) and keeps the free tier's token-per-minute
  rate limit from being eaten by oversized prompts.

WHY TIKTOKEN WHEN WE'RE CALLING GROQ, NOT OPENAI:
  tiktoken's cl100k_base encoding doesn't perfectly match Llama's own
  tokenizer, but it's close enough for BUDGETING purposes (deciding
  whether to keep or drop a chunk) — we don't need exact billing-grade
  precision here, just a consistent, fast way to compare chunk sizes.
  If tiktoken isn't installed, we fall back to a character-count
  approximation (~4 chars per token) rather than crashing.
"""

import asyncio
import logging
from groq import AsyncGroq

logger = logging.getLogger(__name__)
_llm = AsyncGroq()

GROQ_MODEL = "llama-3.1-8b-instant"

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    logger.warning("tiktoken unavailable — falling back to ~4 chars/token estimate")
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


MAX_CONTEXT_TOKENS   = 2500   # hard cap on total context sent to the final LLM call
MAX_TOKENS_PER_CHUNK = 150   # budget for each individual compression call



def select_context(chunks: list[dict], max_tokens: int = MAX_CONTEXT_TOKENS):
    production = [
      c for c in chunks
       if (
           "/tests/" not in ("/" + c["file"].replace("\\", "/"))
           and "/examples/" not in ("/" + c["file"].replace("\\", "/"))
       )
    ]
 
    if production:
      chunks = production
    chunks = sorted(
      chunks,
      key=lambda c: c.get(
          "rerank_score",
          c.get("graph_score", c.get("score", 0)),
      ),
      reverse=True,
)
    selected = []
    used = 0

    for chunk in chunks:
      # Do not cut a selected implementation at an arbitrary character
      # boundary.  The existing token budget below is the single authority
      # for deciding whether a complete chunk fits.
      text = chunk["text"]

      tokens = count_tokens(text)

      if used + tokens > max_tokens:
       # Never produce an empty prompt merely because the strongest chunk is
       # larger than the context budget.
       if not selected and max_tokens > 0:
        if "_enc" in globals():
         text = _enc.decode(_enc.encode(text)[:max_tokens])
        else:
         text = text[:max_tokens * 4]
        tokens = count_tokens(text)
       else:
        continue

      chunk = {**chunk}
      chunk["text"] = text

      selected.append(chunk)
      used += tokens

    logger.info(
        f"Context budget: {used}/{max_tokens} tokens "
        f"({len(selected)}/{len(chunks)} chunks)"
    )
    logger.debug(
        "Selected context: %s",
        [(c["name"], c["type"], c.get("rerank_score")) for c in selected],
    )

    return selected



def order_by_call_graph(chunks: list[dict]) -> list[dict]:
    """
    Order selected chunks so execution-flow questions are presented in a
    natural request lifecycle rather than pure rerank order.
    """

    if len(chunks) <= 1:
        return chunks

    def bare_name(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    by_name: dict[str, list[dict]] = {}
    for chunk in chunks:
        name = chunk.get("name")
        if not name:
            continue
        by_name.setdefault(name, []).append(chunk)
        bare = bare_name(name)
        if bare != name:
            by_name.setdefault(bare, []).append(chunk)

    called_ids = set()
    for chunk in chunks:
        for callee in chunk.get("calls", []):
            called_ids.update(candidate["id"] for candidate in by_name.get(callee, []))

    roots = [
        c
        for c in chunks
        if c.get("id") not in called_ids
    ]

    if not roots:
        roots = chunks[:1]

    ordered = []
    seen = set()

    def visit(chunk):
        chunk_id = chunk.get("id")

        if chunk_id in seen:
            return

        seen.add(chunk_id)
        ordered.append(chunk)

        for callee in chunk.get("calls", []):
            for candidate in by_name.get(callee, []):
                visit(candidate)

    for root in roots:
        visit(root)

    for chunk in chunks:
        if chunk.get("id") not in seen:
            visit(chunk)

    # -------------------------------------------------------
    # Force important Flask lifecycle symbols into the proper
    # execution order if they are present.
    # -------------------------------------------------------

    FLOW_ORDER = [
        "__call__",
        "Flask.__call__",
        "wsgi_app",
        "Flask.wsgi_app",
        "request_context",
        "Flask.request_context",
        "push",
        "AppContext.push",
        "full_dispatch_request",
        "Flask.full_dispatch_request",
        "preprocess_request",
        "Flask.preprocess_request",
        "dispatch_request",
        "Flask.dispatch_request",
        "finalize_request",
        "Flask.finalize_request",
        "process_response",
        "Flask.process_response",
        "pop",
        "AppContext.pop",
        "do_teardown_request",
        "Flask.do_teardown_request",
    ]

    priority = []

    for symbol in FLOW_ORDER:
        for chunk in ordered:
            if chunk.get("name") == symbol and chunk not in priority:
                priority.append(chunk)

    remaining = [
        chunk
        for chunk in ordered
        if chunk not in priority
    ]

    return priority + remaining



def build_prompt(
    question: str,
    chunks: list[dict],
    intent: str,
):
    system_prompt = """
You are answering questions ONLY from the supplied repository context.

Rules:

1. Use ONLY the provided repository context.

2. Never use outside knowledge.

3. Never invent:
- classes
- functions
- methods
- files
- APIs
- execution steps
- call relationships

4. Treat Calls, Called by, and Registered via metadata as the source of truth.
5. Use the code only to explain what a symbol does.
Do not derive new caller/callee relationships from the code.

6. Read every retrieved symbol before answering.

7. Combine multiple retrieved symbols when they together answer the question.

8. If the repository context is incomplete, explicitly say what is missing instead of guessing.

9. Never mention functions or classes that are not present in the retrieved context.

10. For execution-flow questions, follow the execution order of the retrieved symbols.
"""

    # --------------------------------------------------
    # Intent-specific instructions
    # --------------------------------------------------

    if intent == "find_function":

        task = """
Identify the class or function that best matches the user's question.

Explain its purpose from the repository context.

If multiple retrieved symbols together implement the concept,
explain how they relate.

Use the "Called by" metadata to explain where it is used.

Only mention callers that appear in the retrieved context.

Do not invent usages.
"""

    elif intent == "understand_flow":

        task = """
Explain the execution in chronological order.

Start from the request entry point.

Then describe each function in the order it executes.

Do not reorder functions based on importance.

Use the Calls and Called by metadata as the source of truth.

Use the code only to explain what each function does.

Do not invent missing execution steps.

If the execution path is incomplete,
explicitly state which important runtime functions are missing.
"""

        chunks = order_by_call_graph(chunks)

    elif intent == "find_usage":

        task = """
Explain where this symbol is used.

Use the "Called by" metadata as the source of truth.

Only mention callers that appear in the retrieved context.

Do not invent usages.
"""

    elif intent == "debug":

        task = """
Identify the code most likely responsible for the reported issue.

Use the Calls and Called by metadata to trace execution.

Use the code only to explain behavior.

Do not invent call relationships.

If the retrieved context is insufficient,
say so.
"""

    else:

        task = """
Answer using only the retrieved repository context.

If the repository context is insufficient,
say what information is missing instead of guessing.
"""

    system_prompt += "\n\n" + task

    # --------------------------------------------------
    # Build repository context
    # --------------------------------------------------

    context = []

    for i, c in enumerate(chunks, start=1):

        calls = c.get("calls") or []
        called_by = c.get("called_by") or []
        registered_by = c.get("registered_by") or []

        context.append(
            f"""
Symbol {i}

Type: {c.get("type")}
Name: {c.get("name")}
File: {c.get("file")}
Lines: {c.get("line_start")}-{c.get("line_end")}
Calls: {", ".join(calls) if calls else "(none)"}
Called by: {", ".join(called_by) if called_by else "(none)"}
Registered via: {", ".join(registered_by) if registered_by else "(none)"}

Code:

{c["text"]}
"""
        )

    context = "\n\n".join(context)

    logger.debug("Repository context:\n%s", context)

    user_msg = f"""
Repository context:

{context}

Question:
{question}

Answer using only the repository context.

If multiple retrieved symbols together answer the question,
combine them.

If the repository context is insufficient,
state what information is missing instead of guessing.
"""

    return system_prompt, user_msg
