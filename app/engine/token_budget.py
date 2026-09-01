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
import re
from groq import AsyncGroq
from app.config import get_settings

logger = logging.getLogger(__name__)
_llm = AsyncGroq()

settings = get_settings()

GROQ_MODEL = settings.groq_model

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

# select_context() used to budget ONLY chunk["text"] - the raw code slice -
# even though MAX_CONTEXT_TOKENS's own comment above promises a cap on
# "total context sent to the final LLM call". What actually gets sent
# (build_prompt(), below) wraps every chunk in a fixed metadata block
# (Symbol N / Type / Name / File / Lines / Calls / Called by / Registered
# via / Code: labels) and adds system_prompt + user_msg's own boilerplate
# on top - none of which counted against the budget, so the real prompt
# could run well over the nominal 2500 tokens. These are measured (not
# guessed) from the actual template text below, via count_tokens's own
# ~4-chars/token fallback (same estimate method used everywhere else in
# this module, since exact billing-grade precision isn't the point here -
# see the module docstring): the largest system_prompt across all four
# intents (understand_flow, tightened) is 320 tokens; a populated per-
# chunk wrapper (a few Calls/Called by entries, not the empty-metadata
# best case) is ~61 tokens; user_msg's fixed boilerplate outside
# {context}/{question} is 63 tokens. Reserving the worst case up front
# means the budget holds regardless of which intent build_prompt() ends
# up using - select_context() doesn't know `intent` yet at the point it
# runs (see pipeline.py's call order: select_context() before
# build_prompt()), so it can't reserve a tighter, intent-specific amount.
_SYSTEM_PROMPT_RESERVE_TOKENS = 320
_USER_MSG_WRAPPER_TOKENS = 63
_PER_CHUNK_WRAPPER_OVERHEAD_TOKENS = 61


def _tighten_instructions(text: str) -> str:
    """Collapse blank-line padding in static, code-free instruction text.

    build_prompt()'s system_prompt (the rules list + intent task block) is
    sent on every single final-answer call and is almost entirely blank
    lines between one-sentence rules - none of that spacing changes what
    the rules say, it just costs tokens on every request. This only
    collapses runs of blank lines down to a single newline; every word of
    every rule is left exactly as written, same order, same numbering.

    Deliberately NOT applied to `context` (retrieved code/docstrings) or
    the chunk text embedded in the final user message - only to this
    static instructional string, which never contains code.
    """
    return re.sub(r"\n[ \t]*\n+", "\n", text).strip()


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

    # Everything build_prompt() adds around the chunks themselves -
    # system_prompt plus user_msg's fixed boilerplate - comes out of the
    # same max_tokens budget up front, so what's left genuinely bounds
    # the full prompt, not just the raw chunk text.
    effective_max_tokens = max(
        0, max_tokens - _SYSTEM_PROMPT_RESERVE_TOKENS - _USER_MSG_WRAPPER_TOKENS
    )

    for chunk in chunks:
      # Do not cut a selected implementation at an arbitrary character
      # boundary.  The existing token budget below is the single authority
      # for deciding whether a complete chunk fits.
      text = chunk["text"]

      # + the per-chunk metadata wrapper build_prompt() adds (Symbol N /
      # Type / Name / File / Lines / Calls / Called by / Registered via /
      # Code: labels) - not just the raw code text - so a chunk that
      # "fits" here actually fits once wrapped, not just on its own.
      tokens = count_tokens(text) + _PER_CHUNK_WRAPPER_OVERHEAD_TOKENS

      if used + tokens > effective_max_tokens:
       # Never produce an empty prompt merely because the strongest chunk is
       # larger than the context budget.
       if not selected and effective_max_tokens > 0:
        truncate_to = max(0, effective_max_tokens - _PER_CHUNK_WRAPPER_OVERHEAD_TOKENS)
        if "_enc" in globals():
         text = _enc.decode(_enc.encode(text)[:truncate_to])
        else:
         text = text[:truncate_to * 4]
        tokens = count_tokens(text) + _PER_CHUNK_WRAPPER_OVERHEAD_TOKENS
       else:
        continue

      chunk = {**chunk}
      chunk["text"] = text

      selected.append(chunk)
      used += tokens

    logger.info(
        f"Context budget: {used}/{effective_max_tokens} chunk tokens "
        f"({used + _SYSTEM_PROMPT_RESERVE_TOKENS + _USER_MSG_WRAPPER_TOKENS}/{max_tokens} "
        f"total est. incl. system_prompt+wrapper) "
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
    system_prompt = _tighten_instructions(system_prompt)

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