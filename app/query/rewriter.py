"""
rewriter.py — HyDE query rewriting + intent detection (one combined call)

HyDE (Hypothetical Document Embeddings):
  "Where is JWT auth handled?" is an English question. The code says
  `def verify_jwt_token(token):`. These two strings embed to DIFFERENT
  regions of vector space — a question about code and the code itself
  are not semantically close just because one describes the other.

  HyDE's fix: ask the LLM to write a short HYPOTHETICAL code snippet that
  would answer the question, then embed THAT instead of the raw question.
  The snippet lives in code-shaped vector space, much closer to the real
  implementation than the English question ever was.

WHY INTENT DETECTION IS FOLDED INTO THE SAME CALL:
  A separate "classify intent" call would be another ~150ms and another
  paid LLM round trip for a single classification token. Since we're
  already paying for one call to generate the HyDE snippet, we ask it
  to ALSO classify intent in the same JSON response — zero extra calls.

  intent values:
    find_function   — looking for one specific function/class
    understand_flow  — "how does X work end-to-end" -> triggers graph expansion
    find_usage       — "everywhere X is called" -> BM25-leaning, also graph expansion
    debug            — "why does X fail" -> graph expansion (need full context)

RUNNING THIS ON GROQ (llama-3.1-8b-instant) INSTEAD OF A FRONTIER MODEL:
  Groq's free tier has no per-token cost and the model is extremely fast
  (~560 tok/s), which is exactly what you want for a small structured-output
  call like this one. The tradeoff: an 8B instruct model is somewhat less
  reliable at strictly-valid JSON than a larger frontier model. The fenced-
  code-block stripping below plus the broad except-and-fallback already
  covers the realistic failure mode (a stray ```json fence or minor
  formatting slip) without needing a heavier JSON-repair step.
"""
import json
import logging
import re

from app.cache.redis_cache import get_repo_profile
logger = logging.getLogger(__name__)
 # reads GROQ_API_KEY from the environment automatically
from groq import AsyncGroq
from app.config import get_settings

settings = get_settings()

_llm = AsyncGroq(
    api_key=settings.groq_api_key
)


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PROMPT_TEMPLATE = """You are preparing a retrieval query for a code search system.

Your task is NOT to answer the user's question.

Your goal is to generate a retrieval-oriented implementation summary that is likely to retrieve the correct source code.

Write as if you have already opened the relevant implementation and are briefly describing how it works to another engineer.

Use repository identifiers whenever they are available.

When repository-specific identifiers are unavailable, you may describe generic implementation concepts such as request handling, validation, parsing, rendering, caching, serialization, routing, middleware, retries, sessions, authentication, token verification, configuration loading, template rendering, database access, background jobs, etc.

Never invent repository-specific function names, class names, files or variables.
Repository vocabulary:

{repo_profile}

User question:

{question}

Instructions:

1. Use repository identifiers from the repository vocabulary whenever possible.
2. Never invent class names, function names, methods, files, variables, or configuration keys.
If repository-specific identifiers are unavailable, describe the implementation using generic engineering concepts.
Describe the sequence of operations that the implementation is likely to perform.

Focus on implementation behavior rather than user-facing concepts.
5. Do not teach, define, or explain concepts.
6. Write in the style of a developer summarizing source code after reading it.
7. Keep the summary under 80 words.
Do not invent repository-specific identifiers.

Generic implementation descriptions are encouraged if they improve retrieval.
Then generate 5–10 retrieval phrases.
Only describe implementation details that appear in the supplied context.

If a function is only mentioned but not shown,
state that it is referenced but its implementation was not retrieved.

Do not describe Flask behavior from prior knowledge.

Do not infer missing implementation.

Do not describe APIs not present in the retrieved code.
Include:

- important repository identifiers
- generic implementation concepts
- helper operations
- architectural terms

Do not generate English explanations.
Rules for retrieval phrases:

Return only phrases that are useful for retrieval.
Do not return explanatory English sentences.
Never invent repository identifiers.

Intent classification:

find_function
- User wants one symbol.

understand_flow
- User asks how something works.
- User asks about request flow.
- User asks about lifecycle.
- User asks about pipeline.
- User asks about architecture.

find_usage
- User asks where something is used.
- User asks who calls something.
- User asks for references.
Where is X used?

Where is X referenced?

Who calls X?
debug
- User is diagnosing an error.
- User asks why something failed.
- User asks about exceptions.
- User provides stack traces.

Return ONLY valid JSON.
Do not wrap the JSON in markdown.
Do not include explanations before or after the JSON.
{{
  "implementation_summary": "...",
  "phrases": [...],
  "intent": "understand_flow",

  "graph": {{
      "direction":"both",
      "depth"=1,
      "entry":"request"
  }}
}}

find_function

The user wants ONE symbol.

Examples:

Where is login implemented?

Which class validates JWT?

Which file defines Config?

---

understand_flow

The user wants to understand how a feature is implemented.

Examples:

How does authentication work?

How does connection pooling work?

Explain the request flow.

How are retries handled?

---

find_usage

The user wants references.

Examples:

Where is authenticate called?

Who uses RedisCache?

Find every usage of UserRepository.

---

debug

The user wants to diagnose a problem.

Examples:

Why does login fail?

Why am I getting KeyError?

Why is this request timing out?

"""






async def rewrite_query(
    question: str,
    repo_id: str,
    redis_client,
) -> dict:
    """
    Rewrite a user question into retrieval-oriented information.

    Returns:
    {
        "implementation_summary": str,
        "phrases": list[str],
        "intent": str,
        "graph": {
            "direction": "callers|callees|both",
            "depth": int,
            "entry": str,
        },
    }
    """

    repo_profile = await get_repo_profile(redis_client, repo_id)

    words = IDENTIFIER_RE.findall(question)

    ignore = {
        "what", "where", "when", "why", "how",
        "is", "are", "does", "do",
        "the", "a", "an", "of", "to", "in",
        "for", "about", "explain", "describe",
        "tell", "show", "find",
    }

    symbols = [
        w for w in words
        if w.lower() not in ignore
    ]

    symbol = symbols[0] if len(symbols) == 1 else None

    CODE_WORDS = {
        "client",
        "request",
        "response",
        "config",
        "cache",
        "builder",
        "manager",
        "factory",
        "service",
    }

    prompt = _PROMPT_TEMPLATE.format(
        question=question,
        repo_profile=repo_profile,
    )

    try:
        resp = await _llm.chat.completions.create(
            model=settings.groq_model,
            max_completion_tokens=1200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY one valid JSON object. "
                        "Do not use markdown fences. "
                        "Do not include any text outside the JSON object. "
                        "Keep implementation_summary concise. "
                        "Return at most 8 phrases. "
                        "Keep graph.depth between 1 and 2."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = resp.choices[0].message.content or ""

        logger.debug("HYDE RAW RESPONSE: %r", content)

        if not content.strip():
            raise ValueError("HyDE returned empty content")

        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("HyDE response is not a JSON object")

    except Exception as e:
        logger.warning(
            "HyDE rewrite failed (%s), falling back to raw query",
            e,
        )

        return {
            "implementation_summary": question,
            "phrases": [question],
            "intent": "find_function",
            "graph": {
                "direction": "both",
                "depth": 1,
                "entry": symbol or "",
            },
        }

    # ---------------------------------------------------------
    # Validate and normalize the model response
    # ---------------------------------------------------------

    # implementation_summary
    implementation_summary = data.get("implementation_summary")

    if not isinstance(implementation_summary, str):
        implementation_summary = question

    implementation_summary = implementation_summary.strip()

    if not implementation_summary:
        implementation_summary = question

    data["implementation_summary"] = implementation_summary

    # phrases
    phrases = data.get("phrases")

    if not isinstance(phrases, list):
        phrases = [question]

    phrases = [
        str(p).strip()
        for p in phrases
        if str(p).strip()
    ]

    if not phrases:
        phrases = [question]

    # Keep retrieval expansion bounded.
    data["phrases"] = phrases[:8]

    # intent
    valid_intents = {
        "find_function",
        "understand_flow",
        "find_usage",
        "debug",
    }

    intent = data.get("intent")

    if intent not in valid_intents:
        intent = "find_function"

    data["intent"] = intent

    # ---------------------------------------------------------
    # graph validation
    # ---------------------------------------------------------

    graph = data.get("graph")

    if not isinstance(graph, dict):
        graph = {}

    direction = graph.get("direction", "both")

    if direction not in {
        "callers",
        "callees",
        "both",
    }:
        direction = "both"

    depth = graph.get("depth", 1)

    if not isinstance(depth, int) or isinstance(depth, bool):
        depth = 1

    # Keep graph traversal bounded.
    depth = max(1, min(depth, 2))

    entry = graph.get("entry", "")

    if not isinstance(entry, str):
        entry = ""

    entry = entry.strip()

    # If the model didn't provide an entry but we identified
    # exactly one symbol from the user's question, use that.
    if not entry and symbol:
        entry = symbol

    data["graph"] = {
        "direction": direction,
        "depth": depth,
        "entry": entry,
    }

    return data


# Intents that benefit from walking the call graph outward
# from the top retrieved symbol.
GRAPH_EXPAND_INTENTS = {
    "understand_flow",
    "find_usage",
    "debug",
}