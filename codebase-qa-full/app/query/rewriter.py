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
    repo_profile = await get_repo_profile(redis_client, repo_id)
    words = IDENTIFIER_RE.findall(question)

    ignore = {
      "what", "where", "when", "why", "how",
      "is", "are", "does", "do",
      "the", "a", "an", "of", "to", "in",
      "for", "about", "explain", "describe",
      "tell", "show", "find",
    }
    symbols = [w for w in words if w.lower() not in ignore]

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
    
    """Returns {"implementation_summary": str, "phrases": list[str], "intent": str}."""
    prompt = _PROMPT_TEMPLATE.format(
    question=question,
    repo_profile=repo_profile,
)
    try:
        resp = await _llm.chat.completions.create(
         model=settings.groq_model,
         max_completion_tokens=300,
         messages=[{"role": "user", "content": prompt}],
)
        text = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(text)
        if not isinstance(data.get("phrases"), list):
         data["phrases"] = [question]

        if data.get("intent") not in {
          "find_function",
          "understand_flow",
          "find_usage",
          "debug",
        }:
         data["intent"] = "find_function"
        data.setdefault("implementation_summary", question)
        data.setdefault("phrases", [question])
        data.setdefault("intent", "find_function")
        return data
    except Exception as e:
        logger.warning(f"HyDE rewrite failed ({e}), falling back to raw query")
        
        return {
    "implementation_summary": question,
    "phrases": question.split(),
    "intent": "find_function",
}


# Intents that benefit from walking the call graph outward from the top hit.
GRAPH_EXPAND_INTENTS = {"understand_flow", "find_usage", "debug"}
