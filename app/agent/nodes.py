"""
agent/nodes.py — node functions for the ReAct loop.

Reuses build_prompt()/select_context() from token_budget.py for the final
answer (same prompt construction the static pipeline already uses) and the
same AsyncGroq client pattern rewriter.py uses, rather than inventing a
second, inconsistent way of talking to Groq.
"""
import ast
import json
import logging
import re
import textwrap
import time

from groq import AsyncGroq, RateLimitError
from openai import AsyncOpenAI

from app.agent.state import AgentState
from app.agent.tools import retrieval_tool, expand_graph_tool
from app.engine.token_budget import select_context, build_prompt, count_tokens
from app.engine.reranker import rerank, is_low_confidence
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# _chat_with_fallback below already implements its own retry strategy:
# on a Groq RateLimitError it falls over to OpenRouter immediately. But
# with no explicit timeout/max_retries here, both SDKs fall back to
# their own library defaults - which, per both SDKs' documented
# behavior, means a long per-request timeout AND a small number of
# automatic retries-with-backoff on retryable statuses, 429 included.
# That means a single rate-limited Groq call can be silently retried
# INSIDE the SDK, with backoff sleeps between attempts, before
# RateLimitError is ever raised back to the `except` below - our own
# fallback can't run until the SDK's own hidden retries are exhausted,
# turning "one call" into several real network round trips no log line
# here shows. Since we already have our own fallback destination
# (OpenRouter), the SDK's internal retry adds latency without adding
# resilience - it's retrying the exact call we're about to abandon
# anyway. Explicit values here don't change which provider ultimately
# answers or what it returns; they only change how fast a stalled/
# rate-limited Groq call is recognized as such.
_groq_llm = AsyncGroq(
    api_key=settings.groq_api_key,
    timeout=20.0,
    max_retries=0,
)

_openrouter_llm = (
    AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=20.0,
        max_retries=1,
    )
    if settings.openrouter_api_key
    else None
)


MAX_AGENT_ITERATIONS = 6

REASONING_PROMPT = """You are investigating a codebase to answer a question,
deciding one step at a time.

Question: {question}

Chunks gathered so far ({num_chunks}):
{context_summary}

Reasoning trace so far:
{trace}

Retrieval status: {retrieval_status}

Choose exactly one next action:
- "retrieve": an entirely new concept/symbol is missing - not something
  already connected via Calls/Called by to a gathered chunk. Give a
  refined search query, more specific than the original. Never restate a
  topic already searched.
- "expand_graph": the missing piece is a caller, callee, or execution step
  connected to a gathered symbol's Calls or Called by. Prefer this over a
  second search on the same topic for flow/usage/"how does this work"
  questions - give the exact symbol name to expand.
- "answer": only once the gathered code covers every major part of the
  question with an actual runtime/business implementation - not just a
  declaration, wrapper, config class, or registration helper. Never use
  outside/framework knowledge to bridge a gap; if something is missing,
  name it in "missing_information" and choose "retrieve" or
  "expand_graph" instead.

Respond with exactly one JSON object and nothing else: no markdown, no
code fences, no text before or after it. The object must have exactly
these four keys:
  "thought": ONE short sentence, max ~15 words
  "action": exactly one of "retrieve", "expand_graph", or "answer"
  "action_input": a string - the refined search query for "retrieve", the
    exact symbol name to expand for "expand_graph", or "" for "answer"
  "missing_information": an array of strings naming any symbol or
    execution step still missing (an empty array [] if nothing is missing)

Example of the required shape (these values are illustrative only - use
your own thought/action/action_input for this question):
{{"thought": "Need the caller of validate_token to see how it's invoked", "action": "expand_graph", "action_input": "validate_token", "missing_information": []}}
"""

# Sent as the system message on every reasoning call. Kept short but now
# states the JSON schema explicitly and in the same terms as the user
# message's schema block below, instead of leaving the system message
# schema-free (generic "follow the user message" boilerplate) while the
# user message carried all the real detail — Groq's JSON-mode validator is
# more reliable when the schema is stated plainly and consistently rather
# than only implied by one ambiguous example, which is what caused
# `400 json_validate_failed` with an empty `failed_generation`.
_REASONING_SYSTEM_MESSAGE = (
    "You are investigating a codebase to decide the next ReAct step. "
    "Respond with exactly one JSON object and nothing else - no markdown, "
    "no code fences, no explanation before or after it. The object has "
    "exactly four keys: \"thought\" (a short string), \"action\" (exactly "
    "one of the strings \"retrieve\", \"expand_graph\", or \"answer\"), "
    "\"action_input\" (a string, use \"\" when action is \"answer\"), and "
    "\"missing_information\" (an array of strings, use [] when nothing is "
    "missing). Follow the detailed policy in the user message to choose "
    "which action to take."
)


_TOPIC_STOP_WORDS = {
    "a", "an", "and", "application", "code", "does", "do", "for",
    "from", "how", "in", "into", "is", "load", "loading", "of", "on",
    "or", "the", "this", "to", "values", "what", "with",
    # Generic meta-question words: describe the SHAPE of a "how does X
    # work/happen" style question, not its subject. Left uncovered, these
    # count toward the lexical-coverage denominator in
    # _gathered_evidence_sufficient even though they essentially never
    # appear literally in source code - which caps coverage at 50% for
    # any two-term topic like "sessions work" no matter how completely
    # the actual subject ("sessions") is covered by gathered evidence.
    # Generic across any codebase/question, not specific to this repo.
    "work", "works", "working", "happen", "happens", "happening",
    "explain", "explains", "understand", "describe", "describes",
    "show", "shows",
}

_TOPIC_ALIASES = {
    "configuration": "config",
    "configure": "config",
    "configured": "config",
    "configuring": "config",
    "environment": "env",
    "variable": "env",
    "variables": "env",
}

def _is_groq_json_validate_failure(exc: Exception) -> bool:
    """True only for Groq's structured-output validation rejection: an
    HTTP 400 with error code "json_validate_failed" (an empty
    `failed_generation`, per the note below). Checked via attributes the
    Groq/OpenAI-style SDK error objects expose, with a string fallback,
    rather than importing a specific exception class name that may not
    exist across SDK versions - this keeps the check narrow (only this
    exact failure matches) without adding an import that could itself
    fail at module load.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    body = getattr(exc, "body", None)
    error_code = None
    if isinstance(body, dict):
        error_code = (body.get("error") or {}).get("code")
    return error_code == "json_validate_failed" or "json_validate_failed" in str(exc)


async def _chat_with_fallback(
    *,
    messages: list[dict],
    max_completion_tokens: int,
    response_format: dict | None = None,
):
    """Try Groq first; on a Groq rate-limit error, fall back to OpenRouter.
    response_format (e.g. {"type": "json_object"} for reasoning_node's JSON
    mode) is forwarded to whichever provider actually serves the request,
    not just Groq - an OpenRouter fallback that silently dropped it would
    turn reasoning_node's JSON-mode call into free text on every fallback.
    """
    kwargs = {
        "model": settings.groq_model,
        "max_completion_tokens": max_completion_tokens,
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        logger.info(
            "Agent reasoning: calling Groq model '%s'",
            settings.groq_model,
        )

        _t0 = time.perf_counter()
        resp = await _groq_llm.chat.completions.create(**kwargs)
        logger.info(
            "Agent reasoning: groq_call_ms=%.1f model=%s",
            (time.perf_counter() - _t0) * 1000,
            settings.groq_model,
        )
        return resp

    except RateLimitError as exc:
        logger.warning(
            "Agent reasoning: Groq rate limit reached after groq_call_ms=%.1f. "
            "Falling back to OpenRouter. Error: %s",
            (time.perf_counter() - _t0) * 1000,
            exc,
        )

        if _openrouter_llm is None:
            raise

        logger.info(
            "Agent reasoning: calling OpenRouter model '%s'",
            settings.openrouter_model,
        )

        kwargs["model"] = settings.openrouter_model
        _t1 = time.perf_counter()
        resp = await _openrouter_llm.chat.completions.create(**kwargs)
        logger.info(
            "Agent reasoning: openrouter_call_ms=%.1f model=%s",
            (time.perf_counter() - _t1) * 1000,
            settings.openrouter_model,
        )
        return resp

    except Exception as exc:
        # Groq's JSON-mode validator can reject the request outright with
        # a 400 ("json_validate_failed") before any content is generated -
        # this is a request-level rejection, not a malformed-but-present
        # JSON string, so it never reaches reasoning_node's own
        # json.loads()-based recovery. Only this exact, known failure is
        # handled here (never a blanket catch) - anything else re-raises
        # unchanged, so no other error class is masked.
        if response_format is None or not _is_groq_json_validate_failure(exc):
            raise

        logger.warning(
            "Agent reasoning: Groq JSON-mode request rejected "
            "(json_validate_failed). Error: %s",
            exc,
        )

        if _openrouter_llm is not None:
            logger.info(
                "Agent reasoning: calling OpenRouter model '%s'",
                settings.openrouter_model,
            )
            kwargs["model"] = settings.openrouter_model
            return await _openrouter_llm.chat.completions.create(**kwargs)

        # No fallback provider configured. Retry Groq once without
        # response_format instead of letting the whole reasoning step
        # crash. The reasoning prompt already instructs the model to
        # return raw JSON, and reasoning_node's existing parser (markdown-
        # fence stripping, then json.loads() inside its own try/except
        # that already recovers from malformed JSON) is unchanged and
        # still runs exactly as before on whatever content comes back.
        logger.warning(
            "Agent reasoning: no OpenRouter fallback configured. "
            "Retrying Groq without JSON response_format."
        )
        retry_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        return await _groq_llm.chat.completions.create(**retry_kwargs)


def _expanded_symbols(trace: list[str]) -> set[str]:
    """Return symbols that have already had a graph expansion attempted."""
    return {
        match.group(1)
        for line in trace
        if (match := re.search(r"call-graph expansion from '([^']+)'", line))
    }


def _unresolved_symbols(
    requested_symbols: list[object], available_symbols: set[str]
) -> list[str]:
    """Keep only symbol requirements not already supported by evidence.

    Matches via _matches_symbol (bare-vs-qualified, e.g. "dispatch_request"
    vs "Flask.dispatch_request") rather than exact string membership. The
    reasoning model routinely names a requirement by its bare method name
    even when the gathered chunk is stored under its qualified name -
    with exact matching, "dispatch_request" was never recognized as
    already satisfied by an already-gathered "Flask.dispatch_request"
    chunk, so it stayed "unresolved" forever and the model kept
    retrieving/wandering for something it already had.
    """
    return [
        symbol.strip()
        for symbol in requested_symbols
        if isinstance(symbol, str)
        and symbol.strip()
        and not any(_matches_symbol(symbol.strip(), s) for s in available_symbols)
    ]


def _requested_gathered_symbol(query: str, available_symbols: set[str]) -> str | None:
    """Recognize the agent's exact-symbol follow-up without parsing prose
    queries. Returns the matching gathered (canonical) name via
    _matches_symbol, not just the query string, for the same
    bare-vs-qualified reason as _unresolved_symbols above."""
    symbol = query.removeprefix("function ").strip()
    return next((s for s in available_symbols if _matches_symbol(symbol, s)), None)


def _retrieve_target_symbol(query: str, available_symbols: set[str]) -> str | None:
    """Recognize when a retrieve query - regardless of which code path
    produced it ('function X', 'subclass of X', or a bare exact name) -
    is chasing a symbol that's already gathered. Generic by construction:
    matches on whatever's already in evidence, not on any specific name.
    Returns the matching gathered (canonical) name via _matches_symbol so
    a bare query like "dispatch_request" is recognized against an
    already-gathered "Flask.dispatch_request", the same as above."""
    candidate = query.strip()
    for prefix in ("function ", "subclass of "):
        if candidate.startswith(prefix):
            candidate = candidate.removeprefix(prefix).strip()
            break
    return next((s for s in available_symbols if _matches_symbol(candidate, s)), None)


def _last_expansion_found_nothing(trace: list[str]) -> bool:
    """True if the most recent action was a call-graph expansion that
    added zero new chunks. That means THIS symbol's expansion was a
    dead end - it does not mean the graph itself is exhausted, so it's
    a signal to try a different connected symbol, not to fall back to
    semantic search."""
    if not trace:
        return False
    match = re.search(
        r"call-graph expansion from '[^']+' found (\d+) new chunks",
        trace[-1],
    )
    return bool(match) and match.group(1) == "0"


def _expansion_unlikely_to_add_evidence(
    chunk: dict, known_symbols: set[str]
) -> bool:
    """Predict whether expanding this symbol can surface new evidence,
    before paying for a full retrieve/expand round trip.

    expand_by_graph walks a symbol's `calls`/`called_by` names and returns
    whichever of those aren't already visited. If every one of those names
    already matches a symbol we've already gathered or already expanded,
    the walk can only rediscover what's already known - it is structurally
    guaranteed to add 0 new chunks. This is exactly what happened wastefully
    expanding Flask.dispatch_request after an earlier expansion had already
    pulled in its neighbors (full_dispatch_request, finalize_request, etc.).

    Purely a name-overlap prediction over data nodes.py already has
    (gathered chunk names + the trace's already-expanded set) - it does not
    change how expand_by_graph itself walks the graph, and it can't be
    100% precise (a name can resolve to a different, not-yet-seen chunk
    elsewhere in the repo - see call_graph.py's own matching caveat), so
    it's used to order preference, never to hard-block an expansion.
    """
    neighbor_names = set(chunk.get("calls", [])) | set(chunk.get("called_by", []))
    if not neighbor_names:
        return True
    return neighbor_names.issubset(known_symbols)


def _connected_unexpanded_symbols(
    chunks: list[dict], trace: list[str]
) -> list[dict]:
    """Rank graph entry points that have not yet been explored this run.

    Concrete (non-abstract) candidates are preferred over abstract ones.
    An abstract method (@abstractmethod, or a body that's just `raise
    NotImplementedError`) has no runtime behavior of its own to expand
    into - picking it as "the next symbol to try" just reproduces the
    same zero-result dead end this function exists to route around (e.g.
    an interface method like Flask's SessionInterface.open_session, which
    has called_by edges but nothing to walk from). Abstract candidates are
    only returned when nothing concrete and connected is left, so callers
    still get *something* to try rather than stalling out.

    Within each tier, candidates predicted to actually add new evidence
    (see _expansion_unlikely_to_add_evidence) are preferred over ones
    whose neighbors are all already known - so a doomed-to-be-empty
    expansion is only ever chosen when it's genuinely the only option
    left, not as the default pick.
    """
    expanded = _expanded_symbols(trace)
    known_symbols = expanded | {c.get("name") for c in chunks if c.get("name")}

    def is_connected(chunk: dict) -> bool:
        return (
            chunk.get("name") not in expanded
            and chunk.get("type") in {"function", "method"}
            and (chunk.get("calls") or chunk.get("called_by"))
        )

    def ranked(candidates: list[dict]) -> list[dict]:
        return sorted(
            candidates,
            key=lambda chunk: chunk.get("rerank_score", chunk.get("score", 0.0)),
            reverse=True,
        )

    def promising_first(candidates: list[dict]) -> list[dict]:
        promising = [
            c for c in candidates
            if not _expansion_unlikely_to_add_evidence(c, known_symbols)
        ]
        return ranked(promising) if promising else ranked(candidates)

    concrete = [
        chunk for chunk in chunks
        if is_connected(chunk) and not _is_abstract_chunk(chunk)
    ]
    if concrete:
        return promising_first(concrete)

    connected = [chunk for chunk in chunks if is_connected(chunk)]
    return promising_first(connected)


def _retrieval_queries(trace: list[str]) -> list[str]:
    """Extract prior retrieval intents without changing the state schema."""
    queries = []
    for line in trace:
        match = re.search(r"retrieved \d+ new chunks for '([^']+)'", line)
        if match:
            queries.append(match.group(1))
    return queries


def _topic_terms(query: str) -> set[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.lower())
    return {
        _TOPIC_ALIASES.get(word, word)
        for word in words
        if word not in _TOPIC_STOP_WORDS
    }


def _is_repeated_retrieval_topic(query: str, trace: list[str]) -> bool:
    """Catch paraphrased retrieval loops, not only byte-for-byte duplicates."""
    current = _topic_terms(query)
    if not current:
        return False
    for previous_query in _retrieval_queries(trace):
        previous = _topic_terms(previous_query)
        if not previous:
            continue
        overlap = len(current & previous) / min(len(current), len(previous))
        if overlap >= 0.5:
            return True
    return False


def _preferred_graph_symbol(
    state: AgentState, proposed_action: str, proposed_query: str
) -> str | None:
    """Deterministically prefer structural investigation for flow questions."""
    # Do not override a grounded completion merely because another graph edge
    # exists.  This policy is a safety net while evidence is incomplete, not
    # a requirement to exhaust every connected symbol in the repository.
    if (
        state["intent"] != "understand_flow"
        or not _execution_evidence_incomplete(state["gathered_chunks"])
    ):
        return None

    repeats_topic = proposed_action == "retrieve" and _is_repeated_retrieval_topic(
        proposed_query, state["reasoning_trace"]
    )

    # A failed semantic search never proves the structural investigation is
    # complete. After the first search, graph traversal is more informative
    # than another search over the same execution-flow topic. A zero-gain
    # graph expansion is the structural equivalent of an exhausted search -
    # it means THAT symbol was a dead end, not that traversal is done.
    if (
        state.get("retrieval_exhausted")
        or proposed_action == "answer"
        or repeats_topic
        or _last_expansion_found_nothing(state["reasoning_trace"])
    ):
        # select_execution_path_symbol() is the same execution-chain-aware
        # walker used everywhere else in this file for a structural
        # follow-up: it prefers a DIRECT callee of wherever the walk
        # currently is (see its docstring), and only falls back to a
        # relevance-first ranking (question/missing-requirement overlap,
        # then connectivity, then implementation, then unexplored status)
        # when there's no established position yet or no direct callee
        # matched. This function used to pick _connected_unexpanded_symbols(...)[0]
        # instead - a flat rerank_score sort with no notion of "current
        # position" at all - which let a merely topically-similar but
        # graph-unrelated symbol (e.g. make_response, url_for, get)
        # outscore the actual next hop in the execution chain (e.g.
        # match_request, dispatch_request) on embedding similarity alone,
        # since both look equally "connected." Delegating to the same
        # chain-aware walker used elsewhere fixes that without adding a
        # second, different selection policy.
        return select_execution_path_symbol(
            state["gathered_chunks"],
            state["reasoning_trace"],
            question=state["question"],
            missing_requirements=state.get("outstanding_requirements", []),
        )
    return None


def _is_runtime_implementation(chunk: dict) -> bool:
    """Distinguish executable implementation from boundary/API declarations."""
    if chunk.get("type") not in {"function", "method"}:
        return False

    source = chunk.get("raw_source") or chunk.get("text") or ""
    if _is_abstract_chunk(chunk):
        return False

    # A forwarding function is useful graph evidence, but not the runtime or
    # business implementation that completes a flow investigation.
    try:
        function = next(
            node for node in ast.parse(textwrap.dedent(source)).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except (SyntaxError, StopIteration):
        return False

    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) == 1:
        statement = body[0]
        value = statement.value if isinstance(statement, (ast.Return, ast.Expr)) else None
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            return False

    return True


def _has_runtime_implementation(chunks: list[dict]) -> bool:
    return any(_is_runtime_implementation(chunk) for chunk in chunks)


def _is_abstract_chunk(chunk: dict) -> bool:
    """Identify an abstract implementation from code, not its docstring."""
    source = chunk.get("raw_source") or chunk.get("text") or ""
    try:
        function = next(
            node for node in ast.parse(textwrap.dedent(source)).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except (SyntaxError, StopIteration):
        return False

    def is_abstractmethod(decorator: ast.expr) -> bool:
        return (
            isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
        ) or (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "abstractmethod"
        )

    if any(is_abstractmethod(decorator) for decorator in function.decorator_list):
        return True

    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if not isinstance(statement, ast.Raise):
        return False

    exception = statement.exc
    if isinstance(exception, ast.Call):
        exception = exception.func
    return (
        isinstance(exception, ast.Name)
        and exception.id == "NotImplementedError"
    )


def _execution_evidence_incomplete(chunks: list[dict]) -> bool:
    """Keep flow investigations open until graph and runtime evidence agree."""
    return needs_more_context(
        chunks,
        min_connected_symbols=3,
    ) or not _has_runtime_implementation(chunks)


def _stem(word: str) -> str:
    """Coarse, dependency-free suffix stripping so 'loaded' matches
    'load' and 'validates' matches 'validate'. Not a real stemmer - just
    enough to stop ordinary English word-form mismatches from defeating
    the lexical-coverage check below with false negatives."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _haystack_terms(chunk: dict) -> set[str]:
    """Stemmed terms found in a chunk's name/file/source, split on
    identifier boundaries AND on underscores - so a snake_case symbol
    like `load_config` contributes both "load" and "config" as
    independently matchable terms, not just the whole compound name."""
    text = " ".join([
        chunk.get("name", "") or "",
        chunk.get("file", "") or "",
        chunk.get("raw_source") or chunk.get("text") or "",
    ])
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    terms = set()
    for word in words:
        terms.add(_stem(word))
        for part in word.split("_"):
            if part:
                terms.add(_stem(part))
    return terms


def _relevance_terms(question: str, missing_requirements: list[str] | None) -> set[str]:
    """Stemmed, stop-word-filtered terms drawn from the question AND any
    outstanding missing-requirement text, combined - the same lexical
    vocabulary _gathered_evidence_sufficient already uses for whole-
    question topic coverage (_topic_terms + _stem), just also folding in
    whatever the investigation has explicitly said it still needs (e.g.
    "request_context internals showing request, session, g creation and
    push/pop sequence"). That's frequently more specific than the
    original question and is exactly what should steer which of several
    equally-connected symbols to expand next.
    """
    terms = {_stem(term) for term in _topic_terms(question or "")}
    for requirement in missing_requirements or []:
        terms |= {_stem(term) for term in _topic_terms(requirement)}
    return terms


def _relevance_score(chunk: dict, query_terms: set[str]) -> float:
    """Fraction of query_terms (question + outstanding requirements) that
    actually appear in this candidate's own name/file/source
    (_haystack_terms - the same per-chunk term set _gathered_evidence_
    sufficient already builds for lexical coverage).

    Returns a real 0.0, not a small positive floor, when there's no
    overlap at all - see select_execution_path_symbol's use of this: a
    weakly related symbol needs to be genuinely outranked by anything
    with even partial overlap, not merely nudged down by a soft signal
    that graph connectivity or rerank score could still out-vote.
    """
    if not query_terms:
        return 0.0
    chunk_terms = _haystack_terms(chunk)
    if not chunk_terms:
        return 0.0
    overlap = query_terms & chunk_terms
    return len(overlap) / len(query_terms)


def _gathered_evidence_sufficient(
    question: str,
    chunks: list[dict],
    *,
    intent: str | None = None,
    reasoning_trace: list[str] | None = None,
) -> bool:
    """Deterministic, cheap (no LLM call) check for whether gathered
    evidence already covers the question well enough to answer now,
    instead of paying for another retrieve/expand_graph round trip plus a
    full reasoning call.

    Deliberately built as a refinement of _execution_evidence_incomplete,
    not a separate/looser bar: this only ever returns True in cases where
    that existing (already carefully-tuned) completeness check would also
    say evidence is complete, so it can't cause a premature answer that
    the existing gates wouldn't already have allowed - it can only skip
    the LLM call in cases the model would very likely have answered "yes,
    answer" anyway. The added lexical check on top guards against gathered
    evidence being structurally rich (well-connected, has runtime code)
    about a different part of the codebase than what was actually asked.

    Generic: no framework-specific names or terms anywhere - only the
    same structural signals (_has_runtime_implementation via
    _execution_evidence_incomplete) and question-derived term overlap
    (_topic_terms) already used elsewhere in this file.

    understand_flow gets one extra requirement on top of all the above.
    A chunk's `calls`/`called_by` are populated at ingest time - present
    whether or not the graph has ever actually been walked this run. So a
    SINGLE initial retrieval can already look "connected" (satisfying
    _execution_evidence_incomplete) and lexically on-topic using only
    chunks from the FIRST stage of a multi-step pipeline - reasoning_node
    calls this before the reasoning LLM ever runs, so on iteration 1
    there's no missing_information/outstanding-requirement history yet
    for the caller's own named-requirement gate to check against either.
    So for understand_flow specifically: (a) require that at least one
    expand_graph has actually run this investigation before early-stop is
    even considered, and (b) once one has, defer to the existing graph-
    ranking logic (_connected_unexpanded_symbols) - if it still finds a
    promising, unexpanded, connected symbol in what's been gathered,
    there's more of the pipeline's stages left to walk (e.g. a routing
    match step whose result leads to a dispatch step that hasn't been
    expanded yet) and this returns False rather than declaring the
    investigation done. Generic BFS-frontier reasoning, not a fixed
    per-question stage count or any framework-specific symbol name.
    """
    if not chunks or _execution_evidence_incomplete(chunks):
        return False

    trace = reasoning_trace or []

    if intent == "understand_flow":
        if not _expanded_symbols(trace):
            return False

        remaining = _connected_unexpanded_symbols(chunks, trace)
        known_symbols = _expanded_symbols(trace) | {
            c.get("name") for c in chunks if c.get("name")
        }
        promising = [
            c for c in remaining
            if not _expansion_unlikely_to_add_evidence(c, known_symbols)
        ]
        if promising:
            return False

    topic = {_stem(term) for term in _topic_terms(question)}
    if not topic:
        # Nothing distinctive to check lexical coverage against - the
        # structural completeness check above is all there is to go on.
        return True

    haystack_terms: set[str] = set()
    for chunk in chunks:
        haystack_terms |= _haystack_terms(chunk)

    covered = topic & haystack_terms
    # A flat 60% ratio means any two-term topic needs BOTH terms covered
    # to pass at all - too strict for naturally short, broad questions
    # even after stopword filtering removes the obvious meta-words (a
    # term can still fail to literally appear in source for reasons that
    # have nothing to do with evidence quality: synonyms, abbreviations,
    # a term split differently than _haystack_terms's tokenizer expects).
    # 50% with at least one term covered is still a genuine relevance
    # requirement - 0% coverage (the off-topic case) is always rejected
    # regardless of this threshold - just a less brittle one for terse
    # questions.
    return bool(covered) and len(covered) / len(topic) >= 0.5


def _summarize_context(chunks: list[dict]) -> str:
    """Render gathered evidence for the reasoning model.

    Deliberately does NOT cap to a fixed chunk count here. reasoning_node's
    own "Keep shrinking until the ENTIRE prompt fits" loop is what enforces
    the 2500-token budget, by trimming the tail of `prompt_chunks` and
    re-calling this function. A hardcoded slice here would silently hide
    already-gathered evidence (e.g. a chunk that's genuinely relevant but
    ranks 7th) from the model regardless of whether it would actually fit -
    which is exactly what causes the model to "not recognize" evidence it
    already has and re-retrieve the same topic.
    """
    if not chunks:
        return "(nothing gathered yet)"
    lines = []

    for c in chunks:
      line = (
        f"- {c.get('name','?')} "
        f"({c.get('type','?')}) "
        f"in {c.get('file','?')}:{c.get('line_start','?')}"
      )
      calls = c.get("calls", [])
      if calls:
        line += f"\n  Calls: {', '.join(calls[:5])}"

      called_by = c.get("called_by", [])
      if called_by:
        line += f"\n  Called by: {', '.join(called_by[:5])}"
      source = c.get("raw_source") or c.get("text") or ""
      if source:
        # Was source[:800]. The reasoning prompt has a fixed ~2500-token
        # budget (reasoning_node trims whole chunks off the tail until it
        # fits) - a bulky per-chunk source snippet crowds that budget so
        # hard that trimming drops entire chunks, taking their Calls/
        # Called by lines with them. Those lines are exactly the evidence
        # "expand_graph" decisions run on and cost only a few tokens each,
        # so a shorter source snippet leaves more chunks - and therefore
        # more graph edges - actually surviving into the prompt.
        line += f"\n  Source:\n{source[:400]}"
      lines.append(line)

      
    return "\n".join(lines)




def needs_more_context(
    chunks: list[dict],
    *,
    min_symbols: int = 0,
    min_connected_symbols: int = 0,
    require_runtime_code: bool = False,
) -> bool:
    """Apply the existing execution-completeness thresholds consistently."""
    connected_symbols = sum(
        1 for chunk in chunks if chunk.get("calls") or chunk.get("called_by")
    )
    has_runtime_code = any(
        len((chunk.get("text") or "").splitlines()) > 10
        for chunk in chunks
    )
    return (
        len(chunks) < min_symbols
        or connected_symbols < min_connected_symbols
        or (require_runtime_code and not has_runtime_code)
    )


def _token_usage(response, input_text: str, output_text: str) -> tuple[int, int]:
    """Use Groq's usage when available, with the normal trace estimate as fallback."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    return (
        input_tokens if isinstance(input_tokens, int) else count_tokens(input_text),
        output_tokens if isinstance(output_tokens, int) else count_tokens(output_text),
    )


def _response_text(resp) -> str:
    """Pull the model's generated text out of a Groq/OpenAI-SDK chat
    completion response object.

    Both SDKs expose an OpenAI-style `message.content` string in the
    normal case, and that's all `_extract_json_object` used to be given.
    But `_chat_with_fallback` can route this same call to an OpenRouter
    reasoning model (e.g. after a Groq rate limit or json_validate_failed
    rejection - see `_chat_with_fallback` above), and those responses
    routinely come back with `message.content` empty/None while the
    actual generation sits on a separate attribute the SDK still exposes
    - `reasoning` or `reasoning_content`, depending on the provider/model
    - because the provider treats "reasoning" and "final content" as
    distinct fields. Handing an empty `content` straight to
    `_extract_json_object` is exactly the `Expecting value: line 1
    column 1` failure this exists to avoid: there was never any JSON in
    the field being read, not a parsing bug.

    Only used to pick which attribute to treat as the raw response text -
    it does not touch how that text is subsequently cleaned or parsed.
    """
    message = resp.choices[0].message

    content = getattr(message, "content", None)
    if content and content.strip():
        return content.strip()

    for fallback_attr in ("reasoning", "reasoning_content"):
        fallback = getattr(message, fallback_attr, None)
        if fallback and fallback.strip():
            return fallback.strip()

    return ""


def _matches_symbol(reference: str, symbol: str) -> bool:
    return (
        reference == symbol
        or reference == symbol.rsplit(".", 1)[-1]
        or reference.rsplit(".", 1)[-1] == symbol.rsplit(".", 1)[-1]
    )


def _symbol_has_gathered_implementation(name: str, chunks: list[dict]) -> bool:
    """A named requirement is only genuinely resolved once we have that
    symbol's own real content - not just its name present somewhere among
    gathered evidence.

    Plain name matching (`any(_matches_symbol(name, s) for s in
    available_symbols)`) treats a requirement as satisfied the moment ANY
    gathered chunk happens to be named that, even if that chunk is an
    abstract stub, a bare declaration, or otherwise has nothing but a
    signature - the exact "relevant symbol is already in sources but its
    implementation/details aren't gathered" gap: the name is technically
    "there", but there's nothing to actually explain or continue the flow
    from, so continuing to retrieve/expand it is still the right call,
    not treating it as done. Reuses the same AST-based
    _is_abstract_chunk distinction already used elsewhere in this file
    for the same purpose (not a new, separate heuristic).
    """
    for chunk in chunks:
        if not _matches_symbol(name, chunk.get("name", "")):
            continue
        source = (chunk.get("raw_source") or chunk.get("text") or "").strip()
        if not source:
            continue
        if chunk.get("type") not in {"function", "method"}:
            # A class/module-level match (e.g. a Rule or MapAdapter type
            # itself, not one of its methods) still counts as long as we
            # actually have its content - not everything named in a flow
            # is itself a callable with a body to be abstract or not.
            return True
        if not _is_abstract_chunk(chunk):
            return True
    return False


def _get_execution_chain(reasoning_trace: list[str]) -> list[str]:
    """Recover the sequence of symbols the deterministic policy has already
    walked, by reading its own trace entries back."""
    chain = []
    for step in reasoning_trace:
        if "Policy selected structural follow-up from '" in step:
            symbol = step.split("'")[1]
            chain.append(symbol)
    return chain


def select_execution_path_symbol(
    chunks: list[dict],
    trace: list[str],
    *,
    runtime_only: bool = False,
    question: str = "",
    missing_requirements: list[str] | None = None,
) -> str | None:
    """Choose the next runtime step from graph structure, not semantics.

    Module-level (not a reasoning_node closure) so the control-flow logic
    can be exercised directly in tests without an LLM call. Concrete
    (non-abstract) candidates are always preferred over abstract ones -
    falling through to abstract only when nothing concrete and connected
    remains - so a dead-end interface method (e.g. an abstract
    open_session()) is never deterministically chosen over a real
    implementation that's still available to expand.

    Ranking, in priority order, whenever there are multiple viable
    candidates and no single direct-callee hop already decides it (see
    below): (1) relevance to the question/outstanding missing
    requirements, (2) graph connectivity, (3) implementation availability,
    (4) unexplored status - a symbol not yet visited as the walk's current
    position. Weakly related candidates (zero term overlap with
    question/missing_requirements) are demoted as a group behind anything
    with real overlap, not merely nudged down by a soft score - see the
    "relevant first" partition below.

    This ranking is applied at two points. (1) Full ranked_choice() below,
    where there's genuinely no single correct next hop to pick from
    structure alone: no established execution position yet, or the
    direct-callee walk finds no match at all. (2) As a narrow tiebreak
    WITHIN the direct-callee walk itself: when the current position's
    `calls` name MORE THAN ONE real, eligible callee, the first such
    callee (in declaration order) with positive relevance is preferred
    over an earlier-declared but irrelevant one - e.g. create_app calling
    both register_user/send_registration_email and a genuinely relevant
    step, in that order, no longer wanders into the former just because
    they were declared first. A single unambiguous edge is always taken
    regardless of relevance - this never introduces a candidate that
    ISN'T a real direct callee - and when none of several real edges have
    any relevance-term overlap, the original declaration-order choice is
    preserved exactly as before. An earlier fix here specifically
    replaced a flat semantic-similarity sort with chain-aware graph
    walking because similarity alone let a topically-close but graph-
    unrelated symbol (make_response, url_for, get) outrank the actual
    next hop (match_request, dispatch_request) - see
    _preferred_graph_symbol's comment; this tiebreak only ever chooses
    among ACTUAL direct callees, so it doesn't reintroduce that failure
    mode.
    """
    expanded = _expanded_symbols(trace)

    def eligible(chunk: dict, *, allow_abstract: bool) -> bool:
        return (
            chunk.get("name") not in expanded
            # expand_by_graph() explicitly refuses to expand FROM a class
            # chunk (`if chunk.get("type") == "class": continue`) — a class
            # declaration has no runtime call of its own to walk. Picking
            # one here as "the next symbol to expand" is a guaranteed
            # zero-result dead end, indistinguishable from the interface-
            # method dead end _connected_unexpanded_symbols already guards
            # against below. Match that same {"function", "method"} filter
            # so this selector never proposes something expansion can't act on.
            and chunk.get("type") in {"function", "method"}
            and (chunk.get("calls") or chunk.get("called_by"))
            and (not runtime_only or _is_runtime_implementation(chunk))
            and (allow_abstract or not _is_abstract_chunk(chunk))
        )

    candidates = [c for c in chunks if eligible(c, allow_abstract=False)]
    if not candidates:
        candidates = [c for c in chunks if eligible(c, allow_abstract=True)]
    if not candidates:
        return None

    # Prefer a symbol actually predicted to surface new evidence over one
    # whose calls/called_by are all already known (see
    # _expansion_unlikely_to_add_evidence - the Flask.dispatch_request
    # case, where every neighbor was already gathered from an earlier
    # expansion). Falls back to the full candidate set when every option
    # looks doomed, since the prediction isn't certain and shouldn't block
    # progress entirely.
    known_symbols = expanded | {c.get("name") for c in chunks if c.get("name")}
    promising = [
        c for c in candidates
        if not _expansion_unlikely_to_add_evidence(c, known_symbols)
    ]
    if promising:
        candidates = promising

    query_terms = _relevance_terms(question, missing_requirements)

    def ranked_choice(pool: list[dict]) -> str:
        # Penalize weakly related symbols HERE ONLY: partition this
        # specific pool into "shares at least one term with the
        # question/missing requirements" vs. not, and prefer the former as
        # a GROUP - so a weakly related candidate can only win by having
        # no relevant alternative at all, never by merely outscoring one
        # on connectivity or rerank score. Deliberately scoped to this
        # inner function rather than reassigning the outer `candidates` -
        # the direct-callee walk below reads `candidates` directly and
        # must NOT be filtered by relevance (see the docstring above: a
        # real graph edge from a known position outranks any heuristic).
        ranking_pool = pool
        if query_terms:
            relevant = [c for c in pool if _relevance_score(c, query_terms) > 0]
            if relevant:
                ranking_pool = relevant

        visited = set(_get_execution_chain(trace))
        return max(
            ranking_pool,
            key=lambda c: (
                _relevance_score(c, query_terms),
                len(c.get("calls", [])) + len(c.get("called_by", [])),
                int(_is_runtime_implementation(c)),
                int(c.get("name") not in visited),
                c.get("rerank_score", c.get("score", 0.0)),
            ),
        )["name"]

    execution_chain = _get_execution_chain(trace)
    current_symbol = execution_chain[-1] if execution_chain else None

    current_chunks = [
        chunk for chunk in chunks
        if current_symbol and _matches_symbol(chunk.get("name", ""), current_symbol)
    ]

    if not current_chunks:
        return ranked_choice(candidates)

    current = max(
        current_chunks,
        key=lambda c: c.get("rerank_score", c.get("score", 0.0)),
    )
    direct_calls = current.get("calls", [])

    # Walk direct callees of the current execution symbol in declaration
    # order (the order calls actually appear in `current`'s source) by
    # default, not by rerank/semantic score - e.g. full_dispatch_request
    # calls both match_request and dispatch_request, in that order, and a
    # higher rerank score for dispatch_request must not visit it first.
    # This is the same loop that used to only run as a fallback when no
    # direct callee was found at all (a redundant second lookup of
    # `current` under a different name, `current_chunk`) - it's the only
    # place that should ever choose among direct callees.
    #
    # Among MULTIPLE real direct callees, relevance to the question/
    # outstanding missing requirements breaks the tie: create_app calling
    # both register_user/send_registration_email and a step actually
    # relevant to the question, in that order, should prefer the relevant
    # one rather than wandering into whichever was declared first. This
    # only ever chooses among candidates that ARE real direct callees -
    # it never reaches for something unrelated the way a flat semantic-
    # similarity sort would - and declaration order is preserved exactly
    # as before whenever there's only one match, or when none of several
    # matches have any relevance-term overlap at all.
    matched_direct_callees = []
    for call in direct_calls:
        next_chunk = next(
            (c for c in candidates if _matches_symbol(call, c.get("name", ""))),
            None,
        )
        if next_chunk:
            matched_direct_callees.append(next_chunk)

    if matched_direct_callees:
        if query_terms:
            relevant_direct_callees = [
                c for c in matched_direct_callees
                if _relevance_score(c, query_terms) > 0
            ]
            if relevant_direct_callees:
                return relevant_direct_callees[0]["name"]
        return matched_direct_callees[0]["name"]

    # Fall back to the best remaining runtime implementation.
    runtime = [c for c in candidates if _is_runtime_implementation(c)]
    if runtime:
        candidates = runtime

    return ranked_choice(candidates)


def _extract_json_object(raw: str) -> str:
    """Pull a JSON object out of a raw reasoning response defensively.

    `response_format={"type": "json_object"}` is only reliably honored by
    Groq. The OpenRouter fallback path can route to a model that behaves
    differently in ways that all surfaced as the same opaque
    `Expecting value: line 1 column 1 (char 0)` from json.loads():
      - wraps the JSON in prose, e.g. "Here is the JSON:\n```json\n{...}\n```" -
        the old regex only stripped a fence anchored to the exact start/end
        of the string (`^...` / `...$`), so any leading sentence defeated
        it entirely and json.loads() got prose it can't parse at all;
      - fences without a "json" language tag;
      - for reasoning-style models that put chain-of-thought in a separate
        response field, returns an empty `content` altogether - passed
        straight to json.loads(""), which is exactly the "Expecting
        value" error.

    This does not change what a clean {...} response looks like - it only
    recovers the object from noise around it. A response with no JSON
    object anywhere still reaches json.loads() unchanged and still raises
    the same JSONDecodeError as before, so no failure is silently hidden.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty reasoning response content")

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence_match:
        return fence_match.group(1)

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair for JSON cut off mid-generation by
    max_completion_tokens - the exact cause of "Unterminated string..."
    and similar errors from json.loads(): the model was still inside a
    string literal, array, or object when generation stopped, so the
    text is a valid PREFIX of a JSON object but not a complete one.

    Walks the text once to find any string literal left open and any
    brackets/braces left unclosed, then closes them in the order they
    were opened. thought/action/action_input are emitted first by the
    prompt's schema, so they're almost always complete before a cutoff -
    this recovers them (and whatever prefix of missing_information made
    it out) instead of losing the whole response to one truncated tail
    field.

    Pure string manipulation, no network call - this is not a retry and
    adds no latency; it either fixes the local string or it doesn't, in
    microseconds.
    """
    text = text.rstrip()
    if not text:
        return text

    in_string = False
    escape = False
    stack: list[str] = []

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    repaired = text
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


async def reasoning_node(state: AgentState) -> dict:
    # --------------------------------------------------
    # Nothing gathered yet
    # --------------------------------------------------
    if not state["gathered_chunks"]:
        return {
            "next_action": "retrieve",
            "action_query": state["question"],
            "reasoning_trace": state["reasoning_trace"]
            + ["No chunks gathered yet. Starting retrieval."],
        }

    # --------------------------------------------------
    # Rerank gathered chunks
    # --------------------------------------------------
    reranked = await rerank(
        state["question"],
        state["gathered_chunks"],
        top_n=min(30, len(state["gathered_chunks"])),
    )
    available_chunks = [
        c for c in state["gathered_chunks"]
        if c.get("name")
    ]

    available_symbols = {
        c["name"]
        for c in available_chunks
    }

    # --------------------------------------------------
    # Early-stop: skip the reasoning LLM call entirely once evidence
    # already covers the question (deterministic, no extra LLM call -
    # see _gathered_evidence_sufficient's docstring for why this can't
    # answer more eagerly than the existing completeness gates already
    # allow).
    #
    # For understand_flow specifically, also require that nothing the
    # model has EVER explicitly named as still-needed (missing_information
    # from an earlier reasoning call) remains unmatched by gathered
    # evidence. _gathered_evidence_sufficient only ever looks at generic
    # structural/lexical signals on THIS call's chunks - it has no memory
    # of a specific requirement named a turn or two ago, so a structurally
    # rich but unrelated detour (e.g. expanding a well-connected symbol
    # that has nothing to do with what was actually asked) could otherwise
    # satisfy it and let the loop stop before the named requirement was
    # ever evidenced. Note this alone can't stop a PREMATURE first-
    # iteration answer, though: on iteration 1 no reasoning call has run
    # yet, so outstanding_requirements is trivially empty - that's what
    # _gathered_evidence_sufficient's own understand_flow requirement
    # (passed intent/reasoning_trace below) exists to close.
    # --------------------------------------------------
    still_outstanding = [
        item for item in state.get("outstanding_requirements", [])
        if not _symbol_has_gathered_implementation(item, available_chunks)
    ]

    if (
        (state["intent"] != "understand_flow" or not still_outstanding)
        and _gathered_evidence_sufficient(
            state["question"],
            reranked,
            intent=state["intent"],
            reasoning_trace=state["reasoning_trace"],
        )
    ):
        logger.info(
            "Gathered evidence already covers the question; answering "
            "without a reasoning call."
        )
        return {
            "next_action": "answer",
            "action_query": "",
            "gathered_chunks": reranked,
            "reasoning_trace": state["reasoning_trace"]
            + [
                "Thought: gathered evidence already sufficiently covers "
                "the question -> Action: answer"
            ],
            "agent_input_tokens": state.get("agent_input_tokens", 0),
            "agent_output_tokens": state.get("agent_output_tokens", 0),
            "outstanding_requirements": still_outstanding,
        }

    # `reranked` is what gets persisted back into state["gathered_chunks"]
    # below, and it is also what the runtime-evidence gate further down
    # uses to pick the next symbol to expand. Everything from here to the
    # prompt build operates on an independent display copy instead, so
    # trimming it for the prompt can never leak into evidence propagation.
    prompt_chunks = list(reranked)

    # Cap the STARTING candidate count, not just the eventual token
    # budget. `reranked` can carry up to 30 chunks, and without this every
    # iteration of a long-running investigation paid to build (and
    # summarize) an increasingly large first-draft prompt before the
    # token-budget while-loop below ever got a chance to trim it -
    # `_summarize_context` isn't called on `prompt_chunks` until after
    # this cap, specifically so the cap actually bounds the FIRST summary
    # built, not just later shrink passes. `reranked` is already sorted
    # most-relevant-first (reranker.py), so taking its top N keeps
    # prompt-build cost roughly constant across a run instead of growing
    # with it. Display-only - `reranked` (the full list) is still what's
    # returned as gathered_chunks below, unaffected by this cap.
    MAX_PROMPT_CHUNKS = 12
    if len(prompt_chunks) > MAX_PROMPT_CHUNKS:
        prompt_chunks = prompt_chunks[:MAX_PROMPT_CHUNKS]

    summary = _summarize_context(prompt_chunks)
    # Cap each line too, not just the window - a single verbose entry
    # (e.g. a parse-failure fallback thought) shouldn't be able to
    # dominate what's meant to be a compact recent-history summary.
    trace = "\n".join(
        line if len(line) <= 300 else line[:300] + "…"
        for line in state["reasoning_trace"][-4:]
    ) or "(none yet)"
    retrieval_status = (
        "The latest retrieval added little or no new evidence."
        if state.get("retrieval_exhausted")
        else "The latest retrieval produced useful new evidence."
    )

    # --------------------------------------------------
    # Keep shrinking until the ENTIRE prompt fits
    # --------------------------------------------------
    prompt = REASONING_PROMPT.format(
        question=state["question"],
        num_chunks=len(prompt_chunks),
        context_summary=summary,
        trace=trace,
        retrieval_status=retrieval_status,
    )

    while count_tokens(prompt) > 2500 and len(prompt_chunks) > 1:
        prompt_chunks = prompt_chunks[:-1]
        summary = _summarize_context(prompt_chunks)

        prompt = REASONING_PROMPT.format(
            question=state["question"],
            num_chunks=len(prompt_chunks),
            context_summary=summary,
            trace=trace,
            retrieval_status=retrieval_status,
        )

    # --------------------------------------------------
    # Ask the reasoning model
    # --------------------------------------------------
    resp = await _chat_with_fallback(
        # Reasoning output is one small JSON object (thought + action +
        # action_input + a short list) - it never legitimately needs
        # anywhere near 2048 tokens. Capping it tighter bounds the worst
        # case latency of every reasoning call (a confused/rambling
        # generation used to be free to run for up to 2048 tokens before
        # stopping) without giving the model less room than the schema
        # actually needs.
        max_completion_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": _REASONING_SYSTEM_MESSAGE,
            },

            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = _response_text(resp)
    input_tokens, output_tokens = _token_usage(
        resp,
        _REASONING_SYSTEM_MESSAGE + prompt,
        raw,
    )

    try:
        cleaned = _extract_json_object(raw)
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError(
                f"reasoning response was not a JSON object (got {type(parsed).__name__})"
            )

        if parsed.get("action") not in {"retrieve", "expand_graph", "answer"}:
            raise ValueError(f"invalid or missing action: {parsed.get('action')!r}")

        # Defensive defaults for the other three keys: a JSON object that
        # validated on "action" but is missing (or has the wrong type for)
        # "thought"/"action_input"/"missing_information" must not crash
        # further down with an uncaught KeyError/TypeError - every
        # downstream branch already assumes these are present with these
        # types, exactly as a well-formed response always provided them.
        if not isinstance(parsed.get("thought"), str):
            parsed["thought"] = str(parsed.get("thought", ""))
        if not isinstance(parsed.get("action_input"), str):
            parsed["action_input"] = str(parsed.get("action_input") or "")
        if not isinstance(parsed.get("missing_information"), list):
            parsed["missing_information"] = []

        missing_information = parsed["missing_information"]
        BAD_RETRIEVAL_QUERIES = {
            "missing function/class name",
            "missing implementation",
            "more information",
            "implementation",
            "details",
            "unknown",
        }

        # --------------------------------------------------
        # Cumulative named-requirement tracking (understand_flow)
        # --------------------------------------------------
        # Prune anything already matched by gathered evidence, then add any
        # new item this call's missing_information names that isn't already
        # matched or already tracked. This is the run's full history of
        # "the model said we need X" - not just this one call's list - so a
        # requirement named a turn or two ago (e.g. right before a graph
        # expansion dead end) survives even if a later call's own
        # missing_information no longer restates it.
        outstanding = [
            item for item in state.get("outstanding_requirements", [])
            if not _symbol_has_gathered_implementation(item, available_chunks)
        ]
        for item in missing_information:
            if (
                item
                and item.lower() not in BAD_RETRIEVAL_QUERIES
                and not _symbol_has_gathered_implementation(item, available_chunks)
                and item not in outstanding
            ):
                outstanding.append(item)

        if (
            parsed["action"] == "retrieve"
            and parsed.get("action_input", "").strip().lower() in BAD_RETRIEVAL_QUERIES
        ):
            logger.info("Ignoring generic retrieval query from reasoning model.")
            parsed["action"] = "answer"
            parsed["action_input"] = ""



        if parsed["action"] == "retrieve":
            unresolved = _unresolved_symbols(
                missing_information, available_symbols
            )
            gathered_symbol = _requested_gathered_symbol(
                parsed.get("action_input", ""), available_symbols
            )

            if unresolved:
                parsed["action_input"] = unresolved[0]
            elif missing_information or gathered_symbol:
                # The stated requirement is already gathered. Let the normal
                # evidence gate choose a different missing step, rather than
                # issuing the same exact-symbol lookup again.
                parsed["action"] = "answer"
                parsed["action_input"] = ""

            

        if state["gathered_chunks"]:
            logger.debug("Gathered chunk fields: %s", state["gathered_chunks"][0].keys())
        if parsed.get("action") not in {
            "retrieve",
            "expand_graph",
            "answer",
        }:
            raise ValueError(
                f"invalid action: {parsed.get('action')}"
            )

        # Keep the model's semantic judgment, but deterministically route
        # structural follow-ups through the graph once an entry point exists.
        graph_symbol = _preferred_graph_symbol(
            state,
            parsed["action"],
            parsed.get("action_input", ""),
        )
        if graph_symbol:
            logger.info(
                "Policy prefers graph expansion from '%s' over %s.",
                graph_symbol,
                parsed["action"],
            )
            parsed["thought"] = (
                f"{parsed.get('thought', '')} "
                f"Policy selected structural follow-up from '{graph_symbol}'."
            ).strip()

        FLOW_KEYWORDS = {
            "flow",
            "how",
            "works",
            "implemented",
            "implementation",
            "where",
            "participate",
            "calls",
            "execution",
        }




        if graph_symbol:
            parsed["action"] = "expand_graph"
            parsed["action_input"] = graph_symbol

        # --------------------------------------------------
        # Validate expand_graph requests
        # --------------------------------------------------
        if parsed["action"] == "expand_graph":

        

            already_expanded = _expanded_symbols(state["reasoning_trace"])

            symbol = parsed.get("action_input", "")
            selected_chunk = next(
                (
                    # Bare-vs-qualified fix (see _unresolved_symbols above):
                    # the model can ask to expand "dispatch_request" while
                    # the gathered chunk is named "Flask.dispatch_request" -
                    # exact `== symbol` never found it, so it fell straight
                    # into the "not gathered yet, retrieve first" branch
                    # below for a symbol that was already sitting in
                    # available_chunks.
                    c
                    for c in available_chunks 
                    if _matches_symbol(symbol, c.get("name", ""))
                ),
                None
            )
            

            if selected_chunk and _is_abstract_chunk(selected_chunk):
                logger.info(
                    "Abstract symbol '%s' detected. Looking for concrete implementation.",
                    symbol,
                )

                parsed["action"] = "retrieve"
                parsed["action_input"] = f"subclass of {symbol}"

            # Symbol not gathered yet → retrieve first
            # (selected_chunk is None iff no gathered chunk matches `symbol`
            # via _matches_symbol above - kept as one source of truth
            # instead of a second, separately-exact `not in available_symbols`
            # check that could disagree with the fuzzy lookup above it.)
            if selected_chunk is None:

                logger.info(
                    "Symbol '%s' not gathered yet. Retrieving first.",
                    symbol,
                )

                parsed["action"] = "retrieve"
                parsed["action_input"] = f"function {symbol}"

            # Don't expand twice
            elif any(_matches_symbol(symbol, e) for e in already_expanded):
                unresolved = _unresolved_symbols(
                    missing_information, available_symbols
                )
                if unresolved:
                    parsed["action"] = "retrieve"
                    parsed["action_input"] = unresolved[0]
                else:
                    # This SYMBOL is exhausted - that is not the same as the
                    # INVESTIGATION being exhausted. Before concluding there
                    # is nothing left, follow another connected-but-
                    # unexpanded symbol using the calls/called_by evidence
                    # already gathered (generic graph traversal, not tied to
                    # any specific framework or symbol name).
                    next_symbol = select_execution_path_symbol(
                        reranked,
                        state["reasoning_trace"],
                        question=state["question"],
                        missing_requirements=outstanding,
                    )
                    if next_symbol and _execution_evidence_incomplete(
                        state["gathered_chunks"]
                    ):
                        logger.info(
                            "'%s' already expanded; following connected "
                            "symbol '%s' instead.",
                            symbol,
                            next_symbol,
                        )
                        parsed["action"] = "expand_graph"
                        parsed["action_input"] = next_symbol
                    else:
                        logger.info("Requested graph symbol '%s' is already expanded.", symbol)
                        parsed["action"] = "answer"
                        parsed["action_input"] = ""

            # Predicted to add nothing new: every one of this symbol's
            # calls/called_by names already matches evidence we've already
            # gathered or already expanded, so expand_by_graph's walk is
            # very likely to return 0 new chunks - this is exactly the
            # Flask.dispatch_request case, expanded after its neighbors
            # (full_dispatch_request, finalize_request, ...) were already
            # pulled in earlier. The prediction isn't certain (a name can
            # still resolve to an unseen chunk elsewhere in the repo - see
            # call_graph.py's matching caveat), so this only redirects when
            # a more promising connected symbol is actually available; with
            # nothing better to try, it lets the original request through
            # rather than blocking on an unverified guess.
            elif selected_chunk and _expansion_unlikely_to_add_evidence(
                selected_chunk, already_expanded | available_symbols
            ):
                next_symbol = select_execution_path_symbol(
                    reranked,
                    state["reasoning_trace"],
                    question=state["question"],
                    missing_requirements=outstanding,
                )
                if next_symbol and next_symbol != symbol:
                    logger.info(
                        "'%s' is unlikely to add new evidence (all "
                        "neighbors already known); trying '%s' instead.",
                        symbol,
                        next_symbol,
                    )
                    parsed["action"] = "expand_graph"
                    parsed["action_input"] = next_symbol
                else:
                    logger.debug(
                        "'%s' predicted low-yield but no better connected "
                        "symbol is available; proceeding anyway.",
                        symbol,
                    )

        # --------------------------------------------------
        # Never retrieve a symbol/topic we already have.
        # --------------------------------------------------
        # Catches this regardless of which path produced the retrieve (the
        # model's own choice, or the abstract-chunk "subclass of X" redirect
        # above) and regardless of intent - if the target is already-
        # gathered evidence, retrieving it again can only repeat, never
        # progress. This used to be gated to `intent == "understand_flow"`
        # despite the comment above claiming to catch it "regardless of
        # which path" - for any other intent (e.g. a "how is Flask
        # configured" question classified as find_function), a query that
        # already-gathered chunks answer was never recognized as redundant,
        # so the model could retrieve a paraphrase of the same topic every
        # iteration until MAX_AGENT_ITERATIONS was hit with no answer.
        #
        # Two ways a retrieve can be redundant:
        #  - it names an exact symbol we already have (_retrieve_target_symbol)
        #  - it's a paraphrase of a topic already searched this run
        #    (_is_repeated_retrieval_topic)
        # Either way: follow a different connected, unexpanded symbol from
        # the same evidence if one exists; if none does, answer with what's
        # gathered (the runtime evidence gate right below still validates
        # that) rather than issuing the same search again.
        if parsed["action"] == "retrieve":
            action_input = parsed.get("action_input", "")
            retrieve_target = _retrieve_target_symbol(action_input, available_symbols)
            already_covered = bool(retrieve_target) or _is_repeated_retrieval_topic(
                action_input, state["reasoning_trace"]
            )

            if already_covered:
                other_chunks = [
                    c for c in reranked
                    if not retrieve_target or c.get("name") != retrieve_target
                ]
                next_symbol = select_execution_path_symbol(
                    other_chunks,
                    state["reasoning_trace"],
                    question=state["question"],
                    missing_requirements=outstanding,
                )
                if next_symbol:
                    logger.info(
                        "'%s' is already covered by gathered evidence; "
                        "following connected symbol '%s' instead of "
                        "retrieving it again.",
                        action_input,
                        next_symbol,
                    )
                    parsed["action"] = "expand_graph"
                    parsed["action_input"] = next_symbol
                else:
                    logger.info(
                        "'%s' is already covered by gathered evidence and no "
                        "further connected symbol is available; answering "
                        "instead of repeating retrieval.",
                        action_input,
                    )
                    parsed["action"] = "answer"
                    parsed["action_input"] = ""

        # --------------------------------------------------
        # NEW SAFETY CHECK
        # Don't allow answering before reaching runtime code.
        # --------------------------------------------------
        if parsed["action"] == "answer" and state["intent"] == "understand_flow":
           # --------------------------------------------------
# Missing information from the reasoning model
# --------------------------------------------------
            if missing_information:

                missing_symbol = missing_information[0]

                # Bare-vs-qualified fix (see _unresolved_symbols above): a
                # missing_information entry like "add_url_rule" must be
                # recognized against an already-gathered "Flask.add_url_rule"
                # chunk via _matches_symbol, not exact `in available_symbols`
                # membership - that exact check is what let the model
                # conclude "add_url_rule"/"dispatch_request" were still
                # missing and answer accordingly, when the real chunk was
                # already sitting in available_symbols the whole time.
                graph_symbol = None
                for m in missing_information:
                    match = next(
                        (s for s in available_symbols if _matches_symbol(m, s)),
                        None,
                    )
                    if match:
                        graph_symbol = match
                        break

                if graph_symbol:
                    logger.info(
                        "Missing symbol '%s' already gathered. Expanding graph.",
                        graph_symbol,
                    )
                    parsed["action"] = "expand_graph"
                    parsed["action_input"] = graph_symbol

                elif missing_symbol.lower() not in BAD_RETRIEVAL_QUERIES:
                    logger.info(
                        "Retrieving missing symbol '%s'.",
                        missing_symbol,
                    )
                    parsed["action"] = "retrieve"
                    parsed["action_input"] = missing_symbol

# --------------------------------------------------
# Named-requirement gate (understand_flow)
# --------------------------------------------------
# A structural detour that happens to satisfy the generic runtime-
# evidence gate below (well-connected, non-abstract code) is not the
# same as covering what the investigation itself said was still
# needed - e.g. expanding an unrelated-but-well-connected symbol after
# the originally-requested one turned out to be a graph dead end.
# Block "answer" until every requirement ever named this run either
# matches gathered evidence or gets a fresh retrieval attempt.
        if (
            parsed["action"] == "answer"
            and state["intent"] == "understand_flow"
            and outstanding
        ):
            target = outstanding[0]
            logger.info(
                "Named requirement '%s' still unsupported by gathered "
                "evidence; continuing investigation instead of answering.",
                target,
            )
            parsed["action"] = "retrieve"
            parsed["action_input"] = target

# --------------------------------------------------
# Runtime evidence gate
# --------------------------------------------------
        if (
            parsed["action"] == "answer"
            and _execution_evidence_incomplete(state["gathered_chunks"])
        ):
            logger.info(
                "Execution evidence is incomplete; continuing investigation."
            )
            continuation_symbol = select_execution_path_symbol(
                reranked,
                state["reasoning_trace"],
                runtime_only=True,
                question=state["question"],
                missing_requirements=outstanding,
            )
            if not continuation_symbol:
                # No unexpanded symbol is ALREADY a proven runtime
                # implementation - that doesn't mean the graph is exhausted,
                # only that the best-known candidate hasn't been reached
                # yet. Follow any connected, unexpanded symbol (thin
                # wrappers included) rather than falling back to a fresh
                # semantic search.
                continuation_symbol = select_execution_path_symbol(
                    reranked,
                    state["reasoning_trace"],
                    runtime_only=False,
                    question=state["question"],
                    missing_requirements=outstanding,
                )

            if continuation_symbol:
                parsed["action"] = "expand_graph"
                parsed["action_input"] = continuation_symbol
            else:
                parsed["action"] = "retrieve"
                parsed["action_input"] = state["question"]

        # A malformed model response is not evidence that an investigation is
        # complete. Keep the runtime/evidence gate inside reasoning_node.
    except (json.JSONDecodeError, ValueError) as e:

        logger.warning(
            "Reasoning parse failed (%s).",
            e,
        )

        # No new information was gained from this call - carry forward
        # whatever was already outstanding, pruned against evidence
        # gathered since it was named (available_symbols doesn't change
        # within a single call, so this is safe to compute here too).
        outstanding = [
            item for item in state.get("outstanding_requirements", [])
            if not _symbol_has_gathered_implementation(item, available_chunks)
        ]

        # A parse failure carries no new information - it must never be
        # treated as license to answer while evidence is genuinely
        # incomplete. This used to only apply the check below for
        # intent == "understand_flow" and hardcode "answer" for every
        # other intent (find_function, find_usage, debug, ...) - but the
        # real Runtime evidence gate a few lines above (which this exists
        # to mirror) applies _execution_evidence_incomplete unconditionally,
        # with no intent restriction at all. A parse failure on, say, a
        # debug question with incomplete evidence was falling straight
        # through to "answer" with none of that scrutiny. Using the exact
        # same predicate, unconditionally, is what "route it through the
        # existing deterministic investigation/action-selection logic"
        # means here - not a separate, looser rule for this one path.
        if _execution_evidence_incomplete(state["gathered_chunks"]):
            continuation_symbol = select_execution_path_symbol(
                reranked,
                state["reasoning_trace"],
                question=state["question"],
                missing_requirements=outstanding,
            )

            parsed = {
                "thought": f"Reasoning parse failed ({e}); continuing evidence collection.",
                "action": "expand_graph" if continuation_symbol else "retrieve",
                "action_input": continuation_symbol or state["question"],
            }

        else:
            parsed = {
                "thought": f"Reasoning parse failed ({e}); evidence already appears complete.",
                "action": "answer",
                "action_input": "",
            }
            # A parse failure must not silently narrow the evidence handed
            # to the final answer step. `reranked` above is deliberately
            # capped (top_n=min(30, ...)) for choosing what to investigate
            # NEXT - a reasonable cap while evidence-gathering is still in
            # progress, but the wrong thing to hand off the moment we're
            # about to answer: on a multi-file investigation that's
            # gathered more than 30 chunks, that cap would silently drop
            # some of the very symbols retrieval already found, and the
            # final answer would then correctly (from its own narrowed
            # view) say they're "missing" - even though they were
            # genuinely retrieved earlier and simply didn't survive this
            # one rerank's top 30. answer_node already reranks and applies
            # its own token-budget selection on whatever it's given, so
            # handing it everything actually gathered so far costs
            # nothing but correctness. Scoped to this one branch only -
            # every other exit from this function (including "continue
            # investigating" a few lines above, where the cap is exactly
            # the right thing for choosing a next symbol) is unaffected.
            reranked = state["gathered_chunks"]

    # --------------------------------------------------
    # Code-generation intent: only once every gate above has left
    # "answer" standing (i.e. evidence is judged complete) do we
    # switch to proposing a diff instead of a prose answer. This must
    # run after the runtime-evidence gate, not before it, so a code
    # edit is never proposed from declarations/registration code alone.
    # --------------------------------------------------
    if parsed["action"] == "answer" and state.get("intent") == "generate_code_change":
        parsed["action"] = "generate_diff"

    # --------------------------------------------------
    # Keep the execution chain marker consistent
    # --------------------------------------------------
    # _get_execution_chain() recovers "where the structural walk currently
    # is" by re-reading trace lines for "Policy selected structural
    # follow-up from '<symbol>'" — but that marker was only ever written
    # from the _preferred_graph_symbol branch above. Every other place in
    # this function that also lands on "expand_graph" with a symbol chosen
    # by policy (missing-symbol-already-gathered, the already-covered
    # retrieve redirect, the already-expanded redirect, the runtime
    # evidence gate, and the JSON-parse-failure fallback) skipped it, so
    # _get_execution_chain() silently lost track of the real path on those
    # turns and select_execution_path_symbol() fell back to picking a
    # fresh entry point instead of the true next hop. Recording it once,
    # centrally, after every branch has settled on its final action, is
    # what makes every graph-selected symbol — not just the
    # _preferred_graph_symbol one — show up in the chain.
    if parsed["action"] == "expand_graph" and parsed.get("action_input"):
        marker = f"Policy selected structural follow-up from '{parsed['action_input']}'."
        if marker not in parsed.get("thought", ""):
            parsed["thought"] = f"{parsed.get('thought', '')} {marker}".strip()

    parsed.setdefault("thought", "")
    parsed.setdefault("action_input", "")
    return {
        "next_action": parsed["action"],
        "action_query": parsed.get(
            "action_input",
            "",
        ),
        "gathered_chunks": reranked,
        "reasoning_trace": state["reasoning_trace"]
        + [
            f"Thought: {parsed['thought']} -> Action: {parsed['action']}"
        ],
        "agent_input_tokens": state.get("agent_input_tokens", 0) + input_tokens,
        "agent_output_tokens": state.get("agent_output_tokens", 0) + output_tokens,
        "outstanding_requirements": outstanding,
    }

async def retrieve_node(state: AgentState, qdrant_client, redis_client, cfg) -> dict:
    query = state["action_query"] or state["question"]
    exact_symbol = None
    if query.startswith("function "):
        exact_symbol = query.removeprefix("function ").strip() or None
    top_k = max(
      4,
      20 - state["iteration"] * 6,
    )
    new_chunks = await retrieval_tool(
        query=query,
        repo_id=state["repo_id"],
        intent=state["intent"],   # a targeted follow-up lookup, not the original broad intent
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        cfg=cfg,
        top_k=top_k,
        exact_symbol=exact_symbol,
    )

    logger.debug(
        "Retrieved chunks: %s",
        [(c.get("id"), c.get("name"), c.get("file")) for c in new_chunks],
    )

    existing_ids = {c["id"] for c in state["gathered_chunks"]}
    logger.debug("Existing chunk IDs: %s", existing_ids)
    deduped = [c for c in new_chunks if c["id"] not in existing_ids]
    gain_ratio = (len(deduped) / max(1, len(new_chunks)))

    trace = state["reasoning_trace"] + [
        f"Observation: retrieved {len(deduped)} new chunks for '{query}'"
    ]
    if (
        not deduped
        or gain_ratio < 0.20
    ):
        trace.append(
             "Observation: retrieval no longer improving context."
        )

    return {
        "gathered_chunks": state["gathered_chunks"] + deduped,
        "iteration": state["iteration"] + 1,
        "retrieval_exhausted": (not deduped or gain_ratio < 0.20),
        # Retrieval reports evidence gain only. Investigation completion is
        # decided exclusively by reasoning_node after it can consider graph
        # expansion and runtime coverage.
        "next_action": None,
        "action_query": "" ,
        "reasoning_trace": trace,
    }


async def expand_graph_node(state: AgentState, qdrant_client, cfg) -> dict:
    symbol = state["action_query"]

    # Expand from the symbol chosen by the reasoning model.
    # If it isn't present, fall back to the current gathered chunks.
    entry_points = [
        c for c in state["gathered_chunks"]
        if c.get("name") == symbol
    ] 

    if not entry_points:
        return {
            "iteration": state["iteration"] + 1,
            "reasoning_trace": state["reasoning_trace"]
            + [
                f"Observation: no entry chunks to expand from '{symbol}'"
            ],
            "next_action": "retrieve",
            "action_query": symbol,
        }

    expanded, name_index, qualified_index = await expand_graph_tool(
        entry_points,
        state["repo_id"],
        qdrant_client,
        cfg,
        name_index=state.get("graph_name_index"),
        qualified_index=state.get("graph_qualified_index"),
    )

    # --------------------------------------------------
    # Merge + deduplicate
    # --------------------------------------------------

    existing_ids = {chunk["id"] for chunk in state["gathered_chunks"]}
    newly_expanded = [
        chunk for chunk in expanded if chunk["id"] not in existing_ids
    ]
    combined = state["gathered_chunks"] + expanded

    unique = {}

    for chunk in combined:
        unique[chunk["id"]] = chunk

    combined = list(unique.values())

    # --------------------------------------------------
    # Rerank and keep only the best context
    # --------------------------------------------------

    combined = await rerank(
        state["question"],
        combined,
        top_n=min(30, len(combined)),
    )

    return {
        "gathered_chunks": combined,
        "iteration": state["iteration"] + 1,
        "graph_name_index": name_index,
        "graph_qualified_index": qualified_index,
        "reasoning_trace": state["reasoning_trace"]
        + [
            f"Observation: call-graph expansion from '{symbol}' found {len(newly_expanded)} new chunks"
        ],
    }

async def answer_node(state: AgentState) -> dict:
    """
    Final answer generation node.

    Reranks every gathered chunk (retrieval + graph expansion),
    selects the best context within the token budget,
    and generates the final answer.
    """

    cfg = get_settings()

    logger.debug(
        "Gathered chunks (%d): %s",
        len(state["gathered_chunks"]),
        [c.get("name") for c in state["gathered_chunks"]],
    )

    # --------------------------------------------------
    # Nothing gathered
    # --------------------------------------------------
    if not state["gathered_chunks"]:
        return {
            "final_answer": (
                "I couldn't retrieve any relevant code from the repository "
                "to answer this question."
            ),
            "low_confidence": True,
        }

    # --------------------------------------------------
    # Rerank ALL gathered chunks
    # --------------------------------------------------
    _rerank_t0 = time.perf_counter()
    reranked = await rerank(
      state["question"],
      state["gathered_chunks"],
      top_n=min(30, len(state["gathered_chunks"]))
    )
    logger.info(
        "answer_node rerank_ms=%.1f chunks=%d",
        (time.perf_counter() - _rerank_t0) * 1000,
        len(state["gathered_chunks"]),
    )
    logger.debug(
        "Agent rerank scores: %s",
        [(c.get("name"), c.get("rerank_score")) for c in reranked[:10]],
    )

    # --------------------------------------------------
    # Confidence check
    # --------------------------------------------------
    if is_low_confidence(reranked, cfg):
        return {
            "final_answer": (
                "I couldn't find code in this repository that confidently "
                "answers this question, even after retrieval and graph "
                "expansion."
            ),
            "low_confidence": True,
        }

    # --------------------------------------------------
    # Select context within token budget
    # --------------------------------------------------
    final_chunks = select_context(reranked)

    system_prompt, user_msg = build_prompt(
        state["question"],
        final_chunks,
        state["intent"],
    )

    logger.debug(
        "Prompt sizes: system_chars=%d user_chars=%d total_chars=%d est_tokens=%d",
        len(system_prompt), len(user_msg), len(system_prompt) + len(user_msg),
        (len(system_prompt) + len(user_msg)) // 4,
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------
    resp = await _chat_with_fallback(
        
        # Was 400 - too tight for a repository-grounded explanation once the
        # answer needs to walk an execution path or cite more than one
        # symbol, so it cut answers off mid-sentence rather than the model
        # choosing to stop. 1200 gives a real flow explanation room to
        # finish while still bounding cost/latency.
        max_completion_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
    )
    if resp.choices[0].finish_reason == "length":
        logger.warning(
            "Final answer hit max_completion_tokens and was truncated."
        )
    logger.info(
        "answer_node final_answer finish_reason=%s model=%s",
        resp.choices[0].finish_reason,
        getattr(resp, "model", "unknown"),
    )

    answer = resp.choices[0].message.content.strip()
    input_tokens, output_tokens = _token_usage(
        resp,
        system_prompt + user_msg,
        answer,
    )

    return {
        "final_answer": answer,
        "agent_input_tokens": state.get("agent_input_tokens", 0) + input_tokens,
        "agent_output_tokens": state.get("agent_output_tokens", 0) + output_tokens,
    }

def check_stop_condition(state: AgentState) -> str:
    """
    Decide which node to execute next.
    """

    action = state.get("next_action")
    terminal = (
        "generate_diff" if state.get("intent") == "generate_code_change" else "answer"
    )

    # Safety: if the model produced an invalid action,
    # stop and go to this request's terminal node with current context.
    if action not in {"retrieve", "expand_graph", "answer", "generate_diff"}:
        return terminal

    # Model (or the code-gen redirect above) believes it has enough context.
    if action in {"answer", "generate_diff"}:
        return action

    # Hard stop to prevent infinite loops.
    if state["iteration"] >= MAX_AGENT_ITERATIONS:
        return terminal

    # Continue the reasoning loop.
    return action