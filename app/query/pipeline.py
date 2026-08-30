"""
pipeline.py — the complete query pipeline, fully traced

Stage order (every stage wrapped in a StageTimer, see cost_tracker.py):
  1. L1 cache check (exact question+repo+filters match)         -> ~1ms on hit
  2. HyDE rewrite + intent detection (one Groq call)             -> ~60ms
  3. Embed the HyDE snippet (L2 cache checked first)             -> ~25ms or ~1ms cached
  4. Hybrid retrieval: vector + BM25 + RRF (+ graph expand)      -> ~25-60ms
  5. Cross-encoder rerank: top 20 -> top 5                       -> ~80ms, $0
  6. Parallel context compression + token budget enforcement     -> ~200ms
  7. Final LLM call (llama-3.1-8b-instant via Groq), streamed or not -> Groq's
     LPU hardware makes this the fastest stage in the whole pipeline, not
     the slowest — first tokens typically arrive well under 150ms.
  8. Cache the result for next time

Both a blocking `run_query` (returns the full answer + trace) and a
streaming `stream_query` (SSE generator) are provided — same pipeline,
different output shape.

NOTE ON STREAMING SHAPE: Anthropic's SDK provides a `.messages.stream()`
async context manager with a convenience `.text_stream` iterator. Groq's
SDK is OpenAI-style instead — `stream=True` on the normal `.create()` call
returns an async-iterable stream directly, and each chunk's text lives at
`chunk.choices[0].delta.content` (which can be `None` for the very first
chunk and the final chunk, so it's guarded below).
"""

import json
import logging
logger = logging.getLogger(__name__)   # was imported but never instantiated — needed below
from groq import AsyncGroq, RateLimitError
from openai import AsyncOpenAI
from app.schemas.api import SearchResponse
from app.engine.cost_tracker import RequestTrace
from app.engine.reranker import rerank, is_low_confidence
from app.engine.token_budget import  select_context, build_prompt, count_tokens
from app.query.rewriter import rewrite_query, GRAPH_EXPAND_INTENTS
from app.query.retriever import retrieve
from app.cache.redis_cache import get_cached_query, set_cached_query
from app.config import get_settings
from app.agent.graph import run_agent
import time
settings = get_settings()

_groq_llm = AsyncGroq(
    api_key=settings.groq_api_key
)
_openrouter_llm = (
    AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    if settings.openrouter_api_key
    else None
)


async def _chat_with_fallback(
    *,
    messages: list[dict],
    max_completion_tokens: int,
    stream: bool = False,
    response_format: dict | None = None,
):
    """Try Groq first; on a Groq rate-limit error, fall back to OpenRouter.

    `stream` and `response_format` are forwarded to whichever provider
    actually serves the request. Previously this hardcoded stream=True for
    Groq and stream=False for the OpenRouter fallback regardless of what a
    caller wanted - run_query (which needs a normal completion with
    .choices[0].message) and stream_query (which needs an AsyncStream of
    .choices[0].delta chunks) got the same, wrong, unconditional behavior
    either way.
    """
    kwargs = {
        "model": settings.groq_model,
        "max_completion_tokens": max_completion_tokens,
        "messages": messages,
        "stream": stream,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        logger.info(
            "Calling Groq model '%s'%s.",
            settings.groq_model,
            " with streaming" if stream else "",
        )

        return await _groq_llm.chat.completions.create(**kwargs)

    except RateLimitError as exc:
        logger.warning(
            "Groq rate limit reached%s. Falling back to OpenRouter. Error: %s",
            " during streaming" if stream else "",
            exc,
        )

        if _openrouter_llm is None:
            raise

        logger.info(
            "Calling OpenRouter model '%s'%s.",
            settings.openrouter_model,
            " with streaming" if stream else "",
        )

        kwargs["model"] = settings.openrouter_model
        return await _openrouter_llm.chat.completions.create(**kwargs)

async def _do_retrieval_and_rerank(
    question: str,
    repo_id: str,
    qdrant_client,
    redis_client,
    cfg,
    top_k: int,
    trace: RequestTrace,
):
    """Shared by both run_query and stream_query — stages 2 through 5."""

    stage_rewrite = trace.start_stage("hyde_rewrite")
    rewrite = await rewrite_query(question, repo_id, redis_client)
    stage_rewrite.input_tokens = count_tokens(question) + 150
    stage_rewrite.output_tokens = count_tokens(json.dumps(rewrite))
    stage_rewrite.finish()

    graph_expand = rewrite["intent"] in GRAPH_EXPAND_INTENTS
    graph = rewrite.get("graph") if isinstance(rewrite.get("graph"), dict) else {}
    graph_depth = graph.get("depth", 1)
    graph_direction = graph.get("direction", "both")
    if not isinstance(graph_depth, int) or graph_depth < 1:
        graph_depth = 1
    if graph_direction not in {"callers", "callees", "both"}:
        graph_direction = "both"

    stage_retrieve = trace.start_stage(
        "hybrid_retrieval" + ("_graph_expanded" if graph_expand else "")
    )

    candidates = await retrieve(
        question=question,
        hyde_snippet=rewrite["implementation_summary"],
        phrases=rewrite["phrases"],
        intent=rewrite["intent"],
        repo_id=repo_id,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        cfg=cfg,
        # The public top_k controls returned/cached response shape; retrieval
        # keeps the configured wider candidate pool for reranking.
        top_k=cfg.query_top_k,
        graph_expand=graph_expand,
        graph_depth=graph_depth,
        graph_direction=graph_direction,
    )

    stage_retrieve.finish()

    logger.debug("Candidates: %d", len(candidates))

    if not candidates:
        return [], rewrite

    stage_rerank = trace.start_stage("reranker")

    reranked = await rerank(question, candidates, top_n=12)

    stage_rerank.finish()

    logger.debug(
        "Reranked chunks (%d): %s",
        len(reranked),
        [(c["name"], c["type"], c.get("rerank_score")) for c in reranked],
    )

    return reranked, rewrite

async def run_query(question: str, repo_id: str, qdrant_client, redis_client, cfg, top_k: int = 5) -> dict:
    trace = RequestTrace(query=question, repo_id=repo_id)
    t0 = time.perf_counter()
    # ── Stage 1: L1 cache ────────────────────────────────────────────────
    stage_cache = trace.start_stage("query_cache_l1")
    cached = await get_cached_query(redis_client, repo_id, question, top_k)

    logger.debug("Query cache hit: %s", cached is not None)

    if cached:
      logger.debug("Cached first source: %s", cached.get("sources", [None])[0])
      cached["latency_ms"] = {
        "embed_ms": 0,
        "search_ms": 0,
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "cache_hit": True,
    }
      return SearchResponse(**cached)
    reranked, rewrite = await _do_retrieval_and_rerank(question, repo_id, qdrant_client, redis_client, cfg, top_k, trace)

    if not reranked:
        return {
            "question": question, "answer": "No relevant code found in this repository.",
            "sources": [], "cache_hit": False, "trace": trace.summary(),
        }

    # "Keep top-N regardless of score" (below) is right for RANKING —
    # CrossEncoder scores are relative to each other within a query. But a
    # low TOP score is still a real signal that nothing in the corpus is
    # actually relevant (e.g. a conceptual/comparative question against a
    # codebase that only contains implementation code) — ranking chunks
    # relative to each other doesn't mean any of them are actually good.
    if is_low_confidence(reranked, cfg):
        return {
            "question": question,
            "answer": (
                "I couldn't find code in this repository that confidently answers this. "
                "This usually means the question is asking for something outside what's "
                "indexed here (general framework knowledge, a comparison to something not "
                "in this repo, etc.) rather than about this specific codebase's implementation."
            ),
            "sources": [{"name": c.get("name"), "file": c.get("file"), "rerank_score": c.get("rerank_score")} for c in reranked[:3]],
            "cache_hit": False,
            "low_confidence": True,
            "trace": trace.summary(),
        }

    # ── Compress + budget ────────────────────────────────────────────────
    stage_select = trace.start_stage("context_selection")
# Keep the top-N ranked chunks regardless of score.
# CrossEncoder scores are relative, not absolute.

    # Context selection must use the cross-encoder ordering for every intent.
    # For graph intents, ``reranked`` includes the post-expansion candidates.
    context_candidates = reranked

    final_chunks = select_context(context_candidates)
    logger.debug(
        "Final context: %s",
        [(c["name"], c["type"], c.get("rerank_score")) for c in final_chunks],
    )
    stage_select.finish()

    # ── Final LLM call ───────────────────────────────────────────────────
    stage_llm = trace.start_stage("llm_generation")
    system_prompt, user_msg = build_prompt(
    question,
    final_chunks,
    rewrite["intent"],
)
    stage_llm.input_tokens = count_tokens(system_prompt) + count_tokens(user_msg)

    response = await _chat_with_fallback(
        
        # Was 400 - too tight for a repository-grounded explanation and
        # cut answers off mid-sentence. Matches the agent loop's answer_node.
        max_completion_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        stream=False,
    )
    answer = response.choices[0].message.content
    stage_llm.output_tokens = count_tokens(answer)
    stage_llm.finish()

    result = {
     "question": question,
    "answer": answer,
    "rewritten_query": rewrite["implementation_summary"],
    "intent": rewrite["intent"],
    "sources": [
        {
            "id": c["id"],
            "name": c.get("name", ""),
            "type": c.get("type", ""),
            "file": c.get("file", ""),
            "language": c.get("language", ""),
            "line_start": c.get("line_start", 0),
            "line_end": c.get("line_end", 0),
            "docstring": c.get("docstring", ""),
            "calls": c.get("calls", []),
            "raw_source": c.get("raw_source", ""),
            "score": c.get("rerank_score", c.get("score", 0.0)),
        }
        for c in reranked
    ],
        "cache_hit": False,
}
    
    if answer.strip():
      await set_cached_query(
        redis_client,
        repo_id,
        question,
        top_k,
        result,
    )
    return {
       **result,
       "latency_ms": round(
        (time.perf_counter() - t0) * 1000,
        1,
      ),
      "trace": trace.summary(),
}


async def stream_query(
    question: str,
    repo_id: str,
    qdrant_client,
    redis_client,
    cfg,
    top_k: int = 5,
):
    """SSE generator. Emits sources BEFORE the LLM starts, then streams tokens."""

    trace = RequestTrace(query=question, repo_id=repo_id)

    cached = await get_cached_query(redis_client, repo_id, question, top_k)
    if cached:
        yield f"data: {json.dumps({'type':'sources','sources':cached.get('sources', []),'cached':True})}\n\n"
        yield f"data: {json.dumps({'type':'token','text':cached['answer']})}\n\n"
        yield f"data: {json.dumps({'type':'done','trace':trace.summary()})}\n\n"
        return

    reranked, rewrite = await _do_retrieval_and_rerank(
        question,
        repo_id,
        qdrant_client,
        redis_client,
        cfg,
        top_k,
        trace,
    )

    if not reranked:
        yield f"data: {json.dumps({'type':'error','text':'No relevant code found.'})}\n\n"
        return

    # NEW: Cannot-answer-from-repository detection
    if is_low_confidence(reranked, cfg):
        payload = {
         "type": "error",
          "text": (
            "I couldn't find code in this repository that confidently answers this. "
            "This usually means the question is asking about something outside "
            "this repository rather than its implementation."
          ),
        }

        yield f"data: {json.dumps(payload)}\n\n"
        return

    sources = [
        {
            "id": c["id"],
            "name": c.get("name", ""),
            "type": c.get("type", ""),
            "file": c.get("file", ""),
            "language": c.get("language", ""),
            "line_start": c.get("line_start", 0),
            "line_end": c.get("line_end", 0),
            "docstring": c.get("docstring", ""),
            "calls": c.get("calls", []),
            "raw_source": c.get("raw_source", ""),
            "score": c.get("rerank_score", c.get("score", 0.0)),
        }
        for c in reranked
    ]

    payload = {
      "type": "sources",
      "sources": sources,
      "rewrite": rewrite["implementation_summary"],
      "intent": rewrite["intent"],
    }

    yield f"data: {json.dumps(payload)}\n\n"

    stage_select = trace.start_stage("context_selection")

    final_chunks = select_context(reranked)

    stage_select.finish()

    system_prompt, user_msg = build_prompt(
        question,
        final_chunks,
        rewrite["intent"],
    )

    stage_llm = trace.start_stage("llm_generation")
    stage_llm.input_tokens = (
        count_tokens(system_prompt)
        + count_tokens(user_msg)
    )

    full_answer = []

    stream = await _chat_with_fallback(
        
        # Was 400 - too tight for a repository-grounded explanation and cut
        # answers off mid-sentence. Matches the agent loop's answer_node.
        max_completion_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        # Was stream=False here (backwards - this function iterates the
        # response as a stream of .delta chunks right below, which only
        # works when the provider was actually asked to stream).
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer.append(delta)
            yield f"data: {json.dumps({'type':'token','text':delta})}\n\n"

    answer = "".join(full_answer)

    stage_llm.output_tokens = count_tokens(answer)
    stage_llm.finish()

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "intent": rewrite["intent"],
    }

    if answer.strip():
        await set_cached_query(
            redis_client,
            repo_id,
            question,
            top_k,
            result,
        )

    yield f"data: {json.dumps({'type':'done','trace':trace.summary()})}\n\n"

async def run_agentic_query(question: str, repo_id: str, qdrant_client, redis_client, cfg) -> dict:
    """
    Iterative ReAct loop (app/agent/), for the SAME intents that already
    trigger one-shot graph expansion above (GRAPH_EXPAND_INTENTS) — the
    difference is this loop can re-retrieve or expand AGAIN based on what
    the first pass actually found, instead of a single fixed depth=2 walk.

    Deliberately NOT wired into run_query/stream_query or their cache —
    this is a separate, explicit endpoint (see routers/query.py's new
    /ask/agent route) so the existing fast path stays exactly as it was,
    and so a cached final answer here can't silently hide the iterative
    reasoning trace that's the actual point of hitting this path.

    find_function questions don't need this — they're one specific symbol,
    graph expansion adds nothing — so those still go through run_query.
    """
    trace = RequestTrace(query=question, repo_id=repo_id)
    t0 = time.perf_counter()

    stage_rewrite = trace.start_stage("hyde_rewrite")
    rewrite = await rewrite_query(question, repo_id, redis_client)
    # Keep this endpoint's rewrite accounting consistent with the normal
    # /ask pipeline.  rewrite_query does not expose the SDK response object.
    stage_rewrite.input_tokens = count_tokens(question) + 150
    stage_rewrite.output_tokens = count_tokens(json.dumps(rewrite))
    stage_rewrite.finish()

    if rewrite["intent"] not in GRAPH_EXPAND_INTENTS:
        logger.info(f"Intent '{rewrite['intent']}' doesn't need the agent loop, falling back to run_query")
        return await run_query(question, repo_id, qdrant_client, redis_client, cfg)

    stage_agent = trace.start_stage("agent_loop")
    final_state = await run_agent(
        question=question,
        repo_id=repo_id,
        intent=rewrite["intent"],
        hyde_snippet=rewrite["implementation_summary"],
        phrases=rewrite["phrases"],
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        cfg=cfg,
    )
    stage_agent.input_tokens = final_state.get("agent_input_tokens", 0)
    stage_agent.output_tokens = final_state.get("agent_output_tokens", 0)
    stage_agent.finish()

    return {
        "question": question,
        "answer": final_state["final_answer"] or "Could not produce an answer within the iteration limit.",
        "intent": rewrite["intent"],
        "iterations": final_state["iteration"],
        "reasoning_trace": final_state["reasoning_trace"],
        "sources": [
            {
                "id": c["id"],
                "name": c.get("name", ""),
                "type": c.get("type", ""),
                "file": c.get("file", ""),
                "line_start": c.get("line_start", 0),
                "line_end": c.get("line_end", 0),
                "score": c.get("rerank_score", c.get("score", 0.0)),
            }
            for c in final_state["gathered_chunks"]
        ],
        "cache_hit": False,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "trace": trace.summary(),
    }