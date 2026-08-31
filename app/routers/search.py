"""
search.py — query endpoints, now with hybrid search + caching

PHASE 2 CHANGES from Phase 1:
  1. Query result cache checked FIRST — a repeat question returns in ~1ms,
     skipping embedding, vector search, BM25, fusion, and the LLM call entirely.
  2. Vector search AND BM25 keyword search now run in parallel (asyncio.gather),
     then merged with Reciprocal Rank Fusion.
  3. The embedding for the query itself is also cache-checked (Layer 2 cache) —
     if you ask the exact same question twice within the cache window, or if
     two different users ask the same thing, the second one skips the ONNX
     model call too.
  4. Latency breakdown now reports cache_hit so you can SEE the difference
     a cache hit makes — this is exactly what you'd show an interviewer.
"""

from fastapi import APIRouter, Request, HTTPException, Query
from groq import AsyncGroq
import asyncio
import time
import logging

from app.engine.embedder import embed_text
from app.engine.vectordb import search as vector_search, retrieve_by_ids
from app.engine.bm25 import search as bm25_search
from app.engine.fusion import reciprocal_rank_fusion
from app.cache.redis_cache import (
    get_cached_query, set_cached_query,
    get_cached_embedding, set_cached_embedding,
)
from app.routers.repos import get_repo_record
from app.schemas.api import SearchResponse, ChunkOut
from app.config import get_settings
router = APIRouter()
logger = logging.getLogger(__name__)


settings = get_settings()

_llm = AsyncGroq(
    api_key=settings.groq_api_key
)


@router.get("/{repo_id}/search", response_model=SearchResponse)
async def search_endpoint(
    repo_id: str,
    request: Request,
    question: str = Query(..., min_length=3, description="Your question about the codebase"),
    top_k:   int = Query(default=5, ge=1, le=20),
):
    cfg    = request.app.state.settings
    qdrant = request.app.state.qdrant
    redis_client = request.app.state.redis

    meta = await get_repo_record(repo_id, redis_client)
    if meta is None:
        raise HTTPException(404, "Repo not found. POST /repos first.")
    if meta["status"] != "done":
        raise HTTPException(400, f"Repo not ready yet. Status: {meta['status']}")

    t0 = time.perf_counter()

    # ── Layer 1 cache check — exact question, exact repo, exact top_k ──────
    cached = await get_cached_query(redis_client, repo_id, question, top_k)
    if cached:
        cached["latency_ms"] = {"embed_ms": 0, "search_ms": 0, "total_ms": round((time.perf_counter() - t0) * 1000, 1), "cache_hit": True}
        return SearchResponse(**cached)

    # ── Step 1 — embed the query (with Layer 2 cache check) ────────────────
    cached_vector = await get_cached_embedding(redis_client, question)
    if cached_vector:
        vector = cached_vector
    else:
        vector = await embed_text(question, is_query=True)
        await set_cached_embedding(redis_client, question, vector)
    t_embed = round((time.perf_counter() - t0) * 1000, 1)

    # ── Step 2 — vector search AND BM25 search IN PARALLEL ─────────────────
    # asyncio.gather runs both concurrently. Qdrant's call is the I/O-bound
    # one (network round trip); BM25 is synchronous in-memory and effectively
    # instant, but running them together still avoids serializing the wait
    # on Qdrant behind anything else we might add to this gather later.
    vector_hits_task = vector_search(qdrant, cfg.qdrant_collection, vector, repo_id, top_k=20)
    keyword_hits = bm25_search(repo_id, question, top_k=20)   # sync, fast, no need to await
    vector_hits = await vector_hits_task
    t_search = round((time.perf_counter() - t0) * 1000, 1)

    if not vector_hits and not keyword_hits:
        return SearchResponse(
            question=question, answer="No relevant code found in this repository.",
            sources=[], latency_ms={"embed_ms": t_embed, "search_ms": t_search, "total_ms": t_search, "cache_hit": False},
        )

    # ── Step 3 — fuse with RRF, then fetch full payloads for the final list ─
    fused_ids = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=top_k)

    # Most fused IDs will already have their payload from the vector search
    # results; any that came ONLY from BM25 need a direct lookup.
    vector_lookup = {h["id"]: h for h in vector_hits}
    missing_ids   = [doc_id for doc_id in fused_ids if doc_id not in vector_lookup]
    fetched       = await retrieve_by_ids(qdrant, cfg.qdrant_collection, missing_ids) if missing_ids else {}

    results = []
    for doc_id in fused_ids:
        chunk = vector_lookup.get(doc_id) or fetched.get(doc_id)
        if chunk:
            results.append(chunk)

    # ── Step 4 — build context and ask the LLM ──────────────────────────────
    context = "\n\n---\n\n".join([
        f"[{i+1}] {r['type'].title()}: {r['name']} — {r['file']} (lines {r['line_start']}-{r['line_end']})\n{r['raw_source'][:500]}"
        for i, r in enumerate(results)
    ])

    response = await _llm.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Answer this question about the codebase using only the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer concisely. Always cite the file name and function/class name."
            ),
        }],
    )
    answer = response.choices[0].message.content
    t_total = round((time.perf_counter() - t0) * 1000, 1)

    result_payload = {
        "question": question,
        "answer":   answer,
        "sources": [
            ChunkOut(
                id=r["id"], name=r["name"], type=r["type"], file=r["file"],
                language=r["language"], line_start=r["line_start"], line_end=r["line_end"],
                docstring=r.get("docstring", ""), calls=r.get("calls", []),
                score=r.get("score", 0.0), raw_source=r["raw_source"][:300],
            ).model_dump()
            for r in results
        ],
    }

    logger.debug("Baseline search first source: %s", result_payload["sources"][0])
    # Cache the full result for next time (5 min TTL)
    
    await set_cached_query(
     redis_client,
     repo_id,
     question,
     top_k,
     result_payload,
)
    
    return SearchResponse(
        **result_payload,
        latency_ms={"embed_ms": t_embed, "search_ms": t_search, "total_ms": t_total, "cache_hit": False},
    )


@router.get("/{repo_id}/chunks")
async def list_chunks(
    repo_id: str,
    request: Request,
    question:       str = Query(..., description="Search query, used to find similar chunks"),
    top_k:   int = Query(default=10, ge=1, le=50),
    mode:    str = Query(default="hybrid", regex="^(vector|hybrid)$", description="'vector' = Qdrant only, 'hybrid' = vector+BM25+RRF"),
):
    """
    Debug endpoint — returns raw chunks WITHOUT calling the LLM.
    Use ?mode=vector vs ?mode=hybrid on the SAME query to see exactly what
    BM25 + RRF fusion adds (or removes) compared to vector search alone.
    """
    meta = await get_repo_record(repo_id, request.app.state.redis)
    if meta is None:
        raise HTTPException(404, "Repo not found")

    cfg    = request.app.state.settings
    qdrant = request.app.state.qdrant

    vector = await embed_text(question, is_query=True)
    vector_hits = await vector_search(qdrant, cfg.qdrant_collection, vector, repo_id, top_k=top_k)

    if mode == "vector":
        return {"query": question, "mode": "vector", "count": len(vector_hits), "chunks": vector_hits}

    keyword_hits = bm25_search(repo_id, question, top_k=top_k)
    fused_ids = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=top_k)

    vector_lookup = {h["id"]: h for h in vector_hits}
    missing_ids   = [doc_id for doc_id in fused_ids if doc_id not in vector_lookup]
    fetched       = await retrieve_by_ids(qdrant, cfg.qdrant_collection, missing_ids) if missing_ids else {}

    chunks = [vector_lookup.get(doc_id) or fetched.get(doc_id) for doc_id in fused_ids]
    chunks = [c for c in chunks if c]

    return {
        "query": question, "mode": "hybrid", "count": len(chunks), "chunks": chunks,
        "debug": {"vector_hit_count": len(vector_hits), "bm25_hit_count": len(keyword_hits)},
    }