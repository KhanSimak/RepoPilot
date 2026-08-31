"""
query.py — the final pipeline's public API

POST /repos/{id}/ask     — full pipeline, blocking, returns a cost/latency trace
GET  /repos/{id}/stream  — same pipeline, SSE streaming

This supersedes search.py's /search endpoint for anything that needs
HyDE rewriting, reranking, token budgeting, graph expansion, or a cost
trace. /search (Phase 2) is left in place as the simpler baseline you can
diff against — hit both with the same question and compare the `sources`
and latency to see exactly what each added stage changes.
"""

from fastapi import APIRouter, Request, HTTPException, Query as QParam
from fastapi.responses import StreamingResponse

from app.query.pipeline import run_query, stream_query, run_agentic_query
from app.agent.graph import run_agent
from app.routers.repos import get_repo_record

router = APIRouter()


@router.post("/{repo_id}/ask/agent")
async def ask_agent(
    repo_id: str,
    request: Request,
    question: str = QParam(..., min_length=3),
):
    """
    Iterative ReAct loop (app/agent/) — for understand_flow / find_usage /
    debug questions, this can re-retrieve or expand the call graph AGAIN
    based on what it finds, instead of /ask's single fixed-depth expansion.
    Falls back to the regular /ask pipeline automatically for find_function
    questions, where iteration doesn't add anything.

    Response includes `reasoning_trace` — every Thought/Action/Observation
    step, useful for seeing exactly why the agent gathered what it gathered.
    """
    redis_client = request.app.state.redis
    meta = await get_repo_record(repo_id, redis_client)
    if meta is None:
        raise HTTPException(404, "Repo not found")
    if meta["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {meta['status']}")

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant
    return await run_agentic_query(question, repo_id, qdrant, redis_client, cfg)


@router.post("/{repo_id}/generate-diff")
async def generate_diff(
    repo_id: str,
    request: Request,
    change_request: str = QParam(
        ..., min_length=3,
        description="Plain-English description of the code change to make, e.g. "
                     "'add input validation to the login endpoint'",
    ),
):
    """
    Proposes a code change as a unified diff.

    Runs the SAME ReAct investigation loop as /ask/agent - retrieve +
    expand_graph, gated by the same evidence-completeness checks (won't
    propose a change from declarations/registration code alone) - but
    forces intent="generate_code_change" so it terminates in
    generate_diff_node instead of a prose answer. Unlike /ask/agent,
    intent here is NOT auto-detected: the caller is explicitly asking for
    a change, not a question to classify.

    READ-ONLY: nothing in this call touches the repository. `proposed_diff`
    is a unified diff (or null if the model couldn't safely propose one
    from the gathered context) for you to review and apply yourself -
    applying it is a separate, sandboxed step this endpoint doesn't do.
    """
    redis_client = request.app.state.redis
    meta = await get_repo_record(repo_id, redis_client)
    if meta is None:
        raise HTTPException(404, "Repo not found")
    if meta["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {meta['status']}")

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant

    result = await run_agent(
        question=change_request,
        repo_id=repo_id,
        # The agent's own reasoning IS the query refinement (same rationale
        # as tools.py's retrieval_tool) - no separate HyDE rewrite needed.
        hyde_snippet=change_request,
        intent="generate_code_change",
        phrases=[],
        qdrant_client=qdrant,
        redis_client=redis_client,
        cfg=cfg,
    )

    return {
        "proposed_diff": result.get("proposed_diff"),
        "diff_explanation": result.get("diff_explanation"),
        "stop_reason": result.get("stop_reason"),
        "iterations": result.get("iteration"),
        "reasoning_trace": result.get("reasoning_trace"),
    }


@router.post("/{repo_id}/ask")
async def ask(
    repo_id: str,
    request: Request,
    question: str = QParam(..., min_length=3),
    top_k: int = QParam(default=5, ge=1, le=10),
):
    redis_client = request.app.state.redis
    meta = await get_repo_record(repo_id, redis_client)
    if meta is None:
        raise HTTPException(404, "Repo not found")
    if meta["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {meta['status']}")

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant
    return await run_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k)


@router.get("/{repo_id}/stream")
async def ask_stream(
    repo_id: str,
    request: Request,
    question: str = QParam(..., min_length=3),
    top_k: int = QParam(default=5, ge=1, le=10),
):
    """
    SSE events emitted, in order:
      {"type":"sources","sources":[...],"rewrite":"...","intent":"..."}
      {"type":"token","text":"..."}  (repeated as tokens stream in)
      {"type":"done","trace":{...}}
    """
    redis_client = request.app.state.redis
    meta = await get_repo_record(repo_id, redis_client)
    if meta is None:
        raise HTTPException(404, "Repo not found")
    if meta["status"] != "done":
        raise HTTPException(400, "Repo not ready")

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant
    return StreamingResponse(
        stream_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )