"""
query.py — the final pipeline's public API

POST /repos/{id}/ask                — full pipeline, blocking, returns a cost/latency trace
GET  /repos/{id}/stream             — same pipeline, SSE streaming
POST /repos/{id}/generate-diff      — Phase 1/2: propose a code change as a diff
POST /repos/{id}/apply-diff         — Phase 3: apply a diff in an isolated worktree
POST /repos/{id}/applies/{id}/commit — Phase 4: human approves, merge into base branch
POST /repos/{id}/applies/{id}/reject — Phase 4: human rejects, discard worktree+branch
GET  /repos/{id}/applies            — Phase 4: list applies pending review
"""

from fastapi import APIRouter, Request, HTTPException, Query as QParam, Body
from fastapi.responses import StreamingResponse
from dataclasses import asdict
from pathlib import Path
import asyncio

from app.query.pipeline import run_query, stream_query, run_agentic_query
from app.ingest.incremental import run_incremental_ingest
from app.agent.graph import run_agent
from app.agent.sandbox import (
    apply_diff_in_sandbox,
    commit_pending_apply,
    reject_pending_apply,
    list_pending_applies,
)
from app.agent.post_commit_sync import sync_after_commit
from app.routers.repos import get_registry

router = APIRouter()


@router.post("/{repo_id}/ask/agent")
async def ask_agent(repo_id: str, request: Request, question: str = QParam(..., min_length=3)):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {registry[repo_id]['status']}")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    return await run_agentic_query(question, repo_id, qdrant, redis_client, cfg)


@router.post("/{repo_id}/generate-diff")
async def generate_diff(
    repo_id: str, request: Request,
    change_request: str = QParam(..., min_length=3),
):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {registry[repo_id]['status']}")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    result = await run_agent(
        question=change_request, repo_id=repo_id, hyde_snippet=change_request,
        intent="generate_code_change", phrases=[],
        qdrant_client=qdrant, redis_client=redis_client, cfg=cfg,
    )
    return {
        "proposed_diff": result.get("proposed_diff"),
        "diff_explanation": result.get("diff_explanation"),
        "stop_reason": result.get("stop_reason"),
        "iterations": result.get("iteration"),
        "reasoning_trace": result.get("reasoning_trace"),
    }


@router.post("/{repo_id}/apply-diff")
async def apply_diff(
    repo_id: str, request: Request,
    proposed_diff: str = Body(..., embed=True, description="Unified diff, typically from /generate-diff's response"),
    diff_explanation: str = Body("", embed=True),
):
    """
    Phase 3: applies a proposed diff in an isolated git worktree, on a
    brand-new branch. Returns an apply_id for Phase 4's review/commit/
    reject flow. Read-only with respect to the actual repo.
    """
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    meta = registry[repo_id]
    if meta["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {meta['status']}")

    cfg, redis_client = request.app.state.settings, request.app.state.redis
    repo_path = Path(cfg.repos_dir) / repo_id
    base_branch = meta.get("branch", "main")

    result = await apply_diff_in_sandbox(
        repo_id=repo_id, repo_path=repo_path, base_branch=base_branch,
        diff_text=proposed_diff, diff_explanation=diff_explanation,
        redis_client=redis_client,
    )
    return asdict(result)


@router.get("/{repo_id}/applies")
async def list_applies(repo_id: str, request: Request):
    """
    Phase 4: lists applies currently pending human review for this repo
    - each includes the proposed_diff and diff_explanation so a reviewer
    doesn't need to go back to /generate-diff's original response.
    """
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")

    redis_client = request.app.state.redis
    pending = await list_pending_applies(redis_client, repo_id)
    return {"repo_id": repo_id, "pending_applies": pending}


@router.post("/{repo_id}/applies/{apply_id}/commit")
async def commit_apply(repo_id: str, apply_id: str, request: Request):
    """
    Phase 4: the human approves this change. Merges it into the repo's
    base branch (checkout-free — see app/agent/sandbox.py's module
    docstring for why).

    Phase 5: on success, kicks off a background re-index so the next
    query sees the new code. This is backgrounded rather than awaited
    because re-indexing can take a while and the reviewer shouldn't wait
    on it — poll GET /repos/{repo_id} for status, same as after an
    ingest. The working tree is refreshed FIRST (see
    app/agent/post_commit_sync.py — a checkout-free merge leaves the
    working tree holding a staged reversal of the change, which would
    otherwise get indexed instead of the committed code).
    """
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")

    redis_client = request.app.state.redis
    result = await commit_pending_apply(redis_client, apply_id)

    if not result.committed:
        if result.error == "apply_id not found or expired":
            raise HTTPException(404, result.error)
        return asdict(result)

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant
    meta = registry[repo_id]
    repo_path = Path(cfg.repos_dir) / repo_id
    base_branch = meta.get("branch", "main")

    registry[repo_id]["status"] = "ingesting"

    async def _background_resync():
        try:
            sync_result = await sync_after_commit(
                repo_id=repo_id,
                repo_path=repo_path,
                base_branch=base_branch,
                previous_commit=result.previous_commit,
                new_commit=result.commit_hash,
                run_incremental_ingest=run_incremental_ingest,
                github_url=meta["github_url"],
                qdrant_client=qdrant,
                redis_client=redis_client,
                cfg=cfg,
            )
            if sync_result.synced:
                registry[repo_id]["status"] = "done"
                registry[repo_id]["last_commit"] = sync_result.new_commit
            else:
                registry[repo_id]["status"] = "failed"
                registry[repo_id]["error"] = sync_result.error
        except Exception as e:
            registry[repo_id]["status"] = "failed"
            registry[repo_id]["error"] = str(e)

    asyncio.create_task(_background_resync())

    return {
        **asdict(result),
        "reindex": "started",
        "message": "Change committed. Re-indexing started — poll GET /repos/{repo_id} for progress.",
    }


@router.post("/{repo_id}/applies/{apply_id}/reject")
async def reject_apply(repo_id: str, apply_id: str, request: Request):
    """Phase 4: the human rejects this change. Discards the worktree
    and branch entirely — nothing about it is reachable again."""
    redis_client = request.app.state.redis
    result = await reject_pending_apply(redis_client, apply_id)
    if not result.rejected and result.error == "apply_id not found or expired":
        raise HTTPException(404, result.error)
    return asdict(result)


@router.post("/{repo_id}/ask")
async def ask(repo_id: str, request: Request, question: str = QParam(..., min_length=3), top_k: int = QParam(default=5, ge=1, le=10)):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {registry[repo_id]['status']}")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    return await run_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k)


@router.get("/{repo_id}/stream")
async def ask_stream(repo_id: str, request: Request, question: str = QParam(..., min_length=3), top_k: int = QParam(default=5, ge=1, le=10)):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, "Repo not ready")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    return StreamingResponse(
        stream_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )