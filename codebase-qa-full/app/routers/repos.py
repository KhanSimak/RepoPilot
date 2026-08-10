"""
repos.py — repo registration, ingest trigger, and incremental sync

NOTE on storage: the repo registry is still an in-memory dict. This stays
a deliberate simplification through every phase of this project — it
matters once you run multiple server processes and need them to share
state, which is a horizontal-scaling concern outside this project's scope.
The registry now also tracks `last_commit`, which is what makes
POST /repos/{id}/sync able to do an INCREMENTAL re-ingest instead of
starting over from scratch every time.
"""

from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, timezone
import asyncio
import uuid

from app.schemas.api import RepoCreate, RepoStatus
from app.ingest.pipeline import run_ingest
from app.ingest.incremental import run_incremental_ingest
from app.engine.vectordb import count_repo_chunks, delete_repo
from app.engine.bm25 import delete_index

router = APIRouter()

# In-memory registry: repo_id -> metadata dict
_registry: dict[str, dict] = {}


@router.post("/", response_model=RepoStatus, status_code=202)
async def create_repo(body: RepoCreate, request: Request):
    """
    Register a repo and kick off ingest in the background.
    Returns immediately with status="ingesting" — poll GET /repos/{id} for progress.
    """
    repo_id      = str(uuid.uuid4())[:8]
    qdrant       = request.app.state.qdrant
    redis_client = request.app.state.redis
    cfg          = request.app.state.settings

    _registry[repo_id] = {
        "id":          repo_id,
        "github_url":  body.github_url,
        "branch":      body.branch,
        "status":      "ingesting",
        "chunk_count": 0,
        "file_count":  0,
        "languages":   [],
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "error":       None,
    }

    async def _background_ingest():
        try:
            result = await run_ingest(repo_id, body.github_url, body.branch, qdrant, redis_client, cfg)
            _registry[repo_id].update(result)
        except Exception as e:
            _registry[repo_id]["status"] = "failed"
            _registry[repo_id]["error"]  = str(e)

    asyncio.create_task(_background_ingest())

    return RepoStatus(**_registry[repo_id])


@router.get("/{repo_id}", response_model=RepoStatus)
async def get_repo(repo_id: str, request: Request):
    if repo_id not in _registry:
        raise HTTPException(404, "Repo not found")

    meta = dict(_registry[repo_id])

    # If ingest finished, get the live chunk count straight from Qdrant
    if meta["status"] == "done":
        cfg    = request.app.state.settings
        qdrant = request.app.state.qdrant
        meta["chunk_count"] = await count_repo_chunks(qdrant, cfg.qdrant_collection, repo_id)

    return RepoStatus(**meta)


@router.delete("/{repo_id}", status_code=204)
async def delete_repo_endpoint(repo_id: str, request: Request):
    if repo_id not in _registry:
        raise HTTPException(404, "Repo not found")

    cfg    = request.app.state.settings
    qdrant = request.app.state.qdrant
    await delete_repo(qdrant, cfg.qdrant_collection, repo_id)
    delete_index(repo_id)   # drop the in-memory BM25 index too
    del _registry[repo_id]


@router.post("/{repo_id}/sync", status_code=202)
async def sync_repo(repo_id: str, request: Request):
    """
    Incremental re-ingest: git pull, diff against the last ingested commit,
    re-chunk only the changed files, and within those only re-embed chunks
    whose content hash actually changed. See app/ingest/incremental.py.

    Falls back to a full re-walk automatically if there's no previous
    commit on record (e.g. you've never run /sync on this repo before —
    use POST /repos for the very first ingest, then /sync afterward).
    """
    if repo_id not in _registry:
        raise HTTPException(404, "Repo not found")

    meta = _registry[repo_id]
    if meta["status"] == "ingesting":
        raise HTTPException(409, "Ingest already in progress for this repo")

    qdrant, redis_client, cfg = request.app.state.qdrant, request.app.state.redis, request.app.state.settings
    last_commit = meta.get("last_commit")

    _registry[repo_id]["status"] = "ingesting"

    async def _background_sync():
        try:
            result = await run_incremental_ingest(
                repo_id, meta["github_url"], meta["branch"], last_commit, qdrant, redis_client, cfg,
            )
            _registry[repo_id]["status"] = "done"
            _registry[repo_id]["last_commit"] = result["new_commit"]
            _registry[repo_id]["last_sync"] = result
        except Exception as e:
            _registry[repo_id]["status"] = "failed"
            _registry[repo_id]["error"]  = str(e)

    asyncio.create_task(_background_sync())
    return {"repo_id": repo_id, "status": "ingesting", "message": "Incremental sync started — poll GET /repos/{id} for progress"}


def get_registry() -> dict:
    """Used by other routers (search.py, query.py, eval.py) to check if a repo exists/is ready."""
    return _registry
