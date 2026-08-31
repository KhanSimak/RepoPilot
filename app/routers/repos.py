"""
repos.py — repo registration, ingest trigger, and incremental sync

STORAGE: repo metadata/status is persisted in Redis (key `repo:{repo_id}`,
a JSON blob), not just held in a process-local dict. Render can run
multiple app instances/workers, and `asyncio.create_task` background
ingest continues on whichever process started it — a plain in-memory
dict meant GET /repos/{id} could 404 even right after POST /repos/
returned 202, if the poll landed on a different process than the one
doing the ingest. Redis is already a hard dependency of run_ingest() /
run_incremental_ingest() (see app/ingest/pipeline.py,
app/ingest/incremental.py), so this doesn't add a new piece of
infrastructure to local dev — it uses the same redis_client those
already require.

`_registry` (in-memory) is kept too, but now purely as a same-process
mirror: every write goes to Redis first, then updates `_registry` as a
side effect. It exists for get_registry()'s synchronous, dict-shaped
contract — see that function's docstring for why it isn't a full fix for
every consumer, and get_repo_record() for the async alternative.
"""

from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, timezone
import asyncio
import json
import logging
import uuid

from app.schemas.api import RepoCreate, RepoStatus
from app.ingest.pipeline import run_ingest
from app.ingest.incremental import run_incremental_ingest
from app.engine.vectordb import count_repo_chunks, delete_repo
from app.engine.bm25 import delete_index

logger = logging.getLogger(__name__)
router = APIRouter()

# Same-process mirror of what's in Redis - see module docstring and
# get_registry()'s docstring for exactly what this is (and isn't) for.
_registry: dict[str, dict] = {}

_REDIS_KEY_PREFIX = "repo:"


def _redis_key(repo_id: str) -> str:
    return f"{_REDIS_KEY_PREFIX}{repo_id}"


async def _save_repo(redis_client, repo_id: str, meta: dict) -> None:
    """Write-through: Redis is the source of truth (what makes this
    survive across processes/restarts), _registry is just the local
    mirror kept in sync as a side effect."""
    await redis_client.set(_redis_key(repo_id), json.dumps(meta))
    _registry[repo_id] = meta


async def _load_repo(redis_client, repo_id: str) -> dict | None:
    """Read from Redis first - this is what actually fixes the reported
    bug, since it's the same data regardless of which process/restart
    handles the request. Falls back to the local mirror only if Redis
    itself errors, so a transient Redis blip doesn't turn a repo this
    process already knows about into a hard 404; a genuinely unknown
    repo_id still correctly returns None either way.
    """
    try:
        raw = await redis_client.get(_redis_key(repo_id))
    except Exception:
        logger.warning("Redis read failed for repo %s; falling back to local mirror", repo_id, exc_info=True)
        raw = None

    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        meta = json.loads(raw)
        _registry[repo_id] = meta
        return meta

    return _registry.get(repo_id)


async def _delete_repo_record(redis_client, repo_id: str) -> None:
    try:
        await redis_client.delete(_redis_key(repo_id))
    except Exception:
        logger.warning("Redis delete failed for repo %s", repo_id, exc_info=True)
    _registry.pop(repo_id, None)


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

    meta = {
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
    await _save_repo(redis_client, repo_id, meta)

    async def _background_ingest():
        try:
            result = await run_ingest(repo_id, body.github_url, body.branch, qdrant, redis_client, cfg)
            current = await _load_repo(redis_client, repo_id) or dict(meta)
            current.update(result)
            await _save_repo(redis_client, repo_id, current)
        except Exception as e:
            current = await _load_repo(redis_client, repo_id) or dict(meta)
            current["status"] = "failed"
            current["error"]  = str(e)
            await _save_repo(redis_client, repo_id, current)

    asyncio.create_task(_background_ingest())

    return RepoStatus(**meta)


@router.get("/{repo_id}", response_model=RepoStatus)
async def get_repo(repo_id: str, request: Request):
    redis_client = request.app.state.redis
    meta = await _load_repo(redis_client, repo_id)
    if meta is None:
        raise HTTPException(404, "Repo not found")

    meta = dict(meta)

    # If ingest finished, get the live chunk count straight from Qdrant
    if meta["status"] == "done":
        cfg    = request.app.state.settings
        qdrant = request.app.state.qdrant
        meta["chunk_count"] = await count_repo_chunks(qdrant, cfg.qdrant_collection, repo_id)

    return RepoStatus(**meta)


@router.delete("/{repo_id}", status_code=204)
async def delete_repo_endpoint(repo_id: str, request: Request):
    redis_client = request.app.state.redis
    meta = await _load_repo(redis_client, repo_id)
    if meta is None:
        raise HTTPException(404, "Repo not found")

    cfg    = request.app.state.settings
    qdrant = request.app.state.qdrant
    await delete_repo(qdrant, cfg.qdrant_collection, repo_id)
    delete_index(repo_id)   # drop the in-memory BM25 index too
    await _delete_repo_record(redis_client, repo_id)


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
    redis_client = request.app.state.redis
    meta = await _load_repo(redis_client, repo_id)
    if meta is None:
        raise HTTPException(404, "Repo not found")

    if meta["status"] == "ingesting":
        raise HTTPException(409, "Ingest already in progress for this repo")

    qdrant, cfg = request.app.state.qdrant, request.app.state.settings
    last_commit = meta.get("last_commit")

    meta = dict(meta)
    meta["status"] = "ingesting"
    await _save_repo(redis_client, repo_id, meta)

    async def _background_sync():
        try:
            result = await run_incremental_ingest(
                repo_id, meta["github_url"], meta["branch"], last_commit, qdrant, redis_client, cfg,
            )
            current = await _load_repo(redis_client, repo_id) or dict(meta)
            current["status"] = "done"
            current["last_commit"] = result["new_commit"]
            current["last_sync"] = result
            await _save_repo(redis_client, repo_id, current)
        except Exception as e:
            current = await _load_repo(redis_client, repo_id) or dict(meta)
            current["status"] = "failed"
            current["error"]  = str(e)
            await _save_repo(redis_client, repo_id, current)

    asyncio.create_task(_background_sync())
    return {"repo_id": repo_id, "status": "ingesting", "message": "Incremental sync started — poll GET /repos/{id} for progress"}


def get_registry() -> dict:
    """Used by other routers (search.py, query.py, eval.py) to check if a
    repo exists/is ready.

    Kept synchronous and dict-shaped on purpose, so those existing
    callers don't need to change (Redis reads are async; making this
    function itself async would require touching every call site, and
    those files weren't part of this fix). This still returns exactly
    what it always did: the process-local `_registry` mirror.

    THE CATCH: that mirror is only as fresh as whatever create_repo/
    get_repo/sync_repo/delete_repo_endpoint have already done ON THIS
    SAME PROCESS. If a caller in search.py/query.py/eval.py runs on a
    DIFFERENT process than the one that ingested a repo, and GET
    /repos/{id} was never polled on this process to warm the mirror
    (e.g. the frontend polled a different instance the whole time), this
    can still under-report — the same class of bug this file's own
    endpoints were just fixed for, just not closed here because I don't
    have visibility into how those files consume this. get_repo_record()
    below is a drop-in async replacement that reads Redis directly (no
    staleness window) for any of them to adopt.
    """
    return _registry


async def get_repo_record(repo_id: str, redis_client) -> dict | None:
    """Async, Redis-backed equivalent of `get_registry().get(repo_id)` -
    reads the source of truth directly, so it doesn't have
    get_registry()'s same-process staleness window. Available for
    search.py/query.py/eval.py to adopt if/when they're updated; existing
    synchronous get_registry() callers are unaffected."""
    return await _load_repo(redis_client, repo_id)