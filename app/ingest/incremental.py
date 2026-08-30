"""
incremental.py — re-ingest only what changed since last time

THE PROBLEM:
  A full re-ingest of a 1,000-file repo means re-cloning, re-chunking,
  re-embedding, and re-upserting EVERYTHING, even if you only changed
  3 files. For a repo with a few thousand chunks, that's tens of seconds
  of ONNX embedding work for a change that touched almost nothing.

THE FIX, TWO LAYERS OF "DID THIS ACTUALLY CHANGE":
  Layer 1 — git diff: which FILES changed between the last ingested
            commit and HEAD? Only re-chunk those files, not the whole repo.
  Layer 2 — content hash: even within a changed file, did THIS SPECIFIC
            function's body actually change? CodeChunk.content_hash()
            (models/chunk.py) is an MD5 of the raw source. We compare the
            new chunk's hash against what's already stored in Qdrant for
            that chunk ID. If unchanged (e.g. you only edited a comment
            elsewhere in the file, or another function in the same file),
            we skip re-embedding it — the OLD vector is still perfectly
            valid, because nothing about the function's content changed.

WHAT THIS DOESN'T HANDLE (be honest about the limits):
  - Renamed/moved functions are treated as delete-old + add-new, not as
    a rename. That's fine for retrieval quality, just not maximally efficient.
  - Call graph rebuild re-walks the WHOLE repo's chunks after any change,
    because a single changed function can alter `called_by` edges anywhere
    (e.g. delete a function and everyone who called it needs that edge
    removed). This rebuild is cheap (in-memory, no embedding involved) so
    it isn't a performance concern — just worth knowing it isn't
    "incremental" in the same sense the embedding step is.
"""

import logging
import git

from app.ingest.cloner import clone_repo
from app.engine.ast_chunker import chunk_python_file
from app.engine.embedder import embed_batch
from app.engine.vectordb import upsert_chunks, scroll_repo_chunks, retrieve_by_ids
from qdrant_client.models import PointIdsList, SetPayload, SetPayloadOperation
from app.cache.redis_cache import clear_cached_queries_for_repo
from app.engine.bm25 import build_index
from app.engine.call_graph import build_called_by
from app.models.chunk import CodeChunk
from app.ingest.pipeline import _walk_python_files

logger = logging.getLogger(__name__)


def get_changed_files(repo_path: str, since_commit: str | None) -> list[str] | None:
    """
    Returns relative .py file paths changed since `since_commit`, or None
    if we can't determine a diff (e.g. first-ever ingest) — caller should
    fall back to a full re-walk in that case.
    """
    if since_commit is None:
        return None
    try:
        repo = git.Repo(repo_path)
        diff_output = repo.git.diff(since_commit, "HEAD", "--name-only")
        return [f for f in diff_output.strip().split("\n") if f.endswith(".py")]
    except Exception as e:
        logger.warning(f"Could not compute git diff ({e}) — falling back to full re-ingest")
        return None


async def run_incremental_ingest(
    repo_id: str, github_url: str, branch: str, last_commit: str | None,
    qdrant_client, redis_client, cfg,
) -> dict:
    """
    If `last_commit` is None or the diff can't be computed, transparently
    falls back to chunking every .py file — same code path handles both
    "first ingest" and "incremental re-ingest", just with a different
    starting file list.
    """
    local_path = clone_repo(github_url, repo_id, cfg.repos_dir, branch)
    repo       = git.Repo(local_path)
    new_commit = repo.head.commit.hexsha

    changed_files = get_changed_files(local_path, last_commit)

    if changed_files is None:
        files_to_process = _walk_python_files(local_path)
        mode = "full"
    else:
        all_files = dict(_walk_python_files(local_path))
        files_to_process = [(f, all_files[f]) for f in changed_files if f in all_files]
        mode = "incremental"

    logger.info(f"[{mode}] Re-chunking {len(files_to_process)} files")

    new_chunks: list[CodeChunk] = []
    for rel_path, source in files_to_process:
        new_chunks.extend(chunk_python_file(source, rel_path, repo_id))

    # Keep a snapshot before upserts so removed files/symbols can be deleted.
    existing_repo_chunks = await scroll_repo_chunks(qdrant_client, cfg.qdrant_collection, repo_id)
    changed_paths = (
        set(changed_files)
        if changed_files is not None
        else {chunk.get("file") for chunk in existing_repo_chunks}
    )
    new_ids = {chunk.id for chunk in new_chunks}
    obsolete_ids = [
        chunk["id"] for chunk in existing_repo_chunks
        if chunk.get("file") in changed_paths and chunk["id"] not in new_ids
    ]
    if obsolete_ids:
        await qdrant_client.delete(
            collection_name=cfg.qdrant_collection,
            points_selector=PointIdsList(points=obsolete_ids),
        )

    # ── Compare against what's already stored, by content hash ────────────
    new_ids_list = [c.id for c in new_chunks]
    existing = await retrieve_by_ids(qdrant_client, cfg.qdrant_collection, new_ids_list)

    to_embed_chunks, unchanged_chunks = [], []
    for chunk in new_chunks:
        existing_chunk = existing.get(chunk.id)
        if existing_chunk and existing_chunk.get("content_hash") == CodeChunk.content_hash(chunk.raw_source):
            unchanged_chunks.append(chunk)
        else:
            to_embed_chunks.append(chunk)

    logger.info(f"Content hash check: {len(unchanged_chunks)} unchanged, {len(to_embed_chunks)} need (re-)embedding")

    if to_embed_chunks:
        vectors = await embed_batch([c.text for c in to_embed_chunks])
        points  = [{"id": c.id, "vector": v, "payload": c.to_payload()} for c, v in zip(to_embed_chunks, vectors)]
        await upsert_chunks(qdrant_client, cfg.qdrant_collection, points)

    # ── Rebuild the call graph across the WHOLE repo (cheap, in-memory) ────
    all_chunks_now = await scroll_repo_chunks(qdrant_client, cfg.qdrant_collection, repo_id)

    class _Stub:
        def __init__(self, d):
            self.name, self.calls, self.called_by = d["name"], d.get("calls", []), []

    stubs = [_Stub(c) for c in all_chunks_now]
    build_called_by(stubs)

    payload_updates = [
        SetPayloadOperation(
            set_payload=SetPayload(
                payload={"called_by": stub.called_by},
                points=[chunk_dict["id"]],
            )
        )
        for chunk_dict, stub in zip(all_chunks_now, stubs)
        if set(stub.called_by) != set(chunk_dict.get("called_by", []))
    ]
    if payload_updates:
        await qdrant_client.batch_update_points(
            collection_name=cfg.qdrant_collection,
            update_operations=payload_updates,
        )

    # ── Rebuild BM25 — must reflect the full current chunk set ─────────────
    build_index(repo_id, [{"id": c["id"], "text": c["text"], "name": c["name"]} for c in all_chunks_now])
    await clear_cached_queries_for_repo(redis_client, repo_id)

    return {
        "status": "done", "mode": mode,
        "changed_files":   len(files_to_process),
        "embedded_chunks": len(to_embed_chunks),
        "skipped_chunks":  len(unchanged_chunks),
        "total_chunks":    len(all_chunks_now),
        "new_commit":      new_commit,
    }
