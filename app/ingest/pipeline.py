"""
pipeline.py — orchestrates the full ingest flow

  clone repo -> walk files -> chunk each .py file with AST
  -> check embedding cache -> embed only the misses -> upsert into Qdrant
  -> build BM25 index

PHASE 2 CHANGE: embeddings now go through the Redis cache first.
If you re-ingest a repo where 90% of functions haven't changed, those
90% skip the ONNX model entirely — we already have their vectors cached
from last time (the chunk's enriched embed text is identical, so the
cache key, which is md5(text), is identical too).
"""

import asyncio
import os
import time
import logging
from pathlib import Path
import git

from app.models.chunk import CodeChunk
from app.ingest.cloner import clone_repo
from app.engine.ast_chunker import chunk_python_file
from app.engine.embedder import embed_batch
from app.engine.vectordb import upsert_chunks, delete_repo
from app.engine.bm25 import build_index
from app.engine.call_graph import build_called_by
from app.cache.redis_cache import batch_get_embeddings, set_cached_embedding

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".pytest_cache"}
MAX_FILE_SIZE_BYTES = 500_000   # skip huge generated/minified files


async def build_repo_profile(all_chunks: list[CodeChunk]) -> str:
    """
    Build a short summary of the repository vocabulary.
    This is stored in Redis and later injected into the HyDE prompt so
    query rewriting uses the repo's own symbols instead of generic Python.
    """

    class_names = [
        c.name
        for c in all_chunks
        if c.type == "class"
    ][:20]

    function_names = [
        c.name
        for c in all_chunks
        if c.type in ("function", "method")
    ][:30]

    imports = []
    for chunk in all_chunks:
        imports.extend(chunk.imports)

    # remove duplicates while preserving order
    imports = list(dict.fromkeys(imports))[:15]

    return (
        f"Framework/imports: {', '.join(imports)}\n"
        f"Key classes: {', '.join(class_names)}\n"
        f"Key functions: {', '.join(function_names)}"
    )


def _walk_python_files(repo_path: str) -> list[tuple[str, str]]:
    """
    Returns a list of (relative_path, source_code) for every .py file in the repo,
    skipping excluded directories and oversized files.
    """
    repo_root = Path(repo_path)
    found = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = Path(dirpath) / filename
            if filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue

            try:
                source = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")
                continue

            rel_path = str(filepath.relative_to(repo_root))
            found.append((rel_path, source))

    return found


async def run_ingest(repo_id: str, github_url: str, branch: str, qdrant_client, redis_client, cfg) -> dict:
    """
    The full pipeline. Returns a summary dict (chunk count, file count, etc.)
    that gets stored as the repo's metadata.
    """
    t0 = time.perf_counter()

    # 1. Clone (or pull) the repo - clone_repo() is a blocking, synchronous
    #    subprocess call (see app/ingest/cloner.py) that can legitimately
    #    take up to CLONE_TIMEOUT_SECONDS (120s). Calling it directly here
    #    would block THIS event loop for that entire duration - freezing
    #    every other request this FastAPI process is serving, not just
    #    this one ingest. asyncio.to_thread() runs it on a worker thread
    #    instead, so the event loop stays free to serve other requests
    #    while this clone/pull is in progress.
    local_path = await asyncio.to_thread(clone_repo, github_url, repo_id, cfg.repos_dir, branch)

    # 2. Walk every .py file
    files = _walk_python_files(local_path)
    logger.info(f"Found {len(files)} Python files in {repo_id}")

    # 3. Chunk every file with the AST chunker
    all_chunks = []
    for rel_path, source in files:
        all_chunks.extend(chunk_python_file(source, rel_path, repo_id))

    if not all_chunks:
        return {"status": "failed", "error": "No chunks extracted — is this a Python repo?"}

    logger.info(f"Extracted {len(all_chunks)} chunks from {len(files)} files")
    # Build a repository profile for HyDE query rewriting.
    repo_profile = await build_repo_profile(all_chunks)

    await redis_client.set(
    f"repo_profile:{repo_id}",
    repo_profile,
    ex=60 * 60 * 24 * 30,  # 30 days
)
    # 3b. Build the call graph (calls -> called_by) across ALL chunks in
    #     this repo, BEFORE embedding. called_by is part of the embed text's
    #     payload (not the embedding itself — see CodeChunk.to_payload),
    #     so it needs to exist before we build the Qdrant points below.
    build_called_by(all_chunks)

    # 4. Clear any old chunks for this repo (handles re-ingest cleanly)
    await delete_repo(qdrant_client, cfg.qdrant_collection, repo_id)

    # 5. Embed — but check the cache FIRST, in one batched round trip.
    #    Only the chunks that miss the cache actually hit the ONNX model.
    texts        = [c.text for c in all_chunks]
    cache_lookup = await batch_get_embeddings(redis_client, texts)

    cached_count = sum(1 for v in cache_lookup.values() if v is not None)
    to_embed_texts  = [t for t in texts if cache_lookup[t] is None]
    to_embed_chunks = [c for c, t in zip(all_chunks, texts) if cache_lookup[t] is None]

    logger.info(f"Embedding cache: {cached_count}/{len(texts)} hits, embedding {len(to_embed_texts)} new chunks")

    fresh_vectors = await embed_batch(to_embed_texts)

    # Write the freshly-computed ones back to cache for next time
    for text, vector in zip(to_embed_texts, fresh_vectors):
        await set_cached_embedding(redis_client, text, vector)

    # Stitch cached + fresh vectors back together in original chunk order
    fresh_lookup = dict(zip(to_embed_texts, fresh_vectors))
    vectors = [cache_lookup[t] if cache_lookup[t] is not None else fresh_lookup[t] for t in texts]

    # 6. Upsert into Qdrant
    points = [
        {"id": chunk.id, "vector": vector, "payload": chunk.to_payload()}
        for chunk, vector in zip(all_chunks, vectors)
    ]
    await upsert_chunks(qdrant_client, cfg.qdrant_collection, points)

    # 7. Build the BM25 keyword index for this repo
    build_index(repo_id, [{"id": c.id, "text": c.text, "name": c.name} for c in all_chunks])

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"Ingest complete for {repo_id}: {len(all_chunks)} chunks in {elapsed}s")

    commit_hash = git.Repo(local_path).head.commit.hexsha

    return {
        "status":          "done",
        "chunk_count":     len(all_chunks),
        "file_count":      len(files),
        "languages":       ["python"],
        "ingest_seconds":  elapsed,
        "embeddings_cached": cached_count,
        "embeddings_fresh":  len(to_embed_texts),
        "last_commit":     commit_hash,
    }