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

PHASE 2.1 CHANGE — MEMORY-BOUNDED INGESTION (fixes Render OOM):
  Render's free/starter instances cap this process at 512 MB. The old
  implementation materialized several O(repo size) structures at once:

    files      = every file's full source, all in memory together
    all_chunks = every chunk, all in memory together
    texts      = a SECOND full copy of every chunk's text
    vectors    = every embedding vector, all in memory together
    points     = every Qdrant point (id + vector + payload), all at once

  For a large repo those five lists can each be tens to hundreds of MB,
  and they were all alive *simultaneously* at the peak of run_ingest.
  That's what OOM-killed the ingest worker on Render.

  The fix has two parts:

  1. STREAM FILE READING. `_walk_python_files` is now a generator instead
     of a function that returns a fully materialized list. Only one
     file's source is ever resident in memory — as soon as a file has
     been chunked, its raw source falls out of scope and can be
     collected before the next file is even read.

  2. BATCH THE EMBED/UPSERT STAGE. Chunking, call-graph construction,
     and the repo profile still need a global view of the repo — a
     function's caller can live in a different file, and the call
     graph must exist before embedding text is finalized (see the
     comment on build_called_by below). So `all_chunks` (chunk
     metadata + source snippets, NOT full files/vectors/points) is
     still built up across the whole repo — this is expected and is
     the same tradeoff bm25.py documents for its own in-memory index.

     But nothing downstream of chunking needs a global view. Embedding
     lookups, ONNX calls, and Qdrant upserts are all done in small,
     fixed-size batches (EMBED_BATCH_SIZE chunks at a time): look up
     the batch in the cache, embed only that batch's misses, upsert
     just that batch, then let the batch's texts/vectors/points be
     released before moving to the next one. Peak memory for that
     stage is now O(batch_size), not O(repo size), regardless of how
     many chunks the repo has.

PHASE 2.2 CHANGE — THE REMAINING LEAK: all_chunks ITSELF (fixes OOM
still happening after 2.1's batching):
  2.1 bounded the *vectors and Qdrant points* per batch, but it left
  something bigger untouched: `all_chunks` holds full CodeChunk
  objects for the WHOLE repo, and each CodeChunk (see
  app/models/chunk.py) carries THREE separate large strings per chunk —
  `text` (enriched embed text: file + name + docstring + code),
  `raw_source` (the code again, unenriched), and `docstring` — so each
  chunk holds roughly 2x its own source size, and all_chunks holds all
  of that for the entire repo, for the ENTIRE ingest run, because
  nothing ever freed it. Two consequences:

    a) `raw_source` and `docstring` are only ever read once each, when
       chunk.to_payload() is built for that chunk's own Qdrant upsert.
       After that batch is upserted, nothing in the rest of the
       pipeline ever reads them again — but the old code kept every
       chunk's raw_source/docstring alive all the way through the
       *entire* embed/upsert loop (for every other batch) and into the
       BM25 build at the end, for no reason.

    b) `text` is read twice: once for embedding (per batch) and once
       more for the BM25 seed — but the old code only extracted the
       BM25 seed in ONE FINAL PASS after every batch had already run:
       `build_index(repo_id, [{"id": c.id, "text": c.text, ...} for c
       in all_chunks])`. That means right at the moment BM25Okapi is
       building its own tokenized corpus (itself a repo-sized
       allocation), `all_chunks` STILL had every chunk's full text
       resident too — the single highest memory instant in the whole
       pipeline, and it came right after the loop that was supposed to
       have bounded things.

  The fix: extend `_embed_and_upsert_in_batches` to also build the
  BM25 seed incrementally, per batch (capturing each chunk's `text`
  into `bm25_seed` in the same pass that already reads it for
  embedding), and to null out `text`, `raw_source`, and `docstring` on
  each chunk immediately after that chunk's Qdrant payload and BM25
  seed entry have both been captured. Nothing downstream ever reads
  those fields off the chunk object again (verified against
  call_graph.py and bm25.py — call-graph construction only touches
  `name`/`type`/`calls`/`called_by`, and BM25 now reads from
  `bm25_seed`, not from the chunk objects). What's left resident in
  `all_chunks` for the rest of the run is just the small fields
  (id, name, type, calls, called_by, imports, decorators, ...), not
  the multi-KB source/text/docstring per chunk.

  This does NOT touch call_graph.py's build_called_by (which never
  reads text/raw_source/docstring to begin with — verified below) or
  bm25.py's build_index (same input shape it always took: a list of
  {"id","text","name"} dicts — it's just assembled incrementally now
  instead of in one final full-repo pass).

  One structural peak remains, and is NOT eliminated by this change:
  between "all files chunked" and "call graph built", `all_chunks`
  necessarily holds every chunk's text/raw_source/docstring for the
  whole repo at once, because build_called_by() is repo-global (a
  caller can be in a different file than its callee) and must run
  before any chunk's embed text is finalized. Shrinking THAT peak
  would require changing how/when ast_chunker produces `text` and
  `raw_source`, which is out of scope here (ast_chunker.py wasn't
  part of this fix). See test_ingest_memory_accumulation.py for a
  profiled measurement of this remaining, documented baseline.

  Everything else — the Redis cache, Qdrant upsert semantics, the BM25
  index contents/build call, the call graph, the repo profile, and the
  run_ingest return contract — is unchanged.
"""

import asyncio
import gc
import os
import time
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator
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

# How many chunks to embed/upsert per batch. This — not the repo size —
# is what bounds the embed/upsert stage's peak memory. Overridable via
# cfg.ingest_embed_batch_size for tuning against a given Render plan's
# memory limit.
DEFAULT_EMBED_BATCH_SIZE = 100

# How often (in files) to log chunking progress on large repos.
FILE_LOG_INTERVAL = 200

# How often (in batches) to force a gc pass during the embed/upsert
# stage. CPython's refcounting usually reclaims batch-local lists
# immediately, but a periodic explicit collect gives an extra margin
# against reference cycles (e.g. from numpy-backed vectors) on a
# memory-capped host.
GC_EVERY_N_BATCHES = 5


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


def _walk_python_files(repo_path: str) -> Iterator[tuple[str, str]]:
    """
    Yields (relative_path, source_code) for every .py file in the repo,
    skipping excluded directories and oversized files.

    This is a GENERATOR, not a list-returning function. That's the key
    memory fix here: the caller processes (chunks) one file's source and
    lets it go out of scope before the next file is even read off disk.
    Nothing about this function's own behavior — which files it finds,
    which it skips, and why — changed from before.
    """
    repo_root = Path(repo_path)

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
            yield rel_path, source


def _chunk_all_files(local_path: str, repo_id: str) -> tuple[list[CodeChunk], int]:
    """
    Walks and chunks every .py file in the repo, one file at a time.

    Returns (all_chunks, file_count). all_chunks does have to hold every
    chunk from the whole repo — the call graph and BM25 index both need
    that global view — but that's chunk-level metadata and source
    snippets, not whole raw files, embedding vectors, or Qdrant points.
    Those are the structures that made the old implementation blow past
    512 MB, and they're what the batching below avoids materializing in
    full.
    """
    all_chunks: list[CodeChunk] = []
    file_count = 0

    for rel_path, source in _walk_python_files(local_path):
        file_count += 1
        all_chunks.extend(chunk_python_file(source, rel_path, repo_id))
        # `source` falls out of scope on the next loop iteration — only
        # one file's raw text is ever alive at a time.

        if file_count % FILE_LOG_INTERVAL == 0:
            logger.info(
                f"[{repo_id}] chunked {file_count} files so far "
                f"({len(all_chunks)} chunks)..."
            )

    logger.info(f"[{repo_id}] found {file_count} Python files")
    return all_chunks, file_count


def _release_heavy_fields(chunk: CodeChunk) -> None:
    """
    Drop the three large string fields a CodeChunk carries once nothing
    downstream will read them off this object again.

    Safe iff called AFTER:
      - chunk.to_payload() has already been built for this chunk (it's
        the last reader of raw_source and docstring, and one of the
        readers of text), AND
      - this chunk's entry has already been appended to bm25_seed (the
        last reader of text).

    Verified against every other module that touches CodeChunk:
      - call_graph.build_called_by reads only name/type/calls/called_by
        — never text/raw_source/docstring — so it's unaffected no matter
        when this runs relative to it (and in practice this always runs
        well after build_called_by, since call graph construction
        happens before the embed/upsert batches even start).
      - build_repo_profile reads only name/type/imports.
      - bm25.build_index no longer reads chunk objects at all once this
        change lands — it's called with `bm25_seed`, a plain list of
        {"id","text","name"} dicts assembled *before* this function
        runs against each chunk.

    Emptying to "" (not None) keeps the field's declared `str` type
    intact in case anything downstream does string operations on it
    without a None-check.
    """
    chunk.text = ""
    chunk.raw_source = ""
    chunk.docstring = ""


async def _embed_and_upsert_in_batches(
    repo_id: str,
    all_chunks: list[CodeChunk],
    qdrant_client,
    redis_client,
    cfg,
) -> tuple[int, int, list[dict]]:
    """
    Embeds and upserts all_chunks in fixed-size batches so that at most
    one batch's worth of texts/vectors/Qdrant points is ever resident in
    memory, instead of the whole repo's worth at once — AND incrementally
    builds the BM25 seed list while doing so, freeing each chunk's heavy
    fields (text/raw_source/docstring) the moment they're no longer
    needed, instead of leaving them resident until a separate final pass.

    Per batch:
      1. Look up that batch's chunk texts in the Redis embedding cache.
      2. Run ONNX embedding only on that batch's cache misses.
      3. Write freshly-computed vectors back to the cache (same as before).
      4. Build Qdrant points for just this batch (this reads text,
         raw_source, and docstring via chunk.to_payload()) and upsert them.
      5. Capture this batch's {"id","text","name"} entries into the
         running bm25_seed list — same shape bm25.build_index has always
         expected, just assembled incrementally instead of in one final
         list comprehension over the whole repo.
      6. NOW that nothing will read this chunk's text/raw_source/
         docstring again, null them out (see _release_heavy_fields).
      7. Let the rest of the batch's local texts/vectors/points be
         released before moving on to the next batch.

    Returns (cached_count, fresh_count, bm25_seed) — the same
    cached/fresh figures the old single-shot implementation returned
    (just accumulated across batches), plus the BM25 seed list that
    run_ingest now passes straight to build_index instead of building it
    itself in a second full pass over all_chunks.
    """
    batch_size = getattr(cfg, "ingest_embed_batch_size", DEFAULT_EMBED_BATCH_SIZE) or DEFAULT_EMBED_BATCH_SIZE

    total = len(all_chunks)
    total_batches = (total + batch_size - 1) // batch_size

    cached_count = 0
    fresh_count = 0
    bm25_seed: list[dict] = []

    for batch_num, batch_start in enumerate(range(0, total, batch_size), start=1):
        batch = all_chunks[batch_start: batch_start + batch_size]
        batch_texts = [c.text for c in batch]

        cache_lookup = await batch_get_embeddings(redis_client, batch_texts)

        to_embed_texts = [t for t in batch_texts if cache_lookup[t] is None]

        if to_embed_texts:
            fresh_vectors = await embed_batch(to_embed_texts)
            for text, vector in zip(to_embed_texts, fresh_vectors):
                await set_cached_embedding(redis_client, text, vector)
            fresh_lookup = dict(zip(to_embed_texts, fresh_vectors))
        else:
            fresh_vectors = []
            fresh_lookup = {}

        vectors = [
            cache_lookup[t] if cache_lookup[t] is not None else fresh_lookup[t]
            for t in batch_texts
        ]

        # Last reads of raw_source/docstring (via to_payload) and one of
        # the two reads of text (the other being batch_texts above).
        points = [
            {"id": chunk.id, "vector": vector, "payload": chunk.to_payload()}
            for chunk, vector in zip(batch, vectors)
        ]
        await upsert_chunks(qdrant_client, cfg.qdrant_collection, points)

        # Last read of text: capture this batch's BM25 seed entries
        # BEFORE freeing anything below.
        bm25_seed.extend(
            {"id": chunk.id, "text": text, "name": chunk.name}
            for chunk, text in zip(batch, batch_texts)
        )

        # Nothing downstream reads text/raw_source/docstring off these
        # chunk objects again — release them now instead of letting them
        # ride resident through every remaining batch plus the BM25 build.
        for chunk in batch:
            _release_heavy_fields(chunk)

        batch_cached = len(batch_texts) - len(to_embed_texts)
        batch_fresh = len(to_embed_texts)
        cached_count += batch_cached
        fresh_count += batch_fresh

        logger.info(
            f"[{repo_id}] embed batch {batch_num}/{total_batches}: "
            f"{len(batch)} chunks ({batch_fresh} fresh, {batch_cached} cached), "
            f"heavy fields released for {batch_start + len(batch)}/{total} chunks so far"
        )

        # Explicitly drop batch-local references so this batch's texts,
        # vectors, and points can be collected before the next batch
        # starts — this is the "release the batch" step that keeps peak
        # memory at O(batch_size) instead of O(repo size).
        del batch, batch_texts, cache_lookup, to_embed_texts
        del fresh_vectors, fresh_lookup, vectors, points

        if batch_num % GC_EVERY_N_BATCHES == 0:
            gc.collect()

    return cached_count, fresh_count, bm25_seed


async def run_ingest(repo_id: str, github_url: str, branch: str, qdrant_client, redis_client, cfg) -> dict:
    """
    The full pipeline. Returns a summary dict (chunk count, file count, etc.)
    that gets stored as the repo's metadata.

    Return contract, error behavior, and the meaning of every field are
    unchanged from before — only how memory is used along the way changed.
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

    # 2 & 3. Walk every .py file and chunk it with the AST chunker, one
    #    file at a time (see _chunk_all_files / _walk_python_files above
    #    for why this no longer materializes every file's source at once).
    all_chunks, file_count = _chunk_all_files(local_path, repo_id)

    if not all_chunks:
        return {"status": "failed", "error": "No chunks extracted — is this a Python repo?"}

    logger.info(f"Extracted {len(all_chunks)} chunks from {file_count} files")
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
    #     This is inherently repo-global (a caller can live in a different
    #     file than its callee), which is why all_chunks has to be fully
    #     assembled before this step — unlike the embed/upsert stage below,
    #     which is batched.
    build_called_by(all_chunks)

    # 4. Clear any old chunks for this repo (handles re-ingest cleanly)
    await delete_repo(qdrant_client, cfg.qdrant_collection, repo_id)

    # 5 & 6. Embed (checking the Redis cache first) and upsert into Qdrant,
    #    in fixed-size batches so that at most one batch's worth of
    #    texts/vectors/points is ever resident in memory at once — not the
    #    whole repo's worth. This also incrementally builds bm25_seed and
    #    frees each chunk's text/raw_source/docstring as soon as they're
    #    no longer needed, instead of letting them sit resident through
    #    every remaining batch. See _embed_and_upsert_in_batches for
    #    details on both.
    cached_count, fresh_count, bm25_seed = await _embed_and_upsert_in_batches(
        repo_id, all_chunks, qdrant_client, redis_client, cfg
    )
    logger.info(
        f"[{repo_id}] embedding cache: {cached_count}/{len(all_chunks)} hits, "
        f"embedded {fresh_count} new chunks"
    )

    # 7. Build the BM25 keyword index for this repo. Same input shape as
    #    before — a list of {"id","text","name"} dicts — and BM25 still
    #    needs that full corpus at once to compute its index (see
    #    bm25.py); that part is unavoidable and unchanged. What changed
    #    is that bm25_seed was assembled incrementally during the batch
    #    loop above, one batch's worth of text at a time, rather than in
    #    a single final pass over all_chunks while every chunk's text was
    #    still fully resident — see the PHASE 2.2 note at the top of this
    #    file for why that final pass was the single worst memory moment
    #    in the old pipeline.
    build_index(repo_id, bm25_seed)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"Ingest complete for {repo_id}: {len(all_chunks)} chunks in {elapsed}s")

    commit_hash = git.Repo(local_path).head.commit.hexsha

    return {
        "status":          "done",
        "chunk_count":     len(all_chunks),
        "file_count":      file_count,
        "languages":       ["python"],
        "ingest_seconds":  elapsed,
        "embeddings_cached": cached_count,
        "embeddings_fresh":  fresh_count,
        "last_commit":     commit_hash,
    }