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

  2.1 and 2.2 left one structural peak undocumented as a known
  baseline rather than actually fixing it (see the old note that used
  to live here): between "all files chunked" and "call graph built",
  `all_chunks` necessarily held every chunk's text/raw_source/
  docstring for the whole repo at once, because build_called_by() is
  repo-global (a caller can be in a different file than its callee)
  and must run before any chunk's embed text is finalized.

PHASE 2.3 CHANGE — SPOOL FULL CHUNKS TO DISK (fixes the OOM that
survived 2.1/2.2, without touching ast_chunker or the repo-global call
graph requirement):
  The insight: build_called_by only ever needs name/type/calls/
  called_by, and build_repo_profile only ever needs name/type/imports
  — neither reads text, raw_source, or docstring, the three fields
  that make a full CodeChunk heavy. So there's no real reason those
  heavy fields need to be *resident in RAM* across the whole
  chunking -> call-graph window; they just need to still *exist* by
  the time the embed/upsert stage reads them, batch by batch.

  So now:
    - Every full CodeChunk is written to a ChunkSpool (see
      app/ingest/chunk_spool.py) — a disk-backed, append-only,
      read-back-in-original-order store — the instant it's produced
      during chunking, and immediately falls out of scope. Nothing
      repo-sized holding full chunks is ever resident in RAM.
    - In its place, a lightweight ChunkMeta (see app/models/chunk.py)
      is kept resident for the whole repo, for the whole run — one per
      chunk, holding only id/name/type/file/calls/called_by/imports.
      This is what build_repo_profile and build_called_by now run
      against instead of full chunks. build_called_by mutates
      ChunkMeta.called_by in place exactly as it used to mutate
      CodeChunk.called_by — call_graph.py itself needed NO changes,
      since it only ever accessed those same few attributes generically.
    - The embed/upsert stage (_embed_and_upsert_in_batches) now reads
      full chunks back from the spool one batch at a time, in the same
      order they were written — which is guaranteed to match the
      chunk_metas batch it's paired with (see chunk_spool.py's
      docstring for why no offset/id index is needed for this). Each
      chunk's called_by is stale on disk (still empty — it was spooled
      before build_called_by ran), so it's patched in from the
      corresponding ChunkMeta right after the batch is read back and
      before chunk.to_payload() is built.
    - _release_heavy_fields() is gone — it existed only to null out
      heavy fields on chunks that otherwise stayed resident in
      `all_chunks` for the rest of the run. Now, full chunks read back
      from the spool live only inside one batch-loop iteration to
      begin with, so there's nothing to null out; they're simply
      garbage the moment the loop moves to the next batch.

  Net effect: at NO point in the pipeline is more than one batch's
  worth of full CodeChunk objects resident in memory at once — not
  during chunking (spooled immediately), not during the call graph /
  repo profile steps (only ChunkMeta is resident), and not during
  embed/upsert (batch-sized, as it already was in 2.1). Peak memory
  for the whole ingest is now O(batch_size + repo_chunk_count *
  sizeof(ChunkMeta)), and a ChunkMeta is a handful of short strings and
  small lists per chunk — orders of magnitude smaller than a CodeChunk
  carrying that chunk's full text/raw_source/docstring. See
  test_ingest_memory_accumulation.py, which asserts directly (via
  gc.get_objects()) that no more than one batch's worth of full
  CodeChunk objects are ever alive at once during a simulated ingest.

  Everything else — the Redis cache, Qdrant upsert semantics, the BM25
  index contents/build call, the call graph algorithm, the repo
  profile algorithm, and the run_ingest return contract — is unchanged.
"""

import asyncio
import gc
import os
import time
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator
import git

from app.models.chunk import CodeChunk, ChunkMeta
from app.ingest.chunk_spool import ChunkSpool
from app.ingest.cloner import clone_repo
from app.engine.ast_chunker import chunk_python_file
from app.engine.embedder import embed_batch
from app.engine.vectordb import upsert_chunks, delete_repo
from app.engine.bm25 import build_index
from app.engine.call_graph import build_called_by
from app.cache.redis_cache import batch_get_embeddings, set_cached_embedding
def _log_memory(stage: str, repo_id: str) -> None:
    """Log current process RSS in MB for Render memory diagnostics."""
    try:
        # Linux reports ru_maxrss in KB.
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"[{repo_id}] MEMORY {stage}: {rss_mb:.1f} MB PEAK_RSS")
    except Exception as e:
        logger.warning(f"[{repo_id}] Could not read memory usage: {e}")

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


async def build_repo_profile(chunk_metas: list[ChunkMeta]) -> str:
    """
    Build a short summary of the repository vocabulary.
    This is stored in Redis and later injected into the HyDE prompt so
    query rewriting uses the repo's own symbols instead of generic Python.

    Takes the lightweight ChunkMeta list (Phase 2.3), not full chunks —
    only name/type/imports are used, which is all ChunkMeta carries.
    """

    class_names = [
        c.name
        for c in chunk_metas
        if c.type == "class"
    ][:20]

    function_names = [
        c.name
        for c in chunk_metas
        if c.type in ("function", "method")
    ][:30]

    imports = []
    for chunk in chunk_metas:
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


def _chunk_all_files(
    local_path: str, repo_id: str, spool: ChunkSpool
) -> tuple[list[ChunkMeta], int]:
    """
    Walks and chunks every .py file in the repo, one file at a time.

    PHASE 2.3: every full CodeChunk is written to `spool` (see
    chunk_spool.py) the instant it's produced and then falls out of
    scope — it is NEVER accumulated into a repo-sized list. What IS
    accumulated into a repo-sized list is `chunk_metas`, the lightweight
    projection (see ChunkMeta in app/models/chunk.py) that the call
    graph and repo profile steps actually need. That's orders of
    magnitude smaller per-chunk than a full CodeChunk, since it carries
    none of text/raw_source/docstring.

    Returns (chunk_metas, file_count).
    """
    chunk_metas: list[ChunkMeta] = []
    file_count = 0

    for rel_path, source in _walk_python_files(local_path):
        file_count += 1

        for chunk in chunk_python_file(source, rel_path, repo_id):
            spool.write(chunk)
            chunk_metas.append(chunk.to_meta())
            # `chunk` (the full CodeChunk, with text/raw_source/
            # docstring) falls out of scope right here — only its
            # pickled bytes on disk and its lightweight ChunkMeta
            # survive past this iteration.

        # `source` falls out of scope on the next loop iteration — only
        # one file's raw text is ever alive at a time.

        if file_count % FILE_LOG_INTERVAL == 0:
            logger.info(
                f"[{repo_id}] chunked {file_count} files so far "
                f"({len(chunk_metas)} chunks)..."
            )

    logger.info(f"[{repo_id}] found {file_count} Python files")
    return chunk_metas, file_count


async def _embed_and_upsert_in_batches(
    repo_id: str,
    chunk_metas: list[ChunkMeta],
    spool: ChunkSpool,
    qdrant_client,
    redis_client,
    cfg,
) -> tuple[int, int, list[dict]]:
    """
    Embeds and upserts all chunks in fixed-size batches so that at most
    one batch's worth of full CodeChunks/texts/vectors/Qdrant points is
    ever resident in memory, instead of the whole repo's worth at once —
    AND incrementally builds the BM25 seed list while doing so.

    PHASE 2.3: full chunks are no longer sitting in an `all_chunks` list
    waiting to be sliced — they live on disk in `spool`, and this
    function reads exactly one batch's worth back at a time, in lockstep
    with the matching slice of `chunk_metas`. Per batch:
      1. Slice this batch's ChunkMeta entries out of chunk_metas (cheap
         — these were already resident).
      2. Read that many full CodeChunk objects back off the spool. This
         relies on the spool having been written in the same order as
         chunk_metas (see chunk_spool.py and _chunk_all_files) — a
         mismatch is a bug, so it's checked, not assumed.
      3. Patch each chunk's called_by from its ChunkMeta: the on-disk
         copy is stale (empty) because it was spooled before
         build_called_by ran against chunk_metas.
      4. Look up that batch's chunk texts in the Redis embedding cache.
      5. Run ONNX embedding only on that batch's cache misses.
      6. Write freshly-computed vectors back to the cache (same as before).
      7. Build Qdrant points for just this batch (this reads text,
         raw_source, and docstring via chunk.to_payload()) and upsert them.
      8. Capture this batch's {"id","text","name"} entries into the
         running bm25_seed list — same shape bm25.build_index has always
         expected, just assembled incrementally instead of in one final
         list comprehension over the whole repo.
      9. Let the batch's full chunks and local texts/vectors/points be
         released before moving on to the next batch — there's no
         `_release_heavy_fields` step anymore, because these chunk
         objects were never going to outlive this loop iteration to
         begin with (they weren't read off disk until step 2, and
         nothing outside this function ever holds a reference to them).

    Returns (cached_count, fresh_count, bm25_seed) — the same
    cached/fresh figures the old single-shot implementation returned
    (just accumulated across batches), plus the BM25 seed list that
    run_ingest passes straight to build_index instead of building it
    itself in a second full pass over all chunks.
    """
    batch_size = getattr(cfg, "ingest_embed_batch_size", DEFAULT_EMBED_BATCH_SIZE) or DEFAULT_EMBED_BATCH_SIZE

    total = len(chunk_metas)
    total_batches = (total + batch_size - 1) // batch_size

    cached_count = 0
    fresh_count = 0
    bm25_seed: list[dict] = []

    for batch_num, batch_start in enumerate(range(0, total, batch_size), start=1):
        batch_metas = chunk_metas[batch_start: batch_start + batch_size]

        # Pull this batch's full chunks (text/raw_source/docstring and
        # all) back off disk — sequential read, same order they were
        # spooled in during chunking, which is guaranteed to line up
        # with batch_metas' order (see chunk_spool.py).
        batch = spool.read_batch(len(batch_metas))
        if len(batch) != len(batch_metas):
            raise RuntimeError(
                f"[{repo_id}] chunk spool out of sync at batch {batch_num}: "
                f"expected {len(batch_metas)} chunks, read {len(batch)}. "
                f"Spool and chunk_metas must be written/read in lockstep."
            )

        # called_by was empty when each chunk was spooled (chunking
        # happens before build_called_by runs against chunk_metas) — the
        # up-to-date value lives on the meta now, so patch it in before
        # anything reads chunk.to_payload().
        for chunk, meta in zip(batch, batch_metas):
            if chunk.id != meta.id:
                raise RuntimeError(
                    f"[{repo_id}] chunk spool out of order at batch {batch_num}: "
                    f"expected chunk id {meta.id!r}, read {chunk.id!r}."
                )
            chunk.called_by = meta.called_by

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
        _log_memory(
            f"after embed/upsert batch {batch_num}/{total_batches}",
            repo_id,
        )

        # Capture this batch's BM25 seed entries.
        bm25_seed.extend(
            {"id": chunk.id, "text": text, "name": chunk.name}
            for chunk, text in zip(batch, batch_texts)
        )

        batch_cached = len(batch_texts) - len(to_embed_texts)
        batch_fresh = len(to_embed_texts)
        cached_count += batch_cached
        fresh_count += batch_fresh

        logger.info(
            f"[{repo_id}] embed batch {batch_num}/{total_batches}: "
            f"{len(batch)} chunks ({batch_fresh} fresh, {batch_cached} cached), "
            f"{batch_start + len(batch)}/{total} chunks processed so far"
        )

        # Explicitly drop batch-local references — including `batch`
        # itself, the full CodeChunk objects just read off the spool —
        # so this batch's chunks, texts, vectors, and points can all be
        # collected before the next batch is read from disk. This is
        # the "release the batch" step that keeps peak memory at
        # O(batch_size) instead of O(repo size).
        del batch, batch_metas, batch_texts, cache_lookup, to_embed_texts
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
    _log_memory("start", repo_id)

    # 1. Clone (or pull) the repo - clone_repo() is a blocking, synchronous
    #    subprocess call (see app/ingest/cloner.py) that can legitimately
    #    take up to CLONE_TIMEOUT_SECONDS (120s). Calling it directly here
    #    would block THIS event loop for that entire duration - freezing
    #    every other request this FastAPI process is serving, not just
    #    this one ingest. asyncio.to_thread() runs it on a worker thread
    #    instead, so the event loop stays free to serve other requests
    #    while this clone/pull is in progress.
    local_path = await asyncio.to_thread(clone_repo, github_url, repo_id, cfg.repos_dir, branch)
    _log_memory("after clone", repo_id)

    # PHASE 2.3: everything from here on runs against a ChunkSpool, which
    # is what holds full CodeChunk objects on disk instead of in RAM for
    # the whole repo at once. The `with` block guarantees the spool's
    # backing temp file is deleted whether this finishes or raises.
    with ChunkSpool() as spool:

        # 2 & 3. Walk every .py file and chunk it with the AST chunker,
        #    one file at a time (see _walk_python_files for why this
        #    doesn't materialize every file's source at once). Every
        #    full chunk is written straight to `spool`; only the
        #    lightweight chunk_metas list comes back resident in RAM
        #    (see _chunk_all_files and ChunkMeta).
        chunk_metas, file_count = _chunk_all_files(local_path, repo_id, spool)
        _log_memory("after chunking", repo_id)

        if not chunk_metas:
            return {"status": "failed", "error": "No chunks extracted — is this a Python repo?"}

        logger.info(f"Extracted {len(chunk_metas)} chunks from {file_count} files")
        # Build a repository profile for HyDE query rewriting.
        repo_profile = await build_repo_profile(chunk_metas)
        _log_memory("after repo profile", repo_id)

        await redis_client.set(
            f"repo_profile:{repo_id}",
            repo_profile,
            ex=60 * 60 * 24 * 30,  # 30 days
        )
        # 3b. Build the call graph (calls -> called_by) across ALL chunks
        #     in this repo, BEFORE embedding. called_by is part of the
        #     embed text's payload (not the embedding itself — see
        #     CodeChunk.to_payload), so it needs to exist before we build
        #     the Qdrant points below. This is inherently repo-global (a
        #     caller can live in a different file than its callee), which
        #     is why chunk_metas has to be fully assembled before this
        #     step — unlike the embed/upsert stage below, which is
        #     batched. Runs against the lightweight chunk_metas, not full
        #     chunks — call_graph.py only ever touched name/type/calls/
        #     called_by, so it needed no changes for this.
        build_called_by(chunk_metas)
        _log_memory("after repo profile", repo_id)

        # 4. Clear any old chunks for this repo (handles re-ingest cleanly)
        await delete_repo(qdrant_client, cfg.qdrant_collection, repo_id)
        

        # Chunking is done — flip the spool from write mode to read mode
        # before the embed/upsert stage starts pulling batches back off it.
        spool.finish_writing()
        _log_memory("after repo profile", repo_id)

        # 5 & 6. Embed (checking the Redis cache first) and upsert into
        #    Qdrant, in fixed-size batches so that at most one batch's
        #    worth of full chunks/texts/vectors/points is ever resident
        #    in memory at once — not the whole repo's worth. This also
        #    incrementally builds bm25_seed. See
        #    _embed_and_upsert_in_batches for details.
        cached_count, fresh_count, bm25_seed = await _embed_and_upsert_in_batches(
            repo_id, chunk_metas, spool, qdrant_client, redis_client, cfg
        )
        logger.info(
            f"[{repo_id}] embedding cache: {cached_count}/{len(chunk_metas)} hits, "
            f"embedded {fresh_count} new chunks"
        )

        # 7. Build the BM25 keyword index for this repo. Same input shape
        #    as before — a list of {"id","text","name"} dicts — and BM25
        #    still needs that full corpus at once to compute its index
        #    (see bm25.py); that part is unavoidable and unchanged. What
        #    changed (Phase 2.2, still true here) is that bm25_seed was
        #    assembled incrementally during the batch loop above, one
        #    batch's worth of text at a time, rather than in a single
        #    final pass over every full chunk at once.
        build_index(repo_id, bm25_seed)
        _log_memory("after BM25", repo_id)

        elapsed = round(time.perf_counter() - t0, 1)
        logger.info(f"Ingest complete for {repo_id}: {len(chunk_metas)} chunks in {elapsed}s")

        commit_hash = git.Repo(local_path).head.commit.hexsha
        _log_memory("ingest complete", repo_id)

        return {
            "status":          "done",
            "chunk_count":     len(chunk_metas),
            "file_count":      file_count,
            "languages":       ["python"],
            "ingest_seconds":  elapsed,
            "embeddings_cached": cached_count,
            "embeddings_fresh":  fresh_count,
            "last_commit":     commit_hash,
        }