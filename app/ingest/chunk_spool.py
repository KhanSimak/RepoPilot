"""
chunk_spool.py — spools full CodeChunk objects to disk during ingest

PHASE 2.3 FIX (see app/ingest/pipeline.py's module docstring for the
full history of this bug): the previous fix (2.1/2.2) bounded the
embed/upsert stage's memory to O(batch_size), but `all_chunks` still
held every chunk's full text/raw_source/docstring, for the WHOLE repo,
for the entire window between "all files chunked" and "call graph
built" — and, before 2.2, all the way through the embed/upsert loop
and the BM25 build too.

ChunkSpool removes that peak entirely: full CodeChunk objects are
written to a temp file the instant they're produced (during chunking)
and never held in a repo-sized list in RAM. Only a lightweight
ChunkMeta per chunk (see app/models/chunk.py) stays resident for the
call-graph and repo-profile steps. Full chunks come back off disk only
during the embed/upsert stage, one batch at a time.

Why pickle + a single append-only stream, not a file-per-chunk or a
seek/offset index:
  CodeChunk is a plain dataclass of built-in types (str, int, list) —
  pickle round-trips it exactly, with no schema to maintain here.
  Writing every chunk to ONE file via repeated pickle.dump() calls, and
  reading it back via repeated pickle.load() calls, returns objects in
  exactly the order they were written — no offset bookkeeping needed,
  because both the writer (chunking, file-by-file, chunk-by-chunk) and
  the reader (embed/upsert batching) walk the repo's chunks in the same
  order. That's the same ordering guarantee the old in-memory
  `all_chunks` list gave for free; the spool just gives it back from
  disk instead of RAM.

Lifecycle (see pipeline.run_ingest for the real usage):
  1. write() once per chunk, during chunking.
  2. finish_writing() once, when chunking is done.
  3. read_batch(n) repeatedly, during the embed/upsert stage.
  4. close() (or use as a context manager) to delete the temp file.
Steps 1-2 and 3 never interleave — chunking finishes for the whole repo
before any batch is read back, same as the old pipeline's ordering.
"""

import pickle
import tempfile


class ChunkSpool:
    """Disk-backed, write-then-read-once queue of full CodeChunk objects."""

    def __init__(self):
        # A plain (unnamed) temp file: we never need to reopen it by
        # path — the same handle is used for both writing and, later,
        # reading — and it's removed automatically as soon as it's
        # closed, so a crash mid-ingest can't leave a stray large file
        # behind on a disk-constrained Render instance.
        self._file = tempfile.TemporaryFile(prefix="chunk_spool_")
        self._closed = False
        self._writing_done = False

    def write(self, chunk) -> None:
        """Append one full chunk to the spool. Call only before finish_writing()."""
        if self._writing_done:
            raise RuntimeError("ChunkSpool.write() called after finish_writing()")
        pickle.dump(chunk, self._file, protocol=pickle.HIGHEST_PROTOCOL)

    def finish_writing(self) -> None:
        """
        Call once, after every chunk has been written and before the
        first read_batch(). Flushes and rewinds the spool so reads start
        from the first chunk written.
        """
        self._file.flush()
        self._file.seek(0)
        self._writing_done = True

    def read_batch(self, count: int) -> list:
        """
        Read the next `count` chunks off disk, in the order they were
        written. Returns fewer than `count` only if the spool has fewer
        chunks left (the final batch of the repo).
        """
        if not self._writing_done:
            raise RuntimeError("ChunkSpool.read_batch() called before finish_writing()")

        batch = []
        for _ in range(count):
            try:
                batch.append(pickle.load(self._file))
            except EOFError:
                break
        return batch

    def close(self) -> None:
        """Delete the underlying temp file. Safe to call more than once."""
        if not self._closed:
            self._file.close()  # TemporaryFile deletes itself on close
            self._closed = True

    def __enter__(self) -> "ChunkSpool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()