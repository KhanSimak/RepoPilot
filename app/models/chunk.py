"""
chunk.py — CodeChunk: the core data structure of the whole project

Every other file reads or writes this shape. Get this right first.

Why a dataclass and not a Pydantic model?
  CodeChunk is created thousands of times per repo ingest (one per function).
  Dataclasses have near-zero overhead compared to Pydantic's validation layer.
  Pydantic is reserved for API request/response boundaries (see schemas/).
"""

from dataclasses import dataclass, field
import uuid
import hashlib


@dataclass
class CodeChunk:
    # ── Identity ─────────────────────────────────────────────────
    id:           str
    repo_id:      str
    text:         str    # enriched text used for embedding (file + name + docstring + code)
    raw_source:   str  
       # just the code itself, used for display to the user
    qualified_name: str
    # ── Location ─────────────────────────────────────────────────
    name:         str     # function/class/method name
    type:         str     # "function" | "class" | "method"
    file:         str     # relative path inside the repo
    language:     str
    line_start:   int
    line_end:     int
    decorators: list[str] = field(default_factory=list)


    registered_by: list[str] = field(default_factory=list)
    calls: list = field(default_factory=list)
    called_by: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    

    # ── Semantics — extracted once at ingest time ───────────────
    docstring:    str  = ""
 # module-level imports in this file
    complexity:   int  = 1                              # cyclomatic complexity

    @staticmethod
    def make_id(repo_id: str, file: str, name: str, line: int) -> str:
        """
        Deterministic ID — same file+name+line always produces the same ID.
        This matters later (Phase 2+) for incremental re-ingest: if a chunk's
        ID is unchanged, we can skip re-embedding it.
        """
        raw = f"{repo_id}:{file}:{name}:{line}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

    @staticmethod
    def content_hash(source: str) -> str:
        """Hash of the raw code — lets us detect if a function's body actually changed."""
        return hashlib.md5(source.encode()).hexdigest()

    def to_meta(self) -> "ChunkMeta":
        """Project this chunk down to the lightweight fields ChunkMeta needs."""
        return ChunkMeta(
            id=self.id,
            name=self.name,
            type=self.type,
            file=self.file,
            calls=self.calls,
            imports=self.imports,
        )

    def to_payload(self) -> dict:
        """Everything we store in Qdrant alongside the vector."""
        return {
            "repo_id":     self.repo_id,
            "qualified_name": self.qualified_name,
            
            "name":        self.name,
            "type":        self.type,
            "file":        self.file,
            "language":    self.language,
            "line_start":  self.line_start,
            "line_end":    self.line_end,
            "docstring":   self.docstring,
            "calls":       self.calls,
            "called_by":   self.called_by,
            "decorators": self.decorators,

            "registered_by": self.registered_by,
            "imports":     self.imports,
            "complexity":  self.complexity,
            "text":        self.text,
            "raw_source":  self.raw_source,
            "content_hash": self.content_hash(self.raw_source),
        }


@dataclass
class ChunkMeta:
    """
    The lightweight, repo-global-resident companion to CodeChunk.

    PHASE 2.3 (fixes the OOM that survived Phase 2.1/2.2 — see the
    "Phase 2.3" section of app/ingest/pipeline.py's module docstring):
    every full CodeChunk gets spooled to disk (app/ingest/chunk_spool.py)
    the instant it's created, instead of sitting in an `all_chunks` list
    for the whole repo, for the whole ingest run. ChunkMeta is what stays
    resident in RAM instead, for every chunk, for the whole run — because
    the two repo-global steps between chunking and embedding only ever
    need a handful of small fields, never text/raw_source/docstring:

      - call_graph.build_called_by needs name/type/calls/called_by
      - pipeline.build_repo_profile needs name/type/imports

    `called_by` starts empty (nothing calls anything yet at chunk time)
    and is filled in place by build_called_by, exactly like it would be
    on a full CodeChunk. Because it's the only field that changes after
    chunking, it's also the only field that has to be patched back onto
    the full CodeChunk when that chunk is later read off disk for its
    embed/upsert batch (see pipeline._embed_and_upsert_in_batches) — the
    on-disk copy's called_by is stale (still empty) by construction.

    `id` and `file` aren't read by build_called_by or build_repo_profile,
    but are kept here anyway: `id` lets _embed_and_upsert_in_batches
    sanity-check that a disk-read batch lines up with the meta batch
    it's paired with, and `file` is cheap and useful for logging.
    """
    id:        str
    name:      str
    type:      str
    file:      str
    calls:     list = field(default_factory=list)
    called_by: list = field(default_factory=list)
    imports:   list = field(default_factory=list)