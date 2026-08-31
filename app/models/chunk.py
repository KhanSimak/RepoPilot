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
