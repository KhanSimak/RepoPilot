"""
call_graph.py — building and walking the codebase's call graph

THE CENTRAL IDEA OF THIS WHOLE PHASE:
  A codebase is not a bag of independent chunks — it's a graph. Functions
  call other functions. "How does payment processing work?" is not answered
  by ONE function; it's answered by an entry point PLUS everything it calls
  PLUS (often) everything that calls it. Phases 1-3 only ever return isolated
  chunks. This file is what turns "find a function" into "understand a flow."

STEP 1 — INVERSION (calls -> called_by)
  Every chunk already knows what it calls (extracted by ast_chunker.py at
  parse time — that's a per-function, local operation). called_by is the
  GLOBAL inverse of that: "who calls me?" can only be answered once you've
  seen every chunk in the repo, because the caller could be in any file.
  So inversion happens once, at the END of ingest, across all chunks together.

STEP 2 — MATCHING CAVEAT (read this before you trust the graph too much)
  `calls` stores bare names extracted from the AST — e.g. a call like
  `self._get_user(x)` is recorded as "_get_user", not as a fully qualified
  symbol. When we invert, we match callee names against chunk NAMES across
  the whole repo. This means two unrelated classes that both happen to
  define a method called `save()` will be treated as the same callee.
  This is a deliberate, honest simplification — full symbol resolution
  would need a real type checker (or something like Jedi/LSP). For RAG
  retrieval purposes, "approximately right call graph" still meaningfully
  improves flow-style answers; it does not need to be a perfect compiler.

STEP 3 — GRAPH EXPANSION AT QUERY TIME
  Given an entry chunk, walk `calls` (downstream/callees) and/or `called_by`
  (upstream/callers) outward to a fixed depth using BFS. This is what lets
  "explain the payment flow" return the entry point AND its immediate
  collaborators, instead of just the one function that ranked #1.
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def _resolve_callee_chunks(callee_name: str, name_to_chunks: dict[str, list]) -> list:
    """
    Resolve a bare callee name extracted from a call site to the chunk(s)
    it actually invokes.

    A direct name match covers the common case. One systematic mismatch:
    a constructor call `ClassName(...)` is extracted from the AST as a
    call to the bare class name ("Flask", "Foo", whatever the class is
    called) — see the STEP 2 matching caveat at the top of this file. But
    the code that actually runs when that call executes is the class's
    __init__ method, whose chunk is named with this repo's qualified
    method naming scheme, "ClassName.__init__" — never the bare class
    name. So a direct name-to-name match alone never connects an
    instantiation site to its constructor: EVERY class's __init__ is
    call-graph-isolated the moment it's only ever reached via `ClassName(
    ...)`, not because of anything Flask-specific but because of this
    generic bare-name-vs-qualified-name mismatch. Resolve it once, here,
    for any class in the repo: if the bare name matches a class chunk,
    also route the edge to that class's "<ClassName>.__init__" chunk(s),
    if one exists.
    """
    matches = name_to_chunks.get(callee_name, [])
    resolved = list(matches)

    for candidate in matches:
        if getattr(candidate, "type", None) == "class":
            for init_chunk in name_to_chunks.get(f"{callee_name}.__init__", []):
                if init_chunk not in resolved:
                    resolved.append(init_chunk)

    return resolved


def build_called_by(all_chunks: list) -> None:
    """
    Mutates each CodeChunk in `all_chunks` in place, filling in `called_by`.

    O(N * avg_calls_per_chunk) — for a repo with a few thousand chunks and
    each chunk calling ~5 other functions on average, this is a few
    thousand dict insertions. Fast, done once per ingest.
    """
    name_to_chunks: dict[str, list] = {}
    for chunk in all_chunks:
        name_to_chunks.setdefault(chunk.name, []).append(chunk)

    for chunk in all_chunks:
        for callee_name in chunk.calls:
            for callee_chunk in _resolve_callee_chunks(callee_name, name_to_chunks):
                if chunk.name not in callee_chunk.called_by:
                    callee_chunk.called_by.append(chunk.name)

    total_edges = sum(len(c.called_by) for c in all_chunks)
    logger.info(f"Call graph built: {len(all_chunks)} nodes, {total_edges} called_by edges")


def _resolve_graph_neighbors(
    current_chunk: dict,
    callee_name: str,
    name_index: dict[str, list[dict]],
    qualified_index: dict[tuple[str, str], dict],
) -> list[dict]:
    """
    Resolve a call-graph edge from a call-site name to actual chunks.

    AST extraction stores calls as bare names:
        self.dispatch_request() -> "dispatch_request"

    Chunks may be stored with qualified names:
        "Flask.dispatch_request"

    Resolve in increasingly broad scopes so bare names can reach their
    qualified definitions without requiring a full type checker.
    """

    resolved = []
    seen_ids = set()

    def add(chunks):
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                resolved.append(chunk)

    current_file = current_chunk["file"]
    current_name = current_chunk.get("name", "")

    # ---------------------------------------------------------
    # 1. Exact qualified/same-file lookup
    # ---------------------------------------------------------
    exact = qualified_index.get((current_file, callee_name))
    if exact is not None:
        add([exact])

    # ---------------------------------------------------------
    # 2. Direct exact name lookup
    #
    # Useful for genuinely unqualified functions/classes.
    # ---------------------------------------------------------
    add(name_index.get(callee_name, []))

    # ---------------------------------------------------------
    # 3. Same owner/class lookup
    #
    # Example:
    #   current = Flask.full_dispatch_request
    #   call    = dispatch_request
    #
    #   -> Flask.dispatch_request
    # ---------------------------------------------------------
    if "." in current_name:
        owner = current_name.rsplit(".", 1)[0]
        qualified_name = f"{owner}.{callee_name}"
        add(name_index.get(qualified_name, []))

    # ---------------------------------------------------------
    # 4. Same-file qualified suffix lookup
    #
    # Handles cases where the caller's owner cannot be inferred
    # cleanly but the target exists in the same module.
    # ---------------------------------------------------------
    suffix = f".{callee_name}"

    for chunks in name_index.values():
        for chunk in chunks:
            if chunk.get("file") != current_file:
                continue

            if chunk.get("name", "").endswith(suffix):
                add([chunk])

    # ---------------------------------------------------------
    # 5. Global qualified suffix fallback
    #
    # This is deliberately last because:
    #
    #   foo.save()
    #
    # could correspond to several Class.save methods.
    #
    # The earlier same-owner/same-file resolutions are preferred.
    # ---------------------------------------------------------
    if not resolved:
        for chunks in name_index.values():
            for chunk in chunks:
                if chunk.get("name", "").endswith(suffix):
                    add([chunk])

    # ---------------------------------------------------------
    # Constructor resolution
    #
    # ClassName(...) -> ClassName.__init__
    # ---------------------------------------------------------
    for chunk in list(resolved):
        if chunk.get("type") == "class":
            init_name = f"{callee_name}.__init__"
            add(name_index.get(init_name, []))

    return resolved

def expand_by_graph(
    entry_chunks: list[dict],
    name_index: dict[str, list[dict]],
    qualified_index,
    depth: int = 1,
    direction: str = "both",
    
    max_expanded: int = 15,
) -> list[dict]:

    visited_ids = {c["id"] for c in entry_chunks}

    result = []
    frontier = []

    for chunk in entry_chunks:
        chunk = dict(chunk)
        chunk["graph_distance"] = 0
        chunk["graph_score"] = chunk.get(
            "rerank_score",
            chunk.get("score", 0.0),
        )

        result.append(chunk)
        frontier.append(chunk)

    # ---------------------------------------------------------
    # BFS
    # ---------------------------------------------------------

    for current_depth in range(1, depth + 1):

        next_frontier = []

        for chunk in frontier:

            # Don't expand classes
            if chunk.get("type") == "class":
                continue

            neighbor_names = set()

            if direction in ("callees", "both"):
                neighbor_names.update(chunk.get("calls", []))

            if direction in ("callers", "both"):
                neighbor_names.update(chunk.get("called_by", []))
            

            # Flask special cases

            for name in neighbor_names:

                neighbors = _resolve_graph_neighbors(
                    current_chunk=chunk,
                    callee_name=name,
                    name_index=name_index,
                    qualified_index=qualified_index,
                )

                for neighbor in neighbors:

                    normalized = neighbor["file"].replace("\\", "/")

                    # Ignore tests/examples
                    if (
                        "/tests/" in normalized
                        or "/examples/" in normalized
                    ):
                        continue

                    # Already seen
                    if neighbor["id"] in visited_ids:
                        continue

                    visited_ids.add(neighbor["id"])

                    neighbor = dict(neighbor)

                    neighbor["graph_distance"] = current_depth

                    parent_score = chunk.get(
                        "graph_score",
                        chunk.get(
                            "rerank_score",
                            chunk.get("score", 0.0),
                        ),
                    )

                    neighbor["graph_score"] = max(
                        parent_score - 0.05,
                        0,
                    )

                    result.append(neighbor)
                    next_frontier.append(neighbor)

                    if len(result) - len(entry_chunks) >= max_expanded:
                        break

                if len(result) - len(entry_chunks) >= max_expanded:
                    break

            if len(result) - len(entry_chunks) >= max_expanded:
                break

        frontier = next_frontier

        if not frontier:
            break

    result.sort(
        key=lambda c: (
            int(c.get("graph_distance", 0) == 1),
            c.get("graph_score", c.get("rerank_score", c.get("score", 0.0))),
            -c.get("graph_distance", 0),
        ),
        reverse=True,
    )

    return result

def build_name_index(chunks):

    index = {}
    qualified = {}
    decorator_index = {}

    for chunk in chunks:
       for decorator in chunk["decorators"]:
          decorator_index.setdefault(decorator, []).append(chunk)

    for chunk in chunks:

        index.setdefault(chunk["name"], []).append(chunk)

        key = (
            chunk["file"],
            chunk["name"],
        )

        qualified[key] = chunk

    return index, qualified,decorator_index