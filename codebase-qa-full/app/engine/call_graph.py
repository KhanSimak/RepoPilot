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


def _bare_symbol_name(name: str) -> str:
    """Return the terminal method/function name used by AST call extraction."""
    return name.rsplit(".", 1)[-1]


def build_called_by(all_chunks: list) -> None:
    """
    Mutates each CodeChunk in `all_chunks` in place, filling in `called_by`.

    O(N * avg_calls_per_chunk) — for a repo with a few thousand chunks and
    each chunk calling ~5 other functions on average, this is a few
    thousand dict insertions. Fast, done once per ingest.
    """
    name_to_chunks: dict[str, list] = {}
    for chunk in all_chunks:
        # Calls extracted from ``obj.method()`` are bare names (``method``),
        # while method chunks are stored as ``Class.method``.  Index both
        # representations so graph inversion uses the same identifier form as
        # traversal.
        name_to_chunks.setdefault(chunk.name, []).append(chunk)
        bare_name = _bare_symbol_name(chunk.name)
        if bare_name != chunk.name:
            name_to_chunks.setdefault(bare_name, []).append(chunk)

    for chunk in all_chunks:
        for callee_name in chunk.calls:
            # Prefer the caller's own class for ``self.method()``-shaped
            # bare calls before considering repository-wide bare matches.
            scoped_name = None
            if "." in chunk.name and "." not in callee_name:
                scoped_name = f"{chunk.name.rsplit('.', 1)[0]}.{callee_name}"
            candidates = name_to_chunks.get(scoped_name, []) if scoped_name else []
            if not candidates:
                exact = name_to_chunks.get(callee_name, [])
                exact_named = [c for c in exact if c.name == callee_name]
                candidates = exact_named or exact
            for callee_chunk in candidates:
                if chunk.name not in callee_chunk.called_by:
                    callee_chunk.called_by.append(chunk.name)

    total_edges = sum(len(c.called_by) for c in all_chunks)
    logger.info(f"Call graph built: {len(all_chunks)} nodes, {total_edges} called_by edges")


def expand_by_graph(
    entry_chunks: list[dict],
    name_index: dict[str, list[dict]],
    qualified_index,
    depth: int = 1,
    direction: str = "both",
    
    max_expanded: int = 10,
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

            for name in sorted(neighbor_names):

                current_file = chunk["file"]

                # Resolve a bare method call to its current class first.
                scoped_name = None
                if "." in chunk.get("name", "") and "." not in name:
                    scoped_name = f"{chunk['name'].rsplit('.', 1)[0]}.{name}"

                neighbor = (
                    qualified_index.get((current_file, scoped_name))
                    if scoped_name else None
                )
                if neighbor:
                    neighbors = [neighbor]
                else:
                    exact_neighbors = [
                        candidate for candidate in name_index.get(name, [])
                        if candidate.get("name") == name
                    ]
                    neighbors = exact_neighbors or name_index.get(name, [])
                neighbors = sorted(
                    neighbors,
                    key=lambda candidate: (
                        candidate.get("file", ""),
                        candidate.get("line_start", 0),
                        candidate.get("id", ""),
                    ),
                )

                for neighbor in neighbors:

                    normalized = "/" + neighbor["file"].replace("\\", "/")

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
                        parent_score - 0.15,
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
            c.get(
                "graph_score",
                c.get("rerank_score", c.get("score", 0.0)),
            ),
            -c.get("graph_distance", 0),
        ),
        reverse=True,
    )
    if result:
        logger.debug(
            "Graph expansion returned %d chunks; first=%s",
            len(result),
            result[0].get("name"),
        )

    return result

def build_name_index(chunks):

    index = {}
    qualified = {}
    decorator_index = {}

    for chunk in chunks:
       for decorator in chunk.get("decorators", []):
          decorator_index.setdefault(decorator, []).append(chunk)

    for chunk in chunks:
        name = chunk["name"]
        index.setdefault(name, []).append(chunk)
        bare_name = _bare_symbol_name(name)
        if bare_name != name:
            index.setdefault(bare_name, []).append(chunk)

        key = (
            chunk["file"],
            name,
        )

        qualified[key] = chunk
        # Same-file bare-name lookup resolves calls such as
        # ``self.add_url_rule()`` to ``Scaffold.add_url_rule``.
        qualified.setdefault((chunk["file"], bare_name), chunk)

    return index, qualified,decorator_index
