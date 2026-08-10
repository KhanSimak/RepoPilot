"""
routers/graph.py — backend for the Call Graph Explorer frontend.

Two endpoints:
  GET /{repo_id}/symbols            — search chunk names (for the explorer's
                                       search box, since a user won't usually
                                       know an exact function name upfront)
  GET /{repo_id}/symbols/{name}/graph — nodes+edges for a chosen symbol,
                                       reusing the SAME expand_by_graph()
                                       your retrieval pipeline already uses —
                                       this endpoint doesn't compute anything
                                       new, it just exposes existing data.
"""
import logging

from fastapi import APIRouter, Request, HTTPException, Query as QParam

from app.engine.call_graph import expand_by_graph, build_name_index
from app.engine.vectordb import scroll_repo_chunks
from app.routers.repos import get_registry

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_app_state(request):
    return request.app.state.settings, request.app.state.qdrant


@router.get("/{repo_id}/symbols")
async def search_symbols(repo_id: str, request: Request, q: str = QParam("", min_length=0), limit: int = QParam(20, ge=1, le=100)):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")

    cfg, qdrant = _get_app_state(request)
    all_chunks = await scroll_repo_chunks(qdrant, cfg.qdrant_collection, repo_id)

    q_lower = q.lower()
    matches = [
        {"name": c["name"], "type": c.get("type"), "file": c.get("file")}
        for c in all_chunks
        if c.get("name") and q_lower in c["name"].lower()
    ]
    # De-dupe by name (same name can exist in multiple files/classes)
    seen = set()
    deduped = []
    for m in matches:
        if m["name"] not in seen:
            seen.add(m["name"])
            deduped.append(m)

    return {"symbols": deduped[:limit], "total_matches": len(deduped)}


@router.get("/{repo_id}/symbols/{name}/graph")
async def get_symbol_graph(repo_id: str, name: str, request: Request, depth: int = QParam(2, ge=1, le=4)):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")

    cfg, qdrant = _get_app_state(request)
    all_chunks = await scroll_repo_chunks(qdrant, cfg.qdrant_collection, repo_id)
    name_index, qualified_index, _ = build_name_index(all_chunks)

    entry_chunks = name_index.get(name, [])
    if not entry_chunks:
        raise HTTPException(404, f"No symbol named '{name}' found in this repo")
    
    expanded = expand_by_graph(
        entry_chunks, name_index, qualified_index,
        depth=depth, direction="both", max_expanded=40,
    )
    entry_ids = {c["id"] for c in entry_chunks}

    nodes = [
        {
            "id": c["id"],
            "name": c.get("name"),
            "type": c.get("type"),
            "file": c.get("file"),
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
            "is_entry": c["id"] in entry_ids,
        }
        for c in expanded
    ]

    # Edges: only between nodes actually present in this expanded set —
    # a node's full calls/called_by may reference symbols outside max_expanded,
    # which we don't draw an edge to since there'd be nothing to point at.
    node_names_present = {n["name"] for n in nodes}
    edges = []
    seen_edges = set()
    for c in expanded:
        for callee in c.get("calls", []):
            if callee in node_names_present:
                key = (c["name"], callee, "calls")
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({"source": c["name"], "target": callee, "kind": "calls"})

    return {"entry_symbol": name, "depth": depth, "nodes": nodes, "edges": edges}
