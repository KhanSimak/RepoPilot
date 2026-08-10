"""
agent/tools.py — thin wrappers around the EXISTING retrieve() and
expand_by_graph() functions. Nothing here reimplements retrieval; it just
adapts your existing app/query/retriever.py and app/engine/call_graph.py
functions to the shape the agent loop's nodes call.
"""
import logging
import re

from app.query.retriever import retrieve
from app.engine.call_graph import expand_by_graph, build_name_index
from app.engine.vectordb import scroll_repo_chunks

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "a", "an", "and", "are", "does", "do", "for", "from", "how",
    "in", "into", "is", "of", "on", "or", "the", "to", "what",
    "where", "which", "with", "why",
}


def _extract_phrases(query: str) -> list[str]:
    """Keep searchable words and code-shaped identifiers from an agent query."""
    phrases = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        query,
    )
    return [phrase for phrase in phrases if phrase.lower() not in _STOP_WORDS]


async def retrieval_tool(
    query: str,
    repo_id: str,
    intent: str,
    qdrant_client,
    redis_client,
    cfg,
    top_k: int = 20,
) -> list[dict]:
    """
    One retrieval pass, reusing your existing retrieve() exactly as the
    non-agentic pipeline does — HyDE snippet is just the query itself here
    (the agent's own reasoning IS the query refinement; a second nested
    HyDE call would be redundant and slower).
    """
    return await retrieve(
        question=query,
        hyde_snippet=query,
        phrases=_extract_phrases(query),
        intent=intent,
        repo_id=repo_id,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        cfg=cfg,
        top_k=top_k,
        graph_expand=False,   # the agent loop does its OWN expansion via expand_graph_tool
    )


async def expand_graph_tool(
    entry_chunks: list[dict],
    repo_id: str,
    qdrant_client,
    cfg,
    depth: int = 1,
    direction: str = "both",
    name_index: dict | None = None,
    qualified_index: dict | None = None,
) -> tuple[list[dict], dict, dict]:
    """
    Same expand_by_graph() the static pipeline uses, callable repeatedly
    from different/updated entry_chunks — this is what makes expansion
    iterative instead of the single fixed-depth pass retriever.py does today.
    """
    if name_index is None or qualified_index is None:
        all_repo_chunks = await scroll_repo_chunks(qdrant_client, cfg.qdrant_collection, repo_id)
        name_index, qualified_index, _ = build_name_index(all_repo_chunks)

    expanded = expand_by_graph(
        entry_chunks,
        name_index,
        qualified_index,
        depth=depth,
        direction=direction,
        max_expanded=15,
    )
    return expanded, name_index, qualified_index
