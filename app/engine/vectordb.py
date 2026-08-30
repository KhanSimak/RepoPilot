"""
vectordb.py — Qdrant client wrapper

Qdrant is a vector database written in Rust. We use ONE collection for
all repos and filter by repo_id at query time, rather than creating a
separate collection per repo. This scales better — Qdrant collections
have fixed overhead, and pre-filtering by a payload field before the
ANN (approximate nearest neighbor) scan is fast and well-supported.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
import logging

logger = logging.getLogger(__name__)


async def init_qdrant(cfg) -> AsyncQdrantClient:
    """Called once at startup. Creates the collection if it doesn't exist yet."""
    logger.debug("Initializing Qdrant client at %s", cfg.qdrant_url)
    
    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key,)

    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if cfg.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=VectorParams(size=cfg.vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {cfg.qdrant_collection}")
    else:
        logger.info(f"Qdrant collection already exists: {cfg.qdrant_collection}")

    return client


def _repo_filter(repo_id: str) -> Filter:
    """
    Pre-filter by repo_id. This is applied BEFORE Qdrant scans for nearest
    neighbors — so searching repo A never even looks at repo B's vectors,
    no matter how many repos are in the same collection.
    """
    return Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))])


async def upsert_chunks(client: AsyncQdrantClient, collection: str, chunks: list[dict], batch_size: int = 100) -> int:
    """
    Insert (or update) vectors in batches.
    Each item in `chunks` is {"id": ..., "vector": [...], "payload": {...}}.
    """
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        points = [PointStruct(id=c["id"], vector=c["vector"], payload=c["payload"]) for c in batch]
        await client.upsert(collection_name=collection, points=points)
        total += len(batch)
    return total


async def search(client: AsyncQdrantClient, collection: str, vector: list[float], repo_id: str, top_k: int = 20) -> list[dict]:
    """
    Vector similarity search, restricted to one repo.

    NOTE: default top_k bumped to 20 in Phase 2 (was 5 in Phase 1). Hybrid
    search needs a wider candidate pool BEFORE fusion narrows it back down —
    fusing two lists of 5 loses a lot of the value RRF provides. We fuse
    20+20 down to a final top_k chosen by the caller (search.py).
    """
    results = await client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=_repo_filter(repo_id),
        with_payload=True,
    )
    return [{"id": str(r.id), "score": round(r.score, 4), **r.payload} for r in results]


async def delete_repo(client: AsyncQdrantClient, collection: str, repo_id: str):
    """Remove all chunks belonging to one repo — used before re-ingesting."""
    await client.delete(collection_name=collection, points_selector=_repo_filter(repo_id))


async def retrieve_by_ids(client: AsyncQdrantClient, collection: str, ids: list[str]) -> dict[str, dict]:
    """
    Fetch chunk payloads by their exact IDs (no similarity search involved).

    WHY THIS IS NEEDED FOR HYBRID SEARCH:
      RRF fusion (fusion.py) merges vector hits and BM25 hits by ID. Some IDs
      come ONLY from BM25 — a chunk that matched an exact keyword but wasn't
      in the vector search's top-K, because semantically it didn't look close
      enough to the query embedding. We still need that chunk's full payload
      (the actual code) to build the LLM's context. retrieve() (Qdrant's
      points API) does this — a direct lookup by ID, no ANN scan involved.
    """
    if not ids:
        return {}
    points = await client.retrieve(collection_name=collection, ids=ids, with_payload=True)
    return {str(p.id): {"id": str(p.id), "score": 0.0, **p.payload} for p in points}


async def count_repo_chunks(client: AsyncQdrantClient, collection: str, repo_id: str) -> int:
    result = await client.count(collection_name=collection, count_filter=_repo_filter(repo_id), exact=True)
    return result.count


async def scroll_repo_chunks(client: AsyncQdrantClient, collection: str, repo_id: str, batch_size: int = 256) -> list[dict]:
    """
    Fetch EVERY chunk belonging to one repo — not a similarity search,
    just a paginated full scan filtered by repo_id.

    WHY THIS IS NEEDED: call graph expansion (call_graph.py) needs a
    name_index built from the WHOLE repo's chunks, because a caller or
    callee could live in any file. Qdrant's `scroll` API is the right
    tool here — it's designed for exactly this "give me everything
    matching this filter, paginated" use case, as opposed to `search`
    which always ranks by vector similarity.
    """
    all_points, offset = [], None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=_repo_filter(repo_id),
            limit=batch_size,
            offset=offset,
            with_payload=True,
        )
        all_points.extend({"id": str(p.id), "score": 0.0, **p.payload} for p in points)
        if offset is None:
            break
    return all_points
