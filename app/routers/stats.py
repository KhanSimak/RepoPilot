"""
stats.py — observability endpoint

WHY THIS ENDPOINT EXISTS:
  "We added caching" is a claim. "Our cache hit rate is 41.2%" is evidence.
  This endpoint exposes Redis's own internal hit/miss counters so you can
  show, not just tell, that the two-layer cache is doing real work.

  Try this: ask the same question twice via /repos/{id}/search, then hit
  this endpoint — you'll see hits go up by exactly 1 (the embedding cache)
  plus the query cache hit on the second call.
"""

from fastapi import APIRouter, Request
from app.cache.redis_cache import get_cache_stats

router = APIRouter()


@router.get("/cache")
async def cache_stats(request: Request):
    redis_client = request.app.state.redis
    stats = await get_cache_stats(redis_client)
    return {
        "redis_stats": stats,
        "note": "hit_rate_pct reflects ALL Redis GETs (both query cache and embedding cache combined).",
    }
