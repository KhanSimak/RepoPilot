"""
redis_cache.py — two-layer cache

LAYER 1 — query result cache (short TTL, 5 minutes)
  Key:   md5(repo_id + question + filters)
  Value: the full search response (answer + sources + latency)
  Why a TTL and not permanent: the repo can be re-ingested with new code,
  and a stale "answer" pointing at deleted code is worse than a slow
  fresh answer. 5 minutes balances staleness risk against hit rate —
  in practice, repeat questions tend to cluster in short bursts (a whole
  team asking the same thing within the same hour while debugging
  something together).

LAYER 2 — embedding cache (no TTL, effectively permanent)
  Key:   md5(text)
  Value: the embedding vector
  Why permanent: the text being embedded never changes meaning. If you
  embedded "where is auth handled" yesterday, you'll get the exact same
  vector embedding it again today — the model is deterministic and the
  input text is identical. There's no staleness risk, so we cache forever
  and let Redis's own memory eviction policy (maxmemory-policy allkeys-lru
  in docker-compose.yml) clean up old entries if memory pressure occurs.

WHY TWO SEPARATE LAYERS INSTEAD OF ONE CACHE:
  They have completely different lifetimes and completely different
  cost/benefit tradeoffs. A query cache hit skips the ENTIRE pipeline
  (embed + search + LLM call) — huge win, but only valid for 5 minutes.
  An embedding cache hit skips just the ~25ms ONNX call but is valid
  forever. Mixing them into one cache with one TTL would either expire
  embeddings needlessly or keep stale answers too long.
"""

import json
import hashlib
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)


async def get_repo_profile(redis_client, repo_id: str) -> str:
    value = await redis_client.get(f"repo_profile:{repo_id}")

    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


async def init_redis(cfg) -> redis.Redis:
    client = redis.from_url(cfg.redis_url, encoding="utf-8", decode_responses=True)
    
    await client.ping()
    logger.info("Redis connected")
    return client



def _query_key(repo_id: str, question: str, top_k: int) -> str:
    raw = f"q:{repo_id}:{question}:{top_k}"
    return "qcache:" + hashlib.md5(raw.encode()).hexdigest()


def _embed_key(text: str) -> str:
    return "emb:" + hashlib.md5(text.encode()).hexdigest()


# ── Layer 1: query result cache ──────────────────────────────────────────────

async def get_cached_query(r: redis.Redis, repo_id: str, question: str, top_k: int) -> dict | None:
    try:
        raw = await r.get(_query_key(repo_id, question, top_k))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"Cache read failed (treating as miss): {e}")
        return None

    
async def set_cached_query(r: redis.Redis, repo_id: str, question: str, top_k: int, result: dict, ttl_seconds: int = 300) -> None:
    try:
        key = _query_key(repo_id, question, top_k)
        pipe = r.pipeline()
        pipe.setex(key, ttl_seconds, json.dumps(result))
        pipe.sadd(f"qcache_repo:{repo_id}", key)
        pipe.expire(f"qcache_repo:{repo_id}", ttl_seconds)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"Cache write failed (non-fatal): {e}")


async def clear_cached_queries_for_repo(r: redis.Redis, repo_id: str) -> None:
    """Invalidate answers produced against an older ingest of this repo."""
    try:
        index_key = f"qcache_repo:{repo_id}"
        keys = await r.smembers(index_key)
        if keys:
            await r.delete(*keys)
        await r.delete(index_key)
    except Exception as e:
        logger.warning(f"Query-cache invalidation failed (non-fatal): {e}")


# ── Layer 2: embedding cache ─────────────────────────────────────────────────

async def get_cached_embedding(r: redis.Redis, text: str) -> list[float] | None:
    try:
        raw = await r.get(_embed_key(text))
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def set_cached_embedding(r: redis.Redis, text: str, vector: list[float]) -> None:
    try:
        await r.set(_embed_key(text), json.dumps(vector))   # no TTL — permanent
    except Exception as e:
        logger.warning(f"Embedding cache write failed (non-fatal): {e}")


async def batch_get_embeddings(r: redis.Redis, texts: list[str]) -> dict[str, list[float] | None]:
    """
    Look up MANY embeddings in ONE round trip using Redis pipelining,
    instead of N separate await r.get() calls (which would be N round trips).

    This matters at ingest time: a repo with 500 chunks means 500 potential
    cache hits if you're re-ingesting after a small code change. Checking
    each one with a separate network round trip at ~1ms each = 500ms wasted
    purely on network latency. Pipelining batches all 500 GETs into a single
    request/response round trip.
    """
    if not texts:
        return {}
    keys = [_embed_key(t) for t in texts]
    try:
        pipe = r.pipeline()
        for key in keys:
            pipe.get(key)
        values = await pipe.execute()
        return {
            text: (json.loads(v) if v else None)
            for text, v in zip(texts, values)
        }
    except Exception as e:
        logger.warning(f"Batch embedding cache read failed: {e}")
        return {t: None for t in texts}


# ── Cache stats — useful for the /stats/cache endpoint ───────────────────────

async def get_cache_stats(r: redis.Redis) -> dict:
    try:
        info   = await r.info("stats")
        hits   = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total  = hits + misses
        return {
            "hits":     hits,
            "misses":   misses,
            "hit_rate_pct": round(hits / max(1, total) * 100, 1),
        }
    except Exception:
        return {"hits": 0, "misses": 0, "hit_rate_pct": 0.0}
