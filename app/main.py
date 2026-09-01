"""
main.py — FastAPI application entry point.

Initializes the application's shared infrastructure during startup and
stores the initialized components on app.state for reuse across requests.

Startup components:
  - Qdrant: vector storage and similarity search
  - ONNX Runtime embedder: local CPU embedding inference
  - CrossEncoder reranker: candidate reranking
  - Redis: embedding/conversation/cache storage

Application routes:
  /repos
      Repository registration, ingestion, status, deletion, and sync.

  /repos/{repo_id}/search
      Baseline hybrid retrieval using vector search, BM25, and RRF.

  /repos/{repo_id}/ask
      Full retrieval pipeline including query rewriting, hybrid retrieval,
      call-graph expansion, reranking, context compression, and token
      budget enforcement.

  /repos/{repo_id}/stream
      Streaming version of the full query pipeline using Server-Sent Events.

  /repos/{repo_id}/graph
      Call-graph exploration endpoints.

  /stats
      Cache and system statistics.

  /eval
      Retrieval evaluation including Recall@K, MRR, and latency metrics.

All heavyweight components are initialized once during application startup
rather than being recreated for individual requests.

The initialized components are exposed through app.state so routers and
services can share the same Qdrant client, ONNX session, reranker, Redis
connection, and application configuration.

The application shuts down by closing the shared Redis connection.
"""
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.warning("===== MAIN IMPORT START =====")

from fastapi.middleware.cors import CORSMiddleware
logger.warning("===== FASTAPI IMPORTED =====")

from contextlib import asynccontextmanager
from fastapi import FastAPI
logger.warning("===== CORE FASTAPI IMPORTED =====")

from app.config import get_settings
logger.warning("===== CONFIG IMPORTED =====")

from app.engine.embedder import load_embedder
logger.warning("===== EMBEDDER MODULE IMPORTED =====")

from app.engine.reranker import load_reranker
logger.warning("===== RERANKER MODULE IMPORTED =====")

from app.engine.vectordb import init_qdrant
logger.warning("===== VECTORDB MODULE IMPORTED =====")

from app.cache.redis_cache import init_redis
logger.warning("===== REDIS MODULE IMPORTED =====")

from app.routers import (
    repos,
    search,
    stats,
    query,
    eval as eval_router,
    graph as graph_router,
)
logger.warning("===== ALL ROUTERS IMPORTED =====")
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)
logger.warning("===== MAIN.PY IMPORT START =====")


def _log_startup_memory(stage: str) -> None:
    try:
        if resource is not None:
            mb = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / 1024
            logger.info(
                f"STARTUP MEMORY {stage}: {mb:.1f} MB PEAK_RSS"
            )
        elif psutil is not None:
            mb = psutil.Process(
                os.getpid()
            ).memory_info().rss / (1024 * 1024)
            logger.info(
                f"STARTUP MEMORY {stage}: {mb:.1f} MB RSS"
            )
    except Exception as e:
        logger.warning(
            f"Startup memory check failed: {e}"
        )
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()

    _log_startup_memory("begin")

    app.state.settings = cfg

    app.state.qdrant = await init_qdrant(cfg)
    _log_startup_memory("after qdrant")

    app.state.embedder = load_embedder(cfg)
    _log_startup_memory("after embedder")

    app.state.reranker = load_reranker(cfg)
    _log_startup_memory("after reranker")

    app.state.redis = await init_redis(cfg)
    _log_startup_memory("after redis")

    logger.info("===== STARTUP COMPLETE =====")

    yield

    await app.state.redis.aclose()


app = FastAPI(
    title="Codebase Q&A Engine — Final Phase",
    description=(
        "AST chunking + call graph + ONNX embeddings + hybrid search (vector+BM25+RRF) "
        "+ HyDE rewriting + cross-encoder reranking + token budget + Redis caching "
        "+ incremental ingest + Recall@K/MRR evaluation + SSE streaming, with a full "
        "per-stage cost and latency trace on every request."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://34.100.245.18:8080",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "https://codebase-qa-agent-eosin.vercel.app",
   # keep this for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(repos.router,       prefix="/repos", tags=["repos"])
app.include_router(search.router,      prefix="/repos", tags=["search (baseline)"])
app.include_router(query.router,       prefix="/repos", tags=["query (final pipeline)"])
app.include_router(graph_router.router, prefix="/repos", tags=["call graph explorer"])
app.include_router(stats.router,       prefix="/stats", tags=["stats"])
app.include_router(eval_router.router, prefix="/eval",  tags=["eval"])


@app.get("/health")
async def health():
    return {
        "phase": "final",
        "status": "ok",
        "features": [
            "ast_chunking", "call_graph", "onnx_embeddings", "qdrant_vector_search",
            "bm25_keyword_search", "rrf_fusion", "redis_two_layer_cache",
            "hyde_query_rewriting", "intent_detection", "graph_aware_retrieval",
            "cross_encoder_reranking", "token_budget_enforcement",
            "parallel_context_compression", "per_request_cost_trace",
            "sse_streaming", "incremental_ingest", "recall_mrr_eval_framework",
        ],
    }

