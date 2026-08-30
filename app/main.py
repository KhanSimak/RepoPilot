"""
main.py — FastAPI application entry point, final phase

Every engine loaded once at startup and stashed on app.state:
  qdrant, embedder (ONNX), reranker (CrossEncoder), redis.

Routers:
  /repos    — register, status, delete, AND /sync (incremental ingest)
  /repos/.../search   — Phase 2 baseline: hybrid search, no HyDE/rerank/trace
  /repos/.../ask       — final pipeline: HyDE + hybrid + graph + rerank + budget + trace
  /repos/.../stream    — same pipeline, SSE
  /stats    — cache hit rate
  /eval     — retrieval benchmark (Recall@5, Recall@10, MRR, latency, worst files)
"""
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import get_settings
from app.engine.embedder import load_embedder
from app.engine.reranker import load_reranker
from app.engine.vectordb import init_qdrant
from app.cache.redis_cache import init_redis
from app.routers import repos, search, stats, query, eval as eval_router, graph as graph_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    app.state.settings = cfg
    app.state.qdrant    = await init_qdrant(cfg)
    app.state.embedder  = load_embedder(cfg)
    app.state.reranker = load_reranker(cfg) 
    app.state.redis     = await init_redis(cfg)
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
        "http://127.0.0.1:5501",   # keep this for local development
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

