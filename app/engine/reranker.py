"""
reranker.py — Cohere Rerank API, no local model, no PyTorch

WHY THIS CHANGED FROM A LOCAL CROSSENCODER:
  The previous version ran ms-marco-MiniLM-L-6-v2 locally via
  sentence-transformers (CrossEncoder), which pulls in PyTorch as a
  transitive dependency. That's fine on a dev machine; on Render's free
  tier (512MB RAM, small disk budget) PyTorch alone is the single biggest
  line item in the whole dependency tree. Moving reranking to Cohere's
  hosted API removes PyTorch from the runtime entirely - nothing is
  loaded into memory, nothing runs on CPU here anymore.

BI-ENCODER (what Qdrant/embedder.py does, still 100% local/ONNX,
unchanged) vs CROSS-ENCODER (what this file does, now via Cohere's
hosted rerank-v4.0-fast model instead of a local model) - the tradeoff
explained in the original docstring still applies: bi-encoder search
narrows thousands of chunks to ~20-35 candidates cheaply, cross-encoder
reranking (now a network call instead of local CPU inference) picks the
real top N from that narrowed set.

COST: no longer $0 - Cohere's Rerank API is billed per request/document.
Still cheap at this volume (bounded to MAX_RERANK=35 documents per call,
same cap as before), but no longer literally free. Requires COHERE_API_KEY.

OUTPUT CONTRACT: unchanged from the CrossEncoder version. rerank() still
returns chunks with **chunk spread plus a "rerank_score" float, sorted by
that score descending, capped at top_n. Nothing downstream (token_budget.py,
nodes.py, codegen_nodes.py) needs to change.

SCORE SCALE CHANGED - READ THIS BEFORE TRUSTING is_low_confidence():
  ms-marco-MiniLM's CrossEncoder produced raw, unbounded logit-like scores
  (roughly -11 to +11 in practice), which is why the old low-confidence
  threshold was `< -10.0`. Cohere's relevance_score is a normalized
  probability-like value in [0, 1]. The old threshold would NEVER fire
  against these scores (0-1 is always >= -10.0) - it would have silently
  gone dead. The threshold below has been rescaled to fit the new range,
  but 0.05 is a starting point, not a validated cutoff - tune it against
  your own eval set (app/eval/) once you have real Cohere score
  distributions to look at.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cohere

logger = logging.getLogger(__name__)

_client: "cohere.ClientV2 | None" = None
_configured_model = None
_executor = ThreadPoolExecutor(max_workers=2)

COHERE_RERANK_MODEL = "rerank-v4.0-fast"

# Same bound as the old CrossEncoder path: a retrieval pass can contain
# 20 fused candidates plus at most 15 graph neighbors.
MAX_RERANK = 35


def load_reranker(cfg):
    """
    Called once at startup (same call site as before: main.py's lifespan
    does app.state.reranker = load_reranker(cfg)) - now creates a Cohere
    client instead of loading a local model. No model weights, no
    PyTorch, nothing held in memory between requests.
    """
    global _client, _configured_model
    _configured_model = getattr(cfg, "cohere_rerank_model", None) or COHERE_RERANK_MODEL
    logger.info(f"Initializing reranker: Cohere Rerank API ({_configured_model})")
    _client = cohere.ClientV2(api_key=cfg.cohere_api_key)
    logger.info("Reranker ready — Cohere Rerank API, no local model")
    return _client


def _document_text(chunk: dict) -> str:
    """Identical document construction to the old CrossEncoder path, so
    relevance is judged on the same signal as before - only the model
    scoring it changed."""
    return f"""
Name: {chunk.get("name", "")}
Type: {chunk.get("type", "")}
File: {chunk.get("file", "")}

Docstring:
{chunk.get("docstring", "")}

Calls:
{", ".join(chunk.get("calls", []))}

Called by:
{", ".join(chunk.get("called_by", []))}

Code:
{chunk.get("text", "")}
"""


def _rerank_sync(question: str, chunks: list[dict], top_n: int, model: str) -> list[dict]:
    if not chunks:
        return []

    if _client is None:
        raise RuntimeError("Reranker has not been loaded.")

    chunks = chunks[:MAX_RERANK]
    top_n = min(int(top_n), MAX_RERANK, len(chunks))

    documents = [_document_text(c) for c in chunks]

    response = _client.rerank(
        model=model,
        query=question,
        documents=documents,
        top_n=top_n,
    )

    ranked = [
        (chunks[result.index], result.relevance_score)
        for result in response.results
    ]

    logger.debug("Top reranked chunks:")
    for chunk, score in ranked[:5]:
        logger.debug(
            "%s | %s | %.4f",
            chunk.get("name"),
            chunk.get("file"),
            float(score),
        )

    return [
        {
            **chunk,
            "rerank_score": round(float(score), 4),
        }
        for chunk, score in ranked
    ]


async def rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    model = _configured_model or COHERE_RERANK_MODEL
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _rerank_sync, question, chunks, top_n, model
    )


def is_low_confidence(reranked_chunks: list[dict], cfg) -> bool:
    """
    Cohere's relevance_score is a 0-1 probability-like value, NOT the old
    CrossEncoder's unbounded logit scale - see the module docstring.
    Reject only when the top score is genuinely poor, same intent as
    before, rescaled to the new range.
    """
    if not reranked_chunks:
        return True

    top_score = reranked_chunks[0]["rerank_score"]

    logger.debug("Top rerank score: %s", top_score)

    threshold = getattr(cfg, "cohere_low_confidence_threshold", 0.05)
    return top_score < threshold