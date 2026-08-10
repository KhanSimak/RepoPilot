"""
reranker.py — cross-encoder reranking, local and free

BI-ENCODER (what Qdrant/embedder.py does) vs CROSS-ENCODER (this file):
  Bi-encoder: embed the query, embed the document, SEPARATELY, then compare
  with a dot product. Fast (you can pre-compute document embeddings once),
  but the model never actually sees the query and document together —
  it's comparing two independent summaries.

  Cross-encoder: feed (query, document) into the model TOGETHER as one
  input. The model's attention layers can directly relate specific words
  in the query to specific words in the document. Far more accurate, but
  too slow to run against every chunk in a large repo — which is exactly
  why we use bi-encoder search FIRST to narrow thousands of chunks down
  to ~20 candidates, then cross-encoder reranking to pick the real top 5.

COST: $0. ms-marco-MiniLM-L-6-v2 is ~130MB, runs on CPU in ~80ms for
20 pairs, no API call involved.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder
import numpy as np


logger = logging.getLogger(__name__)

_reranker = None
_executor = ThreadPoolExecutor(max_workers=2)


def load_reranker(cfg):
    global _reranker
    logger.info(f"Loading reranker: {cfg.rerank_model}")
    _reranker = CrossEncoder(cfg.rerank_model)
    logger.info("Reranker ready — $0/query, local CPU inference")
    return _reranker

import numpy as np

def _rerank_sync(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []

    if _reranker is None:
        raise RuntimeError("Reranker has not been loaded.")

    # A retrieval pass can contain 20 fused candidates plus at most 15 graph
    # neighbors.  Keep that bounded set intact so graph evidence is scored.
    MAX_RERANK = 35

    chunks = chunks[:MAX_RERANK]
    top_n = min(top_n, MAX_RERANK)

    pairs = []

    for chunk in chunks:
        document = f"""
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

        pairs.append((question, document))

    scores = _reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.debug("Top reranked chunks:")

    for chunk, score in ranked[:5]:
        logger.debug(
            "%s | %s | %.4f",
            chunk.get("name"),
            chunk.get("file"),
            float(score),
        )

    top_n = min(int(top_n), len(ranked))

    return [
        {
            **chunk,
            "rerank_score": round(float(score), 4),
        }
        for chunk, score in ranked[:top_n]
    ]
async def rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _rerank_sync, question, chunks, top_n)


def is_low_confidence(reranked_chunks: list[dict], cfg) -> bool:
    """
    Only reject when we have essentially no evidence.

    MS MARCO CrossEncoder outputs ranking scores, not probabilities,
    so we don't compare against a fixed value like 0.55.
    """

    if not reranked_chunks:
        return True

    top_score = reranked_chunks[0]["rerank_score"]

    logger.debug("Top rerank score: %s", top_score)

    # Only reject if the score is extremely poor.
    return top_score < -10.0
