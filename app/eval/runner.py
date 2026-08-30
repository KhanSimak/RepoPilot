"""
runner.py — runs the golden dataset through retrieval and scores it

IMPORTANT SCOPE NOTE: this benchmarks RETRIEVAL quality (does the right
chunk show up, and at what rank), not the final LLM-generated answer's
correctness. Evaluating generated text quality is a much harder, fuzzier
problem (you'd need an LLM-as-judge or human review). Recall@K and MRR on
retrieval are objective, fast to compute, and directly measure the part
of the pipeline that's actually under our control — if the right chunk
never reaches the LLM, no amount of prompt engineering fixes the answer.

For speed and cost, the benchmark calls embed + hybrid retrieval directly
(skipping HyDE rewriting, reranking, and the final LLM call) — this
isolates "can hybrid search alone find the answer" which is the metric
you want to track for regressions in chunking/embedding/indexing.
"""

import time
import logging

from app.engine.embedder import embed_text
from app.engine.vectordb import search as vector_search, scroll_repo_chunks, retrieve_by_ids
from app.engine.bm25 import search as bm25_search
from app.engine.fusion import reciprocal_rank_fusion
from app.eval.golden_dataset import build_golden_dataset
from app.eval.metrics import recall_at_k, mean_reciprocal_rank, latency_percentiles, per_file_recall, ndcg_at_k

logger = logging.getLogger(__name__)


async def _find_rank(question: str, expected_id: str, repo_id: str, qdrant_client, cfg, top_k: int = 10) -> tuple[int | None, float]:
    """Returns (rank_of_expected_chunk_or_None, latency_ms)."""
    t0 = time.perf_counter()

    vector = await embed_text(question, is_query=True)
    vector_hits  = await vector_search(qdrant_client, cfg.qdrant_collection, vector, repo_id, top_k=top_k)
    keyword_hits = bm25_search(repo_id, question, top_k=top_k)
    fused_ids    = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=top_k)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if expected_id in fused_ids:
        return fused_ids.index(expected_id) + 1, latency_ms   # 1-indexed rank
    return None, latency_ms


async def run_eval(repo_id: str, qdrant_client, cfg, max_questions: int = 100) -> dict:
    all_chunks = await scroll_repo_chunks(qdrant_client, cfg.qdrant_collection, repo_id)
    if not all_chunks:
        return {"error": "No chunks found for this repo — has it been ingested?"}

    golden = build_golden_dataset(all_chunks, max_questions=max_questions)
    if not golden:
        return {"error": "No docstrings found to build a golden dataset from. "
                          "Recall/MRR require at least some documented functions."}

    found_ranks: list[int | None] = []
    latencies: list[float] = []
    file_results: dict[str, list] = {}

    for item in golden:
        rank, latency_ms = await _find_rank(item["question"], item["expected_id"], repo_id, qdrant_client, cfg)
        found_ranks.append(rank)
        latencies.append(latency_ms)
        file_results.setdefault(item["file"], []).append(rank)

    return {
        "repo_id":        repo_id,
        "questions_run":  len(golden),
        "recall_at_5":    recall_at_k(found_ranks, 5),
        "recall_at_10":   recall_at_k(found_ranks, 10),
        "mrr":            mean_reciprocal_rank(found_ranks),
        "ndcg_at_5":      ndcg_at_k(found_ranks, 5),
        "latency_ms":     latency_percentiles(latencies),
        "worst_files":    per_file_recall(file_results, k=5)[:5],
        "note": "Measures hybrid retrieval (vector+BM25+RRF) only — "
                "skips HyDE/reranking/LLM generation for fast, repeatable scoring.",
    }