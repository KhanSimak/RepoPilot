"""
metrics.py — Recall@K, MRR, latency percentiles

RECALL@K:
  "Does the correct chunk appear ANYWHERE in the top K results?"
  A simple yes/no per question, averaged across the whole dataset.
  Recall@5 = 0.847 means: for 84.7% of questions, the right chunk was
  somewhere in the top 5. It says nothing about WHERE in the top 5 —
  rank 1 and rank 5 count equally. That's what MRR is for.

MRR (Mean Reciprocal Rank):
  For each question, score = 1/rank of the correct chunk (0 if not found
  in the searched results at all). Average across all questions.
  MRR rewards the correct answer appearing EARLY, not just somewhere.
  A system that always puts the right chunk at rank 1 scores MRR=1.0.
  A system that always puts it at rank 5 scores MRR=0.2 — same Recall@5
  as the first system if both always include it in top 5, but MRR
  correctly shows the first system is actually better for the user.

LATENCY PERCENTILES (P50/P95/P99), NOT JUST AVERAGE:
  Averages hide tail behavior. If 95 requests take 200ms and 5 requests
  take 5000ms (a cache miss combined with a cold model, say), the AVERAGE
  looks like ~440ms — fine. But P95 correctly reports "5% of your users
  wait 5 SECONDS," which is the number that actually matters for user
  experience and for catching regressions before they compound.
"""

import statistics


def recall_at_k(found_ranks: list[int | None], k: int) -> float:
    """
    found_ranks: for each question, the 1-indexed rank of the correct chunk
                 in the result list, or None if it wasn't found at all.
    """
    if not found_ranks:
        return 0.0
    hits = sum(1 for rank in found_ranks if rank is not None and rank <= k)
    return round(hits / len(found_ranks), 4)


def mean_reciprocal_rank(found_ranks: list[int | None]) -> float:
    if not found_ranks:
        return 0.0
    reciprocal_sum = sum(1.0 / rank if rank is not None else 0.0 for rank in found_ranks)
    return round(reciprocal_sum / len(found_ranks), 4)


def ndcg_at_k(found_ranks: list[int | None], k: int) -> float:
    """
    nDCG@K — like MRR, but discounts by log2(rank+1) instead of 1/rank,
    a gentler penalty for being slightly further down the list.

    CAVEAT, stated honestly: your golden dataset has exactly ONE correct
    chunk per question (binary relevant/not-relevant), not graded relevance
    (this chunk is a perfect match, that one's a 60%-relevant partial
    match). nDCG's real value is ranking among MULTIPLE relevant items at
    different relevance levels — with single binary ground truth, this
    collapses to a smoothed version of MRR, not fully independent signal.
    Still a correct, standard metric — just don't expect it to diverge
    much from MRR until/unless the golden dataset has graded relevance.
    """
    import math
    if not found_ranks:
        return 0.0
    ideal_dcg = 1.0 / math.log2(2)  # best possible: correct chunk at rank 1
    scores = [
        (1.0 / math.log2(rank + 1)) / ideal_dcg
        if rank is not None and rank <= k else 0.0
        for rank in found_ranks
    ]
    return round(sum(scores) / len(scores), 4)


def latency_percentiles(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_lat = sorted(latencies_ms)

    def _percentile(p: float) -> float:
        idx = min(len(sorted_lat) - 1, int(round(p / 100 * (len(sorted_lat) - 1))))
        return round(sorted_lat[idx], 1)

    return {"p50": _percentile(50), "p95": _percentile(95), "p99": _percentile(99)}


def per_file_recall(file_results: dict[str, list[int | None]], k: int = 5) -> list[dict]:
    """
    file_results: {file_path: [rank_or_none, rank_or_none, ...]} — every
    question whose expected chunk lives in that file, with its found rank.

    Returns files sorted WORST recall first — this is the actionable
    output: "these are the files your retrieval struggles with, go look
    at why" (often: functions too long, missing docstrings, or generic
    names that collide with other functions across the repo).
    """
    scored = []
    for file_path, ranks in file_results.items():
        if not ranks:
            continue
        scored.append({
            "file":     file_path,
            "recall":   recall_at_k(ranks, k),
            "question_count": len(ranks),
        })
    return sorted(scored, key=lambda x: x["recall"])