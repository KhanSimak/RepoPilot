"""
test_eval_metrics.py — verify Recall@K, MRR, and latency percentile math

Run with: pytest tests/ -v
"""
from app.eval.metrics import recall_at_k, mean_reciprocal_rank, latency_percentiles, per_file_recall


def test_recall_at_5_all_found_within_5():
    ranks = [1, 3, 5, 2, 4]
    assert recall_at_k(ranks, 5) == 1.0


def test_recall_at_5_some_missed():
    ranks = [1, 3, None, 7, 5]   # rank 7 is outside top-5; None means not found at all
    # found within top 5: ranks 1, 3, 5 -> 3 out of 5
    assert recall_at_k(ranks, 5) == 0.6


def test_recall_at_k_empty_input():
    assert recall_at_k([], 5) == 0.0


def test_recall_at_k_none_means_not_found():
    ranks = [None, None, None]
    assert recall_at_k(ranks, 5) == 0.0


def test_mrr_perfect_rank_one():
    ranks = [1, 1, 1]
    assert mean_reciprocal_rank(ranks) == 1.0


def test_mrr_mixed_ranks():
    ranks = [1, 2, 4]   # reciprocals: 1.0, 0.5, 0.25 -> mean = 0.5833...
    result = mean_reciprocal_rank(ranks)
    assert abs(result - 0.5833) < 0.001


def test_mrr_not_found_counts_as_zero():
    ranks = [1, None]   # reciprocals: 1.0, 0.0 -> mean = 0.5
    assert mean_reciprocal_rank(ranks) == 0.5


def test_mrr_distinguishes_rank_1_vs_rank_5_even_with_same_recall_at_5():
    """The whole point of MRR vs Recall@K: same recall, different MRR."""
    always_rank_1 = [1, 1, 1]
    always_rank_5 = [5, 5, 5]
    assert recall_at_k(always_rank_1, 5) == recall_at_k(always_rank_5, 5) == 1.0
    assert mean_reciprocal_rank(always_rank_1) > mean_reciprocal_rank(always_rank_5)


def test_latency_percentiles_basic():
    latencies = [100.0] * 95 + [5000.0] * 5   # 95 fast requests, 5 very slow ones
    result = latency_percentiles(latencies)
    assert result["p50"] == 100.0
    assert result["p95"] >= 100.0   # right at the boundary between fast and slow
    assert result["p99"] == 5000.0  # tail latency correctly captured


def test_latency_percentiles_empty():
    result = latency_percentiles([])
    assert result == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_per_file_recall_sorts_worst_first():
    file_results = {
        "good_file.py": [1, 1, 2],      # recall@5 = 1.0
        "bad_file.py":  [None, None, 6], # recall@5 = 0.0
        "ok_file.py":   [1, None, 3],    # recall@5 = 0.667
    }
    result = per_file_recall(file_results, k=5)
    assert result[0]["file"] == "bad_file.py"   # worst first
    assert result[-1]["file"] == "good_file.py" # best last
