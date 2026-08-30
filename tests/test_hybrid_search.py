"""
test_hybrid_search.py — verify BM25 indexing and RRF fusion behave correctly

Run with: pytest tests/ -v
"""
from app.engine.bm25 import build_index, search as bm25_search, _tokenize, delete_index
from app.engine.fusion import reciprocal_rank_fusion


# ── BM25 tests ────────────────────────────────────────────────────────────────

def test_tokenize_keeps_underscores():
    """Critical for code: process_payment must tokenize as ONE token, not split on '_'."""
    tokens = _tokenize("def process_payment(amount):")
    assert "process_payment" in tokens


def test_tokenize_lowercases():
    tokens = _tokenize("StripeCharge")
    assert "stripecharge" in tokens


def test_bm25_finds_exact_identifier():
    chunks = [
        {"id": "1", "text": "def process_payment(amount): stripe.Charge.create(amount)", "name": "process_payment"},
        {"id": "2", "text": "def send_email(to, subject): smtp.send(to, subject)", "name": "send_email"},
        {"id": "3", "text": "def log_event(event): logger.info(event)", "name": "log_event"},
    ]
    build_index("test_repo_1", chunks)

    results = bm25_search("test_repo_1", "stripe charge create", top_k=5)
    assert len(results) > 0
    assert results[0]["id"] == "1"   # the payment chunk should rank first
    delete_index("test_repo_1")


def test_bm25_no_index_returns_empty():
    """Searching a repo that hasn't been indexed yet should fail gracefully, not crash."""
    results = bm25_search("nonexistent_repo", "anything", top_k=5)
    assert results == []


def test_bm25_zero_score_filtered_out():
    """A query that matches nothing should return an empty list, not garbage low scores."""
    chunks = [{"id": "1", "text": "def alpha(): pass", "name": "alpha"}]
    build_index("test_repo_2", chunks)
    results = bm25_search("test_repo_2", "zzz_no_match_at_all_qqq", top_k=5)
    assert results == []
    delete_index("test_repo_2")


# ── RRF fusion tests ──────────────────────────────────────────────────────────

def test_rrf_favors_documents_in_both_lists():
    """A doc ranked highly in BOTH vector and keyword search should beat
    a doc ranked highly in only one."""
    vector_hits  = [{"id": "A", "score": 0.9}, {"id": "B", "score": 0.8}, {"id": "C", "score": 0.7}]
    keyword_hits = [{"id": "A", "bm25_score": 5.0}, {"id": "D", "bm25_score": 4.0}]

    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=10)

    # "A" appears at rank 1 in both lists — should be the top fused result
    assert fused[0] == "A"


def test_rrf_includes_keyword_only_hits():
    """A doc found ONLY by BM25 (not in vector results at all) must still
    appear in the fused output — this is the whole point of hybrid search."""
    vector_hits  = [{"id": "A", "score": 0.9}]
    keyword_hits = [{"id": "Z", "bm25_score": 9.0}]   # never seen by vector search

    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=10)
    assert "Z" in fused


def test_rrf_respects_top_k():
    vector_hits  = [{"id": str(i), "score": 1.0 - i * 0.01} for i in range(30)]
    keyword_hits = []
    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=5)
    assert len(fused) == 5


def test_rrf_empty_inputs():
    """Should not crash on empty lists — happens when a repo has no BM25 matches."""
    fused = reciprocal_rank_fusion([], [], top_k=5)
    assert fused == []
