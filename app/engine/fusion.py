"""
fusion.py — Reciprocal Rank Fusion (RRF)

THE PROBLEM RRF SOLVES:
  Vector search returns a cosine similarity score, typically 0–1.
  BM25 returns an unbounded score that depends on document length and
  corpus statistics — it might be 3.2, or 47.8, depending entirely on
  the data. You CANNOT directly add these two numbers together; they're
  not on the same scale and mean completely different things.

  RRF sidesteps the whole problem by ignoring the SCORES and using only
  the RANK (position) in each list. A document at rank 1 in vector search
  contributes 1/(k+1) to its fused score; a document at rank 1 in BM25
  contributes the same. No score normalization needed, no weight tuning
  required to get a sane baseline — which is why RRF is the standard
  choice for hybrid search in production systems (Azure AI Search,
  Elasticsearch's hybrid retriever, Weaviate, all use this).

THE FORMULA:
  score(doc) = Σ over each ranked list: weight / (k + rank)

  k=60 is the constant from the original RRF paper — empirically chosen
  to avoid letting the very top rank dominate too much (with a small k,
  rank 1 vs rank 2 would be an enormous score difference; k=60 smooths
  that out across the whole list).

WHY WE WEIGHT KEYWORD SLIGHTLY LOWER (0.4) THAN VECTOR (0.6):
  For code specifically you might expect keyword matching to dominate
  (exact              s matter a lot), but in practice most natural-language
  questions ("how does X work") are answered better by semantic search.
  We weight keyword slightly lower as the default but expect this to be
  tuned per-codebase if you have eval data (Phase 4 builds exactly that).
"""


def reciprocal_rank_fusion(
    vector_hits:   list[dict],   # [{"id": ..., "score": ...}, ...] sorted by relevance
    keyword_hits:  list[dict],   # [{"id": ..., "bm25_score": ...}, ...] sorted by relevance
    k:             int   = 60,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    top_k:         int   = 20,
) -> list[str]:
    """
    Returns a list of chunk IDs, ranked by fused score (best first).
    Caller is responsible for looking up the actual chunk data by ID.
    """
    scores: dict[str, float] = {}

    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (k + rank + 1)

    for rank, hit in enumerate(keyword_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight / (k + rank + 1)

    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    return ranked_ids[:top_k]
