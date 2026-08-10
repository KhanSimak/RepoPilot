"""
bm25.py — keyword search alongside vector search

WHY YOU NEED THIS EVEN THOUGH YOU HAVE VECTOR SEARCH:
  Vector search finds what something MEANS. It's excellent at "what handles
  user authentication" even if the code never uses the word "authentication".

  But vector search is surprisingly bad at exact tokens. Ask "find every call
  to stripe.Charge.create" and the embedding of that question is semantically
  close to LOTS of payment-related code — it has no special pull toward the
  literal string "stripe.Charge.create". A developer searching for an exact
  function name, a specific error message, or a config key expects an exact
  match, and BM25 (the algorithm behind Elasticsearch/Lucene) is built for
  exactly that: term frequency × inverse document frequency.

  Phase 2 runs BOTH searches and merges them (see fusion.py) — you get
  the best of semantic understanding AND exact keyword matching.

WHY IN-MEMORY AND NOT A SEPARATE SERVICE:
  rank-bm25 builds a simple inverted index in Python memory. For a single
  repo's worth of chunks (hundreds to low thousands), this is fast enough
  and avoids running yet another piece of infrastructure. If you needed to
  scale to many large repos simultaneously, you'd move this to Elasticsearch
  or Qdrant's own sparse vector support — out of scope for this project.
"""

import re
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# One BM25 index per repo_id, kept in process memory.
# Rebuilt every time a repo is (re)ingested.
_indexes: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    """
    Lowercase and split on anything that isn't a letter, digit, or underscore.
    Keeping the underscore matters a lot for code: `process_payment` should
    tokenize as one token, not three (`process`, `payment` losing the link
    between them, and definitely not splitting `_` itself as a token).
    """
    return re.sub(r"[^a-z0-9_]", " ", text.lower()).split()


def build_index(repo_id: str, chunks: list[dict]) -> None:
    """
    chunks: list of {"id": ..., "text": ..., "name": ...} —
    called once per ingest, right after Qdrant upsert.
    """
    corpus = [_tokenize(c["text"] + " " + c["name"]) for c in chunks]
    ids    = [c["id"] for c in chunks]
    names = [c["name"] for c in chunks]

    _indexes[repo_id] = {
        "bm25": BM25Okapi(corpus),
        "ids":  ids,
        "names":names,
    }
    logger.info(f"BM25 index built for {repo_id}: {len(corpus)} documents")


def search(repo_id: str, query: str, top_k: int = 20) -> list[dict]:
    """
    Returns [{"id": chunk_id, "bm25_score": float}, ...] sorted descending.
    Returns [] if no index exists yet for this repo (e.g. still ingesting).
    """
    index = _indexes.get(repo_id)
    if not index:
        return []

    scores = index["bm25"].get_scores(_tokenize(query))
    ranked = sorted(zip(index["ids"], scores), key=lambda x: x[1], reverse=True)

    # Filter out zero-score results — they didn't actually match any term
    return [
        {"id": doc_id, "bm25_score": round(float(score), 4)}
        for doc_id, score in ranked[:top_k]
        if score > 0
    ]


def has_index(repo_id: str) -> bool:
    return repo_id in _indexes


def delete_index(repo_id: str) -> None:
    _indexes.pop(repo_id, None)
def get_chunk_names_for_ids(repo_id: str, ids: list[str]) -> list[str]:
    index = _indexes.get(repo_id)

    if not index:
        return []

    id_to_name = dict(zip(index["ids"], index["names"]))

    return [
        id_to_name[i]
        for i in ids
        if i in id_to_name
    ]