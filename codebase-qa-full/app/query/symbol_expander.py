import logging

from app.engine.bm25 import search as bm25_search, get_chunk_names_for_ids

logger = logging.getLogger(__name__)


async def expand_to_symbols(
    phrases: list[str],
    repo_id: str,
    top_symbols: int = 10,
) -> list[str]:
    """
    Expand generic HyDE phrases into actual repository identifiers.

    Example:
        "jwt auth"
            ↓
        verify_jwt_token
        JWTMiddleware
        decode_token
    """

    STOP_WORDS = {
        "how", "does", "do", "did", "is", "are", "was", "were",
        "a", "an", "the", "to", "into", "of", "for", "in", "on",
        "with", "every", "major", "component", "components",
        "explain", "describe", "what", "which", "why", "when",
    }

    cleaned_phrases = []

    for phrase in phrases:
        phrase = phrase.strip()

        if not phrase:
            continue

        if phrase.lower() in STOP_WORDS:
            continue

        cleaned_phrases.append(phrase)

    phrases = cleaned_phrases

    initial_hits = []

    for phrase in phrases:
        initial_hits.extend(
            bm25_search(repo_id, phrase, top_k=5)
        )

    if not initial_hits:
        return phrases

    ids = []
    seen = set()

    for hit in initial_hits:
        if hit["id"] not in seen:
            ids.append(hit["id"])
            seen.add(hit["id"])

    names = get_chunk_names_for_ids(repo_id, ids[:top_symbols])

    expanded = phrases + names

    return list(dict.fromkeys(expanded))