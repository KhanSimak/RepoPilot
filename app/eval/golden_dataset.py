"""
golden_dataset.py — auto-generate a Q&A benchmark from the repo itself

THE PROBLEM WITH MOST RAG EVAL:
  You need a set of (question, correct_answer) pairs to measure against.
  Hand-writing these is slow and most projects skip it entirely — which is
  why most RAG projects can only ever say "it seems to work well."

THE TRICK:
  Every chunk that already HAS a docstring is a free, self-labeling
  question: "What does {function_name} do?" -> the correct answer is
  THIS chunk, because the docstring is a human description of what it does.
  We don't need an LLM to invent questions OR grade answers — the dataset
  generates itself from metadata extracted at ingest time.

  This isn't a substitute for a hand-curated eval set in a real production
  system, but it requires zero manual labeling effort and degrades
  gracefully: a repo with good docstring coverage gets a rich benchmark;
  a repo with none gets an honest "not enough docstrings" message instead
  of a misleading number.
"""

import logging

logger = logging.getLogger(__name__)

MIN_DOCSTRING_LENGTH = 15   # skip trivial one-word docstrings like "Init."


def build_golden_dataset(chunks: list[dict], max_questions: int = 100) -> list[dict]:
    """
    chunks: full repo chunk dicts (from scroll_repo_chunks).
    Returns: [{"question": str, "expected_id": str, "file": str, "name": str}, ...]
    """
    golden = []
    for chunk in chunks:
        docstring = chunk.get("docstring", "")
        if len(docstring) < MIN_DOCSTRING_LENGTH:
            continue

        golden.append({
            "question":    f"What does {chunk['name']} do?",
            "expected_id": chunk["id"],
            "file":        chunk["file"],
            "name":        chunk["name"],
        })

        if len(golden) >= max_questions:
            break

    logger.info(f"Golden dataset: {len(golden)} questions generated from {len(chunks)} chunks")
    return golden
