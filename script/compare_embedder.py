"""
scripts/compare_embedders.py — validate embedder_direct_onnx.py against
the current embedder.py before switching anything in production.

WHAT THIS PROVES (if it passes):
  Given the SAME model.onnx + tokenizer files at models/bge-small/, the
  optimum/transformers path (embedder.py) and the raw onnxruntime/
  tokenizers path (embedder_direct_onnx.py) produce:
    - identical output shape (384-dim vectors)
    - near-identical vectors for the same input text (cosine sim > threshold)
  for both query-prefixed and document (non-prefixed) embedding, and for
  both single embed_text() calls and batched embed_batch() calls (the
  batch case also exercises dynamic padding, since the sample texts below
  are deliberately different lengths).

WHAT THIS DOES NOT PROVE: retrieval quality, or that either embedder is
"correct" in an absolute sense - only that the two are equivalent to each
other, which is the actual question at hand (removing optimum/torch
without silently changing what gets embedded).

REQUIRES: both dependency stacks installed side by side (the current
requirements.txt's optimum+transformers, AND onnxruntime+tokenizers for
the candidate), plus the real models/bge-small/ files. Run from the repo
root:

    python scripts/compare_embedders.py

Exits non-zero if any comparison fails, so this can gate a later PR.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.config import get_settings
from app.engine import embedder as old_embedder
from app.engine import embedder_direct_onnx as new_embedder

COSINE_SIM_THRESHOLD = 0.999  # near-identical, not just "similar"

SAMPLE_TEXTS = [
    "How does the request context get attached to the response?",
    "def login():\n    return authenticate(request.form)\n",
    "a",  # single-token edge case
    "SessionInterface.open_session raises NotImplementedError in the base class",
    "class SecureCookieBackend(SessionBackend):\n    def open_session(self, request):\n        return self._load(request.cookies)\n",
    "call graph expansion walks calls and called_by edges outward from an entry chunk",
]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def main() -> bool:
    cfg = get_settings()

    print("Loading old embedder (optimum + transformers)...")
    old_embedder.load_embedder(cfg)
    print("Loading new embedder (raw onnxruntime + tokenizers)...")
    new_embedder.load_embedder(cfg)
    print()

    all_passed = True

    # --- embed_text: both query and document (non-query) prefix paths ---
    for is_query in (True, False):
        label = "query-prefixed" if is_query else "document (no prefix)"
        print(f"=== embed_text, {label} ===")
        for text in SAMPLE_TEXTS:
            old_vec = await old_embedder.embed_text(text, is_query=is_query)
            new_vec = await new_embedder.embed_text(text, is_query=is_query)

            shape_ok = len(old_vec) == len(new_vec) == 384
            sim = cosine_sim(old_vec, new_vec)
            sim_ok = sim >= COSINE_SIM_THRESHOLD
            passed = shape_ok and sim_ok
            all_passed &= passed

            status = "PASS" if passed else "FAIL"
            preview = text[:50].replace("\n", "\\n")
            print(f"  [{status}] dim={len(old_vec)}/{len(new_vec)} cos_sim={sim:.6f}  \"{preview}\"")
        print()

    # --- embed_batch: exercises dynamic padding across mixed lengths ---
    print("=== embed_batch (mixed lengths, tests dynamic padding) ===")
    old_batch = await old_embedder.embed_batch(SAMPLE_TEXTS)
    new_batch = await new_embedder.embed_batch(SAMPLE_TEXTS)

    length_ok = len(old_batch) == len(new_batch) == len(SAMPLE_TEXTS)
    all_passed &= length_ok
    print(f"  [{'PASS' if length_ok else 'FAIL'}] batch output length: {len(old_batch)}/{len(new_batch)} (expected {len(SAMPLE_TEXTS)})")

    for i, text in enumerate(SAMPLE_TEXTS):
        old_vec, new_vec = old_batch[i], new_batch[i]
        shape_ok = len(old_vec) == len(new_vec) == 384
        sim = cosine_sim(old_vec, new_vec)
        sim_ok = sim >= COSINE_SIM_THRESHOLD
        passed = shape_ok and sim_ok
        all_passed &= passed

        status = "PASS" if passed else "FAIL"
        preview = text[:50].replace("\n", "\\n")
        print(f"  [{status}] dim={len(old_vec)}/{len(new_vec)} cos_sim={sim:.6f}  \"{preview}\"")

    print()
    print("ALL COMPARISONS PASSED" if all_passed else "SOME COMPARISONS FAILED — do not switch production yet")
    return all_passed


if __name__ == "__main__":
    passed = asyncio.run(main())
    sys.exit(0 if passed else 1)