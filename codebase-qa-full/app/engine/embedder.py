"""
embedder.py — local ONNX embedder (bge-small-en-v1.5)

WHY LOCAL INSTEAD OF AN EMBEDDING API:
  OpenAI's text-embedding-ada-002 costs $0.0001 per 1K tokens and requires
  a network round trip (~100ms). At 10,000 queries/day that's $0.50/day = $182/year.

  bge-small-en-v1.5 with ONNX Runtime runs entirely on your own CPU.
  Cost: $0.00 forever. Latency: ~25ms (4x faster, no network hop).

WHAT ONNX ACTUALLY DOES:
  The model is normally a PyTorch graph. ONNX Runtime converts it into an
  optimized, statically-compiled inference graph — similar to how a JIT
  compiler speeds up code. `export=True` does this conversion once and
  caches the result; every subsequent load is fast.

WHY A QUERY PREFIX:
  BGE models were trained with an asymmetric setup: queries get a special
  instruction prefix, documents don't. This measurably improves retrieval
  quality — skip it and you lose a few points of accuracy for free.
"""

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import asyncio
import logging
import os
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

_tokenizer = None
_model     = None
_executor  = ThreadPoolExecutor(max_workers=4)


def load_embedder(cfg):
    """Called once at app startup (see main.py lifespan)."""
    global _tokenizer, _model
    logger.info(f"Loading ONNX embedder: {cfg.embed_model}")
    MODEL_PATH = "./models/bge-small"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    _model = ORTModelForFeatureExtraction.from_pretrained(
      MODEL_PATH,
      file_name="model.onnx"
)
    logger.info("Embedder ready — $0.00 per query from this point on")
    return _model


def _mean_pool_normalize(outputs, attention_mask) -> np.ndarray:
    """
    A transformer outputs one vector PER TOKEN, not one vector per sentence.
    Mean pooling averages all token vectors (weighted by the attention mask,
    so padding tokens don't count) into a single sentence vector.

    L2 normalization (dividing by the vector's length) means the dot product
    of two vectors equals their cosine similarity — this is what Qdrant's
    COSINE distance metric expects.
    """
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask[..., np.newaxis].astype(float)
    summed   = (token_embeddings * mask).sum(axis=1)
    counts   = mask.sum(axis=1).clip(min=1e-9)
    vectors  = summed / counts
    norms    = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
    return vectors / norms


def _encode_sync(texts: list[str]) -> list[list[float]]:
    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    # ONNX expects int64
    inputs = {
        k: v.astype("int64")
        for k, v in inputs.items()
    }

    outputs = _model(**inputs)
    vectors = _mean_pool_normalize(outputs, inputs["attention_mask"])
    return vectors.tolist()

async def embed_text(text: str, is_query: bool = True) -> list[float]:
    """
    Embed one piece of text.

    WHY run_in_executor: the ONNX model call is CPU-bound, not I/O-bound.
    If we called _encode_sync directly inside an `async def`, it would
    block FastAPI's entire event loop — every other request would freeze
    for the ~25ms duration. run_in_executor moves the work to a thread,
    keeping the event loop free to handle other requests concurrently.
    """
    if is_query:
        # bge-small-en-v1.5's documented query instruction is the full
        # phrase below — a shortened prefix still runs fine (this doesn't
        # crash anything) but doesn't match what the model was actually
        # trained on, which quietly costs you retrieval quality.
        text = f"Represent this sentence for searching relevant passages: {text}"
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _encode_sync, [text])
    return result[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed many texts at once — used during ingest for hundreds of chunks.

    Batching matters: one forward pass over 32 texts is much faster than
    32 separate forward passes, because the model's matrix multiplications
    are more efficient at larger batch sizes (better CPU/SIMD utilization).
    """
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        vectors = await loop.run_in_executor(_executor, _encode_sync, batch)
        all_vectors.extend(vectors)
    

    del vectors
    gc.collect()
    return all_vectors
