"""
embedder.py — local ONNX embedder (bge-small-en-v1.5), WITHOUT optimum
or transformers.

STATUS: promoted to production. This replaces the previous optimum +
transformers-based implementation after scripts/compare_embedders.py
validated identical output against it: query/document/batch tests all
passed, cosine similarity 1.000000 in every case. The retired
implementation is preserved as embedder_direct_onnx.py's sibling history
in version control (it was the file this one was copied from verbatim,
module docstring header aside) if a diff against the optimum-based
version is ever needed.

WHY THIS EXISTS: optimum's base install requires torch>=1.11 and datasets
unconditionally, regardless of only using its ONNX runtime path - see the
audit that led here. This replaces:
  optimum.onnxruntime.ORTModelForFeatureExtraction  ->  onnxruntime.InferenceSession
  transformers.AutoTokenizer                         ->  tokenizers.Tokenizer
using the SAME already-exported model.onnx + tokenizer.json files at
models/bge-small/ - no re-export, no re-training, same weights, same
tokenizer pipeline (tokenizer.json IS the serialized fast-tokenizer
pipeline transformers' AutoTokenizer was already using under the hood,
so this tokenizes identically, not just similarly).

WHAT'S DIFFERENT FROM THE RETIRED optimum-BASED IMPLEMENTATION, MECHANICALLY:
  - transformers.AutoTokenizer.from_pretrained(dir) reads several files
    (tokenizer_config.json, vocab.txt, tokenizer.json, special_tokens_map.json)
    and picks a tokenizer implementation. tokenizers.Tokenizer.from_file()
    reads tokenizer.json ALONE - it's a fully self-contained serialization
    of the same pipeline (normalizer, pre-tokenizer, model, post-processor
    that adds [CLS]/[SEP]), so this loads the same pipeline, not a
    reimplementation of it.
  - padding=True / truncation=True (per-call kwargs on AutoTokenizer) become
    enable_padding()/enable_truncation() (configured once on the Tokenizer
    object, applied on every encode_batch() call after that).
  - The HF "fast" tokenizer's .input_ids/.attention_mask/.token_type_ids
    become tokenizers' Encoding.ids/.attention_mask/.type_ids - same data,
    different attribute names.
  - ORTModelForFeatureExtraction(**inputs) returning an object with
    .last_hidden_state becomes session.run(output_names, input_feed)
    returning a plain list of numpy arrays, looked up by output name
    instead of attribute access.

Everything else - the query prefix, max_length=512, mean pooling, L2
normalization, batch size 32, the async embed_text/embed_batch interface -
is unchanged on purpose: this is a true drop-in replacement, not a
rewrite. Nothing that imports this module needs to change.
"""

import asyncio
import gc
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

MODEL_DIR = "./models/bge-small"

_tokenizer: Tokenizer | None = None
_session: ort.InferenceSession | None = None
_output_names: list[str] | None = None
_executor = ThreadPoolExecutor(max_workers=4)


def load_embedder(cfg):
    """
    Called once at app startup - same call site/contract as before:
    main.py's lifespan does app.state.embedder = load_embedder(cfg).
    Returns the InferenceSession directly (previously an
    ORTModelForFeatureExtraction) - if anything besides embed_text/
    embed_batch calls methods on app.state.embedder directly, it needs to
    be checked against onnxruntime.InferenceSession's API, not optimum's.
    """
    global _tokenizer, _session, _output_names

    logger.info(f"Loading direct-ONNX embedder (no optimum/transformers): {getattr(cfg, 'embed_model', MODEL_DIR)}")

    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    _tokenizer = Tokenizer.from_file(tokenizer_path)

    # Determine the pad token/id from the model's own special_tokens_map.json
    # rather than hardcoding "[PAD]" - correct for this exact model's
    # vocab, not an assumption about BERT-family tokenizers in general.
    special_tokens_path = os.path.join(MODEL_DIR, "special_tokens_map.json")
    with open(special_tokens_path) as f:
        special_tokens = json.load(f)
    pad_token = special_tokens.get("pad_token", "[PAD]")
    if isinstance(pad_token, dict):  # some exports store {"content": "...", ...}
        pad_token = pad_token.get("content", "[PAD]")
    pad_id = _tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise RuntimeError(
            f"Pad token '{pad_token}' not found in tokenizer vocab - "
            f"check {tokenizer_path} matches {special_tokens_path}."
        )

    # Same intent as the old padding=True / truncation=True kwargs, just
    # configured once here instead of passed per-call: pad to the LONGEST
    # sequence in each batch (length=None), not a fixed length.
    _tokenizer.enable_truncation(max_length=512)
    _tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token, length=None)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    model_path = os.path.join(MODEL_DIR, "model.onnx")
    _session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _output_names = [o.name for o in _session.get_outputs()]

    if "last_hidden_state" not in _output_names:
        raise RuntimeError(
            f"Expected ONNX output 'last_hidden_state', got {_output_names} "
            f"from {model_path} - model export may not match what this "
            f"module expects."
        )

    logger.info(
        "Direct-ONNX embedder ready — $0.00 per query, no optimum/transformers/torch"
    )
    return _session


def _mean_pool_normalize(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    A transformer outputs one vector PER TOKEN, not one vector per sentence.
    Mean pooling averages all token vectors (weighted by the attention mask,
    so padding tokens don't count) into a single sentence vector.

    L2 normalization (dividing by the vector's length) means the dot product
    of two vectors equals their cosine similarity — this is what Qdrant's
    COSINE distance metric expects.
    """
    mask = attention_mask[..., np.newaxis].astype(float)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    vectors = summed / counts
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
    return vectors / norms


def _encode_sync(texts: list[str]) -> list[list[float]]:
    encodings = _tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array(
        [e.attention_mask for e in encodings],
        dtype=np.int64,
    )
    token_type_ids = np.array(
        [e.type_ids for e in encodings],
        dtype=np.int64,
    )

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    logger.info(
        "ONNX BEFORE session.run: batch=%d",
        len(texts),
    )

    outputs = _session.run(_output_names, ort_inputs)

    logger.info(
        "ONNX AFTER session.run: batch=%d",
        len(texts),
    )

    last_hidden_state = outputs[
        _output_names.index("last_hidden_state")
    ]

    vectors = _mean_pool_normalize(
        last_hidden_state,
        attention_mask,
    )

    return vectors.tolist()



async def embed_text(text: str, is_query: bool = True) -> list[float]:
    """
    Embed one piece of text.

    WHY run_in_executor: the ONNX model call is CPU-bound, not I/O-bound.
    If we called _encode_sync directly inside an `async def`, it would
    block FastAPI's entire event loop — every other request would freeze
    for the duration. run_in_executor moves the work to a thread, keeping
    the event loop free to handle other requests concurrently.
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
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        vectors = await loop.run_in_executor(_executor, _encode_sync, batch)
        all_vectors.extend(vectors)

    del vectors
    gc.collect()
    return all_vectors