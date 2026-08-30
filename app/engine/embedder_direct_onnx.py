"""
embedder_direct_onnx.py — local ONNX embedder (bge-small-en-v1.5),
WITHOUT optimum or transformers.

STATUS: not wired into production yet. This is a drop-in candidate for
embedder.py, kept as a separate file specifically so it can be validated
against the current embedder.py (see scripts/compare_embedders.py) before
anything gets switched over. embedder.py is untouched.

WHY THIS EXISTS: optimum's base install requires torch>=1.11 and datasets
unconditionally, regardless of only using its ONNX runtime path - see the
audit that led here. This file replaces:
  optimum.onnxruntime.ORTModelForFeatureExtraction  ->  onnxruntime.InferenceSession
  transformers.AutoTokenizer                         ->  tokenizers.Tokenizer
using the SAME already-exported model.onnx + tokenizer.json files at
models/bge-small/ - no re-export, no re-training, same weights, same
tokenizer pipeline (tokenizer.json IS the serialized fast-tokenizer
pipeline transformers' AutoTokenizer was already using under the hood,
so this should tokenize identically, not just similarly).

WHAT'S DIFFERENT FROM embedder.py, MECHANICALLY:
  - transformers.AutoTokenizer.from_pretrained(dir) reads several files
    (tokenizer_config.json, vocab.txt, tokenizer.json, special_tokens_map.json)
    and picks a tokenizer implementation. tokenizers.Tokenizer.from_file()
    reads tokenizer.json ALONE - it's a fully self-contained serialization
    of the same pipeline (normalizer, pre-tokenizer, model, post-processor
    that adds [CLS]/[SEP]), so this is loading the same pipeline, not a
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
is unchanged on purpose, so this can be a true drop-in replacement.
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
    Called once at app startup - same call site/contract as embedder.py's
    load_embedder(cfg): main.py's lifespan does
    app.state.embedder = load_embedder(cfg). No production wiring changes
    yet; this exists so it CAN be swapped in once validated.
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

    # Same intent as embedder.py's padding=True / truncation=True kwargs,
    # just configured once here instead of passed per-call: pad to the
    # LONGEST sequence in each batch (length=None), not a fixed length.
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
    Identical math to embedder.py's _mean_pool_normalize - only the input
    shape (a raw numpy array here, vs an object with a .last_hidden_state
    attribute there) differs, because session.run() returns plain arrays
    instead of a wrapped output object.
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
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    outputs = _session.run(_output_names, ort_inputs)
    last_hidden_state = outputs[_output_names.index("last_hidden_state")]

    vectors = _mean_pool_normalize(last_hidden_state, attention_mask)
    return vectors.tolist()


async def embed_text(text: str, is_query: bool = True) -> list[float]:
    """Same interface, same query-prefix behavior as embedder.py."""
    if is_query:
        text = f"Represent this sentence for searching relevant passages: {text}"
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _encode_sync, [text])
    return result[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Same interface, same batch size 32, as embedder.py."""
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