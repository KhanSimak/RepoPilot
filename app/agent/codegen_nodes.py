"""
agent/codegen_nodes.py — Phase 1+2 of the coding agent: the terminal node
for intent == "generate_code_change".

Deliberately reuses the SAME reasoning_node / retrieve_node /
expand_graph_node loop nodes.py already has for Q&A - context gathering,
evidence gating, and iteration capping don't need a second implementation.
nodes.py already contains the "answer" -> "generate_diff" redirect and
check_stop_condition support for it; this file only adds the node that
redirect points to.

TWO-MODEL SPLIT + FALLBACK:
  Every OTHER call in this loop (reasoning_node's retrieve/expand_graph/
  answer decisions) keeps using nodes.py's existing reasoning model as-is.
  This file's one LLM call - actually WRITING the diff - uses a DEDICATED
  coding model, with a second model as fallback if the primary is
  rate-limited (free-tier endpoints return HTTP 429 once their daily/
  per-minute quota is hit - this is the "token limit is over" case).

  PRIMARY:  cohere/north-mini-code:free  - Cohere's agentic coding model,
            256K context, confirmed live on OpenRouter's free tier.
  FALLBACK: poolside/laguna-xs-2.1:free  - Poolside's current free coding
            agent model. (Poolside's LARGER Laguna M.1:free endpoint - the
            one originally proposed - was delisted from OpenRouter's free
            tier in early August 2026; XS-2.1 is the closest still-free
            replacement.) Qwen3-Coder-480B-A35B:free was considered too,
            but that endpoint has also been delisted as of mid-2026.

  IMPORTANT - this catalog rotates weekly. Both model IDs below are
  read from settings, not hardcoded into the fallback LOGIC, specifically
  so a delisted endpoint is a one-line config change, not a code change.
  Before deploying, verify both IDs are still live at
  https://openrouter.ai/models (filter: Free).

REQUIRED CONFIG (add to app/config.py's settings):
    coding_model_api_key:    str  - OpenRouter API key
    coding_model_base_url:   str  - "https://openrouter.ai/api/v1"
    coding_model:            str  - primary model ID (defaults to
                                     "cohere/north-mini-code:free" if unset)
    coding_model_fallback:   str  - fallback model ID (defaults to
                                     "poolside/laguna-xs-2.1:free" if unset)

READ-ONLY BY DESIGN: this node calls the model and returns text. It never
writes to disk, never touches git, never applies anything. Applying a
proposed diff happens in a later, separately sandboxed step (a later
phase) that this node has no access to.
"""
import json
import logging

from openai import AsyncOpenAI, RateLimitError

from app.agent.state import AgentState
from app.agent.diff_prompt import build_diff_prompt
from app.engine.token_budget import select_context, count_tokens
from app.engine.reranker import rerank, is_low_confidence
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_DEFAULT_PRIMARY_MODEL = "cohere/north-mini-code:free"
_DEFAULT_FALLBACK_MODEL = "poolside/laguna-xs-2.1:free"

PRIMARY_CODING_MODEL = getattr(settings, "coding_model", None) or _DEFAULT_PRIMARY_MODEL
FALLBACK_CODING_MODEL = getattr(settings, "coding_model_fallback", None) or _DEFAULT_FALLBACK_MODEL

# Separate from nodes.py's _groq_llm/_openrouter_llm on purpose - see the
# module docstring. Only created if the coding-model settings are present,
# so an app that hasn't configured Phase 2 yet doesn't fail at import time
# for a code path it isn't using. Both primary and fallback are free
# OpenRouter endpoints, so ONE client (same base_url/api_key) serves both -
# only the `model=` string changes between the two attempts.
_coding_llm = (
    AsyncOpenAI(
        api_key=settings.coding_model_api_key,
        base_url=settings.coding_model_base_url,
        timeout=45.0,
        max_retries=0,  # we drive the retry/fallback decision ourselves
    )
    if getattr(settings, "coding_model_api_key", None)
    and getattr(settings, "coding_model_base_url", None)
    else None
)


def _is_rate_limited(exc: Exception) -> bool:
    """True for a 429 from either SDK error type or a raw HTTP status,
    checked defensively (attributes, not one exception class) since a
    free-tier proxy's error shape can vary by provider."""
    if isinstance(exc, RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def _token_usage(response, input_text: str, output_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    return (
        input_tokens if isinstance(input_tokens, int) else count_tokens(input_text),
        output_tokens if isinstance(output_tokens, int) else count_tokens(output_text),
    )


async def _generate_diff_with_fallback(system_prompt: str, user_msg: str):
    """Try the primary coding model; on a rate-limit response, retry once
    with the fallback model. Any other error propagates - only quota
    exhaustion triggers the fallback, not a genuine request failure."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        logger.info("generate_diff: calling primary model '%s'", PRIMARY_CODING_MODEL)
        resp = await _coding_llm.chat.completions.create(
            model=PRIMARY_CODING_MODEL,
            max_completion_tokens=1200,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return resp, PRIMARY_CODING_MODEL
    except Exception as exc:
        if not _is_rate_limited(exc):
            raise
        logger.warning(
            "generate_diff: primary model '%s' rate-limited (%s), "
            "falling back to '%s'.",
            PRIMARY_CODING_MODEL, exc, FALLBACK_CODING_MODEL,
        )

    resp = await _coding_llm.chat.completions.create(
        model=FALLBACK_CODING_MODEL,
        max_completion_tokens=1200,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return resp, FALLBACK_CODING_MODEL


async def generate_diff_node(state: AgentState) -> dict:
    cfg = get_settings()

    # --------------------------------------------------
    # Nothing gathered
    # --------------------------------------------------
    if not state["gathered_chunks"]:
        return {
            "final_answer": None,
            "proposed_diff": None,
            "diff_explanation": (
                "I couldn't retrieve any relevant code from the repository "
                "to propose a change."
            ),
            "stop_reason": "no_context",
        }

    if _coding_llm is None:
        logger.error(
            "generate_diff_node: coding_model_api_key/coding_model_base_url "
            "not configured - see app/agent/codegen_nodes.py's module "
            "docstring for the required settings."
        )
        return {
            "final_answer": None,
            "proposed_diff": None,
            "diff_explanation": (
                "The coding model isn't configured yet, so I can't propose "
                "a diff. (coding_model_api_key / coding_model_base_url "
                "missing from settings.)"
            ),
            "stop_reason": "coding_model_not_configured",
        }

    # --------------------------------------------------
    # Rerank ALL gathered chunks (same pattern as answer_node)
    # --------------------------------------------------
    reranked = await rerank(
        state["question"],
        state["gathered_chunks"],
        top_n=min(30, len(state["gathered_chunks"])),
    )

    if is_low_confidence(reranked, cfg):
        return {
            "final_answer": None,
            "proposed_diff": None,
            "diff_explanation": (
                "I couldn't find code in this repository that confidently "
                "supports this change, even after retrieval and graph "
                "expansion."
            ),
            "stop_reason": "low_confidence",
        }

    # --------------------------------------------------
    # Select context within token budget, build the diff prompt
    # --------------------------------------------------
    final_chunks = select_context(reranked)
    system_prompt, user_msg = build_diff_prompt(state["question"], final_chunks)

    # --------------------------------------------------
    # Generate the proposed diff (still just text - nothing applied).
    # Tries the primary coding model first, falls back to the secondary
    # one on a rate-limit response.
    # --------------------------------------------------
    try:
        resp, model_used = await _generate_diff_with_fallback(system_prompt, user_msg)
    except Exception as exc:
        logger.error("generate_diff: both coding models failed (%s).", exc)
        return {
            "final_answer": None,
            "proposed_diff": None,
            "diff_explanation": (
                "Both the primary and fallback coding models were "
                "unavailable, so I couldn't propose a diff. Try again "
                "shortly."
            ),
            "stop_reason": "coding_models_unavailable",
        }

    raw = (resp.choices[0].message.content or "").strip()
    input_tokens, output_tokens = _token_usage(resp, system_prompt + user_msg, raw)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
        parsed = json.loads(cleaned)
        diff_text = (parsed.get("diff") or "").strip() or None
        explanation = parsed.get("explanation", "").strip()
    except json.JSONDecodeError as e:
        logger.warning("Diff generation parse failed (%s).", e)
        diff_text = None
        explanation = f"Model returned an unparsable response ({e})."

    return {
        "proposed_diff": diff_text,
        "diff_explanation": explanation,
        # Keep any Q&A-shaped consumer (e.g. a chat UI) working without
        # having to special-case this intent.
        "final_answer": explanation,
        "stop_reason": "diff_proposed" if diff_text else "diff_generation_failed",
        "agent_input_tokens": state.get("agent_input_tokens", 0) + input_tokens,
        "agent_output_tokens": state.get("agent_output_tokens", 0) + output_tokens,
    }