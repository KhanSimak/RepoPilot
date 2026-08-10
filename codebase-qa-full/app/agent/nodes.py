"""
agent/nodes.py — node functions for the ReAct loop.

Reuses build_prompt()/select_context() from token_budget.py for the final
answer (same prompt construction the static pipeline already uses) and the
same AsyncGroq client pattern rewriter.py uses, rather than inventing a
second, inconsistent way of talking to Groq.
"""
import json
import logging
import re

from groq import AsyncGroq

from app.agent.state import AgentState
from app.agent.tools import retrieval_tool, expand_graph_tool
from app.engine.token_budget import select_context, build_prompt, count_tokens
from app.engine.reranker import rerank, is_low_confidence
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
_llm = AsyncGroq(api_key=settings.groq_api_key)

MAX_AGENT_ITERATIONS = 4

REASONING_PROMPT = """You are investigating a codebase to answer a question,
deciding one step at a time.

Question: {question}

Chunks gathered so far ({num_chunks}):
{context_summary}

Reasoning trace so far:
{trace}

Choose exactly one next action:
- "retrieve": you need different/additional code context. Give a refined
  search query — more specific than the original, informed by what you've
  seen so far.
- "expand_graph": you have a specific function/class name from the chunks
  above and the question is about flow/usage ("what calls this", "what
  happens after this runs") — expanding the call graph from that symbol
  would help more than another retrieval pass.
  If the question asks HOW something works, WHAT CALLS WHAT,
or asks for the execution flow,
prefer expand_graph until you have the execution path.

Do NOT answer after only seeing registration helpers
(route, add_url_rule, decorators).

Never answer from framework knowledge.

Only answer if the retrieved context contains the complete execution path.

If any major step of the execution path is missing, choose "retrieve".

Do not infer missing functions.

Do not guess missing calls.

Respond ONLY with JSON, no other text, no markdown fences:
{{"thought": "<brief reasoning>", "action": "retrieve|expand_graph|answer", "action_input": "<refined query or exact symbol name, empty string if answering>"}}
"""


def _summarize_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(nothing gathered yet)"
    lines = []

    for c in chunks[:6]:
      line = (
        f"- {c.get('name','?')} "
        f"({c.get('type','?')}) "
        f"in {c.get('file','?')}:{c.get('line_start','?')}"
      )
      calls = c.get("calls", [])
      if calls:
        line += f"\n  Calls: {', '.join(calls[:5])}"

      called_by = c.get("called_by", [])
      if called_by:
        line += f"\n  Called by: {', '.join(called_by[:5])}"
      lines.append(line)
    return "\n".join(lines)


def needs_more_context(
    chunks: list[dict],
    *,
    min_symbols: int = 0,
    min_connected_symbols: int = 0,
    require_runtime_code: bool = False,
) -> bool:
    """Apply the existing execution-completeness thresholds consistently."""
    connected_symbols = sum(
        1 for chunk in chunks if chunk.get("calls") or chunk.get("called_by")
    )
    has_runtime_code = any(
        len((chunk.get("text") or "").splitlines()) > 10
        for chunk in chunks
    )
    return (
        len(chunks) < min_symbols
        or connected_symbols < min_connected_symbols
        or (require_runtime_code and not has_runtime_code)
    )


async def reasoning_node(state: AgentState) -> dict:
    if state.get("retrieval_exhausted"):
        return {
            "next_action": "answer",
            "action_query": "",
            "reasoning_trace": state["reasoning_trace"],
        }

    # --------------------------------------------------
    # Nothing gathered yet
    # --------------------------------------------------
    if not state["gathered_chunks"]:
        return {
            "next_action": "retrieve",
            "action_query": state["question"],
            "reasoning_trace": state["reasoning_trace"]
            + ["No chunks gathered yet. Starting retrieval."],
        }

    # --------------------------------------------------
    # Rerank gathered chunks
    # --------------------------------------------------
    reranked = await rerank(
        state["question"],
        state["gathered_chunks"],
        top_n=min(30, len(state["gathered_chunks"])),
    )

    summary = _summarize_context(reranked)
    trace = "\n".join(state["reasoning_trace"][-4:]) or "(none yet)"

    # --------------------------------------------------
    # Keep shrinking until the ENTIRE prompt fits
    # --------------------------------------------------
    prompt = REASONING_PROMPT.format(
        question=state["question"],
        num_chunks=len(reranked),
        context_summary=summary,
        trace=trace,
    )

    while count_tokens(prompt) > 2500 and len(reranked) > 1:
        reranked = reranked[:-1]
        summary = _summarize_context(reranked)

        prompt = REASONING_PROMPT.format(
            question=state["question"],
            num_chunks=len(reranked),
            context_summary=summary,
            trace=trace,
        )

    # --------------------------------------------------
    # Ask the reasoning model
    # --------------------------------------------------
    resp = await _llm.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=250,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()

    try:
        cleaned = re.sub(
            r"^```json\s*|\s*```$",
            "",
            raw,
            flags=re.MULTILINE,
        ).strip()

        parsed = json.loads(cleaned)

        if parsed["action"] == "answer":

            gathered = state["gathered_chunks"]

            if (
                state["intent"] == "understand_flow"
                and needs_more_context(
                    gathered,
                    min_symbols=5,
                    min_connected_symbols=3,
                    require_runtime_code=True,
                )
            ):
                logger.info(
                "Execution path still incomplete. Retrieving more context."
            )

                parsed["action"] = "retrieve"
                parsed["action_input"] = state["question"]
        logger.debug("Gathered chunk fields: %s", state["gathered_chunks"][0].keys())
        if parsed.get("action") not in {
            "retrieve",
            "expand_graph",
            "answer",
        }:
            raise ValueError(
                f"invalid action: {parsed.get('action')}"
            )

        # --------------------------------------------------
        # Validate expand_graph requests
        # --------------------------------------------------
        if parsed["action"] == "expand_graph":

            available_chunks = [
                c for c in state["gathered_chunks"]
                if c.get("name")
            ]

            available_symbols = {
                c["name"]
                for c in available_chunks
            }

            already_expanded = {
                line.split("'")[1]
                for line in state["reasoning_trace"]
                if "call-graph expansion from '" in line
            }

            symbol = parsed.get("action_input", "")

            # Symbol not gathered yet → retrieve first
            if symbol not in available_symbols:

                logger.info(
                    "Symbol '%s' not gathered yet. Retrieving first.",
                    symbol,
                )

                parsed["action"] = "retrieve"
                parsed["action_input"] = f"function {symbol}"

            # Don't expand twice
            elif symbol in already_expanded:

                remaining = [
                    c for c in available_chunks
                    if c["name"] not in already_expanded
                ]

                if remaining:

                    best = max(
                        remaining,
                        key=lambda c: c.get(
                            "rerank_score",
                            c.get("score", 0.0),
                        ),
                    )

                    parsed["action_input"] = best["name"]

                else:

                    logger.info(
                        "All symbols already expanded."
                    )

                    parsed["action"] = "answer"
                    parsed["action_input"] = ""

        # --------------------------------------------------
        # NEW SAFETY CHECK
        # Don't allow answering before reaching runtime code.
        # --------------------------------------------------
        if parsed["action"] == "answer" and state["intent"] == "understand_flow":

            connected = [
                c for c in state["gathered_chunks"]
                if c.get("calls") or c.get("called_by")
            ]

            if needs_more_context(
                state["gathered_chunks"],
                min_connected_symbols=3,
            ):

                logger.info(
                    "Not enough connected symbols. Expanding graph."
                )

                ranked = sorted(
                    connected or state["gathered_chunks"],
                    key=lambda c: c.get(
                        "rerank_score",
                        c.get("score", 0.0),
                    ),
                    reverse=True,
                )

                if ranked:

                    parsed["action"] = "expand_graph"
                    parsed["action_input"] = ranked[0]["name"]

                else:

                    parsed["action"] = "retrieve"
                    parsed["action_input"] = state["question"]
    except (json.JSONDecodeError, ValueError) as e:

        logger.warning(
          "Reasoning parse failed (%s).",
          e,
    )

        parsed = {
          "thought": f"Reasoning parse failed ({e}).",
          "action": "answer",
          "action_input": "",
       }

    parsed.setdefault("thought", "")
    parsed.setdefault("action_input", "")
    return {
        "next_action": parsed["action"],
        "action_query": parsed.get(
            "action_input",
            "",
        ),
        "gathered_chunks": reranked,
        "reasoning_trace": state["reasoning_trace"]
        + [
            f"Thought: {parsed['thought']} -> Action: {parsed['action']}"
        ],
    }

async def retrieve_node(state: AgentState, qdrant_client, redis_client, cfg) -> dict:
    query = state["action_query"] or state["question"]
    new_chunks = await retrieval_tool(
        query=query,
        repo_id=state["repo_id"],
        intent=state["intent"],   # a targeted follow-up lookup, not the original broad intent
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        cfg=cfg,
    )

    logger.debug(
        "Retrieved chunks: %s",
        [(c.get("id"), c.get("name"), c.get("file")) for c in new_chunks],
    )

    existing_ids = {c["id"] for c in state["gathered_chunks"]}
    logger.debug("Existing chunk IDs: %s", existing_ids)
    deduped = [c for c in new_chunks if c["id"] not in existing_ids]

    trace = state["reasoning_trace"] + [
        f"Observation: retrieved {len(deduped)} new chunks for '{query}'"
    ]
    if not deduped:
        trace.append("Observation: retrieval produced no new chunks; answering with gathered context.")

    return {
        "gathered_chunks": state["gathered_chunks"] + deduped,
        "iteration": state["iteration"] + 1,
        "retrieval_exhausted": not deduped,
        "next_action": "answer" if not deduped else None,
        "action_query": "" if not deduped else state["action_query"],
        "reasoning_trace": trace,
    }


async def expand_graph_node(state: AgentState, qdrant_client, cfg) -> dict:
    symbol = state["action_query"]

    # Expand from the symbol chosen by the reasoning model.
    # If it isn't present, fall back to the current gathered chunks.
    entry_points = [
        c for c in state["gathered_chunks"]
        if c.get("name") == symbol
    ] or state["gathered_chunks"]

    if not entry_points:
        return {
            "iteration": state["iteration"] + 1,
            "reasoning_trace": state["reasoning_trace"]
            + [
                f"Observation: no entry chunks to expand from '{symbol}'"
            ],
        }

    expanded, name_index, qualified_index = await expand_graph_tool(
        entry_points,
        state["repo_id"],
        qdrant_client,
        cfg,
        name_index=state.get("graph_name_index"),
        qualified_index=state.get("graph_qualified_index"),
    )

    # --------------------------------------------------
    # Merge + deduplicate
    # --------------------------------------------------

    combined = state["gathered_chunks"] + expanded

    unique = {}

    for chunk in combined:
        unique[chunk["id"]] = chunk

    combined = list(unique.values())

    # --------------------------------------------------
    # Rerank and keep only the best context
    # --------------------------------------------------

    combined = await rerank(
        state["question"],
        combined,
        top_n=min(30, len(combined)),
    )

    return {
        "gathered_chunks": combined,
        "iteration": state["iteration"] + 1,
        "graph_name_index": name_index,
        "graph_qualified_index": qualified_index,
        "reasoning_trace": state["reasoning_trace"]
        + [
            f"Observation: call-graph expansion from '{symbol}' found {len(expanded)} chunks"
        ],
    }

async def answer_node(state: AgentState) -> dict:
    """
    Final answer generation node.

    Reranks every gathered chunk (retrieval + graph expansion),
    selects the best context within the token budget,
    and generates the final answer.
    """

    cfg = get_settings()

    logger.debug(
        "Gathered chunks (%d): %s",
        len(state["gathered_chunks"]),
        [c.get("name") for c in state["gathered_chunks"]],
    )

    # --------------------------------------------------
    # Nothing gathered
    # --------------------------------------------------
    if not state["gathered_chunks"]:
        return {
            "final_answer": (
                "I couldn't retrieve any relevant code from the repository "
                "to answer this question."
            ),
            "low_confidence": True,
        }

    # --------------------------------------------------
    # Rerank ALL gathered chunks
    # --------------------------------------------------
    reranked = await rerank(
      state["question"],
      state["gathered_chunks"],
      top_n=min(30, len(state["gathered_chunks"]))
    )
    logger.debug(
        "Agent rerank scores: %s",
        [(c.get("name"), c.get("rerank_score")) for c in reranked[:10]],
    )

    # --------------------------------------------------
    # Confidence check
    # --------------------------------------------------
    if is_low_confidence(reranked, cfg):
        return {
            "final_answer": (
                "I couldn't find code in this repository that confidently "
                "answers this question, even after retrieval and graph "
                "expansion."
            ),
            "low_confidence": True,
        }

    # --------------------------------------------------
    # Select context within token budget
    # --------------------------------------------------
    final_chunks = select_context(reranked)

    system_prompt, user_msg = build_prompt(
        state["question"],
        final_chunks,
        state["intent"],
    )

    logger.debug(
        "Prompt sizes: system_chars=%d user_chars=%d total_chars=%d est_tokens=%d",
        len(system_prompt), len(user_msg), len(system_prompt) + len(user_msg),
        (len(system_prompt) + len(user_msg)) // 4,
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------
    resp = await _llm.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=400,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
    )

    return {
        "final_answer": resp.choices[0].message.content.strip()
    }

def check_stop_condition(state: AgentState) -> str:
    """
    Decide which node to execute next.
    """

    action = state.get("next_action")

    # Safety: if the model produced an invalid action,
    # stop and answer with the current context.
    if action not in {"retrieve", "expand_graph", "answer"}:
        return "answer"

    # Model believes it has enough context.
    if action == "answer":
        return "answer"

    # Hard stop to prevent infinite loops.
    if state["iteration"] >= MAX_AGENT_ITERATIONS:
        return "answer"

    # Continue the reasoning loop.
    return action
