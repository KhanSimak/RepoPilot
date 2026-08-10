"""
agent/state.py — state schema for the iterative ReAct loop

WHY THIS EXISTS: retriever.py's graph expansion today is ONE-SHOT — for
understand_flow / find_usage / debug intents, expand_by_graph() runs exactly
once, at a fixed depth=2, from whatever the initial RRF fusion returned.
That's often not enough: "how does auth work end-to-end" might need to
expand from a DIFFERENT entry point than the top RRF hit, or need a second
expansion pass once the first one reveals which function is actually
central. This state schema is what lets the loop decide that iteratively
instead of guessing depth=2 is alAways right.

Chunk dicts here have the SAME shape as everywhere else in this codebase
(id, name, type, file, calls, called_by, etc. — see models/chunk.py) —
deliberately not a new shape, so gathered_chunks can be passed straight
into select_context() / build_prompt() from token_budget.py unchanged.
"""
from typing import Literal, TypedDict

class AgentState(TypedDict):
    question: str
    repo_id: str
    hyde_snippet: str
    phrases: list[str]
    intent: str

    gathered_chunks: list[dict]
    reasoning_trace: list[str]
    iteration: int

    next_action: Literal["retrieve", "expand_graph", "answer"] | None
    action_query: str | None
    graph_name_index: dict | None
    graph_qualified_index: dict | None
    retrieval_exhausted: bool

    final_answer: str | None
    stop_reason: str | None
