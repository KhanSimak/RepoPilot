"""
agent/graph.py — wires reasoning_node / retrieve_node / expand_graph_node /
answer_node into the ReAct loop: reasoning decides, an action runs, its
observation feeds back into reasoning, repeat until check_stop_condition
routes to answer.
"""
from functools import partial

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    reasoning_node,
    retrieve_node,
    expand_graph_node,
    answer_node,
    check_stop_condition,
)
from app.agent.codegen_nodes import generate_diff_node


def build_agent_graph(qdrant_client, redis_client, cfg):
    graph = StateGraph(AgentState)

    graph.add_node("reasoning", reasoning_node)
    graph.add_node("retrieve", partial(retrieve_node, qdrant_client=qdrant_client, redis_client=redis_client, cfg=cfg))
    graph.add_node("expand_graph", partial(expand_graph_node, qdrant_client=qdrant_client, cfg=cfg))
    graph.add_node("answer", answer_node)
    graph.add_node("generate_diff", generate_diff_node)

    graph.set_entry_point("reasoning")

    graph.add_conditional_edges(
        "reasoning",
        check_stop_condition,
        {
            "retrieve": "retrieve",
            "expand_graph": "expand_graph",
            "answer": "answer",
            "generate_diff": "generate_diff",
        },
    )
    graph.add_edge("retrieve", "reasoning")
    graph.add_edge("expand_graph", "reasoning")
    graph.add_edge("answer", END)
    graph.add_edge("generate_diff", END)

    return graph.compile()


async def run_agent(
    question: str,
    repo_id: str,
    hyde_snippet: str,
    intent: str,
    phrases: list[str],
    qdrant_client,
    redis_client,
    cfg,
) -> AgentState:
    compiled = build_agent_graph(qdrant_client, redis_client, cfg)

    initial_state: AgentState = {
        "question": question,
        "repo_id": repo_id,
        "hyde_snippet": hyde_snippet,
        "phrases": phrases,
        "intent": intent,
        "gathered_chunks": [],
        "reasoning_trace": [],
        "iteration": 0,
        "next_action": None,
        "action_query": None,
        "graph_name_index": None,
        "graph_qualified_index": None,
        "retrieval_exhausted": False,
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
        "final_answer": None,
        "stop_reason": None,
        "proposed_diff": None,
        "diff_explanation": None,
    }

    return await compiled.ainvoke(initial_state)