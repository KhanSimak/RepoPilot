import asyncio
import json
from types import SimpleNamespace

from app.agent import nodes
from app.agent import tools
from app.query import pipeline


class _FakeCompletions:
    async def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({
                    "thought": "The collected implementation is sufficient.",
                    "action": "answer",
                    "action_input": "",
                    "missing_information": [],
                })
            ))],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=17),
        )


class _ExpandCompletions:
    async def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({
                    "thought": "The requested initialization step is not gathered.",
                    "action": "expand_graph",
                    "action_input": "Bootstrap.__init__",
                    "missing_information": [],
                })
            ))],
            usage=SimpleNamespace(prompt_tokens=101, completion_tokens=19),
        )


def _runtime_chunk(identifier, name, calls):
    return {
        "id": identifier,
        "name": name,
        "type": "function",
        "file": "package/settings.py",
        "calls": calls,
        "called_by": [],
        "raw_source": """def load_values(application):
    value = application.settings
    value.load_mapping({})
    application.settings = value
    return value
""",
        "text": "def load_values(application):\n    value = application.settings\n    value.load_mapping({})\n    application.settings = value\n    return value\n",
    }


def test_abstract_detection_uses_python_syntax_not_docstring_words():
    chunk = {
        "raw_source": '''def load_values(self):
    """Pass an option to change the loading behavior."""
    for key, value in {}.items():
        self[key] = value
    return True
''',
    }

    assert not nodes._is_abstract_chunk(chunk)
    assert nodes._is_abstract_chunk({"raw_source": "def operation(self):\n    pass\n"})
    assert nodes._is_abstract_chunk({
        "raw_source": "def operation(self):\n    raise NotImplementedError\n"
    })
    assert nodes._is_abstract_chunk({
        "raw_source": "@abstractmethod\ndef operation(self):\n    return None\n"
    })


def test_runtime_detection_uses_function_body_not_file_path_or_length():
    chunk = {
        "type": "function",
        "file": "docs/example.py",
        "raw_source": "def execute():\n    return 1\n",
    }

    assert nodes._is_runtime_implementation(chunk)


def test_reasoning_summary_includes_retrieved_source():
    summary = nodes._summarize_context([{
        "name": "Worker.execute",
        "type": "method",
        "file": "package/worker.py",
        "line_start": 1,
        "calls": ["resolve"],
        "called_by": [],
        "raw_source": "def execute(self):\n    return resolve()\n",
    }])

    assert "def execute(self):" in summary
    assert "return resolve()" in summary


def test_requested_graph_symbol_is_not_replaced_by_an_unrelated_candidate(monkeypatch):
    fake_llm = SimpleNamespace(chat=SimpleNamespace(completions=_ExpandCompletions()))
    monkeypatch.setattr(nodes, "_llm", fake_llm)

    async def identity_rerank(_question, chunks, top_n):
        return chunks[:top_n]

    monkeypatch.setattr(nodes, "rerank", identity_rerank)
    state = {
        "question": "How does the application initialize its settings?",
        "repo_id": "sample-repository",
        "intent": "understand_flow",
        "gathered_chunks": [_runtime_chunk("1", "Settings.load", ["Bootstrap.__init__"])],
        "reasoning_trace": [],
        "iteration": 1,
        "retrieval_exhausted": False,
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
    }

    result = asyncio.run(nodes.reasoning_node(state))

    assert result["next_action"] == "retrieve"
    assert result["action_query"] == "function Bootstrap.__init__"


def test_exact_symbol_follow_up_includes_indexed_repository_evidence(monkeypatch):
    async def semantic_candidates(**_kwargs):
        return [{"id": "other", "name": "Other.__init__"}]

    async def repository_chunks(*_args):
        return [{"id": "target", "name": "Bootstrap.__init__"}]

    monkeypatch.setattr(tools, "retrieve", semantic_candidates)
    monkeypatch.setattr(tools, "scroll_repo_chunks", repository_chunks)
    cfg = SimpleNamespace(qdrant_collection="chunks")

    result = asyncio.run(tools.retrieval_tool(
        query="function Bootstrap.__init__",
        repo_id="sample-repository",
        intent="understand_flow",
        qdrant_client=None,
        redis_client=None,
        cfg=cfg,
        exact_symbol="Bootstrap.__init__",
    ))

    assert [chunk["name"] for chunk in result] == [
        "Bootstrap.__init__", "Other.__init__"
    ]


def test_flow_question_can_converge_after_runtime_evidence(monkeypatch):
    """A complete flow must not be overridden into another graph expansion."""
    fake_llm = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    monkeypatch.setattr(nodes, "_llm", fake_llm)

    async def identity_rerank(_question, chunks, top_n):
        return chunks[:top_n]

    monkeypatch.setattr(nodes, "rerank", identity_rerank)
    state = {
        "question": "How does this application load settings values?",
        "repo_id": "sample-repository",
        "intent": "understand_flow",
        "gathered_chunks": [
            _runtime_chunk("1", "Settings.load_mapping", ["Settings.load_object"]),
            _runtime_chunk("2", "Settings.load_object", ["Settings.load_environment"]),
            _runtime_chunk("3", "Settings.load_environment", ["Settings.load_file"]),
        ],
        "reasoning_trace": [],
        "iteration": 3,
        "retrieval_exhausted": False,
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
    }

    result = asyncio.run(nodes.reasoning_node(state))

    assert result["next_action"] == "answer"
    assert result["agent_input_tokens"] == 123
    assert result["agent_output_tokens"] == 17


def test_agent_endpoint_populates_trace_token_usage(monkeypatch):
    async def fake_rewrite(*_args):
        return {
            "implementation_summary": "application settings loading",
            "phrases": ["settings"],
            "intent": "understand_flow",
        }

    async def fake_run_agent(**_kwargs):
        return {
            "final_answer": "Configuration is loaded from repository code.",
            "iteration": 1,
            "reasoning_trace": [],
            "gathered_chunks": [],
            "agent_input_tokens": 321,
            "agent_output_tokens": 45,
        }

    monkeypatch.setattr(pipeline, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)

    result = asyncio.run(pipeline.run_agentic_query(
        "How does this application load settings values?",
        "sample-repository",
        None,
        None,
        None,
    ))
    stages = {stage["stage"]: stage for stage in result["trace"]["stages"]}

    assert stages["hyde_rewrite"]["tokens_in"] > 0
    assert stages["hyde_rewrite"]["tokens_out"] > 0
    assert stages["agent_loop"]["tokens_in"] == 321
    assert stages["agent_loop"]["tokens_out"] == 45
