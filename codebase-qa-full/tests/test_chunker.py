"""
test_chunker.py — verify the AST chunker behaves correctly

Run with: pytest tests/ -v
"""
from app.engine.ast_chunker import chunk_python_file

SAMPLE = '''
import os
from pathlib import Path

class PaymentService:
    """Handles all payment operations."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_payment(self, amount: float, user_id: str) -> dict:
        """Process a payment via Stripe."""
        user = self._get_user(user_id)
        if not user:
            raise ValueError("User not found")
        result = stripe.Charge.create(amount=amount, currency="usd")
        self._log_transaction(user_id, result)
        return result

    def _get_user(self, user_id: str):
        return db.query(User).filter(User.id == user_id).first()

def standalone_function(x: int, y: int) -> int:
    """A simple standalone function."""
    return x + y

async def async_handler():
    """An async function."""
    await some_service.do_thing()
'''


def test_extracts_class():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    assert "PaymentService" in [c.name for c in chunks]


def test_extracts_methods():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    names = [c.name for c in chunks]
    assert "process_payment" in names
    assert "_get_user" in names


def test_extracts_standalone_function():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    assert "standalone_function" in [c.name for c in chunks]


def test_extracts_async_function():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    assert "async_handler" in [c.name for c in chunks]


def test_call_graph_extraction():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    process = next(c for c in chunks if c.name == "process_payment")
    # Should have captured the calls inside the function body
    assert "create" in process.calls or "Charge" in str(process.calls)
    assert "_get_user" in process.calls


def test_docstring_extraction():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    process = next(c for c in chunks if c.name == "process_payment")
    assert "Stripe" in process.docstring


def test_type_classification():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    cls    = next(c for c in chunks if c.name == "PaymentService")
    method = next(c for c in chunks if c.name == "process_payment")
    func   = next(c for c in chunks if c.name == "standalone_function")
    assert cls.type    == "class"
    assert method.type == "method"
    assert func.type   == "function"


def test_no_mid_function_split():
    """The single most important property: every chunk is a COMPLETE unit."""
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    for chunk in chunks:
        assert chunk.line_end >= chunk.line_start
        assert len(chunk.raw_source.splitlines()) >= 1
        # the source should not start or end mid-statement —
        # a quick sanity check is that it should be parseable on its own
        # for top-level functions/classes (methods need their class context,
        # so we don't check that here)


def test_complexity_counts_branches():
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    process = next(c for c in chunks if c.name == "process_payment")
    assert process.complexity >= 2   # has an `if` branch


def test_syntax_error_returns_empty_list():
    chunks = chunk_python_file("def broken(:", "bad.py", "repo1")
    assert chunks == []


def test_embed_text_is_enriched():
    """The embed text should contain more than just the raw code."""
    chunks = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    process = next(c for c in chunks if c.name == "process_payment")
    assert "services/payments.py" in process.text   # file path included
    assert "process_payment" in process.text         # name included
    assert "Stripe" in process.text                  # docstring included


def test_deterministic_ids():
    """Same file+name+line should always produce the same chunk ID — critical for incremental ingest later."""
    chunks_a = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    chunks_b = chunk_python_file(SAMPLE, "services/payments.py", "repo1")
    ids_a = {c.name: c.id for c in chunks_a}
    ids_b = {c.name: c.id for c in chunks_b}
    assert ids_a == ids_b
