"""
test_call_graph.py — verify call graph inversion and BFS expansion

Run with: pytest tests/ -v
"""
from dataclasses import dataclass, field
from app.engine.call_graph import build_called_by, expand_by_graph, build_name_index


@dataclass
class FakeChunk:
    name: str
    calls: list = field(default_factory=list)
    called_by: list = field(default_factory=list)


def test_called_by_inversion_basic():
    """If A calls B, then B.called_by should contain A."""
    a = FakeChunk(name="process_payment", calls=["charge_card"])
    b = FakeChunk(name="charge_card", calls=[])
    build_called_by([a, b])
    assert "process_payment" in b.called_by


def test_called_by_no_self_duplicate():
    """Multiple calls to the same function shouldn't duplicate the called_by entry."""
    a = FakeChunk(name="retry_wrapper", calls=["do_thing", "do_thing"])  # calls it twice (e.g. in a loop)
    b = FakeChunk(name="do_thing", calls=[])
    build_called_by([a, b])
    assert b.called_by.count("retry_wrapper") == 1


def test_called_by_multiple_callers():
    a = FakeChunk(name="caller_one", calls=["shared_util"])
    b = FakeChunk(name="caller_two", calls=["shared_util"])
    c = FakeChunk(name="shared_util", calls=[])
    build_called_by([a, b, c])
    assert set(c.called_by) == {"caller_one", "caller_two"}


def test_called_by_no_callers_stays_empty():
    a = FakeChunk(name="orphan_function", calls=[])
    build_called_by([a])
    assert a.called_by == []


# ── Graph expansion (BFS) tests ────────────────────────────────────────────────

def _chunk_dict(id_, name, calls=None, called_by=None):
    return {
        "id": id_, "name": name, "calls": calls or [],
        "called_by": called_by or [], "file": "test.py",
        "type": "function", "decorators": [],
    }


def test_expand_finds_direct_callee():
    entry = _chunk_dict("1", "entry_point", calls=["helper"])
    helper = _chunk_dict("2", "helper")
    index, qualified_index, _ = build_name_index([entry, helper])

    result = expand_by_graph([entry], index, qualified_index, depth=1, direction="callees")
    ids = {c["id"] for c in result}
    assert "2" in ids


def test_expand_respects_max_expanded_cap():
    entry = _chunk_dict("0", "hub", calls=[f"leaf_{i}" for i in range(50)])
    leaves = [_chunk_dict(str(i+1), f"leaf_{i}") for i in range(50)]
    index, qualified_index, _ = build_name_index([entry] + leaves)

    result = expand_by_graph([entry], index, qualified_index, depth=1, direction="callees", max_expanded=10)
    # entry itself + at most 10 expanded
    assert len(result) <= 11


def test_expand_direction_callers_only():
    """direction='callers' should walk called_by, NOT calls."""
    entry = _chunk_dict("1", "target", calls=["should_not_appear"], called_by=["caller_a"])
    decoy = _chunk_dict("2", "should_not_appear")
    caller = _chunk_dict("3", "caller_a")
    index, qualified_index, _ = build_name_index([entry, decoy, caller])

    result = expand_by_graph([entry], index, qualified_index, depth=1, direction="callers")
    ids = {c["id"] for c in result}
    assert "3" in ids        # caller_a found via called_by
    assert "2" not in ids    # should_not_appear NOT found, since direction=callers skips `calls`


def test_expand_deduplicates_entry_chunks():
    """Entry chunks should never appear twice in the result, even if they call each other."""
    a = _chunk_dict("1", "a", calls=["b"])
    b = _chunk_dict("2", "b", calls=["a"])
    index, qualified_index, _ = build_name_index([a, b])

    result = expand_by_graph([a, b], index, qualified_index, depth=2, direction="both")
    ids = [c["id"] for c in result]
    assert len(ids) == len(set(ids))   # no duplicates
