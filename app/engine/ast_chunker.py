"""
ast_chunker.py — Python AST-based code chunker

This is the most important file in the project. It's why this RAG system
is better than a generic "split text every 512 characters" approach.

THE PROBLEM with fixed-size chunking:
  A 100-line function gets cut into chunk[0:512] and chunk[512:1024].
  Neither half makes sense alone. The LLM gets a fragment, not a function.

THE SOLUTION — AST chunking:
  Python's built-in `ast` module parses source code into a tree of nodes.
  We walk the tree and grab every FunctionDef, AsyncFunctionDef, and ClassDef
  node. Each one becomes EXACTLY one chunk — never split, never merged.

WHAT WE EXTRACT BEYOND JUST THE CODE:
  - calls:      what other functions this one calls (for "find usages" later)
  - imports:    module-level dependencies
  - complexity: cyclomatic complexity (branches = harder to understand)
  - docstring:  used to enrich the embedding text

This file has zero external dependencies — `ast` is part of the
Python standard library.
"""

import ast
import uuid
from app.models.chunk import CodeChunk


def _make_id(repo_id: str, file: str, name: str, line: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{repo_id}:{file}:{name}:{line}"))


def _extract_calls(node: ast.AST) -> list[str]:
    """
    Walk a function's body and collect every function/method call inside it.

    ast.Call nodes look like:
      stripe.Charge.create(...)   → func is an ast.Attribute → we want 'create'
      process_payment(...)        → func is an ast.Name      → we want 'process_payment'
    """
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
            elif isinstance(child.func, ast.Name):
                calls.add(child.func.id)
    return sorted(calls)


def _extract_decorators(source: str, node: ast.AST) -> list[str]:
    """
    Extract decorator names.

    @app.route("/")        -> app.route
    @router.get("/")       -> router.get
    @login_required        -> login_required
    """
    decorators = []

    if not hasattr(node, "decorator_list"):
        return decorators

    for dec in node.decorator_list:

        text = ast.get_source_segment(source, dec)

        if text:
          decorators.append("@" + text)

    return decorators

def _extract_imports(tree: ast.Module) -> list[str]:
    """Top-level `import x` and `from x import y` statements in the file."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return list(set(imports))


def _cyclomatic_complexity(node: ast.AST) -> int:
    """
    Count branch points: if/elif, for, while, except, and/or, with.
    complexity = branches + 1.

    A function with complexity 1 has no branches (straight-line code).
    A function with complexity 10 has many decision points — harder to
    understand, more important to chunk and embed carefully.
    """
    branch_nodes = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.BoolOp)
    return sum(1 for _ in ast.walk(node) if isinstance(_, branch_nodes)) + 1


def _parent_class_map(tree: ast.Module) -> dict[int, str]:
    """
    Build a map from node id -> enclosing class name.
    Lets us tell the difference between a top-level `function` and a `method`.
    """
    parents: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if child is not node:
                    parents[id(child)] = node.name
    return parents


def _build_embed_text(
    chunk_type: str,
    decorators: list[str],
    name: str,
    file: str,
    parent: str | None,
    docstring: str,
    calls: list[str],
    source: str,
) -> str:
    """
    Build the text that actually gets embedded.

    KEY INSIGHT: we don't embed just the raw code. We prepend metadata
    so the vector captures PURPOSE, not just syntax.

      "File: services/payments.py
       Function: process_payment
       Description: Process a payment via Stripe.
       Calls: charge, log_transaction

       def process_payment(amount, user_id):
           ..."

    This embedding is much richer than embedding the raw function alone —
    a query like "how is Stripe used" matches on the "Calls: charge" line
    even if the word "Stripe" never appears in the function body itself.
    """
    parts = [f"File: {file}", f"{chunk_type.title()}: {name}"]
    if parent:
        parts.append(f"Class: {parent}")
    if docstring:
        parts.append(f"Description: {docstring[:200]}")
    if calls:
        parts.append(f"Calls: {', '.join(calls[:8])}")
    if decorators:
        parts.append(f" Registered via: {', '.join(decorators)}")
    parts.append("")
    parts.append(source)
    
       
    return "\n".join(parts)

def _qualified_name(filepath: str, parent: str | None, name: str) -> str:
    if parent:
        return f"{filepath}::{parent}.{name}"
    return f"{filepath}::{name}"

def chunk_python_file(source: str, filepath: str, repo_id: str) -> list[CodeChunk]:
    """
    The main entry point. Takes raw Python source + its file path,
    returns a list of CodeChunk objects — one per function/class/method.

    If the file has a syntax error, we return an empty list rather than
    crashing the whole ingest job.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines       = source.splitlines()
    parent_map  = _parent_class_map(tree)
    mod_imports = _extract_imports(tree)
    chunks: list[CodeChunk] = []

    for node in ast.walk(tree):
        is_func  = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_class = isinstance(node, ast.ClassDef)

        if not (is_func or is_class):
            continue
        if not hasattr(node, "end_lineno"):
            continue   # extremely old Python AST without end_lineno — skip

        raw_lines = lines[node.lineno - 1 : node.end_lineno]
        if len(raw_lines) < 2:
            continue   # skip trivial one-liners — not worth a chunk

        raw_source = "\n".join(raw_lines)
        docstring  = ast.get_docstring(node) or ""
        decorators = _extract_decorators(source, node)
        calls      = _extract_calls(node) if is_func else []
        parent     = parent_map.get(id(node))
        qualified_name = _qualified_name(
            filepath,
            parent,
            node.name,
        )
        complexity = _cyclomatic_complexity(node) if is_func else 1

        if is_class:
            chunk_type = "class"
        elif parent:
            chunk_type = "method" 
        else:
            chunk_type = "function"

        embed_text = _build_embed_text(
            chunk_type,decorators, node.name, filepath, parent, docstring, calls, raw_source
        )

        chunks.append(CodeChunk(
            id          = _make_id(repo_id, filepath, node.name, node.lineno),
            repo_id     = repo_id,
            text        = embed_text,
            raw_source  = raw_source,
            name=f"{parent}.{node.name}" if parent else node.name,
            type        = chunk_type,
            file        = filepath,
            qualified_name=qualified_name,
            language    = "python",
            line_start  = node.lineno,
            line_end    = node.end_lineno,
            docstring   = docstring,
            decorators=decorators,
            
            registered_by=decorators,
            calls       = calls,
            imports     = mod_imports,
            complexity  = complexity,
        ))


    return chunks

