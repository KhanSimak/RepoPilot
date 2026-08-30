"""
agent/diff_prompt.py — prompt construction for generate_diff_node.

Deliberately separate from token_budget.build_prompt(): that function is
tuned per-Q&A-intent (find_function/understand_flow/find_usage/debug) and
is exercised by the existing, working Q&A path. Code-gen gets its own
prompt builder so nothing here can regress the Q&A prompt.
"""

DIFF_SYSTEM_PROMPT = """\
You are proposing a code change based ONLY on the supplied repository
context.

Rules:
1. Use ONLY the provided repository context as ground truth for existing
   code - never invent files, functions, classes, or APIs not shown.
2. Produce a MINIMAL, targeted unified diff. Do not rewrite unrelated code,
   reformat untouched lines, or "clean up" code outside the scope of the
   request.
3. The diff must be valid unified diff format (git diff style) that would
   apply cleanly against the file paths and line ranges shown above.
4. Never modify test files unless the request explicitly asks for test
   changes.
5. If the repository context is insufficient to make this change safely,
   return an empty "diff" and explain what's missing instead of guessing.

Respond ONLY with JSON, no other text, no markdown fences:

{
    "explanation": "brief description of the change and why it's correct",
    "files": ["list of file paths touched"],
    "diff": "a single unified diff implementing the change, or an empty string if you cannot safely propose one"
}
"""


def build_diff_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    context = []
    for i, c in enumerate(chunks, start=1):
        calls = c.get("calls") or []
        called_by = c.get("called_by") or []
        context.append(
            f"""
Symbol {i}
Type: {c.get("type")}
Name: {c.get("name")}
File: {c.get("file")}
Lines: {c.get("line_start")}-{c.get("line_end")}
Calls: {", ".join(calls) if calls else "(none)"}
Called by: {", ".join(called_by) if called_by else "(none)"}

Code:

{c.get("text") or c.get("raw_source") or ""}
"""
        )
    context_block = "\n\n".join(context)

    user_msg = f"""
Repository context:

{context_block}

Requested change:
{question}

Propose a minimal unified diff implementing this change, using only the
repository context above. If the context doesn't reach the code that
actually needs to change, say so instead of guessing at code you haven't
seen.
"""
    return DIFF_SYSTEM_PROMPT, user_msg