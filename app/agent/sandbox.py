"""
agent/sandbox.py — Phase 3 (isolation) + Phase 4 (review/commit/reject).

PHASE 3 — WHAT THIS DOES: given a repo's local path and a unified diff
(from generate_diff_node), applies that diff ONLY inside an isolated git
worktree on a brand-new branch — never the repo's actual working
directory or its real branches. Minimum viable sandbox: git worktree +
a temp branch, no OS-level isolation needed yet, because nothing here
executes untrusted code — it only applies a text patch via `git apply`.

PHASE 4 — WHAT'S NEW: a human can now COMMIT (merge the applied change
into the base branch) or REJECT (discard the worktree and branch
entirely) a pending apply. Nothing lands on the real branch without an
explicit commit call — that's the gate this phase exists to build.

MERGE SAFETY (the one non-obvious design choice in Phase 4): committing
requires merging the agent's branch into base_branch. The first design
tried for this used a second, throwaway worktree checked out on
base_branch — but git refuses to check out a branch that's ALREADY
checked out elsewhere, and repo_path's own working directory normally
sits on base_branch precisely because that's what cloning produces —
so that approach is blocked by git itself in the ordinary case, not a
theoretical edge case (confirmed by actually hitting the error against
real git). The fix: merge with NO checkout at all, using git's
checkout-free merge plumbing (merge-tree --write-tree + commit-tree +
update-ref) — see _merge_into_base for the full reasoning. This moves
base_branch's ref directly without ever touching repo_path's working
directory or index, exactly like a bare repo receiving a push.

PENDING-APPLY STATE: stored in Redis, not in-memory — the review/commit
request that eventually follows an apply may land on a different Render
process than the one that did the apply. A per-repo Redis SET
(pending_applies_index:{repo_id}) tracks which apply_ids are currently
awaiting review, so "what's waiting for review on this repo" is a cheap
SMEMBERS, not a scan.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

PENDING_APPLY_TTL_SECONDS = 60 * 60 * 24 * 3  # 3 days


@dataclass
class ApplyResult:
    applied: bool
    repo_id: str
    apply_id: str | None = None
    worktree_path: str | None = None
    branch_name: str | None = None
    error: str | None = None


@dataclass
class CommitResult:
    committed: bool
    apply_id: str
    repo_id: str
    commit_hash: str | None = None
    # base_branch's commit BEFORE the merge — Phase 5's re-index needs
    # this as the "last indexed commit" so incremental ingest diffs the
    # right range instead of re-indexing the whole repo.
    previous_commit: str | None = None
    error: str | None = None


@dataclass
class RejectResult:
    rejected: bool
    apply_id: str
    repo_id: str
    error: str | None = None


class SandboxError(RuntimeError):
    """Infra-level failure — distinct from a diff/merge simply not
    applying cleanly, which is an expected, reportable outcome."""


def _run_git(args: list[str], cwd: Path, input_text: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, input=input_text,
        capture_output=True, text=True, timeout=timeout,
    )


def _create_worktree(repo_path: Path, base_branch: str, branch_prefix: str = "agent-apply") -> tuple[Path, str]:
    branch_name = f"{branch_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    worktree_dir = Path(tempfile.mkdtemp(prefix=f"{branch_prefix}-"))

    result = _run_git(["worktree", "add", "-b", branch_name, str(worktree_dir), base_branch], cwd=repo_path)
    if result.returncode != 0:
        shutil.rmtree(worktree_dir, ignore_errors=True)
        raise SandboxError(f"worktree creation failed: {result.stderr.strip()}")

    return worktree_dir, branch_name


def cleanup_worktree(repo_path: Path, worktree_path: Path, branch_name: str) -> None:
    _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=repo_path)
    _run_git(["branch", "-D", branch_name], cwd=repo_path)
    shutil.rmtree(worktree_path, ignore_errors=True)


def apply_diff_to_worktree(worktree_path: Path, diff_text: str) -> tuple[bool, str | None]:
    result = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=worktree_path, input=diff_text, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, None


async def apply_diff_in_sandbox(
    repo_id: str, repo_path: Path, base_branch: str, diff_text: str,
    diff_explanation: str, redis_client,
) -> ApplyResult:
    if not diff_text or not diff_text.strip():
        return ApplyResult(applied=False, repo_id=repo_id, error="empty diff — nothing to apply")

    try:
        worktree_path, branch_name = _create_worktree(repo_path, base_branch)
    except SandboxError as e:
        logger.error(f"[{repo_id}] sandbox creation failed: {e}")
        return ApplyResult(applied=False, repo_id=repo_id, error=str(e))

    applied, error = apply_diff_to_worktree(worktree_path, diff_text)

    if not applied:
        logger.warning(f"[{repo_id}] diff did not apply cleanly: {error}")
        cleanup_worktree(repo_path, worktree_path, branch_name)
        return ApplyResult(applied=False, repo_id=repo_id, error=error)

    apply_id = str(uuid.uuid4())[:8]
    record = {
        "apply_id": apply_id,
        "repo_id": repo_id,
        "repo_path": str(repo_path),
        "worktree_path": str(worktree_path),
        "branch_name": branch_name,
        "base_branch": base_branch,
        "diff_explanation": diff_explanation,
        "proposed_diff": diff_text,
        "status": "pending_review",
        "created_at": time.time(),
    }
    await redis_client.set(f"pending_apply:{apply_id}", json.dumps(record), ex=PENDING_APPLY_TTL_SECONDS)
    await redis_client.sadd(f"pending_applies_index:{repo_id}", apply_id)

    logger.info(
        f"[{repo_id}] diff applied on isolated branch '{branch_name}' at "
        f"{worktree_path} — pending review (apply_id={apply_id})"
    )
    return ApplyResult(applied=True, repo_id=repo_id, apply_id=apply_id, worktree_path=str(worktree_path), branch_name=branch_name)


async def get_pending_apply(redis_client, apply_id: str) -> dict | None:
    raw = await redis_client.get(f"pending_apply:{apply_id}")
    if raw is None:
        return None
    return json.loads(raw)


async def list_pending_applies(redis_client, repo_id: str) -> list[dict]:
    apply_ids = await redis_client.smembers(f"pending_applies_index:{repo_id}")
    records = []
    for apply_id in apply_ids:
        record = await get_pending_apply(redis_client, apply_id)
        if record is not None:
            records.append(record)
    return records


def _merge_into_base(repo_path: Path, agent_branch: str, base_branch: str, commit_message: str) -> tuple[bool, str | None, str | None]:
    """
    Merges agent_branch into base_branch with NO checkout at all, using
    git's checkout-free merge plumbing (merge-tree --write-tree +
    commit-tree + update-ref) rather than a second worktree.

    WHY NOT A SECOND WORKTREE (the design this replaced): git refuses to
    check out a branch that's ALREADY checked out somewhere else, and
    repo_path's own working directory normally sits on base_branch
    precisely because that's what cloning produces — so a second
    worktree checked out on base_branch is blocked by git itself in the
    ORDINARY case, not just a theoretical edge case. Confirmed by
    actually hitting this exact error against real git before switching
    approaches.

    NOTE: this moves base_branch's ref WITHOUT updating repo_path's own
    working directory or index — exactly like a bare repo receiving a
    push. repo_path's checked-out files will be stale until the next
    explicit sync (Phase 5's /sync endpoint), consistent with how this
    project already treats repo_path: a periodically-refreshed clone
    for ingestion, not a live editing workspace.

    Returns (success, new_commit_hash, error). Conflict details from
    merge-tree land on stdout, not stderr — verified against real git,
    not assumed — so the error message reads from the right stream.
    """
    merge_tree_result = _run_git(["merge-tree", "--write-tree", base_branch, agent_branch], cwd=repo_path)
    if merge_tree_result.returncode != 0:
        error = merge_tree_result.stdout.strip() or merge_tree_result.stderr.strip()
        return False, None, error

    tree_hash = merge_tree_result.stdout.strip()
    base_commit = _run_git(["rev-parse", base_branch], cwd=repo_path).stdout.strip()
    agent_commit = _run_git(["rev-parse", agent_branch], cwd=repo_path).stdout.strip()

    commit_result = _run_git(
        ["commit-tree", tree_hash, "-p", base_commit, "-p", agent_commit, "-m", commit_message],
        cwd=repo_path,
    )
    if commit_result.returncode != 0:
        return False, None, commit_result.stderr.strip()
    new_commit_hash = commit_result.stdout.strip()

    update_ref_result = _run_git(["update-ref", f"refs/heads/{base_branch}", new_commit_hash], cwd=repo_path)
    if update_ref_result.returncode != 0:
        return False, None, update_ref_result.stderr.strip()

    return True, new_commit_hash, None


async def commit_pending_apply(redis_client, apply_id: str) -> CommitResult:
    """
    Phase 4 — the human said yes. Stages + commits the change on the
    agent's own branch (inside its ORIGINAL Phase-3 worktree — untouched
    since apply), merges that commit into base_branch via a throwaway
    merge-worktree, then cleans up the agent's worktree/branch (their
    commits are preserved via the merge commit's history, so deleting
    the branch pointer loses nothing) and marks the record committed.
    """
    record = await get_pending_apply(redis_client, apply_id)
    if record is None:
        return CommitResult(committed=False, apply_id=apply_id, repo_id="", error="apply_id not found or expired")
    if record["status"] != "pending_review":
        return CommitResult(committed=False, apply_id=apply_id, repo_id=record["repo_id"], error=f"apply is not pending review (status={record['status']!r})")

    repo_path = Path(record["repo_path"])
    worktree_path = Path(record["worktree_path"])
    branch_name = record["branch_name"]
    base_branch = record["base_branch"]
    repo_id = record["repo_id"]

    add_result = _run_git(["add", "-A"], cwd=worktree_path)
    if add_result.returncode != 0:
        return CommitResult(committed=False, apply_id=apply_id, repo_id=repo_id, error=f"git add failed: {add_result.stderr.strip()}")

    commit_message = record.get("diff_explanation") or "Agent-proposed change"
    commit_message = f"{commit_message}\n\n(Applied by coding agent, approved by human review.)"
    commit_result = _run_git(["commit", "-m", commit_message], cwd=worktree_path)
    if commit_result.returncode != 0:
        return CommitResult(committed=False, apply_id=apply_id, repo_id=repo_id, error=f"git commit failed: {commit_result.stderr.strip()}")

    # Capture base_branch's tip BEFORE the merge moves it — Phase 5's
    # re-index uses this as the last-indexed commit.
    previous_commit = _run_git(["rev-parse", base_branch], cwd=repo_path).stdout.strip() or None

    success, new_commit_hash, error = _merge_into_base(repo_path, branch_name, base_branch, commit_message)
    if not success:
        logger.error(f"[{repo_id}] merge into {base_branch!r} failed for apply_id={apply_id}: {error}")
        return CommitResult(committed=False, apply_id=apply_id, repo_id=repo_id, error=f"merge failed: {error}")

    cleanup_worktree(repo_path, worktree_path, branch_name)

    record["status"] = "committed"
    record["commit_hash"] = new_commit_hash
    record["previous_commit"] = previous_commit
    record["committed_at"] = time.time()
    await redis_client.set(f"pending_apply:{apply_id}", json.dumps(record), ex=PENDING_APPLY_TTL_SECONDS)
    await redis_client.srem(f"pending_applies_index:{repo_id}", apply_id)

    logger.info(f"[{repo_id}] apply_id={apply_id} committed onto {base_branch!r} as {new_commit_hash}")
    return CommitResult(
        committed=True, apply_id=apply_id, repo_id=repo_id,
        commit_hash=new_commit_hash, previous_commit=previous_commit,
    )


async def reject_pending_apply(redis_client, apply_id: str) -> RejectResult:
    """Phase 4 — the human said no. Discards the worktree and branch
    entirely; nothing about them is ever reachable again."""
    record = await get_pending_apply(redis_client, apply_id)
    if record is None:
        return RejectResult(rejected=False, apply_id=apply_id, repo_id="", error="apply_id not found or expired")
    if record["status"] != "pending_review":
        return RejectResult(rejected=False, apply_id=apply_id, repo_id=record["repo_id"], error=f"apply is not pending review (status={record['status']!r})")

    repo_path = Path(record["repo_path"])
    worktree_path = Path(record["worktree_path"])
    branch_name = record["branch_name"]
    repo_id = record["repo_id"]

    cleanup_worktree(repo_path, worktree_path, branch_name)

    record["status"] = "rejected"
    record["rejected_at"] = time.time()
    await redis_client.set(f"pending_apply:{apply_id}", json.dumps(record), ex=PENDING_APPLY_TTL_SECONDS)
    await redis_client.srem(f"pending_applies_index:{repo_id}", apply_id)

    logger.info(f"[{repo_id}] apply_id={apply_id} rejected, worktree/branch discarded")
    return RejectResult(rejected=True, apply_id=apply_id, repo_id=repo_id)