"""
agent/post_commit_sync.py — Phase 5: re-index the repo after a Phase 4
commit lands, so the next query sees the code that's actually there now.

WHY THIS ISN'T JUST "CALL /sync": Phase 4's commit is deliberately
checkout-free (see sandbox.py's _merge_into_base) — it moves
base_branch's ref via update-ref without touching repo_path's working
directory or index. That leaves repo_path in a state that is NOT merely
stale, but actively misleading, and this was verified against real git
rather than assumed:

    $ git status --short
    M  app.py          <-- a STAGED modification

Git sees the (old) working-tree content as an uncommitted change
*relative to the new merged HEAD* — i.e. as a staged REVERSAL of the
agent's change. Two concrete consequences:

  1. Ingest walks the filesystem (_walk_python_files reads real files
     off disk), so re-indexing without refreshing would index the OLD
     pre-merge source while the repo's ref says otherwise — silently
     desynchronizing the index from the committed truth.
  2. Worse, anything that later ran `git add -A && git commit` in
     repo_path would COMMIT that reversal, undoing the agent's merged
     change without anyone asking it to.

So step one here is always `git reset --hard <base_branch>`, which
fixes both at once (verified: working tree matches the merge, and
`git status` comes back clean). Only then is re-indexing meaningful.

WHAT THIS DELIBERATELY DOES NOT DO: it does not re-implement ingestion.
It refreshes the working tree, works out which files the merge actually
touched, and hands off to the project's existing incremental ingest —
the same one POST /repos/{id}/sync already uses. Incremental ingest
already does the expensive-but-correct thing (diff against the last
ingested commit, re-chunk only changed files, re-embed only chunks whose
content hash changed), and duplicating that here would mean two code
paths to keep in sync.

TWO THINGS TO VERIFY AGAINST YOUR OWN cloner.py / incremental.py BEFORE
TRUSTING THIS IN PRODUCTION (I could not check either — neither file was
available when this was written, so these are flagged rather than
silently assumed):

  A. If run_incremental_ingest internally calls clone_repo, and
     clone_repo does something like `git fetch && git reset --hard
     origin/<branch>`, that would DESTROY the local merge commit Phase 4
     just created, because that commit exists only locally and is ahead
     of origin. A plain `git pull` is fine here (it reports "Already up
     to date" when local is ahead and origin hasn't moved) — a
     `reset --hard origin/...` is not. Check which one cloner.py does.

  B. The merge commit is local-only. Nothing in this project pushes it
     anywhere. That's intentional for now (agent changes stay on the
     server's clone until a human pushes them deliberately), but it does
     mean a future re-clone from scratch loses committed agent changes.
     Worth deciding explicitly rather than discovering later.
"""

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    synced: bool
    repo_id: str
    new_commit: str | None = None
    changed_files: list[str] | None = None
    error: str | None = None


def _run_git(args: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def refresh_working_tree(repo_path: Path, base_branch: str) -> tuple[bool, str | None]:
    """
    Bring repo_path's working tree and index back in line with
    base_branch's ref after Phase 4's checkout-free merge.

    Uses `reset --hard` deliberately: the working tree here is a
    disposable, re-cloneable artifact of ingestion, never a place a
    human edits, so there is no uncommitted human work for --hard to
    destroy. The one thing it WOULD destroy — the phantom staged
    reversal described in this module's docstring — is exactly what
    needs destroying.

    Returns (ok, error).
    """
    result = _run_git(["reset", "--hard", base_branch], cwd=repo_path)
    if result.returncode != 0:
        return False, f"git reset --hard {base_branch} failed: {result.stderr.strip()}"

    status = _run_git(["status", "--short"], cwd=repo_path)
    if status.stdout.strip():
        # Should be empty after a successful hard reset. If it isn't,
        # something else is going on (untracked build output, a
        # concurrent write) and re-indexing from here would index an
        # unknown state - report rather than proceed quietly.
        return False, f"working tree still dirty after reset: {status.stdout.strip()[:200]}"

    return True, None


def changed_files_between(repo_path: Path, old_commit: str, new_commit: str) -> list[str]:
    """
    Which files the merge actually touched. Informational (surfaced in
    the sync result / logs so a human can see what got re-indexed) —
    incremental ingest computes its own diff internally from the commit
    range, so this is not what drives re-indexing.
    """
    result = _run_git(["diff", "--name-only", old_commit, new_commit], cwd=repo_path)
    if result.returncode != 0:
        logger.warning(f"could not compute changed files: {result.stderr.strip()}")
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


async def sync_after_commit(
    repo_id: str,
    repo_path: Path,
    base_branch: str,
    previous_commit: str | None,
    new_commit: str,
    run_incremental_ingest,
    github_url: str,
    qdrant_client,
    redis_client,
    cfg,
) -> SyncResult:
    """
    Phase 5 flow: refresh the working tree, then hand off to the
    project's existing incremental ingest.

    run_incremental_ingest is passed in rather than imported at module
    scope so this module stays independently testable without dragging
    in the whole ingest dependency tree (embedder, qdrant, bm25), and so
    the seam is explicit rather than hidden behind an import.
    """
    ok, error = refresh_working_tree(repo_path, base_branch)
    if not ok:
        logger.error(f"[{repo_id}] post-commit working tree refresh failed: {error}")
        return SyncResult(synced=False, repo_id=repo_id, error=error)

    changed = changed_files_between(repo_path, previous_commit, new_commit) if previous_commit else []
    logger.info(
        f"[{repo_id}] working tree refreshed to {new_commit[:8]}; "
        f"{len(changed)} file(s) changed by the merge — re-indexing"
    )

    try:
        result = await run_incremental_ingest(
            repo_id, github_url, base_branch, previous_commit,
            qdrant_client, redis_client, cfg,
        )
    except Exception as e:
        logger.error(f"[{repo_id}] incremental re-ingest failed after commit: {e}")
        return SyncResult(synced=False, repo_id=repo_id, changed_files=changed, error=str(e))

    indexed_commit = result.get("new_commit", new_commit)
    logger.info(f"[{repo_id}] re-index complete at {indexed_commit[:8]}")

    return SyncResult(
        synced=True, repo_id=repo_id,
        new_commit=indexed_commit, changed_files=changed,
    )