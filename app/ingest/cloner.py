"""
cloner.py — clone a GitHub repo to local disk

Runs `git` directly via subprocess with a hard wall-clock timeout and
every form of interactive prompting disabled, so a clone/pull can never
hang the ingest pipeline indefinitely.

WHY subprocess DIRECTLY INSTEAD OF GitPython'S clone_from/pull:
  GitPython is itself a thin wrapper around these same `git` subprocess
  calls - nothing here is used for anything beyond that, no diffing or
  commit inspection. Recent GitPython versions expose a
  `kill_after_timeout` kwarg on git command execution, but that
  depends on the installed GitPython version actually forwarding it
  correctly through clone_from()/pull(), which isn't something
  reliably verifiable ahead of time. subprocess.run()'s own timeout
  handling is a long-stable stdlib guarantee, so it's used directly -
  the actual git invocations (shallow, single-branch, same
  clone/pull-then-reclone-on-failure structure) are unchanged.

WHY depth=1 (shallow clone):
  We only need the current snapshot of the code, not its entire commit
  history. A shallow clone downloads just the latest commit's files —
  for a large repo this is the difference between a 2-second clone and
  a 2-minute clone.

WHY A HARD TIMEOUT:
  Matches the reported symptom exactly: Render logs showing only
  repeated `GET /repos/{id} -> 200 OK`, none of run_ingest()'s own log
  lines ("Starting clone", "Found X Python files", ...), and the repo
  stuck at status="ingesting" with 0 files/chunks. A request handler
  answering fine while ingestion never reaches its first log line is
  what a blocked, un-timed-out `git` subprocess looks like - nothing
  upstream of clone_repo() was hanging, clone_repo() itself never
  returned.

WHY DISABLE INTERACTIVE PROMPTS:
  A server process has no TTY. Without GIT_TERMINAL_PROMPT=0 and an
  SSH command with BatchMode=yes, `git` silently blocks on stdin for a
  username/password or an SSH host-key "yes/no" prompt that will never
  be answered - indefinitely, timeout or not, since the prompt itself
  is what's actually hanging, not a slow network. This is the more
  likely root cause of the described hang: a bare timeout would
  eventually recover from it (after the full 120s, every time), but a
  disabled prompt fails in under a second with a clear reason instead.
"""

import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard wall-clock ceiling for a single clone/pull invocation. Generous
# enough for a large shallow clone over a slow connection, while still
# failing well within a request's own reasonable lifetime. Tune per
# your actual repo sizes/network if needed.
CLONE_TIMEOUT_SECONDS = 120


class CloneError(Exception):
    """Raised when a git clone/pull fails or exceeds CLONE_TIMEOUT_SECONDS.

    Replaces GitPython's git.GitCommandError/git.exc.GitError for
    anything that previously caught those specifically - update any such
    call site to catch CloneError instead.
    """


def _git_env() -> dict:
    """Environment for the git subprocess: inherits the current
    environment, then disables every form of interactive prompting a
    server process (no TTY) can never actually answer.
    """
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # no HTTPS username/password prompt
    env["GIT_ASKPASS"] = "echo"       # belt-and-suspenders: any askpass call returns empty
    env["GIT_SSH_COMMAND"] = (
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=10"
    )  # no SSH password/host-key prompt, and a fast-failing SSH-level connect timeout too
    return env


def _run_git(args: list[str], cwd: str | None = None) -> None:
    """Run one git subprocess with a hard timeout and no interactive
    prompts. Raises CloneError with a clear message on timeout or
    non-zero exit - never left to hang or fail silently.
    """
    logger.info("Running: git %s", " ".join(args))

    try:
        proc = subprocess.Popen(
            ["git"] + args,
            cwd=cwd,
            env=_git_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group, so a timeout kill
                                      # takes any child process git itself
                                      # spawned (e.g. a stuck ssh helper)
                                      # down with it too, not just git itself
        )
    except FileNotFoundError as exc:
        raise CloneError(f"git executable not found: {exc}") from exc

    try:
        stdout, stderr = proc.communicate(timeout=CLONE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # proc.kill() alone only signals the direct child process; killing
        # the whole process group also catches anything git spawned that
        # is itself stuck (e.g. an ssh subprocess waiting on its own
        # prompt), which a plain kill() would leave running.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        # A second communicate() (not wait()) after the kill: the process
        # is already dead so this returns immediately, but it's what
        # actually closes the stdout/stderr PIPE file handles - wait()
        # alone leaves them open, leaking file descriptors on every
        # timeout in a long-running server process.
        proc.communicate()
        raise CloneError(
            f"git {' '.join(args)} timed out after {CLONE_TIMEOUT_SECONDS}s"
        )

    if proc.returncode != 0:
        raise CloneError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): "
            f"{stderr.strip()[:500]}"
        )

    logger.debug("git %s output: %s", " ".join(args), stdout.strip()[:500])


def clone_repo(github_url: str, repo_id: str, repos_dir: str, branch: str = "main") -> str:
    local_path = str(Path(repos_dir) / repo_id)
    os.makedirs(repos_dir, exist_ok=True)

    if Path(local_path).exists():
        logger.info(f"Repo already cloned, pulling latest: {local_path}")
        try:
            _run_git(["pull"], cwd=local_path)
            return local_path
        except CloneError as e:
            logger.warning(f"Pull failed ({e}), re-cloning from scratch")
            shutil.rmtree(local_path, ignore_errors=True)

    logger.info(f"Cloning {github_url} -> {local_path}")
    _run_git([
        "clone",
        "--branch", branch,
        "--depth", "1",
        "--single-branch",
        github_url,
        local_path,
    ])
    return local_path