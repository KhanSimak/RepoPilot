"""
cloner.py — clone a GitHub repo to local disk

Uses GitPython, a thin wrapper around the `git` CLI.

WHY depth=1 (shallow clone):
  We only need the current snapshot of the code, not its entire commit
  history. A shallow clone downloads just the latest commit's files —
  for a large repo this is the difference between a 2-second clone and
  a 2-minute clone.
"""

import os
import shutil
import logging
from pathlib import Path
import git

logger = logging.getLogger(__name__)


def clone_repo(github_url: str, repo_id: str, repos_dir: str, branch: str = "main") -> str:
    local_path = str(Path(repos_dir) / repo_id)
    os.makedirs(repos_dir, exist_ok=True)

    if Path(local_path).exists():
        logger.info(f"Repo already cloned, pulling latest: {local_path}")
        try:
            git.Repo(local_path).remotes.origin.pull()
            return local_path
        except Exception as e:
            logger.warning(f"Pull failed ({e}), re-cloning from scratch")
            shutil.rmtree(local_path, ignore_errors=True)

    logger.info(f"Cloning {github_url} -> {local_path}")
    git.Repo.clone_from(github_url, local_path, branch=branch, depth=1, single_branch=True)
    return local_path
