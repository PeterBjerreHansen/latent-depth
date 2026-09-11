"""Small helpers for recording the code and input-config provenance of runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _worktree_sha256(repo_root: str | Path) -> str:
    """Hash tracked diffs and non-ignored untracked files in the worktree."""
    root = Path(repo_root)
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff)
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update((root / os.fsdecode(raw_path)).read_bytes())
    return digest.hexdigest()


def git_provenance(repo_root: str | Path) -> dict[str, str | bool | None]:
    """Return commit, dirty state, and a content digest for the worktree."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        worktree_sha256 = _worktree_sha256(repo_root)
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": None, "worktree_dirty": None, "worktree_sha256": None}
    return {
        "code_commit": revision or None,
        "worktree_dirty": bool(status.strip()),
        "worktree_sha256": worktree_sha256,
    }
