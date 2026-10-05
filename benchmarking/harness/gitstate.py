"""The source state of a git checkout: commit, uncommitted diff, untracked files."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: Untracked paths under these prefixes are outputs, not sources.
IGNORED_UNTRACKED_PREFIXES = ("benchmarking/results/", "target/")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(repo: Path) -> dict:
    """Commit, branch, SHA-256 of `git diff HEAD`, and hashes of untracked files."""
    try:
        commit = _git(repo, "rev-parse", "HEAD").strip()
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
        diff = _git(repo, "diff", "HEAD", "--binary").encode()
        untracked = [line for line in _git(repo, "ls-files", "--others",
                                           "--exclude-standard").splitlines()
                     if not line.startswith(IGNORED_UNTRACKED_PREFIXES)]
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    untracked_hashes = {}
    for rel in untracked:
        path = repo / rel
        if path.is_file():
            untracked_hashes[rel] = _sha256_bytes(path.read_bytes())
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(diff) or bool(untracked_hashes),
        "diff_sha256": _sha256_bytes(diff) if diff else None,
        "diff_bytes": len(diff),
        "untracked_sha256": untracked_hashes,
    }
