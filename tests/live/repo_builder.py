"""Build a real git repository on disk for the live harness to operate on.

The agent clones, edits, commits, and pushes for real here. Only GitHub is
faked, so the bare repository created below is a genuine remote: assertions
can read back what the agent actually pushed.

The repository carries a self-contained linter with no third-party
dependencies, so a run neither needs the network nor depends on the resolved
version of any real tool.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

# Split so this module and the linter itself do not trip the check.
BANNED_API = "legacy_" + "call("

LINTER = '''"""Fail if any tracked module calls an API removed in lib 2.0."""

import pathlib
import sys

BANNED = "legacy_" + "call("

problems = []
for path in sorted(pathlib.Path().rglob("*.py")):
    if "tools" in path.parts:
        continue
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if BANNED in line:
            problems.append(
                f"{path}:{lineno}: E900 legacy_call() was removed in lib 2.0; use modern_call() instead",
            )

if problems:
    print("\\n".join(problems))
    sys.exit(1)

print("lint ok")
'''

LIB = '''"""Minimal stand-in library exposing both the old and new entry points."""


def legacy_call(value):
    """Return the value (removed in lib 2.0)."""
    return value


def modern_call(value):
    """Return the value."""
    return value
'''

APP_CLEAN = '''"""Application entry point."""

from lib import modern_call


def main():
    """Print the processed value."""
    print(modern_call(1))
'''

APP_WITH_BANNED_CALL = '''"""Application entry point."""

from lib import legacy_call


def main():
    """Print the processed value."""
    print(legacy_call(1))
'''

WORKFLOW = """name: CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python tools/lint.py

  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
"""

DEPLOY = """#!/bin/sh
echo "deploying"
exit 1
"""


class BuiltRepo(NamedTuple):
    """Paths and identifiers for a freshly built scenario repository.

    Attributes:
        bare: The remote the agent pushes to.
        main_sha: Commit at the tip of the default branch.
        pr_sha: Commit at the tip of the dependency-bot branch.
        pr_branch: Name of the dependency-bot branch.

    """

    bare: Path
    main_sha: str
    pr_sha: str
    pr_branch: str


def _git(*args: str, cwd: Path) -> str:
    """Run a git command in cwd and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bare(*args: str, bare: Path) -> str:
    """Run a git command against a bare repository and return its stdout.

    Addressed by --git-dir rather than by cwd because git refuses to treat a
    bare repository as the working repository when safe.bareRepository is
    'explicit', which is a common hardening setting.
    """
    return _git(f"--git-dir={bare}", *args, cwd=bare.parent)


def _write(root: Path, relative: str, content: str) -> None:
    """Write content to root/relative, creating parents as needed."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_scenario_repo(tmp_path: Path, pr_branch: str = "dependabot/pip/lib-2.0") -> BuiltRepo:
    """Create a bare remote whose default branch is clean and whose PR is not.

    The default branch passes 'lint' and the dependency-bot branch fails it by
    calling an API the bump removed. The 'deploy' job fails on both, standing
    in for infrastructure breakage that no dependency PR can repair. That split
    is the point: it separates a check the base is failing from one it is not.
    """
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "-b", "main", cwd=source)
    _git("config", "user.email", "harness@example.invalid", cwd=source)
    _git("config", "user.name", "Live Harness", cwd=source)

    _write(source, "lib.py", LIB)
    _write(source, "app.py", APP_CLEAN)
    _write(source, "tools/lint.py", LINTER)
    _write(source, ".github/workflows/ci.yml", WORKFLOW)
    _write(source, "deploy.sh", DEPLOY)
    _write(source, "requirements.txt", "lib==1.0\n")
    _git("add", "-A", cwd=source)
    _git("commit", "-m", "Initial commit", cwd=source)
    main_sha = _git("rev-parse", "HEAD", cwd=source)

    # The bump: lib 2.0 removed legacy_call, and app.py was migrated to it
    # incorrectly, so 'lint' fails on the PR while still passing on main.
    _git("checkout", "-b", pr_branch, cwd=source)
    _write(source, "requirements.txt", "lib==2.0\n")
    _write(source, "app.py", APP_WITH_BANNED_CALL)
    _git("add", "-A", cwd=source)
    _git("commit", "-m", "chore(deps): bump lib from 1.0 to 2.0", cwd=source)
    pr_sha = _git("rev-parse", "HEAD", cwd=source)
    _git("checkout", "main", cwd=source)

    bare = tmp_path / "remote.git"
    _git("clone", "--bare", str(source), str(bare), cwd=tmp_path)
    # GitHub exposes PR heads at refs/pull/<n>/head; the agent fetches that ref
    # directly, so the fake remote has to carry it too.
    _git_bare("update-ref", "refs/pull/1/head", pr_sha, bare=bare)
    # A bare clone marks every branch as the clone's own; allow pushes to the
    # branch the agent will target.
    _git_bare("config", "receive.denyCurrentBranch", "ignore", bare=bare)

    return BuiltRepo(bare=bare, main_sha=main_sha, pr_sha=pr_sha, pr_branch=pr_branch)


def seed_workspace_clone(workspace: Path, bare: Path, repo_name: str) -> tuple[Path, Path]:
    """Place the remote and a checkout of it inside the agent's workspace.

    Pre-seeding sidesteps rewriting github.com URLs inside the sandbox: the
    workflow already says to clone only 'if not already', so the agent finds
    the checkout in place and works against a local origin.

    The remote is copied into the workspace rather than left in pytest's tmp
    directory because the sandbox only grants writes inside the workspace, and
    'git push' needs to write to the remote. Copying preserves every SHA, so
    the scenario's identifiers stay valid.

    Returns the relocated remote and the checkout.
    """
    workspace_bare = workspace / "remote.git"
    shutil.copytree(bare, workspace_bare)
    checkout = workspace / repo_name
    _git("clone", str(workspace_bare), str(checkout), cwd=workspace)
    _git("config", "user.email", "agent@example.invalid", cwd=checkout)
    _git("config", "user.name", "Dependency Director", cwd=checkout)
    return workspace_bare, checkout


def file_at_ref(bare: Path, ref: str, relative: str) -> str:
    """Read a file's contents at a ref in the bare remote."""
    return _git_bare("show", f"{ref}:{relative}", bare=bare)


def ref_sha(bare: Path, ref: str) -> str:
    """Resolve a ref to a SHA in the bare remote."""
    return _git_bare("rev-parse", ref, bare=bare)
