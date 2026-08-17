"""Fixtures for the live harness.

These tests call the real model. They are slow, cost tokens, and are not
deterministic, so they are excluded from the default run by the 'live' marker
and skipped outright when no API key is configured.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dependency_director import main as dd_main
from dependency_director.config import Settings
from tests.live.fake_github import FakeGitHub
from tests.live.repo_builder import BuiltRepo, build_scenario_repo, seed_workspace_clone

REPO = "acme/widget"
REPO_NAME = "widget"


def _has_model_credentials() -> bool:
    """Report whether a live run can authenticate to a model backend."""
    if os.environ.get("GEMINI_API_KEY"):
        return True
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")


requires_model = pytest.mark.skipif(
    not _has_model_credentials(),
    reason="live harness needs GEMINI_API_KEY or Vertex configuration",
)


@pytest.fixture
def scenario_repo(tmp_path: Path) -> BuiltRepo:
    """Build the bare remote and dependency-bot branch for one scenario."""
    return build_scenario_repo(tmp_path)


@pytest.fixture
def live_settings() -> Settings:
    """Settings for a live run, inheriting real model credentials from env."""
    return Settings()


@pytest.fixture
def run_live(
    scenario_repo: BuiltRepo,
    live_settings: Settings,
    tmp_path: Path,
) -> Iterator[Callable[..., Any]]:
    """Run the real agent against the scripted GitHub and the local remote.

    Yields a callable taking the FakeGitHub to serve and any run_agent_for_repo
    keyword overrides; it returns the fake once the run has completed, with the
    pushed-to remote attached for inspection.
    """
    seeded: dict[str, Path] = {}
    preserved = tmp_path / "pushed.git"

    def seeding_prepare(workspace_dir: str) -> None:
        # Mirrors the production behaviour it replaces, then plants the clone
        # so the agent finds the repository already checked out.
        path = Path(workspace_dir)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        seeded["workspace"] = path
        seeded["bare"], seeded["checkout"] = seed_workspace_clone(
            path,
            scenario_repo.bare,
            REPO_NAME,
        )

    def preserving_cleanup(workspace_dir: str) -> None:
        # The run deletes its workspace on the way out, which would take the
        # remote the agent pushed to with it. Copy it somewhere durable first
        # so tests can read back what was actually pushed.
        path = Path(workspace_dir)
        remote = path / "remote.git"
        if remote.exists() and not preserved.exists():
            shutil.copytree(remote, preserved)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    async def _run(fake: FakeGitHub, **overrides: Any) -> FakeGitHub:
        kwargs: dict[str, Any] = {
            "max_attempts": 2,
            "dry_run": False,
            "hint": (
                f"The repository is already cloned at <workspace_dir>/{REPO_NAME} "
                "with its origin remote configured. Do not clone it again."
            ),
        }
        kwargs.update(overrides)
        with (
            patch.object(dd_main, "GitHubClient", return_value=fake),
            patch.object(dd_main, "_prepare_workspace", side_effect=seeding_prepare),
            patch.object(dd_main, "_cleanup_workspace", side_effect=preserving_cleanup),
        ):
            await dd_main.run_agent_for_repo(REPO, live_settings, **kwargs)
        fake.workspace = seeded.get("workspace")
        fake.checkout = seeded.get("checkout")
        fake.bare = preserved if preserved.exists() else seeded.get("bare")
        return fake

    yield _run
