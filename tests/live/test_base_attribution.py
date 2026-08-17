"""Live checks on how a red base branch is attributed.

The rule under test: a failing base excuses a dependency PR only for the
checks the base is failing too. A check the PR fails and the base passes
belongs to the PR. Unit tests assert the instruction text says this; these
assert the agent acts on it, against a real git remote it really pushes to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.live.conftest import requires_model
from tests.live.fake_github import FakeGitHub, check_runs
from tests.live.repo_builder import BANNED_API, BuiltRepo, file_at_ref, ref_sha

pytestmark = [pytest.mark.live, requires_model]

BOT = "dependabot[bot]"
TITLE = "chore(deps): bump lib from 1.0 to 2.0"


def _fake(scenario_repo: BuiltRepo, *, base_lint: str) -> FakeGitHub:
    """Script a repository whose base always fails deploy and may fail lint.

    'deploy' fails on both refs throughout, standing in for infrastructure
    breakage. Only 'lint' varies, which isolates the comparison being tested.
    """
    return FakeGitHub(
        prs=[
            {
                "number": 1,
                "title": TITLE,
                "author": BOT,
                "head_sha": scenario_repo.pr_sha,
                "head_ref": scenario_repo.pr_branch,
            },
        ],
        checks_by_ref={
            scenario_repo.pr_sha: check_runs(("lint", "failure"), ("deploy", "failure")),
            "main": check_runs(("lint", base_lint), ("deploy", "failure")),
        },
        branches=["main", scenario_repo.pr_branch],
    )


@pytest.mark.asyncio
async def test_pr_is_fixed_when_it_fails_a_check_the_base_passes(
    scenario_repo: BuiltRepo,
    run_live: Callable[..., Any],
) -> None:
    """A red base must not excuse a failure the base does not have.

    The base is red on 'deploy' while 'lint' passes there, so the PR's lint
    failure is the PR's own. Blaming the base here would silently stop fixing
    every dependency PR in a repository with one broken deploy job.
    """
    fake = await run_live(_fake(scenario_repo, base_lint="success"))
    remote = fake.bare
    assert remote is not None

    pushed = file_at_ref(remote, scenario_repo.pr_branch, "app.py")
    assert BANNED_API not in pushed, "the agent did not fix and push the lint failure"
    assert ref_sha(remote, scenario_repo.pr_branch) != scenario_repo.pr_sha
    # The bump itself must survive the fix.
    assert "lib==2.0" in file_at_ref(remote, scenario_repo.pr_branch, "requirements.txt")


@pytest.mark.asyncio
async def test_pr_is_skipped_when_the_base_fails_the_same_check(
    scenario_repo: BuiltRepo,
    run_live: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the base is failing the same check, the PR cannot go green alone.

    Nothing should be pushed: the fix belongs on the base branch, not in a
    'dependabot/*' branch where it would be poor review hygiene.
    """
    fake = await run_live(_fake(scenario_repo, base_lint="failure"))
    remote = fake.bare
    assert remote is not None

    assert ref_sha(remote, scenario_repo.pr_branch) == scenario_repo.pr_sha
    output = capsys.readouterr().out
    assert "not fixed" in output
    assert "base branch" in output


@pytest.mark.asyncio
async def test_base_health_is_checked_once_for_the_whole_run(
    scenario_repo: BuiltRepo,
    run_live: Callable[..., Any],
) -> None:
    """Re-deriving the same base diagnosis per PR is what made runs expensive."""
    fake = await run_live(_fake(scenario_repo, base_lint="failure"))

    base_lookups = [args for name, args in fake.calls if name == "get_commit_check_runs" and args[2] == "main"]
    assert len(base_lookups) == 1, f"base checked {len(base_lookups)} times, expected once"
