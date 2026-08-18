"""Live checks on a second run over a base fix that has not merged yet.

The base fix branch name is derived from the base ref, not read from GitHub,
so every run over a still-broken base rebuilds the same branch. Nothing about
that is wrong until the previous run's fix PR is still open: then the work is
already done, the branch is under review, and redoing it costs a clone and
overwrites a reviewer's branch before dead-ending on GitHub's duplicate 422.

Unit tests assert 'create_pr' refuses to duplicate and the instructions say to
check first; these assert the agent acts on it against a real git remote.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.live.conftest import requires_model
from tests.live.fake_github import FakeGitHub, check_runs
from tests.live.repo_builder import BuiltRepo, ref_sha

pytestmark = [pytest.mark.live, requires_model]

BOT = "dependabot[bot]"
FIX_BRANCH = "dependency-director/fix-base-main"


def _fake(scenario_repo: BuiltRepo, *, fix_pr_open: bool) -> FakeGitHub:
    """Script a red base, optionally with last run's fix PR still open on it."""
    prs: list[dict[str, Any]] = [
        {
            "number": 1,
            "title": "chore(deps): bump lib from 1.0 to 2.0",
            "author": BOT,
            "head_sha": scenario_repo.pr_sha,
            "head_ref": scenario_repo.pr_branch,
        },
    ]
    branches = ["main", scenario_repo.pr_branch]
    if fix_pr_open:
        prs.append(
            {
                "number": 58,
                "title": "fix(ci): repair lint on main",
                "author": "a-human",
                "head_sha": scenario_repo.pr_sha,
                "head_ref": FIX_BRANCH,
            },
        )
        branches.append(FIX_BRANCH)

    return FakeGitHub(
        prs=prs,
        checks_by_ref={
            scenario_repo.pr_sha: check_runs(("lint", "failure")),
            "main": check_runs(("lint", "failure")),
        },
        branches=branches,
    )


@pytest.mark.asyncio
async def test_open_base_fix_pr_stops_the_run_before_it_clones(
    scenario_repo: BuiltRepo,
    run_live: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unmerged base fix must block the run, not be rebuilt from scratch."""
    fake = await run_live(_fake(scenario_repo, fix_pr_open=True), fix_base=True)
    remote = fake.bare
    assert remote is not None

    assert not fake.created_prs, "opened a duplicate base fix PR"
    # The dependency branch is blocked on the base fix either way, so nothing
    # belongs on it — least of all a second copy of a fix already under review.
    assert ref_sha(remote, scenario_repo.pr_branch) == scenario_repo.pr_sha
    output = capsys.readouterr().out
    assert "58" in output, "the run did not name the fix PR that already covers this base"
    assert "not fixed" in output


@pytest.mark.asyncio
async def test_base_fix_is_still_opened_when_none_exists(
    scenario_repo: BuiltRepo,
    run_live: Callable[..., Any],
) -> None:
    """The guard must block a duplicate, not the first fix.

    Without this, 'check before you clone' could pass by never fixing anything.
    """
    fake = await run_live(_fake(scenario_repo, fix_pr_open=False), fix_base=True)

    assert fake.created_prs, "no base fix PR was opened for a base nobody had fixed"
    assert fake.created_prs[0]["base"] == "main"
