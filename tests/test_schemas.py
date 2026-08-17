"""Tests for the schemas guarding the GitHub boundary.

Two contracts are pinned here. Inbound models must tolerate a payload GitHub
has grown new fields on, because we read a slice of a large evolving API.
Outbound models must keep emitting exactly the keys the agent's instructions
promise, because the model parses them by name.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dependency_director.schemas import (
    BotPrList,
    BotPrSummary,
    BranchCiStatus,
    ChangedFile,
    CheckRunsResponse,
    CheckSummary,
    CommitDetails,
    CommitStatusResponse,
    CommitSummary,
    FileContents,
    PatchedFile,
    PrStatus,
    PullRequest,
    PullRequestRef,
    WorkflowJobsResponse,
    WorkflowRunsResponse,
    summarize_checks,
)

# --- Inbound: unknown fields are ignored, absent fields default ---


def test_pull_request_reads_the_nested_head_and_base_refs() -> None:
    """Verify base.ref survives parsing, so a PR is diagnosed against its own base."""
    pr = PullRequest.model_validate(
        {
            "number": 7,
            "title": "chore(deps): bump lib",
            "user": {"login": "dependabot[bot]"},
            "created_at": "2026-01-02T03:04:05Z",
            "head": {"ref": "dependabot/pip/lib-2.0", "sha": "deadbeef"},
            "base": {"ref": "develop", "sha": "cafe"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    assert pr.base.ref == "develop"
    assert pr.head.sha == "deadbeef"
    assert pr.author == "dependabot[bot]"


def test_pull_request_tolerates_a_payload_github_has_extended() -> None:
    """Verify unknown fields are ignored, so a GitHub addition cannot break a run."""
    pr = PullRequest.model_validate(
        {"number": 1, "title": "t", "some_field_added_next_year": {"nested": [1, 2]}},
    )
    assert pr.number == 1


def test_pull_request_defaults_the_fields_github_omits() -> None:
    """Verify absent optional fields default instead of raising.

    GitHub omits 'mergeable' while it computes mergeability, and omits 'user'
    on some payloads; neither is a reason to abandon a run.
    """
    pr = PullRequest.model_validate({"number": 1, "title": "t"})
    assert pr.mergeable is None
    assert pr.author == ""
    assert pr.base.ref == ""
    assert pr.head.sha == ""


@pytest.mark.parametrize("user", [None, {}, {"login": None}])
def test_a_deleted_account_reads_as_no_author(user: dict[str, object] | None) -> None:
    """Verify a null user or login yields an empty author rather than raising.

    GitHub nulls these out when an account is deleted. An empty author matches
    no configured bot, which is the right outcome.
    """
    assert PullRequest.model_validate({"number": 1, "user": user}).author == ""


def test_pull_request_rejects_a_number_of_the_wrong_type() -> None:
    """Verify a non-numeric PR number is rejected rather than silently coerced."""
    with pytest.raises(ValidationError):
        PullRequest.model_validate({"number": "not-a-number"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-01-02T03:04:05Z", datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
        ("2026-01-02T03:04:05+00:00", datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
        ("2026-01-02T03:04:05", datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
    ],
)
def test_workflow_run_timestamps_are_aware_utc(raw: str, expected: datetime) -> None:
    """Verify timestamps parse to aware UTC however GitHub spells the offset."""
    runs = WorkflowRunsResponse.model_validate({"workflow_runs": [{"id": 1, "created_at": raw}]})
    created = runs.workflow_runs[0].created_at
    assert created.tzinfo is not None
    assert created == expected


def test_workflow_run_offsets_are_normalised_for_comparison() -> None:
    """Verify a non-UTC offset compares correctly against a Z timestamp.

    A lexical sort puts '2026-01-02T00:00:00-05:00' before
    '2026-01-02T01:00:00Z' even though it is four hours later.
    """
    runs = WorkflowRunsResponse.model_validate(
        {
            "workflow_runs": [
                {"id": 1, "created_at": "2026-01-02T01:00:00Z"},
                {"id": 2, "created_at": "2026-01-02T00:00:00-05:00"},
            ],
        },
    )
    newest = max(runs.workflow_runs, key=lambda r: r.created_at)
    assert newest.id == 2


@pytest.mark.parametrize("raw", ["", "not-a-timestamp", None])
def test_unparseable_workflow_run_timestamps_sort_oldest(raw: str | None) -> None:
    """Verify a missing or malformed timestamp sorts last rather than raising."""
    runs = WorkflowRunsResponse.model_validate(
        {"workflow_runs": [{"id": 1, "created_at": raw}, {"id": 2, "created_at": "2020-01-01T00:00:00Z"}]},
    )
    newest = max(runs.workflow_runs, key=lambda r: r.created_at)
    assert newest.id == 2


def test_check_runs_response_defaults_to_empty() -> None:
    """Verify a repository with no checks parses rather than raising."""
    assert CheckRunsResponse.model_validate({}).check_runs == []
    assert CommitStatusResponse.model_validate({}).statuses == []
    assert WorkflowJobsResponse.model_validate({}).jobs == []
    assert WorkflowRunsResponse.model_validate({}).workflow_runs == []


# --- summarize_checks: the same reduction, now over models ---


@pytest.mark.parametrize(
    ("check_runs", "statuses", "state", "expected"),
    [
        ([], [], None, "NONE"),
        ([{"name": "lint", "status": "completed", "conclusion": "success"}], [], None, "GREEN"),
        ([{"name": "lint", "status": "completed", "conclusion": "failure"}], [], None, "RED"),
        ([{"name": "lint", "status": "in_progress", "conclusion": None}], [], None, "PENDING"),
        ([], [{"context": "ci/legacy", "state": "failure"}], "failure", "RED"),
        ([], [{"context": "ci/legacy", "state": "success"}], "success", "GREEN"),
        ([], [{"context": "ci/legacy", "state": "pending"}], "pending", "PENDING"),
    ],
)
def test_summarize_checks_folds_both_reporting_apis(
    check_runs: list[dict[str, object]],
    statuses: list[dict[str, object]],
    state: str | None,
    expected: str,
) -> None:
    """Verify modern check runs and legacy statuses reduce to one verdict."""
    ci_status, _checks = summarize_checks(
        CheckRunsResponse.model_validate({"check_runs": check_runs}),
        CommitStatusResponse.model_validate({"statuses": statuses, "state": state}),
    )
    assert ci_status == expected


def test_summarize_checks_labels_legacy_statuses_by_context() -> None:
    """Verify a legacy status is named by its context, so check names compare."""
    _ci, checks = summarize_checks(
        CheckRunsResponse.model_validate({}),
        CommitStatusResponse.model_validate(
            {"state": "failure", "statuses": [{"context": "ci/circleci", "state": "error"}]},
        ),
    )
    assert checks == [CheckSummary(name="ci/circleci", status="completed", conclusion="failure")]


# --- Outbound: the agent-facing key names are the contract ---


def test_bot_pr_list_exposes_the_base_ref_of_every_pr() -> None:
    """Verify the PR listing names each PR's base, so the agent can check it.

    Without this the agent has nothing to pass to get_branch_ci_status and
    silently falls back to the repository default branch.
    """
    payload = BotPrList(
        bot_prs=[
            BotPrSummary(
                number=7,
                title="chore(deps): bump lib",
                author="dependabot[bot]",
                created_at="2026-01-02T03:04:05Z",
                base_ref="develop",
            ),
        ],
        count=1,
    ).model_dump()
    assert payload["bot_prs"][0]["base_ref"] == "develop"
    assert set(payload["bot_prs"][0]) == {"number", "title", "author", "created_at", "base_ref"}


def test_pr_status_names_the_base_branch() -> None:
    """Verify the PR status payload carries base_ref alongside head_sha."""
    payload = PrStatus(
        pr_number=7,
        title="t",
        head_sha="deadbeef",
        base_ref="develop",
        mergeable=True,
        mergeable_state="clean",
        ci_status="RED",
        merge_status="CLEAN",
        checks=[CheckSummary(name="lint", status="completed", conclusion="failure")],
    ).model_dump()
    assert payload["base_ref"] == "develop"
    assert payload["head_sha"] == "deadbeef"


@pytest.mark.parametrize(
    ("model", "expected_keys"),
    [
        (
            BranchCiStatus(branch="develop", ci_status="GREEN", checks=[]),
            {"branch", "ci_status", "checks"},
        ),
        (
            ChangedFile(filename="a.py", status="modified", additions=1, deletions=2),
            {"filename", "status", "additions", "deletions"},
        ),
        (
            FileContents(name="a.py", path="src/a.py", size=1, content="", encoding="base64"),
            {"name", "path", "size", "content", "encoding"},
        ),
        (
            CommitSummary(sha="abc1234", message="m", author="a", date="2026-01-02T03:04:05Z"),
            {"sha", "message", "author", "date"},
        ),
        (
            CommitDetails(
                sha="abc1234",
                message="m",
                author="a",
                date="2026-01-02T03:04:05Z",
                files=[PatchedFile(filename="a.py", status="modified", patch="@@")],
            ),
            {"sha", "message", "author", "date", "files"},
        ),
    ],
)
def test_outbound_payloads_emit_exactly_their_documented_keys(
    model: BotPrList | BranchCiStatus | ChangedFile | FileContents | CommitSummary | CommitDetails,
    expected_keys: set[str],
) -> None:
    """Verify each agent-facing payload emits the keys its docstring promises.

    The agent reads these by name, so an added or renamed key is a contract
    change and should fail here rather than in a run.
    """
    assert set(model.model_dump()) == expected_keys


def test_pull_request_ref_defaults_are_empty_strings() -> None:
    """Verify a missing ref reads as empty rather than None, so callers can compare."""
    ref = PullRequestRef.model_validate({})
    assert ref.ref == ""
    assert ref.sha == ""
