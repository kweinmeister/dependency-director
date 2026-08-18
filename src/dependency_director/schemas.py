"""Schemas for the two boundaries where a payload has a shape worth enforcing.

**Inbound** models parse the slices of GitHub's API this project reads. They
ignore unknown fields, because GitHub keeps adding them and a run must not
break when it does, and they default the fields GitHub legitimately omits
(``mergeable`` is absent while it is being computed). What they buy over
``dict.get("a", {}).get("b", "")`` is that a field we depend on is declared in
one place instead of being re-derived at every call site.

**Outbound** models are the contract with the agent: the JSON its instructions
tell it to read by key name. Declaring them means a payload cannot quietly lose
a field, which is exactly how the base branch of a pull request went missing.

Raw diffs, file contents, and job logs are deliberately absent. They are
strings with no schema, and validating them would be ceremony.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# Sorts oldest so a run with a missing or malformed timestamp loses ties
# rather than being mistaken for the most recent attempt.
_OLDEST = datetime.min.replace(tzinfo=UTC)

# Check conclusions that mean the check did not pass. 'cancelled' and
# 'timed_out' count because a dependency PR is no more mergeable for them
# having been infrastructure problems rather than test failures.
FAILING_CONCLUSIONS = frozenset({"action_required", "cancelled", "failure", "timed_out"})


def _to_aware_utc(value: object) -> datetime:
    """Coerce a GitHub timestamp to an aware UTC datetime.

    GitHub spells its timestamps with a trailing 'Z', but re-runs and older
    payloads have been seen with numeric offsets. Comparing those as strings
    orders them wrongly, so they are normalised here once.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return _OLDEST
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return _OLDEST


UtcTimestamp = Annotated[datetime, BeforeValidator(_to_aware_utc)]


class _Inbound(BaseModel):
    """Base for payloads GitHub owns: read our slice, ignore the rest."""

    model_config = ConfigDict(extra="ignore")


# --- Inbound: pull requests ---


class PullRequestRef(_Inbound):
    """One end of a pull request — its branch name and tip commit."""

    ref: str = ""
    sha: str = ""


class GitHubUser(_Inbound):
    """The subset of a user payload used to attribute authorship.

    ``login`` is nullable because GitHub returns a null user for accounts that
    have since been deleted.
    """

    login: str | None = None


class PullRequest(_Inbound):
    """A pull request, as much of it as this project reads.

    Everything defaults, including ``number``. Callers that act on a specific
    pull request already hold its number and pass it in; this model is read
    for the surrounding detail, and a partial payload should degrade rather
    than abort a run.
    """

    number: int = 0
    title: str = ""
    user: GitHubUser | None = None
    created_at: str = ""
    head: PullRequestRef = Field(default_factory=PullRequestRef)
    base: PullRequestRef = Field(default_factory=PullRequestRef)
    mergeable: bool | None = None
    mergeable_state: str | None = None

    @property
    def author(self) -> str:
        """Return the login that opened the pull request, or empty if absent.

        A deleted account leaves either no user or a null login, and neither
        matches a configured bot, which is the correct outcome.
        """
        return (self.user.login or "") if self.user else ""


# --- Inbound: checks, statuses, and workflow runs ---


class CheckRun(_Inbound):
    """One check run reported through the modern checks API."""

    name: str = ""
    status: str | None = None
    conclusion: str | None = None


class CheckRunsResponse(_Inbound):
    """The check-runs listing for a commit ref."""

    check_runs: list[CheckRun] = Field(default_factory=list)


class CommitStatus(_Inbound):
    """One status reported through the legacy commit-status API."""

    context: str = ""
    state: str | None = None


class CommitStatusResponse(_Inbound):
    """The combined legacy commit status for a commit ref."""

    state: str | None = None
    statuses: list[CommitStatus] = Field(default_factory=list)


class WorkflowRun(_Inbound):
    """One GitHub Actions workflow run on a commit."""

    id: int = 0
    name: str = ""
    conclusion: str | None = None
    created_at: UtcTimestamp = _OLDEST


class WorkflowRunsResponse(_Inbound):
    """The workflow-run listing for a commit ref."""

    workflow_runs: list[WorkflowRun] = Field(default_factory=list)


class WorkflowJob(_Inbound):
    """One job within a workflow run."""

    id: int = 0
    name: str = ""
    conclusion: str | None = None


class WorkflowJobsResponse(_Inbound):
    """The job listing for a workflow run."""

    jobs: list[WorkflowJob] = Field(default_factory=list)


# --- Outbound: the JSON the agent is instructed to read ---


class CheckSummary(BaseModel):
    """One check, reduced to the three fields the agent compares on."""

    name: str
    status: str | None = None
    conclusion: str | None = None


class BotPrSummary(BaseModel):
    """A dependency-bot pull request as the agent first sees it.

    ``base_ref`` is here so the agent can check the branch this PR actually
    targets. Without it the only branch it can name is the repository default,
    which is the wrong one for any PR aimed elsewhere.
    """

    number: int
    title: str
    author: str
    created_at: str
    base_ref: str


class BotPrList(BaseModel):
    """The reply to ``list_bot_prs``."""

    bot_prs: list[BotPrSummary]
    count: int


class BranchCiStatus(BaseModel):
    """The reply to ``get_branch_ci_status``."""

    branch: str
    ci_status: str
    checks: list[CheckSummary]


class PrStatus(BaseModel):
    """The reply to ``get_pr_status`` and ``wait_for_ci``."""

    pr_number: int
    title: str
    head_sha: str
    base_ref: str
    mergeable: bool | None
    mergeable_state: str | None
    ci_status: str
    merge_status: str
    checks: list[CheckSummary]


class ChangedFile(BaseModel):
    """One entry in the reply to ``get_pr_files``."""

    filename: str
    status: str
    additions: int
    deletions: int


class PatchedFile(BaseModel):
    """One file in the reply to ``get_commit_details``, with its patch."""

    filename: str
    status: str
    patch: str


class FileContents(BaseModel):
    """The reply to ``get_file_contents``.

    Every field defaults to None: GitHub returns a different shape for a
    directory or a file too large to inline, and reporting that back is more
    useful to the agent than failing the call.
    """

    name: str | None = None
    path: str | None = None
    size: int | None = None
    content: str | None = None
    encoding: str | None = None


class CommitSummary(BaseModel):
    """One entry in the reply to ``list_commits``."""

    sha: str
    message: str
    author: str
    date: str


class CommitDetails(BaseModel):
    """The reply to ``get_commit_details``."""

    sha: str
    message: str
    author: str
    date: str
    files: list[PatchedFile]


def summarize_checks(
    check_runs: CheckRunsResponse,
    commit_status: CommitStatusResponse,
) -> tuple[str, list[CheckSummary]]:
    """Reduce a ref's checks to one verdict and a flat list.

    ci_status is GREEN, RED, PENDING, or NONE. The modern checks API and the
    legacy commit-status API are folded together because a repository may
    report through either, and the caller compares check names across refs.
    """
    checks: list[CheckSummary] = [
        CheckSummary(name=run.name, status=run.status, conclusion=run.conclusion) for run in check_runs.check_runs
    ]

    legacy_state = commit_status.state if commit_status.statuses else None
    checks.extend(
        CheckSummary(
            name=status.context,
            status="completed",
            conclusion="success"
            if status.state == "success"
            else ("failure" if status.state in ("failure", "error") else None),
        )
        for status in commit_status.statuses
    )

    has_failures = any(c.conclusion in FAILING_CONCLUSIONS for c in checks)
    has_pending = any(c.status not in ("completed", "success") for c in checks)

    ci_status = "NONE"
    if has_failures or legacy_state in ("failure", "error"):
        ci_status = "RED"
    elif has_pending or legacy_state == "pending":
        ci_status = "PENDING"
    elif checks or legacy_state == "success":
        ci_status = "GREEN"

    return ci_status, checks


def json_payload(model: BaseModel) -> str:
    """Render an outbound model as the indented JSON the agent reads."""
    return model.model_dump_json(indent=2)
