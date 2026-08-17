"""Tests for PR status tools and check suites in dependency-director."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.config import DEFAULT_BOTS, DEFAULT_MAX_FAILED_JOBS, LogLimits
from dependency_director.schemas import PullRequest
from dependency_director.tools import (
    AgentTools,
    GitHubClient,
    GitHubNotFoundError,
    ToolFn,
    _check_ci,
    create_agent_tools,
)


@pytest.mark.asyncio
async def test_github_client_get_pr_details(github_token: str) -> None:
    """Verify GitHub client fetches PR details correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "number": 22,
        "state": "open",
        "title": "chore(deps): bump hono",
        "head": {"sha": "sha123"},
        "mergeable": True,
        "mergeable_state": "clean",
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_pr_details("owner", "repo", 22)
        assert result["number"] == 22
        assert result["head"]["sha"] == "sha123"
        assert result["mergeable"] is True
        assert result["mergeable_state"] == "clean"
        mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo/pulls/22", headers=c.headers)
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_commit_check_runs(github_token: str) -> None:
    """Verify GitHub client fetches check runs for a commit correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_commit_check_runs("owner", "repo", "sha123")
        assert result["total_count"] == 1
        assert result["check_runs"][0]["conclusion"] == "success"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/sha123/check-runs",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_commit_status(github_token: str) -> None:
    """Verify GitHub client fetches commit status correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "state": "success",
        "statuses": [{"context": "ci/circleci", "state": "success"}],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_commit_status("owner", "repo", "sha123")
        assert result["state"] == "success"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/sha123/status",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_workflow_runs_for_commit(github_token: str) -> None:
    """Verify GitHub client fetches workflow runs for a commit correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "workflow_runs": [{"id": 98765, "status": "completed", "conclusion": "success"}],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_workflow_runs_for_commit("owner", "repo", "sha123")
        assert result["total_count"] == 1
        assert result["workflow_runs"][0]["id"] == 98765
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/actions/runs",
            headers=c.headers,
            params={"head_sha": "sha123"},
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_workflow_run_jobs(github_token: str) -> None:
    """Verify GitHub client fetches jobs for a workflow run correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "jobs": [{"id": 111, "name": "build", "status": "completed", "conclusion": "failure"}],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_workflow_run_jobs("owner", "repo", 98765)
        assert result["jobs"][0]["id"] == 111
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/actions/runs/98765/jobs",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_job_logs(github_token: str) -> None:
    """Verify GitHub client fetches job logs correctly."""
    mock_response = MagicMock()
    mock_response.text = "line1\nline2\nline3"
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.get_job_logs("owner", "repo", 111)
        assert result == "line1\nline2\nline3"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/actions/jobs/111/logs",
            headers=c.headers,
            follow_redirects=True,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_job_logs_redirect_handling(github_token: str) -> None:
    """Verify GitHub client handles redirects when fetching job logs."""
    c = GitHubClient(token=github_token)

    async def mock_handle_request(request: httpx_mod.Request) -> httpx_mod.Response:
        url_str = str(request.url)
        if url_str == "https://api.github.com/repos/owner/repo/actions/jobs/111/logs":
            return httpx_mod.Response(
                status_code=302,
                headers={"Location": "https://azure-blob-storage.com/logs-xyz"},
                request=request,
            )
        if url_str == "https://azure-blob-storage.com/logs-xyz":
            return httpx_mod.Response(status_code=200, text="actual log contents", request=request)
        return httpx_mod.Response(status_code=404, request=request)

    mock_transport = MagicMock()
    mock_transport.aclose = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(side_effect=mock_handle_request)
    c.client._transport = mock_transport
    result = await c.get_job_logs("owner", "repo", 111)
    assert result == "actual log contents"
    assert mock_transport.handle_async_request.call_count == 2
    await c.close()


@pytest.fixture
def tools(mock_client: MagicMock) -> AgentTools:
    """Fixture to set up write tools for status checking tests."""
    return create_agent_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=False, review_wait=0)


@pytest.fixture
def get_pr_status(tools: AgentTools) -> ToolFn:
    """Fixture to retrieve the PR status checking tool."""
    return tools.get_pr_status


@pytest.fixture
def get_pr_workflow_run_logs(tools: AgentTools) -> ToolFn:
    """Fixture to retrieve the PR workflow run log fetching tool."""
    return tools.get_pr_workflow_run_logs


@pytest.mark.asyncio
async def test_tool_get_pr_status_green(mock_client: MagicMock, get_pr_status: ToolFn) -> None:
    """Verify get_pr_status tool returns success when checks are passing."""
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "title": "bump",
            "head": {"sha": "sha123"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "total_count": 1,
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"state": "pending", "statuses": []})
    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)
    assert result["pr_number"] == 22
    assert result["mergeable"] is True
    assert result["mergeable_state"] == "clean"
    assert result["ci_status"] == "GREEN"
    assert len(result["checks"]) == 1
    assert result["checks"][0]["name"] == "test"


@pytest.mark.asyncio
async def test_tool_get_pr_status_red_failing_check_run(mock_client: MagicMock, get_pr_status: ToolFn) -> None:
    """Verify get_pr_status tool returns failure when a check run is failing."""
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "title": "bump",
            "head": {"sha": "sha123"},
            "mergeable": True,
            "mergeable_state": "unstable",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "total_count": 1,
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"state": "pending", "statuses": []})
    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)
    assert result["ci_status"] == "RED"


@pytest.mark.asyncio
async def test_tool_get_pr_status_pending(mock_client: MagicMock, get_pr_status: ToolFn) -> None:
    """Verify get_pr_status tool returns pending when checks are still running."""
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "title": "bump",
            "head": {"sha": "sha123"},
            "mergeable": True,
            "mergeable_state": "blocked",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "total_count": 1,
            "check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"state": "pending", "statuses": []})
    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)
    assert result["ci_status"] == "PENDING"


@pytest.mark.asyncio
async def test_tool_get_pr_status_conflict(mock_client: MagicMock, get_pr_status: ToolFn) -> None:
    """Verify get_pr_status tool reports merge conflicts."""
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "title": "bump",
            "head": {"sha": "sha123"},
            "mergeable": False,
            "mergeable_state": "dirty",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(return_value={"total_count": 0, "check_runs": []})
    mock_client.get_commit_status = AsyncMock(return_value={"state": "pending", "statuses": []})
    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)
    assert result["mergeable"] is False
    assert result["mergeable_state"] == "dirty"
    assert result["merge_status"] == "CONFLICT"
    assert result["ci_status"] == "NONE"  # No checks ran, CI is unknown


@pytest.mark.asyncio
async def test_tool_get_pr_workflow_run_logs(mock_client: MagicMock, get_pr_workflow_run_logs: ToolFn) -> None:
    """Verify get_pr_workflow_run_logs tool retrieves logs correctly."""
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={"total_count": 1, "workflow_runs": [{"id": 98765, "name": "CI", "conclusion": "failure"}]},
    )
    mock_client.get_workflow_run_jobs = AsyncMock(
        return_value={
            "jobs": [
                {
                    "id": 111,
                    "name": "build",
                    "status": "completed",
                    "conclusion": "failure",
                },
                {
                    "id": 222,
                    "name": "lint",
                    "status": "completed",
                    "conclusion": "success",
                },
            ],
        },
    )
    large_log = "\n".join(f"log line {i}" for i in range(1, 100))
    mock_client.get_job_logs = AsyncMock(return_value=large_log)
    result = await get_pr_workflow_run_logs("owner", "repo", 22)
    assert "--- FAILED JOB: CI / build (ID: 111) ---" in result
    assert "log line 99" in result
    assert "log line 1" not in result
    assert "log line 50" in result


@pytest.mark.asyncio
async def test_tool_get_pr_workflow_run_logs_api_error_on_logs(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify get_pr_workflow_run_logs handles API errors gracefully."""
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={"total_count": 1, "workflow_runs": [{"id": 98765, "name": "CI", "conclusion": "failure"}]},
    )
    mock_client.get_workflow_run_jobs = AsyncMock(
        return_value={
            "jobs": [
                {
                    "id": 111,
                    "name": "build",
                    "status": "completed",
                    "conclusion": "failure",
                },
            ],
        },
    )
    mock_client.get_job_logs = AsyncMock(side_effect=GitHubNotFoundError("GitHub API error 404"))
    result = await get_pr_workflow_run_logs("owner", "repo", 22)
    assert "--- FAILED JOB: CI / build (ID: 111) ---" in result
    assert "Failed to retrieve log: GitHub API error 404" in result


def _run(run_id: int, name: str, conclusion: str = "failure", created_at: str = "") -> dict[str, object]:
    """Build a minimal workflow-run payload for the log-fetching tests."""
    return {"id": run_id, "name": name, "conclusion": conclusion, "created_at": created_at}


def _job(job_id: int, name: str, conclusion: str = "failure") -> dict[str, object]:
    """Build a minimal job payload for the log-fetching tests."""
    return {"id": job_id, "name": name, "status": "completed", "conclusion": conclusion}


@pytest.mark.asyncio
async def test_workflow_run_logs_covers_every_failing_workflow(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify logs are read from all failing workflow runs, not only the first.

    A commit routinely triggers several workflows. Reading only runs[0] means
    the agent sees one failure, fixes it, and is surprised when CI is still
    red for a reason it was never shown.
    """
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={
            "workflow_runs": [
                _run(1, "CI"),
                _run(2, "CodeQL"),
            ],
        },
    )
    jobs_by_run = {1: {"jobs": [_job(111, "pytest")]}, 2: {"jobs": [_job(222, "analyze")]}}
    mock_client.get_workflow_run_jobs = AsyncMock(side_effect=lambda _o, _r, run_id: jobs_by_run[run_id])
    mock_client.get_job_logs = AsyncMock(return_value="boom")

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "CI / pytest" in result
    assert "CodeQL / analyze" in result


@pytest.mark.asyncio
async def test_workflow_run_logs_keeps_newest_run_per_workflow(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify a re-run supersedes its earlier attempt instead of being shown twice.

    Re-running a failed workflow produces a second run with the same name on
    the same SHA. Only the newest one reflects reality.
    """
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={
            "workflow_runs": [
                _run(9, "CI", created_at="2026-08-17T10:00:00Z"),
                _run(1, "CI", created_at="2026-08-17T09:00:00Z"),
            ],
        },
    )
    jobs_by_run = {9: {"jobs": [_job(999, "newest")]}, 1: {"jobs": [_job(111, "stale")]}}
    mock_client.get_workflow_run_jobs = AsyncMock(side_effect=lambda _o, _r, run_id: jobs_by_run[run_id])
    mock_client.get_job_logs = AsyncMock(return_value="boom")

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "newest" in result
    assert "stale" not in result


@pytest.mark.asyncio
async def test_workflow_run_logs_skips_successful_runs(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify a passing run is not fetched, so the extra API calls are not spent."""
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={
            "workflow_runs": [
                _run(1, "CI", conclusion="success"),
                _run(2, "Lint", conclusion="failure"),
            ],
        },
    )
    mock_client.get_workflow_run_jobs = AsyncMock(return_value={"jobs": [_job(222, "ruff")]})
    mock_client.get_job_logs = AsyncMock(return_value="boom")

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "Lint / ruff" in result
    mock_client.get_workflow_run_jobs.assert_awaited_once_with("owner", "repo", 2)


@pytest.mark.asyncio
async def test_workflow_run_logs_caps_jobs_and_reports_the_cap(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify the job cap holds and says what it dropped.

    A silent cap reads as 'these are all the failures', which is exactly the
    wrong impression when the agent is deciding whether its fix is complete.
    """
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(return_value={"workflow_runs": [_run(1, "CI")]})
    mock_client.get_workflow_run_jobs = AsyncMock(
        return_value={"jobs": [_job(100 + i, f"job{i}") for i in range(5)]},
    )
    mock_client.get_job_logs = AsyncMock(return_value="boom")

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert result.count("--- FAILED JOB:") == DEFAULT_MAX_FAILED_JOBS
    assert "job0" in result
    assert f"{5 - DEFAULT_MAX_FAILED_JOBS} more failed job" in result


@pytest.mark.asyncio
async def test_workflow_run_logs_honour_configured_caps(mock_client: MagicMock) -> None:
    """Verify both CI-log caps come from configuration, not from constants.

    A repo whose failure only makes sense with more jobs or a longer tail has
    no way to say so while the caps are baked in.
    """
    tools = create_agent_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
        log_limits=LogLimits(max_failed_jobs=1, tail_lines=2),
    )
    get_logs = tools.get_pr_workflow_run_logs
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(return_value={"workflow_runs": [_run(1, "CI")]})
    mock_client.get_workflow_run_jobs = AsyncMock(
        return_value={"jobs": [_job(100 + i, f"job{i}") for i in range(3)]},
    )
    mock_client.get_job_logs = AsyncMock(return_value="setup\nline1\nline2")

    result = await get_logs("owner", "repo", 22)

    assert result.count("--- FAILED JOB:") == 1
    assert "2 more failed job" in result
    assert "setup" not in result
    assert "line1\nline2" in result


@pytest.mark.asyncio
async def test_workflow_run_logs_reports_no_failures_across_all_runs(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    """Verify the empty case names every run examined, not just the latest one."""
    mock_client.get_pr_details = AsyncMock(return_value={"number": 22, "head": {"sha": "sha123"}})
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={"workflow_runs": [_run(1, "CI", conclusion="success")]},
    )

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "No failed jobs" in result


@pytest.mark.asyncio
async def test_check_ci_legacy_error_state_is_red() -> None:
    """Legacy commit status with state='error' must produce ci_status='RED'.

    GitHub uses 'error' for checks that fail structurally (e.g. infrastructure
    issues). This must not be classified as PENDING.
    """
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "chore(deps): bump something",
            "head": {"sha": "abc123"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={"check_runs": []},
    )
    mock_client.get_commit_status = AsyncMock(
        return_value={
            "state": "error",
            "statuses": [{"context": "ci/deploy", "state": "error"}],
        },
    )

    ci_status, merge_status, result_json = await _check_ci(mock_client, "owner", "repo", 42)
    result = json.loads(result_json)
    assert ci_status == "RED", f"Expected RED but got {ci_status}"
    assert result["ci_status"] == "RED"
    assert merge_status == "CLEAN"


# --- Issue #1: CONFLICT must not overwrite ci_status ---


@pytest.mark.asyncio
async def test_check_ci_conflict_preserves_green_ci() -> None:
    """When PR has merge conflict but CI is green, ci_status must remain GREEN.

    Previously, CONFLICT unconditionally overwrote ci_status. Now ci_status
    and merge_status are separate: ci_status reflects CI pass/fail while
    merge_status reflects mergeability.
    """
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "bump something",
            "head": {"sha": "abc123"},
            "mergeable": False,
            "mergeable_state": "dirty",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})

    ci_status, merge_status, result_json = await _check_ci(mock_client, "owner", "repo", 42)
    result = json.loads(result_json)
    assert ci_status == "GREEN", f"Expected GREEN but got {ci_status}"
    assert merge_status == "CONFLICT"
    assert result["ci_status"] == "GREEN"
    assert result["merge_status"] == "CONFLICT"


@pytest.mark.asyncio
async def test_check_ci_conflict_preserves_red_ci() -> None:
    """When PR has merge conflict AND CI is red, both statuses must reflect their state."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "bump something",
            "head": {"sha": "abc123"},
            "mergeable": False,
            "mergeable_state": "dirty",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "check_runs": [{"name": "ci", "status": "completed", "conclusion": "failure"}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})

    ci_status, merge_status, result_json = await _check_ci(mock_client, "owner", "repo", 42)
    result = json.loads(result_json)
    assert ci_status == "RED"
    assert merge_status == "CONFLICT"
    assert result["ci_status"] == "RED"
    assert result["merge_status"] == "CONFLICT"


# --- Issue #2: completed + null conclusion must NOT be PENDING ---


@pytest.mark.asyncio
async def test_check_ci_completed_null_conclusion_not_pending() -> None:
    """A check run with status=completed and conclusion=None must not be PENDING.

    GitHub returns this for skipped/neutral checks. Previously, the
    'or conclusion is None' clause classified these as pending, causing
    wait_for_ci to poll all 10 retries pointlessly.
    """
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "bump something",
            "head": {"sha": "abc123"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "check_runs": [
                {"name": "ci", "status": "completed", "conclusion": "success"},
                {"name": "skipped-check", "status": "completed", "conclusion": None},
            ],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})

    ci_status, merge_status, result_json = await _check_ci(mock_client, "owner", "repo", 42)
    result = json.loads(result_json)
    assert ci_status == "GREEN", f"Expected GREEN but got {ci_status} (null conclusion treated as pending)"
    assert merge_status == "CLEAN"
    assert result["ci_status"] == "GREEN"


# --- Issue #6: head_sha must be included in _check_ci response ---


@pytest.mark.asyncio
async def test_check_ci_includes_head_sha() -> None:
    """The _check_ci JSON response must include head_sha for debugging."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "bump something",
            "head": {"sha": "deadbeef123"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})

    _ci_status, _merge_status, result_json = await _check_ci(mock_client, "owner", "repo", 42)
    result = json.loads(result_json)
    assert "head_sha" in result, "head_sha must be present in _check_ci response"
    assert result["head_sha"] == "deadbeef123"


@pytest.fixture
def get_branch_ci_status(tools: AgentTools) -> ToolFn:
    """Fixture to retrieve the branch CI status tool."""
    return tools.get_branch_ci_status


@pytest.mark.parametrize(
    ("check_runs", "expected"),
    [
        ([{"name": "ci", "status": "completed", "conclusion": "success"}], "GREEN"),
        ([{"name": "ci", "status": "completed", "conclusion": "failure"}], "RED"),
        ([{"name": "ci", "status": "in_progress", "conclusion": None}], "PENDING"),
        ([], "NONE"),
    ],
)
@pytest.mark.asyncio
async def test_tool_get_branch_ci_status_verdicts(
    mock_client: MagicMock,
    get_branch_ci_status: ToolFn,
    check_runs: list[dict[str, object]],
    expected: str,
) -> None:
    """Verify branch health is classified with the same verdicts used for PRs."""
    mock_client.get_commit_check_runs = AsyncMock(return_value={"check_runs": check_runs})
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = json.loads(await get_branch_ci_status("owner", "repo", "main"))
    assert result["branch"] == "main"
    assert result["ci_status"] == expected


@pytest.mark.asyncio
async def test_tool_get_branch_ci_status_costs_one_round_of_calls(
    mock_client: MagicMock,
    get_branch_ci_status: ToolFn,
) -> None:
    """The whole point is answering before a clone, so it must not resolve a SHA first.

    GitHub accepts a branch name as a commit ref, so the branch is queried directly.
    """
    mock_client.get_commit_check_runs = AsyncMock(return_value={"check_runs": []})
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    await get_branch_ci_status("owner", "repo", "main")
    mock_client.get_commit_check_runs.assert_awaited_once_with("owner", "repo", "main")
    mock_client.get_commit_status.assert_awaited_once_with("owner", "repo", "main")
    mock_client.get_commit.assert_not_called()


@pytest.mark.asyncio
async def test_tool_get_branch_ci_status_defaults_to_the_default_branch(
    mock_client: MagicMock,
    get_branch_ci_status: ToolFn,
) -> None:
    """Callers asking 'is the base broken?' should not have to know the base's name."""
    mock_client.get_default_branch = AsyncMock(return_value="trunk")
    mock_client.get_commit_check_runs = AsyncMock(return_value={"check_runs": []})
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = json.loads(await get_branch_ci_status("owner", "repo"))
    assert result["branch"] == "trunk"
    mock_client.get_commit_check_runs.assert_awaited_once_with("owner", "repo", "trunk")


@pytest.mark.asyncio
async def test_tool_get_branch_ci_status_reports_which_checks_failed(
    mock_client: MagicMock,
    get_branch_ci_status: ToolFn,
) -> None:
    """A bare RED is not actionable; the agent needs to know what to go fix."""
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "check_runs": [
                {"name": "lint", "status": "completed", "conclusion": "failure"},
                {"name": "test", "status": "completed", "conclusion": "success"},
            ],
        },
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = json.loads(await get_branch_ci_status("owner", "repo", "main"))
    assert result["ci_status"] == "RED"
    failed = [c["name"] for c in result["checks"] if c["conclusion"] == "failure"]
    assert failed == ["lint"]


# --- The base branch the agent is told about must be the PR's own ---


@pytest.fixture
def list_bot_prs(tools: AgentTools) -> ToolFn:
    """Fixture to retrieve the bot PR listing tool."""
    return tools.list_bot_prs


@pytest.mark.asyncio
async def test_list_bot_prs_reports_the_branch_each_pr_targets(
    mock_client: MagicMock,
    list_bot_prs: ToolFn,
) -> None:
    """The listing must name each PR's base, so the agent can check the right branch.

    Without it the agent has nothing to pass to 'get_branch_ci_status' and
    silently falls back to the repository default, diagnosing a branch the PR
    does not target.
    """
    mock_client.list_open_prs = AsyncMock(
        return_value=[
            PullRequest.model_validate(
                {
                    "number": 5,
                    "title": "bump lib",
                    "user": {"login": DEFAULT_BOTS[0].author},
                    "created_at": "2026-08-01T00:00:00Z",
                    "base": {"ref": "develop"},
                },
            ),
        ],
    )
    result = json.loads(await list_bot_prs("owner", "repo"))
    assert result["bot_prs"][0]["base_ref"] == "develop"


@pytest.mark.asyncio
async def test_pr_status_reports_the_branch_the_pr_targets(
    mock_client: MagicMock,
    get_pr_status: ToolFn,
) -> None:
    """A PR's status must carry its base, so a RED verdict can be attributed."""
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 42,
            "title": "bump",
            "head": {"sha": "sha123"},
            "base": {"ref": "release/2.0"},
            "mergeable": True,
            "mergeable_state": "clean",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(return_value={"check_runs": []})
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = json.loads(await get_pr_status("owner", "repo", 42))
    assert result["base_ref"] == "release/2.0"
