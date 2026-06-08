import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependency_director.config import DEFAULT_BOTS
from dependency_director.tools import (
    GitHubClient,
    GitHubNotFoundError,
    ToolFn,
    create_agent_tools,
)

# ============================================================
# GitHubClient Unit Tests
# ============================================================


@pytest.mark.asyncio
async def test_github_client_get_pr_details() -> None:
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
        c = GitHubClient(token="dummy")
        result = await c.get_pr_details("owner", "repo", 22)
        assert result["number"] == 22
        assert result["head"]["sha"] == "sha123"
        assert result["mergeable"] is True
        assert result["mergeable_state"] == "clean"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/pulls/22",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_commit_check_runs() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "check_runs": [
            {"name": "test", "status": "completed", "conclusion": "success"},
        ],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.get_commit_check_runs("owner", "repo", "sha123")
        assert result["total_count"] == 1
        assert result["check_runs"][0]["conclusion"] == "success"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/sha123/check-runs",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_commit_status() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "state": "success",
        "statuses": [{"context": "ci/circleci", "state": "success"}],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.get_commit_status("owner", "repo", "sha123")
        assert result["state"] == "success"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/sha123/status",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_workflow_runs_for_commit() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "workflow_runs": [
            {"id": 98765, "status": "completed", "conclusion": "success"},
        ],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
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
async def test_github_client_get_workflow_run_jobs() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_count": 1,
        "jobs": [
            {
                "id": 111,
                "name": "build",
                "status": "completed",
                "conclusion": "failure",
            },
        ],
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.get_workflow_run_jobs("owner", "repo", 98765)
        assert result["jobs"][0]["id"] == 111
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/actions/runs/98765/jobs",
            headers=c.headers,
        )
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_job_logs() -> None:
    mock_response = MagicMock()
    mock_response.text = "line1\nline2\nline3"
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.get_job_logs("owner", "repo", 111)
        assert result == "line1\nline2\nline3"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/actions/jobs/111/logs",
            headers=c.headers,
        )
        await c.close()


# ============================================================
# High-Level Tools Unit Tests (create_tools)
# ============================================================


@pytest.fixture
def tools(mock_client: MagicMock) -> tuple[ToolFn, ...]:
    return create_agent_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
    )


@pytest.fixture
def get_pr_status(tools: tuple[ToolFn, ...]) -> ToolFn:
    fn = tools[3]
    fn_name = getattr(fn, "__name__", "")
    assert fn_name == "get_pr_status", f"Unexpected tool at index 3: {fn_name!r}"
    return fn


@pytest.fixture
def get_pr_workflow_run_logs(tools: tuple[ToolFn, ...]) -> ToolFn:
    fn = tools[4]
    fn_name = getattr(fn, "__name__", "")
    assert fn_name == "get_pr_workflow_run_logs", (
        f"Unexpected tool at index 4: {fn_name!r}"
    )
    return fn


@pytest.mark.asyncio
async def test_tool_get_pr_status_green(
    mock_client: MagicMock,
    get_pr_status: ToolFn,
) -> None:
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
            "check_runs": [
                {"name": "test", "status": "completed", "conclusion": "success"},
            ],
        },
    )
    mock_client.get_commit_status = AsyncMock(
        return_value={
            "state": "pending",
            "statuses": [],
        },
    )

    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)

    assert result["pr_number"] == 22
    assert result["mergeable"] is True
    assert result["mergeable_state"] == "clean"
    assert result["ci_status"] == "GREEN"
    assert len(result["checks"]) == 1
    assert result["checks"][0]["name"] == "test"


@pytest.mark.asyncio
async def test_tool_get_pr_status_red_failing_check_run(
    mock_client: MagicMock,
    get_pr_status: ToolFn,
) -> None:
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
            "check_runs": [
                {"name": "test", "status": "completed", "conclusion": "failure"},
            ],
        },
    )
    mock_client.get_commit_status = AsyncMock(
        return_value={
            "state": "pending",
            "statuses": [],
        },
    )

    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)

    assert result["ci_status"] == "RED"


@pytest.mark.asyncio
async def test_tool_get_pr_status_pending(
    mock_client: MagicMock,
    get_pr_status: ToolFn,
) -> None:
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
            "check_runs": [
                {"name": "test", "status": "in_progress", "conclusion": None},
            ],
        },
    )
    mock_client.get_commit_status = AsyncMock(
        return_value={
            "state": "pending",
            "statuses": [],
        },
    )

    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)

    assert result["ci_status"] == "PENDING"


@pytest.mark.asyncio
async def test_tool_get_pr_status_conflict(
    mock_client: MagicMock,
    get_pr_status: ToolFn,
) -> None:
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "title": "bump",
            "head": {"sha": "sha123"},
            "mergeable": False,
            "mergeable_state": "dirty",
        },
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={
            "total_count": 0,
            "check_runs": [],
        },
    )
    mock_client.get_commit_status = AsyncMock(
        return_value={
            "state": "pending",
            "statuses": [],
        },
    )

    result_str = await get_pr_status("owner", "repo", 22)
    result = json.loads(result_str)

    assert result["mergeable"] is False
    assert result["mergeable_state"] == "dirty"
    assert result["ci_status"] == "CONFLICT"


@pytest.mark.asyncio
async def test_tool_get_pr_workflow_run_logs(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "head": {"sha": "sha123"},
        },
    )
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={
            "total_count": 1,
            "workflow_runs": [{"id": 98765}],
        },
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

    # Large log file representation
    large_log = "\n".join(f"log line {i}" for i in range(1, 100))
    mock_client.get_job_logs = AsyncMock(return_value=large_log)

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "--- FAILED JOB: build (ID: 111) ---" in result
    assert "log line 99" in result
    # It should have truncated to only include the last 50 lines
    assert "log line 1" not in result
    assert "log line 50" in result


@pytest.mark.asyncio
async def test_tool_get_pr_workflow_run_logs_api_error_on_logs(
    mock_client: MagicMock,
    get_pr_workflow_run_logs: ToolFn,
) -> None:
    mock_client.get_pr_details = AsyncMock(
        return_value={
            "number": 22,
            "head": {"sha": "sha123"},
        },
    )
    mock_client.get_workflow_runs_for_commit = AsyncMock(
        return_value={
            "total_count": 1,
            "workflow_runs": [{"id": 98765}],
        },
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

    mock_client.get_job_logs = AsyncMock(
        side_effect=GitHubNotFoundError("GitHub API error 404"),
    )

    result = await get_pr_workflow_run_logs("owner", "repo", 22)

    assert "--- FAILED JOB: build (ID: 111) ---" in result
    assert "Failed to retrieve log: GitHub API error 404" in result
