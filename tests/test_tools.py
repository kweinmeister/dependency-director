"""Tests for repository client and git command tools in dependency-director."""

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.tools import (
    GitHubAuthenticationError,
    GitHubClient,
    GitHubNotFoundError,
    ToolFn,
    _check_bot_author,
    _make_create_pr,
    _make_wait_for_ci,
    _make_write_tools,
)

from .conftest import make_client_with_status


@pytest.fixture
def tools(mock_client: MagicMock) -> tuple[ToolFn, ToolFn, ToolFn]:
    """Fixture to set up write tools for status checking tests."""
    return _make_write_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=False, review_wait=0)


@pytest.fixture
def dry_run_tools(mock_client: MagicMock) -> tuple[ToolFn, ToolFn, ToolFn]:
    """Fixture to set up write tools in dry-run mode."""
    return _make_write_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=True, review_wait=0)


@pytest.mark.asyncio
async def test_merge_bot_pr_success(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR merge command succeeds when all requirements are met."""
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "Merge successful"})
    result = await merge("owner", "repo", 42)
    assert "Successfully merged PR #42" in result
    mock_client.merge_pr.assert_called_once_with("owner", "repo", 42)


@pytest.mark.asyncio
async def test_merge_bot_pr_dry_run(mock_client: MagicMock, dry_run_tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR merge is simulated correctly in dry-run mode."""
    merge, _, _ = dry_run_tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock()
    result = await merge("owner", "repo", 42)
    assert "[DRY-RUN]" in result
    mock_client.merge_pr.assert_not_called()


@pytest.mark.asyncio
async def test_merge_bot_pr_blocked(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR merge is blocked when requirements are not met."""
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="malicious_user")
    with pytest.raises(PermissionError) as exc_info:
        await merge("owner", "repo", 42)
    assert "malicious_user" in str(exc_info.value)


@pytest.mark.asyncio
async def test_merge_bot_pr_api_error(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR merge handles GitHub API errors gracefully."""
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(side_effect=Exception("GitHub API down"))
    with pytest.raises(Exception, match="GitHub API down"):
        await merge("owner", "repo", 42)


@pytest.mark.asyncio
async def test_merge_api_error_on_merge_call(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify PR merge handles API error during merge call."""
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock(side_effect=Exception("Merge conflict"))
    with pytest.raises(Exception, match="Merge conflict"):
        await merge("owner", "repo", 42)
    mock_client.merge_pr.assert_called_once()


@pytest.mark.asyncio
async def test_merge_empty_author_blocked(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify PR merge is blocked if the author name is empty."""
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="")
    with pytest.raises(PermissionError):
        await merge("owner", "repo", 42)


@pytest.mark.asyncio
async def test_rebase_bot_pr_success(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR rebase command succeeds when all requirements are met."""
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock()
    result = await rebase("owner", "repo", 42)
    assert "Successfully requested rebase" in result
    mock_client.comment_on_pr.assert_called_once_with("owner", "repo", 42, "@dependabot rebase")


@pytest.mark.asyncio
async def test_rebase_bot_pr_dry_run(mock_client: MagicMock, dry_run_tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR rebase is simulated correctly in dry-run mode."""
    _, rebase, _ = dry_run_tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock()
    result = await rebase("owner", "repo", 42)
    assert "[DRY-RUN]" in result
    mock_client.comment_on_pr.assert_not_called()


@pytest.mark.asyncio
async def test_rebase_bot_pr_blocked(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify bot PR rebase is blocked when requirements are not met."""
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="malicious_user")
    with pytest.raises(PermissionError):
        await rebase("owner", "repo", 42)


@pytest.mark.asyncio
async def test_rebase_api_error_on_comment(mock_client: MagicMock, tools: tuple[ToolFn, ToolFn, ToolFn]) -> None:
    """Verify PR rebase handles API error during commenting."""
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock(side_effect=Exception("Forbidden"))
    with pytest.raises(Exception, match="Forbidden"):
        await rebase("owner", "repo", 42)


@pytest.mark.asyncio
async def test_merge_renovate_success(mock_client: MagicMock) -> None:
    """Verify PR merge succeeds for Renovate bot."""
    merge, _, _ = _make_write_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=False, review_wait=0)
    mock_client.get_pr_author = AsyncMock(return_value="renovate[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "OK"})
    result = await merge("owner", "repo", 42)
    assert "Successfully merged" in result


@pytest.mark.asyncio
async def test_merge_bot_pr_405_returns_actionable_message(
    mock_client: MagicMock,
) -> None:
    """Verify PR merge 405 error returns an actionable message."""
    merge, _, _ = _make_write_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=False, review_wait=0)
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_response = MagicMock()
    mock_response.status_code = 405
    mock_client.merge_pr = AsyncMock(
        side_effect=httpx_mod.HTTPStatusError("405", request=MagicMock(), response=mock_response),
    )
    result = await merge("owner", "repo", 99)
    assert "405" in result
    assert "get_pr_status" in result
    assert "rebase_bot_pr" in result


@pytest.mark.asyncio
async def test_rebase_renovate_uses_correct_command(mock_client: MagicMock) -> None:
    """Verify Renovate bot rebase uses the correct CLI command."""
    _, rebase, _ = _make_write_tools(client=mock_client, bots=DEFAULT_BOTS, dry_run=False, review_wait=0)
    mock_client.get_pr_author = AsyncMock(return_value="renovate[bot]")
    mock_client.comment_on_pr = AsyncMock()
    await rebase("owner", "repo", 42)
    mock_client.comment_on_pr.assert_called_once_with("owner", "repo", 42, "@renovatebot rebase")


@pytest.mark.asyncio
async def test_custom_bot_accepted(mock_client: MagicMock) -> None:
    """Verify custom bot configuration is accepted."""
    custom = [BotConfig(author="custom[bot]", rebase_command="@custom rebase")]
    merge, _, _ = _make_write_tools(client=mock_client, bots=custom, dry_run=False, review_wait=0)
    mock_client.get_pr_author = AsyncMock(return_value="custom[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "OK"})
    result = await merge("owner", "repo", 1)
    assert "Successfully merged" in result


@pytest.mark.asyncio
async def test_custom_bot_rejects_default(mock_client: MagicMock) -> None:
    """Verify custom bot configuration rejects default bot commands."""
    custom = [BotConfig(author="custom[bot]", rebase_command="@custom rebase")]
    merge, _, _ = _make_write_tools(client=mock_client, bots=custom, dry_run=False, review_wait=0)
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    with pytest.raises(PermissionError):
        await merge("owner", "repo", 1)


def test_check_bot_author_valid() -> None:
    """Verify check_bot_author permits a valid configured bot author."""
    bot = _check_bot_author("dependabot[bot]", DEFAULT_BOTS)
    assert bot.rebase_command == "@dependabot rebase"


def test_check_bot_author_invalid() -> None:
    """Verify check_bot_author raises PermissionError for invalid authors."""
    with pytest.raises(PermissionError):
        _check_bot_author("malicious_user", DEFAULT_BOTS)


@pytest.mark.asyncio
async def test_wait_for_reviews_disabled(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews exits immediately when disabled."""
    _ = mock_client
    wait = wait_tool(review_wait=0)
    result = await wait("owner", "repo", 42)
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_wait_for_reviews_finds_comments(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews stops waiting when approvals are found in comments."""
    wait = wait_tool(review_wait=5)
    mock_client.get_pr_reviews = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "user": {"login": "bot"},
                    "body": "Use typing.cast",
                    "state": "COMMENTED",
                },
            ],
        ],
    )
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "typing.cast" in result
    assert mock_client.get_pr_reviews.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_reviews_polls_until_timeout(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews polls until timeout when no reviews are found."""
    wait = wait_tool(review_wait=1)
    mock_client.get_pr_reviews = AsyncMock(return_value=[])
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await wait("owner", "repo", 42)
    assert "no review comments" in result.lower()
    assert mock_client.get_pr_reviews.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_reviews_ignores_stale(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews ignores reviews that are older than specified."""
    wait = wait_tool(review_wait=5)
    stale = [{"user": {"login": "old"}, "body": "LGTM", "state": "APPROVED"}]
    new = [
        *stale,
        {"user": {"login": "new"}, "body": "Fix types", "state": "CHANGES_REQUESTED"},
    ]
    mock_client.get_pr_reviews = AsyncMock(side_effect=[stale, stale, new])
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "new" in result
    assert "Fix types" in result


@pytest.mark.asyncio
async def test_wait_for_reviews_empty_body_not_ignored(
    mock_client: MagicMock,
    wait_tool: Callable[..., ToolFn],
) -> None:
    """Verify wait_for_reviews does not ignore empty-bodied comments."""
    wait = wait_tool(review_wait=1)
    mock_client.get_pr_reviews = AsyncMock(
        side_effect=[
            [],
            [{"id": 100, "user": {"login": "tester"}, "body": "", "state": "APPROVED"}],
        ],
    )
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "Review comments found" in result
    assert "tester" in result
    assert "APPROVED" in result


@pytest.mark.asyncio
async def test_wait_for_reviews_handles_dismissed_reviews(
    mock_client: MagicMock,
    wait_tool: Callable[..., ToolFn],
) -> None:
    """Verify wait_for_reviews handles dismissed reviews correctly."""
    wait = wait_tool(review_wait=1)
    mock_client.get_pr_reviews = AsyncMock(
        side_effect=[
            [
                {"id": 1, "user": {"login": "u1"}, "body": "ok", "state": "COMMENTED"},
                {"id": 2, "user": {"login": "u2"}, "body": "ok", "state": "COMMENTED"},
                {"id": 3, "user": {"login": "u3"}, "body": "ok", "state": "COMMENTED"},
            ],
            [
                {"id": 1, "user": {"login": "u1"}, "body": "ok", "state": "COMMENTED"},
                {"id": 3, "user": {"login": "u3"}, "body": "ok", "state": "COMMENTED"},
                {
                    "id": 4,
                    "user": {"login": "u4"},
                    "body": "new feedback",
                    "state": "CHANGES_REQUESTED",
                },
            ],
        ],
    )
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "Review comments found" in result
    assert "u4" in result
    assert "new feedback" in result


@pytest.mark.asyncio
async def test_wait_for_reviews_missing_user_key(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews handles comments missing user metadata."""
    wait = wait_tool(review_wait=1)
    mock_client.get_pr_reviews = AsyncMock(side_effect=[[], [{"body": "Needs fix", "state": "CHANGES_REQUESTED"}]])
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "unknown" in result
    assert "Needs fix" in result


@pytest.mark.asyncio
async def test_wait_for_reviews_negative_timeout(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews raises error for negative timeout values."""
    _ = mock_client
    wait = wait_tool(review_wait=-1)
    result = await wait("owner", "repo", 42)
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_wait_for_reviews_api_error(mock_client: MagicMock, wait_tool: Callable[..., ToolFn]) -> None:
    """Verify wait_for_reviews handles API errors gracefully during polling."""
    wait = wait_tool(review_wait=1)
    mock_client.get_pr_reviews = AsyncMock(side_effect=Exception("API down"))
    with (
        pytest.raises(Exception, match="API down"),
        patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock),
    ):
        await wait("owner", "repo", 42)


@pytest.mark.asyncio
async def test_github_client_get_pr_author(github_token: str) -> None:
    """Verify GitHub client retrieves PR author username correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": {"login": "dependabot[bot]"}}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_pr_author("o", "r", 1) == "dependabot[bot]"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_missing(github_token: str) -> None:
    """Verify GitHub client returns empty string if PR author info is missing."""
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_user_none(github_token: str) -> None:
    """Verify GitHub client returns empty string if author user is None."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": None}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_login_none(github_token: str) -> None:
    """Verify GitHub client returns empty string if author login is None."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": {"login": None}}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_merge_pr(github_token: str) -> None:
    """Verify GitHub client sends merge request correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": "Merged"}
    with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
        mock_put.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.merge_pr("o", "r", 1)
        assert result["message"] == "Merged"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_comment_on_pr(github_token: str) -> None:
    """Verify GitHub client comments on PR correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "body": "test"}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.comment_on_pr("o", "r", 1, "test")
        assert result["body"] == "test"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_reviews(github_token: str) -> None:
    """Verify GitHub client retrieves PR reviews correctly."""
    mock_response = MagicMock()
    mock_response.json.return_value = [{"state": "APPROVED"}]
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        reviews = await c.get_pr_reviews("o", "r", 1)
        assert len(reviews) == 1
        await c.close()


@pytest.mark.asyncio
async def test_github_client_empty_token() -> None:
    """Verify GitHub client initializes headers correctly with an empty token."""
    c = GitHubClient(token="")
    assert "Authorization" not in c.headers
    await c.close()


@pytest.mark.asyncio
async def test_github_client_with_token(github_token: str) -> None:
    """Verify GitHub client initializes headers correctly with a valid token."""
    c = GitHubClient(token=github_token)
    assert c.headers["Authorization"] == f"Bearer {github_token}"
    await c.close()


@pytest.mark.asyncio
async def test_github_client_double_close(github_token: str) -> None:
    """Verify GitHub client can be closed multiple times safely."""
    c = GitHubClient(token=github_token)
    await c.close()
    await c.close()


@pytest.mark.asyncio
async def test_github_client_http_errors(github_token: str) -> None:
    """Verify GitHub client propagates HTTP client errors correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
        message="Not Found",
        request=MagicMock(),
        response=mock_response,
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        with pytest.raises(httpx_mod.HTTPStatusError):
            await c.get_pr_author("o", "r", 999)
        await c.close()


@pytest.mark.asyncio
async def test_github_client_parameter_validation_denied(github_token: str) -> None:
    """Verify GitHub client rejects invalid parameters in owner/repo names."""
    c = GitHubClient(token=github_token)
    with pytest.raises(ValueError, match="Invalid owner name"):
        await c.get_pr_author("owner/../../something", "repo", 42)
    with pytest.raises(ValueError, match="Invalid repository name"):
        await c.get_pr_author("owner", "repo/../../something", 42)
    with pytest.raises(ValueError, match="Invalid owner name"):
        await c.get_pr_author("owner$", "repo", 42)
    with pytest.raises(ValueError, match="Invalid repository name"):
        await c.get_pr_author("owner", "repo$", 42)
    await c.close()


@pytest.mark.asyncio
async def test_github_client_fail_fast_401() -> None:
    """Verify GitHub client fails fast on authentication errors."""
    c = make_client_with_status(401, "https://api.github.com/repos/owner/repo/pulls/1")
    with pytest.raises(GitHubAuthenticationError) as exc_info:
        await c.get_pr_author("owner", "repo", 1)
    assert "GitHub API error 401" in str(exc_info.value)
    await c.close()


@pytest.mark.asyncio
async def test_github_client_get_file_contents_404_raises_not_found() -> None:
    """Verify get_file_contents raises FileNotFoundError on 404."""
    c = make_client_with_status(404, "https://api.github.com/repos/owner/repo/contents/path")
    with pytest.raises(GitHubNotFoundError) as exc_info:
        await c.get_file_contents("owner", "repo", "path")
    assert "GitHub API error 404" in str(exc_info.value)
    await c.close()


# --- wait_for_ci tests (TDD) ---


@pytest.fixture
def ci_tool(mock_client: MagicMock) -> ToolFn:
    """Create a wait_for_ci tool bound to a mock client."""
    return _make_wait_for_ci(mock_client)


@pytest.mark.asyncio
async def test_wait_for_ci_returns_on_green(mock_client: MagicMock, ci_tool: ToolFn) -> None:
    """wait_for_ci returns immediately when CI is already GREEN."""
    mock_client.get_pr_details = AsyncMock(
        return_value={"head": {"sha": "abc123"}, "mergeable": True, "mergeable_state": "clean", "title": "bump foo"},
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={"check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}]},
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = await ci_tool("owner", "repo", 42)
    assert "GREEN" in result
    # Should not need to poll — one call is enough
    assert mock_client.get_pr_details.call_count == 1


@pytest.mark.asyncio
async def test_wait_for_ci_returns_on_red(mock_client: MagicMock, ci_tool: ToolFn) -> None:
    """wait_for_ci returns immediately when CI has failed."""
    mock_client.get_pr_details = AsyncMock(
        return_value={"head": {"sha": "abc123"}, "mergeable": True, "mergeable_state": "clean", "title": "bump foo"},
    )
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={"check_runs": [{"name": "ci", "status": "completed", "conclusion": "failure"}]},
    )
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    result = await ci_tool("owner", "repo", 42)
    assert "RED" in result
    assert mock_client.get_pr_details.call_count == 1


@pytest.mark.asyncio
async def test_wait_for_ci_none_then_green(mock_client: MagicMock, ci_tool: ToolFn) -> None:
    """wait_for_ci polls through NONE → PENDING → GREEN."""
    pr_details = {"head": {"sha": "abc123"}, "mergeable": True, "mergeable_state": "clean", "title": "bump foo"}
    mock_client.get_pr_details = AsyncMock(return_value=pr_details)
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})

    # 1st call: no checks (NONE), 2nd: pending, 3rd: green
    mock_client.get_commit_check_runs = AsyncMock(
        side_effect=[
            {"check_runs": []},
            {"check_runs": [{"name": "ci", "status": "in_progress", "conclusion": None}]},
            {"check_runs": [{"name": "ci", "status": "completed", "conclusion": "success"}]},
        ],
    )
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await ci_tool("owner", "repo", 42)
    assert "GREEN" in result
    assert mock_client.get_pr_details.call_count == 3


@pytest.mark.asyncio
async def test_wait_for_ci_timeout(mock_client: MagicMock, ci_tool: ToolFn) -> None:
    """wait_for_ci gives up after max retries and reports timeout."""
    pr_details = {"head": {"sha": "abc123"}, "mergeable": True, "mergeable_state": "clean", "title": "bump foo"}
    mock_client.get_pr_details = AsyncMock(return_value=pr_details)
    mock_client.get_commit_status = AsyncMock(return_value={"statuses": []})
    mock_client.get_commit_check_runs = AsyncMock(
        return_value={"check_runs": [{"name": "ci", "status": "in_progress", "conclusion": None}]},
    )
    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await ci_tool("owner", "repo", 42)
    assert "pending" in result.lower() or "timeout" in result.lower()
    # Should have polled max retries + 1 initial call
    assert mock_client.get_pr_details.call_count >= 10


# --- create_pr tool ---


@pytest.fixture
def create_pr(mock_client: MagicMock) -> ToolFn:
    """Fixture for the standalone-PR creation tool."""
    return _make_create_pr(mock_client, dry_run=False)


@pytest.mark.asyncio
async def test_create_pr_opens_pull_request(mock_client: MagicMock, create_pr: ToolFn) -> None:
    """Verify create_pr opens a PR and reports its URL.

    The standalone fix strategy pushes to its own branch, which is useless
    without a way to open the PR that carries it.
    """
    mock_client.get_default_branch = AsyncMock(return_value="main")
    mock_client.create_pull_request = AsyncMock(
        return_value={"number": 101, "html_url": "https://github.com/owner/repo/pull/101"},
    )
    result = await create_pr("owner", "repo", "fix: bump ty", "dependency-director/fix-90", "Fixes #90", "")
    assert "#101" in result
    assert "https://github.com/owner/repo/pull/101" in result
    mock_client.create_pull_request.assert_awaited_once_with(
        "owner",
        "repo",
        title="fix: bump ty",
        head="dependency-director/fix-90",
        base="main",
        body="Fixes #90",
    )


@pytest.mark.asyncio
async def test_create_pr_uses_explicit_base_without_lookup(mock_client: MagicMock, create_pr: ToolFn) -> None:
    """Verify an explicit base branch is honoured and skips the default-branch lookup."""
    mock_client.get_default_branch = AsyncMock(return_value="main")
    mock_client.create_pull_request = AsyncMock(return_value={"number": 7, "html_url": "u"})
    await create_pr("owner", "repo", "t", "head", "b", "develop")
    mock_client.create_pull_request.assert_awaited_once_with(
        "owner",
        "repo",
        title="t",
        head="head",
        base="develop",
        body="b",
    )
    mock_client.get_default_branch.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_pr_dry_run_does_not_write(mock_client: MagicMock) -> None:
    """Verify dry-run reports the intended PR without creating one."""
    mock_client.create_pull_request = AsyncMock()
    tool = _make_create_pr(mock_client, dry_run=True)
    result = await tool("owner", "repo", "fix: bump ty", "dependency-director/fix-90", "body", "main")
    assert "[DRY-RUN]" in result
    mock_client.create_pull_request.assert_not_awaited()


# --- rebase clobber guard ---


def _commit(login: str | None) -> dict[str, object]:
    """Build a minimal PR-commit payload attributed to the given login."""
    return {"sha": "s", "author": {"login": login} if login else None}


@pytest.mark.asyncio
async def test_rebase_refuses_to_clobber_agent_commits(mock_client: MagicMock, tools: tuple[ToolFn, ...]) -> None:
    """Verify a rebase is refused once the branch carries non-bot commits.

    Asking Dependabot to rebase force-pushes the branch from scratch, which
    silently discards any fix already pushed to it.
    """
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.list_pr_commits = AsyncMock(return_value=[_commit("dependabot[bot]"), _commit("someone-else")])
    mock_client.comment_on_pr = AsyncMock()

    result = await rebase("owner", "repo", 90)

    assert "someone-else" in result
    mock_client.comment_on_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_rebase_refuses_when_commit_author_is_unattributed(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ...],
) -> None:
    """Verify an unattributed commit blocks the rebase.

    A commit GitHub cannot map to an account is not demonstrably the bot's,
    and wrongly discarding a fix costs more than a rebase the agent has to
    request another way.
    """
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.list_pr_commits = AsyncMock(return_value=[_commit(None)])
    mock_client.comment_on_pr = AsyncMock()

    result = await rebase("owner", "repo", 90)

    assert "unattributed" in result.lower()
    mock_client.comment_on_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_rebase_proceeds_on_untouched_bot_branch(mock_client: MagicMock, tools: tuple[ToolFn, ...]) -> None:
    """Verify the guard leaves an untouched bot branch rebaseable."""
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.list_pr_commits = AsyncMock(return_value=[_commit("dependabot[bot]")])
    mock_client.comment_on_pr = AsyncMock(return_value={})

    result = await rebase("owner", "repo", 90)

    assert "Successfully requested rebase" in result
    mock_client.comment_on_pr.assert_awaited_once()


@pytest.mark.asyncio
async def test_rebase_dry_run_still_reports_the_clobber(
    mock_client: MagicMock,
    dry_run_tools: tuple[ToolFn, ...],
) -> None:
    """Verify dry-run surfaces the refusal rather than promising a rebase it would not do."""
    _, rebase, _ = dry_run_tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.list_pr_commits = AsyncMock(return_value=[_commit("someone-else")])

    result = await rebase("owner", "repo", 90)

    assert "[DRY-RUN]" not in result
    assert "someone-else" in result


# --- client methods backing create_pr ---


@pytest.mark.asyncio
async def test_github_client_get_default_branch(github_token: str) -> None:
    """Verify the client reads the repository's default branch."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"default_branch": "trunk"}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_default_branch("owner", "repo") == "trunk"
        mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo", headers=c.headers)
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_default_branch_falls_back(github_token: str) -> None:
    """Verify a repository with no reported default branch falls back to main."""
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token=github_token)
        assert await c.get_default_branch("owner", "repo") == "main"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_create_pull_request(github_token: str) -> None:
    """Verify the client posts a pull request to the correct endpoint."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"number": 5}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        c = GitHubClient(token=github_token)
        result = await c.create_pull_request("owner", "repo", title="t", head="h", base="b", body="d")
        assert result["number"] == 5
        mock_post.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/pulls",
            headers=c.headers,
            json={"title": "t", "head": "h", "base": "b", "body": "d"},
        )
        await c.close()
