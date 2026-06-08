from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.tools import (
    GitHubClient,
    ToolFn,
    _check_bot_author,
    create_tools,
)


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(spec=GitHubClient)


@pytest.fixture
def tools(mock_client: MagicMock) -> tuple[ToolFn, ToolFn, ToolFn]:
    return create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
    )


@pytest.fixture
def dry_run_tools(mock_client: MagicMock) -> tuple[ToolFn, ToolFn, ToolFn]:
    return create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=True,
        review_wait=0,
    )


# ============================================================
# merge_bot_pr
# ============================================================


@pytest.mark.asyncio
async def test_merge_bot_pr_success(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "Merge successful"})

    result = await merge("owner", "repo", 42)
    assert "Successfully merged PR #42" in result
    mock_client.merge_pr.assert_called_once_with("owner", "repo", 42)


@pytest.mark.asyncio
async def test_merge_bot_pr_dry_run(
    mock_client: MagicMock,
    dry_run_tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = dry_run_tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock()

    result = await merge("owner", "repo", 42)
    assert "[DRY-RUN]" in result
    mock_client.merge_pr.assert_not_called()


@pytest.mark.asyncio
async def test_merge_bot_pr_blocked(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="malicious_user")

    with pytest.raises(PermissionError) as exc_info:
        await merge("owner", "repo", 42)
    assert "malicious_user" in str(exc_info.value)


@pytest.mark.asyncio
async def test_merge_bot_pr_api_error(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(side_effect=Exception("GitHub API down"))

    with pytest.raises(Exception, match="GitHub API down"):
        await merge("owner", "repo", 42)


@pytest.mark.asyncio
async def test_merge_api_error_on_merge_call(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.merge_pr = AsyncMock(side_effect=Exception("Merge conflict"))

    with pytest.raises(Exception, match="Merge conflict"):
        await merge("owner", "repo", 42)
    mock_client.merge_pr.assert_called_once()


@pytest.mark.asyncio
async def test_merge_empty_author_blocked(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    merge, _, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="")

    with pytest.raises(PermissionError):
        await merge("owner", "repo", 42)


# ============================================================
# rebase_bot_pr
# ============================================================


@pytest.mark.asyncio
async def test_rebase_bot_pr_success(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock()

    result = await rebase("owner", "repo", 42)
    assert "Successfully requested rebase" in result
    mock_client.comment_on_pr.assert_called_once_with(
        "owner",
        "repo",
        42,
        "@dependabot rebase",
    )


@pytest.mark.asyncio
async def test_rebase_bot_pr_dry_run(
    mock_client: MagicMock,
    dry_run_tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    _, rebase, _ = dry_run_tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock()

    result = await rebase("owner", "repo", 42)
    assert "[DRY-RUN]" in result
    mock_client.comment_on_pr.assert_not_called()


@pytest.mark.asyncio
async def test_rebase_bot_pr_blocked(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="malicious_user")

    with pytest.raises(PermissionError):
        await rebase("owner", "repo", 42)


@pytest.mark.asyncio
async def test_rebase_api_error_on_comment(
    mock_client: MagicMock,
    tools: tuple[ToolFn, ToolFn, ToolFn],
) -> None:
    _, rebase, _ = tools
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")
    mock_client.comment_on_pr = AsyncMock(side_effect=Exception("Forbidden"))

    with pytest.raises(Exception, match="Forbidden"):
        await rebase("owner", "repo", 42)


# ============================================================
# Multi-bot support
# ============================================================


@pytest.mark.asyncio
async def test_merge_renovate_success(mock_client: MagicMock) -> None:
    merge, _, _ = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
    )
    mock_client.get_pr_author = AsyncMock(return_value="renovate[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "OK"})

    result = await merge("owner", "repo", 42)
    assert "Successfully merged" in result


@pytest.mark.asyncio
async def test_rebase_renovate_uses_correct_command(mock_client: MagicMock) -> None:
    _, rebase, _ = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
    )
    mock_client.get_pr_author = AsyncMock(return_value="renovate[bot]")
    mock_client.comment_on_pr = AsyncMock()

    await rebase("owner", "repo", 42)
    mock_client.comment_on_pr.assert_called_once_with(
        "owner",
        "repo",
        42,
        "@renovatebot rebase",
    )


@pytest.mark.asyncio
async def test_custom_bot_accepted(mock_client: MagicMock) -> None:
    custom = [BotConfig(author="custom[bot]", rebase_command="@custom rebase")]
    merge, _, _ = create_tools(
        client=mock_client,
        bots=custom,
        dry_run=False,
        review_wait=0,
    )
    mock_client.get_pr_author = AsyncMock(return_value="custom[bot]")
    mock_client.merge_pr = AsyncMock(return_value={"message": "OK"})

    result = await merge("owner", "repo", 1)
    assert "Successfully merged" in result


@pytest.mark.asyncio
async def test_custom_bot_rejects_default(mock_client: MagicMock) -> None:
    custom = [BotConfig(author="custom[bot]", rebase_command="@custom rebase")]
    merge, _, _ = create_tools(
        client=mock_client,
        bots=custom,
        dry_run=False,
        review_wait=0,
    )
    mock_client.get_pr_author = AsyncMock(return_value="dependabot[bot]")

    with pytest.raises(PermissionError):
        await merge("owner", "repo", 1)


# ============================================================
# _check_bot_author
# ============================================================


def test_check_bot_author_valid() -> None:
    bot = _check_bot_author("dependabot[bot]", DEFAULT_BOTS)
    assert bot.rebase_command == "@dependabot rebase"


def test_check_bot_author_invalid() -> None:
    with pytest.raises(PermissionError):
        _check_bot_author("malicious_user", DEFAULT_BOTS)


# ============================================================
# wait_for_reviews
# ============================================================


@pytest.mark.asyncio
async def test_wait_for_reviews_disabled(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=0,
    )
    result = await wait("owner", "repo", 42)
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_wait_for_reviews_finds_comments(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=5,
    )
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
async def test_wait_for_reviews_polls_until_timeout(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=1,
    )
    mock_client.get_pr_reviews = AsyncMock(return_value=[])

    with patch(
        "dependency_director.tools.asyncio.sleep",
        new_callable=AsyncMock,
    ) as mock_sleep:
        result = await wait("owner", "repo", 42)
    assert "no review comments" in result.lower()
    assert mock_client.get_pr_reviews.call_count == 3  # 1 baseline + 2 polls
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_reviews_ignores_stale(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=5,
    )
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
async def test_wait_for_reviews_empty_body_not_ignored(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=1,
    )
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
) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=1,
    )
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
async def test_wait_for_reviews_missing_user_key(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=1,
    )
    mock_client.get_pr_reviews = AsyncMock(
        side_effect=[
            [],
            [{"body": "Needs fix", "state": "CHANGES_REQUESTED"}],
        ],
    )

    with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
        result = await wait("owner", "repo", 42)
    assert "unknown" in result
    assert "Needs fix" in result


@pytest.mark.asyncio
async def test_wait_for_reviews_negative_timeout(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=-1,
    )
    result = await wait("owner", "repo", 42)
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_wait_for_reviews_api_error(mock_client: MagicMock) -> None:
    _, _, wait = create_tools(
        client=mock_client,
        bots=DEFAULT_BOTS,
        dry_run=False,
        review_wait=1,
    )
    mock_client.get_pr_reviews = AsyncMock(side_effect=Exception("API down"))

    with pytest.raises(Exception, match="API down"):
        with patch("dependency_director.tools.asyncio.sleep", new_callable=AsyncMock):
            await wait("owner", "repo", 42)


# ============================================================
# GitHubClient
# ============================================================


@pytest.mark.asyncio
async def test_github_client_get_pr_author() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": {"login": "dependabot[bot]"}}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        assert await c.get_pr_author("o", "r", 1) == "dependabot[bot]"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_missing() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_user_none() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": None}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_author_login_none() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"user": {"login": None}}
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        assert await c.get_pr_author("o", "r", 1) == ""
        await c.close()


@pytest.mark.asyncio
async def test_github_client_merge_pr() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": "Merged"}
    with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
        mock_put.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.merge_pr("o", "r", 1)
        assert result["message"] == "Merged"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_comment_on_pr() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "body": "test"}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        c = GitHubClient(token="dummy")
        result = await c.comment_on_pr("o", "r", 1, "test")
        assert result["body"] == "test"
        await c.close()


@pytest.mark.asyncio
async def test_github_client_get_pr_reviews() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = [{"state": "APPROVED"}]
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        reviews = await c.get_pr_reviews("o", "r", 1)
        assert len(reviews) == 1
        await c.close()


@pytest.mark.asyncio
async def test_github_client_empty_token() -> None:
    c = GitHubClient(token="")
    assert "Authorization" not in c.headers
    await c.close()


@pytest.mark.asyncio
async def test_github_client_with_token() -> None:
    c = GitHubClient(token="my-token")
    assert c.headers["Authorization"] == "Bearer my-token"
    await c.close()


@pytest.mark.asyncio
async def test_github_client_double_close() -> None:
    c = GitHubClient(token="dummy")
    await c.close()
    await c.close()


@pytest.mark.asyncio
async def test_github_client_http_errors() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
        message="Not Found",
        request=MagicMock(),
        response=mock_response,
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        c = GitHubClient(token="dummy")
        with pytest.raises(httpx_mod.HTTPStatusError):
            await c.get_pr_author("o", "r", 999)
        await c.close()


@pytest.mark.asyncio
async def test_github_client_parameter_validation_denied() -> None:
    c = GitHubClient(token="dummy")
    # Test path traversal in owner
    with pytest.raises(ValueError, match="Invalid owner name"):
        await c.get_pr_author("owner/../../something", "repo", 42)
    # Test path traversal in repo
    with pytest.raises(ValueError, match="Invalid repository name"):
        await c.get_pr_author("owner", "repo/../../something", 42)
    # Test illegal characters in owner
    with pytest.raises(ValueError, match="Invalid owner name"):
        await c.get_pr_author("owner$", "repo", 42)
    # Test illegal characters in repo
    with pytest.raises(ValueError, match="Invalid repository name"):
        await c.get_pr_author("owner", "repo$", 42)
    await c.close()
