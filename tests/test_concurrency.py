from typing import Any, Never
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.main import get_repositories, run_agent


@pytest.mark.asyncio
async def test_get_repositories_filters_forks() -> None:
    mock_repos_page1 = [
        {"name": "repo-1", "fork": False},
        {"name": "repo-2", "fork": True},
        {"name": "repo-3", "fork": False},
    ]
    mock_repos_page2: list[dict[str, Any]] = []

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_user = MagicMock()
        mock_user.status_code = 200
        mock_user.json.return_value = {"login": "not-test-owner"}

        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = mock_repos_page1

        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = mock_repos_page2

        mock_get.side_effect = [mock_user, mock_response1, mock_response2]

        repos = await get_repositories(owner="test-owner", token="dummy-token")
        assert repos == ["test-owner/repo-1", "test-owner/repo-3"]

        assert mock_get.call_count == 3
        _, kwargs = mock_get.call_args_list[1]
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer dummy-token"


@pytest.mark.asyncio
async def test_get_repositories_pagination() -> None:
    page_1_data = [
        {"name": "repo-1", "fork": False},
    ]
    page_2_data = [
        {"name": "repo-2", "fork": False},
    ]
    page_3_data: list[dict[str, Any]] = []  # Empty list to end pagination

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_user = MagicMock()
        mock_user.status_code = 200
        mock_user.json.return_value = {"login": "not-test-owner"}

        response_1 = MagicMock()
        response_1.status_code = 200
        response_1.json.return_value = page_1_data

        response_2 = MagicMock()
        response_2.status_code = 200
        response_2.json.return_value = page_2_data

        response_3 = MagicMock()
        response_3.status_code = 200
        response_3.json.return_value = page_3_data

        mock_get.side_effect = [mock_user, response_1, response_2, response_3]

        repos = await get_repositories(owner="test-owner", token="dummy-token")
        assert repos == ["test-owner/repo-1", "test-owner/repo-2"]
        assert mock_get.call_count == 4


@pytest.mark.asyncio
@patch("dependency_director.main.run_agent_for_repo", new_callable=AsyncMock)
@patch("dependency_director.main.get_repositories", new_callable=AsyncMock)
async def test_run_agent_concurrency(
    mock_get_repos: MagicMock,
    mock_run_agent_for_repo: MagicMock,
) -> None:
    mock_get_repos.return_value = [
        "test-owner/repo-1",
        "test-owner/repo-2",
        "test-owner/repo-3",
    ]

    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = "dummy"
        mock_settings.no_sandbox = False

        await run_agent(
            "test-owner",
            concurrency=2,
            max_attempts=3,
            repo=None,
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=False,
            review_wait=5,
        )

        mock_get_repos.assert_called_once_with("test-owner", "dummy")

        assert mock_run_agent_for_repo.call_count == 3
        mock_run_agent_for_repo.assert_any_call(
            "test-owner/repo-1",
            mock_settings,
            3,
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=False,
            review_wait=5,
            hint=None,
        )
        mock_run_agent_for_repo.assert_any_call(
            "test-owner/repo-2",
            mock_settings,
            3,
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=False,
            review_wait=5,
            hint=None,
        )
        mock_run_agent_for_repo.assert_any_call(
            "test-owner/repo-3",
            mock_settings,
            3,
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=False,
            review_wait=5,
            hint=None,
        )


# ============================================================
# get_repositories error handling
# ============================================================


@pytest.mark.asyncio
async def test_get_repositories_http_error() -> None:
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_user = MagicMock()
        mock_user.status_code = 200
        mock_user.json.return_value = {"login": "not-test-owner"}

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            message="Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )
        mock_get.side_effect = [mock_user, mock_response]

        with pytest.raises(httpx_mod.HTTPStatusError):
            await get_repositories(owner="test-owner", token="bad-token")


@pytest.mark.asyncio
async def test_get_repositories_empty_token() -> None:
    mock_repos = [{"name": "repo-1", "fork": False}]

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_response1 = MagicMock()
        mock_response1.json.return_value = mock_repos
        mock_response2 = MagicMock()
        mock_response2.json.return_value = []
        mock_get.side_effect = [mock_response1, mock_response2]

        repos = await get_repositories(owner="test-owner", token="")
        assert len(repos) == 1
        _, kwargs = mock_get.call_args_list[0]
        assert "Authorization" not in kwargs["headers"]


@pytest.mark.asyncio
async def test_get_repositories_no_repos() -> None:
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_user = MagicMock()
        mock_user.status_code = 200
        mock_user.json.return_value = {"login": "not-test-owner"}

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.side_effect = [mock_user, mock_response]

        repos = await get_repositories(owner="test-owner", token="dummy")
        assert repos == []


@pytest.mark.asyncio
async def test_get_repositories_falls_back_to_orgs_on_404() -> None:
    mock_user = MagicMock()
    mock_user.status_code = 200
    mock_user.json.return_value = {"login": "not-my-org"}

    mock_404 = MagicMock()
    mock_404.status_code = 404
    mock_404.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
        message="Not Found",
        request=MagicMock(),
        response=mock_404,
    )

    mock_org_page1 = MagicMock()
    mock_org_page1.json.return_value = [
        {"name": "org-repo-1", "fork": False},
        {"name": "org-repo-2", "fork": True},
    ]

    mock_org_page2 = MagicMock()
    mock_org_page2.json.return_value = []

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = [mock_user, mock_404, mock_org_page1, mock_org_page2]

        repos = await get_repositories(owner="my-org", token="dummy")
        assert repos == ["my-org/org-repo-1"]

        calls = mock_get.call_args_list
        assert "/users/my-org/repos?type=owner" in calls[1][0][0]
        assert "/orgs/my-org/repos?type=sources" in calls[2][0][0]


@pytest.mark.asyncio
async def test_get_repositories_invalid_owner() -> None:
    with pytest.raises(ValueError, match="Invalid owner name"):
        await get_repositories(owner="../invalid-owner", token="dummy")


# ============================================================
# run_agent error handling
# ============================================================


@pytest.mark.asyncio
@patch("dependency_director.main.run_agent_for_repo", new_callable=AsyncMock)
@patch("dependency_director.main.get_repositories", new_callable=AsyncMock)
async def test_run_agent_worker_error_isolates(
    mock_get_repos: MagicMock,
    mock_run_agent_for_repo: MagicMock,
) -> None:
    mock_get_repos.return_value = ["test-owner/repo-1", "test-owner/repo-2"]
    mock_run_agent_for_repo.side_effect = [Exception("repo-1 failed"), None]

    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = "dummy"

        await run_agent(
            "test-owner",
            concurrency=2,
            max_attempts=3,
            repo=None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
        )

        assert mock_run_agent_for_repo.call_count == 2


@pytest.mark.asyncio
@patch("dependency_director.main.get_repositories", new_callable=AsyncMock)
async def test_run_agent_no_repos_found(mock_get_repos: MagicMock) -> None:
    mock_get_repos.return_value = []

    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = "dummy"

        await run_agent(
            "test-owner",
            concurrency=1,
            max_attempts=3,
            repo=None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
        )


# ============================================================
# Auth validation
# ============================================================


@pytest.mark.asyncio
@patch("dependency_director.main.run_agent_for_repo", new_callable=AsyncMock)
async def test_run_agent_vertex_no_api_key_succeeds(
    mock_run_agent_for_repo: MagicMock,
) -> None:
    """Vertex AI mode with project+location should not require GEMINI_API_KEY."""
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = ""
        mock_settings.github_token = "dummy"
        mock_settings.vertex = True
        mock_settings.google_cloud_project = "my-project"
        mock_settings.google_cloud_location = "us-central1"

        await run_agent(
            "test-owner",
            concurrency=1,
            max_attempts=3,
            repo="test-owner/repo",
            dry_run=True,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
        )

        mock_run_agent_for_repo.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_no_api_key_no_vertex_exits() -> None:
    """Without Vertex and without API key, should exit."""
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = ""
        mock_settings.github_token = "dummy"
        mock_settings.vertex = False

        with pytest.raises(SystemExit):
            await run_agent(
                "test-owner",
                concurrency=1,
                max_attempts=3,
                repo=None,
                dry_run=False,
                auto_merge=False,
                verify_all=False,
                standalone_fix=False,
                review_wait=0,
            )


@pytest.mark.asyncio
async def test_run_agent_vertex_missing_project_exits() -> None:
    """Vertex AI without project+location should exit."""
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = ""
        mock_settings.github_token = "dummy"
        mock_settings.vertex = True
        mock_settings.google_cloud_project = ""
        mock_settings.google_cloud_location = ""

        with pytest.raises(SystemExit):
            await run_agent(
                "test-owner",
                concurrency=1,
                max_attempts=3,
                repo=None,
                dry_run=False,
                auto_merge=False,
                verify_all=False,
                standalone_fix=False,
                review_wait=0,
            )


@pytest.mark.asyncio
@patch("dependency_director.main.run_agent_for_repo", new_callable=AsyncMock)
@patch("dependency_director.main.get_repositories", new_callable=AsyncMock)
async def test_run_agent_no_github_token_warns(
    mock_get_repos: MagicMock,
    mock_run_agent_for_repo: MagicMock,
) -> None:
    """Missing GITHUB_TOKEN should warn but not exit."""
    mock_get_repos.return_value = ["test-owner/repo-1"]

    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = ""
        mock_settings.vertex = False
        mock_settings.bots = []

        await run_agent(
            "test-owner",
            concurrency=1,
            max_attempts=3,
            repo=None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
        )

        mock_run_agent_for_repo.assert_called_once()


@pytest.mark.asyncio
@patch("dependency_director.main.get_repositories", new_callable=AsyncMock)
async def test_run_agent_get_repos_exception(mock_get_repos: MagicMock) -> None:
    """When get_repositories raises an exception, run_agent should print error and exit."""
    mock_get_repos.side_effect = Exception("GitHub API down")

    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = "dummy-token"
        mock_settings.vertex = False

        with pytest.raises(SystemExit) as exc_info:
            await run_agent(
                "test-owner",
                concurrency=1,
                max_attempts=3,
                repo=None,
                dry_run=False,
                auto_merge=False,
                verify_all=False,
                standalone_fix=False,
                review_wait=0,
            )

        assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_run_agent_verify_all_no_sandbox_incompatible() -> None:
    """run_agent should raise SystemExit(1) if both verify_all and no_sandbox are True."""
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.gemini_api_key = "dummy"
        mock_settings.github_token = "dummy-token"
        mock_settings.vertex = False
        mock_settings.no_sandbox = False

        with pytest.raises(SystemExit) as exc_info:
            await run_agent(
                "test-owner",
                concurrency=1,
                max_attempts=3,
                repo=None,
                dry_run=False,
                auto_merge=False,
                verify_all=True,
                standalone_fix=False,
                review_wait=0,
                no_sandbox=True,
            )

        assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_get_repositories_authenticated_owner() -> None:
    owner = "test-owner"
    token = "dummy-token"

    mock_user_response = MagicMock()
    mock_user_response.status_code = 200
    mock_user_response.json.return_value = {
        "login": "TeSt-OwNeR",
    }  # Mix case to test case-insensitivity

    mock_repos_response = MagicMock()
    mock_repos_response.status_code = 200
    mock_repos_response.json.return_value = [
        {"name": "repo-priv", "fork": False},
        {"name": "repo-fork", "fork": True},
    ]

    mock_repos_empty = MagicMock()
    mock_repos_empty.status_code = 200
    mock_repos_empty.json.return_value = []

    calls = []

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        if "api.github.com/user" in str(url) and "repos" not in str(url):
            return mock_user_response
        if "api.github.com/user/repos" in str(url):
            if len(calls) == 2:
                return mock_repos_response
            return mock_repos_empty
        msg = f"Unexpected URL: {url}"
        raise ValueError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        repos = await get_repositories(owner=owner, token=token)
        assert repos == ["test-owner/repo-priv"]
        assert len(calls) == 3
        assert "user/repos?affiliation=owner" in str(calls[1])


@pytest.mark.asyncio
async def test_get_repositories_other_owner() -> None:
    owner = "other-owner"
    token = "dummy-token"

    mock_user_response = MagicMock()
    mock_user_response.status_code = 200
    mock_user_response.json.return_value = {"login": "test-owner"}

    mock_repos_response = MagicMock()
    mock_repos_response.status_code = 200
    mock_repos_response.json.return_value = [
        {"name": "repo-pub", "fork": False},
    ]

    mock_repos_empty = MagicMock()
    mock_repos_empty.status_code = 200
    mock_repos_empty.json.return_value = []

    calls = []

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        if "api.github.com/user" in str(url) and "repos" not in str(url):
            return mock_user_response
        if f"api.github.com/users/{owner}/repos" in str(url):
            if len(calls) == 2:
                return mock_repos_response
            return mock_repos_empty
        msg = f"Unexpected URL: {url}"
        raise ValueError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        repos = await get_repositories(owner=owner, token=token)
        assert repos == [f"{owner}/repo-pub"]
        assert len(calls) == 3
        assert f"users/{owner}/repos?type=owner" in str(calls[1])


@pytest.mark.asyncio
async def test_get_repositories_user_endpoint_fails() -> None:
    owner = "test-owner"
    token = "dummy-token"

    mock_user_response = MagicMock()
    mock_user_response.status_code = 401

    mock_repos_response = MagicMock()
    mock_repos_response.status_code = 200
    mock_repos_response.json.return_value = [
        {"name": "repo-pub", "fork": False},
    ]

    mock_repos_empty = MagicMock()
    mock_repos_empty.status_code = 200
    mock_repos_empty.json.return_value = []

    calls = []

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        if "api.github.com/user" in str(url) and "repos" not in str(url):
            # This triggers exception/fallback
            raise httpx_mod.HTTPStatusError(
                message="Unauthorized",
                request=MagicMock(),
                response=mock_user_response,
            )
        if f"api.github.com/users/{owner}/repos" in str(url):
            if len(calls) == 2:
                return mock_repos_response
            return mock_repos_empty
        msg = f"Unexpected URL: {url}"
        raise ValueError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        repos = await get_repositories(owner=owner, token=token)
        assert repos == [f"{owner}/repo-pub"]
        assert len(calls) == 3
        assert f"users/{owner}/repos?type=owner" in str(calls[1])


@pytest.mark.asyncio
async def test_get_repositories_scope_error_propagates() -> None:
    owner = "test-owner"
    token = "scope-error-token"

    from dependency_director.tools import GitHubAuthenticationError

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Never:
        msg = "Unauthorized/Bad Scope"
        raise GitHubAuthenticationError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        with pytest.raises(GitHubAuthenticationError, match="Unauthorized/Bad Scope"):
            await get_repositories(owner=owner, token=token)


@pytest.mark.asyncio
async def test_get_repositories_user_repos_404_fallback() -> None:
    owner = "test-owner"
    token = "fallback-404-token"

    mock_user_response = MagicMock()
    mock_user_response.status_code = 200
    mock_user_response.json.return_value = {"login": "test-owner"}

    mock_user_repos_404 = MagicMock()
    mock_user_repos_404.status_code = 404
    mock_user_repos_404.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
        message="Not Found",
        request=MagicMock(),
        response=mock_user_repos_404,
    )

    mock_public_repos = MagicMock()
    mock_public_repos.status_code = 200
    mock_public_repos.json.return_value = [{"name": "public-repo", "fork": False}]

    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.json.return_value = []

    calls = []

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(str(url))
        if "api.github.com/user" in str(url) and "repos" not in str(url):
            return mock_user_response
        if "api.github.com/user/repos" in str(url):
            raise httpx_mod.HTTPStatusError(
                message="Not Found",
                request=MagicMock(),
                response=mock_user_repos_404,
            )
        if f"api.github.com/users/{owner}/repos" in str(url):
            if len(calls) == 3:
                return mock_public_repos
            return mock_empty
        msg = f"Unexpected URL: {url}"
        raise ValueError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        repos = await get_repositories(owner=owner, token=token)
        assert repos == ["test-owner/public-repo"]
        assert len(calls) == 4
        assert "user/repos" in calls[1]
        assert "users/test-owner/repos" in calls[2]


@pytest.mark.asyncio
async def test_get_repositories_token_caching() -> None:
    owner = "test-owner"
    token = "cached-token"

    mock_user_response = MagicMock()
    mock_user_response.status_code = 200
    mock_user_response.json.return_value = {"login": "test-owner"}

    mock_repos_response = MagicMock()
    mock_repos_response.status_code = 200
    mock_repos_response.json.return_value = [{"name": "repo-1", "fork": False}]

    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.json.return_value = []

    calls = []

    def side_effect(url: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(str(url))
        if "api.github.com/user" in str(url) and "repos" not in str(url):
            return mock_user_response
        if "api.github.com/user/repos" in str(url):
            if len(calls) == 2 or len(calls) == 4:
                return mock_repos_response
            return mock_empty
        msg = f"Unexpected URL: {url}"
        raise ValueError(msg)

    with patch("httpx.AsyncClient.get", side_effect=side_effect):
        from dependency_director.tools import GitHubClient

        client = GitHubClient(token=token)
        try:
            repos_1 = await client.get_repositories(owner=owner)
            assert repos_1 == ["test-owner/repo-1"]
            assert len(calls) == 3
            assert "user" in calls[0]
            assert "user/repos" in calls[1]

            repos_2 = await client.get_repositories(owner=owner)
            assert repos_2 == ["test-owner/repo-1"]
            assert len(calls) == 5
            assert "user/repos" in calls[3]
        finally:
            await client.close()
