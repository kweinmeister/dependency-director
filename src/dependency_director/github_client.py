"""GitHub API client and HTTP error handling for dependency bot PR management."""

import asyncio
import functools
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from http import HTTPStatus
from typing import Any, cast

import httpx
from pydantic import ValidationError

from dependency_director.config import (
    RATE_LIMIT_BASE_DELAY_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    RATE_LIMIT_MAX_DELAY_SECONDS,
)
from dependency_director.schemas import (
    ChangedFile,
    CommitDetails,
    CommitResponse,
    CommitSummary,
    GitHubUser,
    PatchedFile,
    PullRequest,
    PullRequestBody,
    RepositoryPayload,
)

OWNER_RE = re.compile(r"^[a-zA-Z0-9-]{1,39}$")
REPO_RE = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")


class GitHubClientError(Exception):
    """Base exception for GitHubClient errors."""


class GitHubAuthenticationError(GitHubClientError):
    """Exception raised for 401/403 errors."""


class GitHubNotFoundError(GitHubClientError):
    """Exception raised for 404 errors."""


class GitHubRateLimitError(GitHubClientError):
    """Exception raised when GitHub refuses a request for rate-limit reasons.

    Attributes:
        retry_after: Seconds GitHub asked the caller to wait, or None when it
            gave no advice and the caller should back off on its own schedule.

    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Record the message and the pause GitHub requested, if any."""
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the pause GitHub asked for, or None if it named none."""
    raw = response.headers.get("retry-after", "")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        # GitHub may spell Retry-After as an HTTP date. Treat that the same as
        # an absent header and let exponential backoff pick the delay.
        return None


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Return the backoff delay in seconds for a rate-limited retry attempt."""
    if retry_after is not None:
        return retry_after
    return float(RATE_LIMIT_BASE_DELAY_SECONDS * 2 ** (attempt - 1))


def _is_rate_limited(response: httpx.Response) -> bool:
    """Report whether a refusal is a rate limit rather than a permission problem.

    A 429 always is. A 403 only counts when GitHub says so — a secondary limit
    sends 'Retry-After' and an exhausted primary limit sends
    'x-ratelimit-remaining: 0'. A missing-scope 403 sends neither, and retrying
    it would stall the run without ever succeeding.
    """
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return True
    if response.status_code != HTTPStatus.FORBIDDEN:
        return False
    return "retry-after" in response.headers or response.headers.get("x-ratelimit-remaining") == "0"


def _github_error_detail(response: httpx.Response) -> str:
    """Summarize GitHub's own explanation of a refusal, or '' when it gave none.

    GitHub puts the actionable part of a 4xx in the body: 'message' names the
    class of problem and 'errors' names the specific one. A 422 on the pulls
    endpoint means 'a pull request already exists', 'no commits between', or
    an invalid head, and the status code alone distinguishes none of them.
    """
    try:
        payload = response.json()
    except (ValueError, httpx.ResponseNotRead):
        return ""
    if not isinstance(payload, dict):
        return ""

    parts: list[str] = []
    message = payload.get("message")
    if isinstance(message, str) and message:
        parts.append(message)

    errors = payload.get("errors")
    if isinstance(errors, list):
        for error in errors:
            detail = error.get("message") or error.get("code") if isinstance(error, dict) else error
            if isinstance(detail, str) and detail:
                parts.append(detail)
    return " — ".join(parts)


def _raise_for_status(response: httpx.Response) -> None:
    """Raise on a 4xx/5xx, folding GitHub's explanation into the message.

    Re-raises the same exception type so callers that branch on
    ``exc.response.status_code`` keep working; only the message grows.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _github_error_detail(response)
        if not detail:
            raise
        msg = f"{exc}\nGitHub said: {detail}"
        raise httpx.HTTPStatusError(msg, request=exc.request, response=response) from exc


def validate_repo_params(owner: str, repo: str) -> None:
    if not OWNER_RE.match(owner):
        msg = f"Invalid owner name: {owner!r}"
        raise ValueError(msg)
    if not REPO_RE.match(repo):
        msg = f"Invalid repository name: {repo!r}"
        raise ValueError(msg)


class GitHubClient:
    """Shared HTTP client for GitHub API calls. Created once per agent run."""

    def __init__(self, token: str | None = None) -> None:
        """Initialize GitHub API client with optional auth token."""
        self._user_cache: dict[str, str] = {}
        self.headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "dependency-director-agent",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        async def check_api_errors(response: httpx.Response) -> None:
            if _is_rate_limited(response):
                msg = f"GitHub API error {response.status_code} on {response.url} - Rate limited."
                raise GitHubRateLimitError(msg, _retry_after_seconds(response))

            if response.status_code in (
                HTTPStatus.UNAUTHORIZED,
                HTTPStatus.FORBIDDEN,
                HTTPStatus.NOT_FOUND,
            ):
                # Don't fail-fast on the organization-fallback check in get_repositories
                if (
                    response.status_code == HTTPStatus.NOT_FOUND
                    and "/users/" in str(response.url)
                    and "/repos" in str(response.url)
                ):
                    return

                msg = f"GitHub API error {response.status_code} on {response.url}"
                if response.status_code == HTTPStatus.UNAUTHORIZED:
                    msg += " - Unauthorized. Please verify your DEPDIRECTOR_GITHUB_TOKEN."
                    raise GitHubAuthenticationError(msg)
                if response.status_code == HTTPStatus.FORBIDDEN:
                    msg += " - Forbidden/Rate Limited. Please verify your token scopes and rate limits."
                    raise GitHubAuthenticationError(msg)
                if response.status_code == HTTPStatus.NOT_FOUND:
                    msg += " - Not Found. Please verify the owner/repository names exist and your token has access."
                    raise GitHubNotFoundError(msg)

        self.client = httpx.AsyncClient(event_hooks={"response": [check_api_errors]})

    async def _send_with_backoff(
        self,
        send: Callable[[], Awaitable[httpx.Response]],
    ) -> httpx.Response:
        """Issue a request, retrying while GitHub reports a rate limit.

        Honours the pause GitHub asks for and otherwise doubles its own, giving
        up once the attempts run out or the requested wait exceeds the ceiling.
        Every other failure propagates untouched on the first attempt.
        """
        attempt = 0
        while True:
            try:
                return await send()
            except GitHubRateLimitError as exc:
                attempt += 1
                wait = _retry_delay(attempt, exc.retry_after)
                if attempt >= RATE_LIMIT_MAX_ATTEMPTS or wait > RATE_LIMIT_MAX_DELAY_SECONDS:
                    raise
                await asyncio.sleep(wait)

    async def _request(
        self,
        method: str,
        path: str,
        owner: str,
        repo: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: Any = None,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        validate_repo_params(owner, repo)
        # rstrip so an empty path addresses the repository itself rather than
        # a trailing-slash URL.
        url = f"https://api.github.com/repos/{owner}/{repo}/{path.lstrip('/')}".rstrip("/")
        req_headers = {**self.headers, **(headers or {})}

        method_upper = method.upper()
        kwargs: dict[str, Any] = {"headers": req_headers}
        if follow_redirects:
            kwargs["follow_redirects"] = True

        send: Callable[[], Awaitable[httpx.Response]]
        if method_upper == "GET":
            if params is not None:
                kwargs["params"] = params
            send = functools.partial(self.client.get, url, **kwargs)
        elif method_upper == "PUT":
            if json_data is not None:
                kwargs["json"] = json_data
            send = functools.partial(self.client.put, url, **kwargs)
        elif method_upper == "POST":
            if json_data is not None:
                kwargs["json"] = json_data
            send = functools.partial(self.client.post, url, **kwargs)
        else:
            if params is not None:
                kwargs["params"] = params
            if json_data is not None:
                kwargs["json"] = json_data
            send = functools.partial(self.client.request, method, url, **kwargs)

        response = await self._send_with_backoff(send)
        _raise_for_status(response)
        return response

    async def _get(
        self,
        path: str,
        owner: str,
        repo: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        return await self._request(
            "GET",
            path,
            owner,
            repo,
            params=params,
            headers=headers,
            follow_redirects=follow_redirects,
        )

    async def _put(
        self,
        path: str,
        owner: str,
        repo: str,
        json_data: Any = None,
    ) -> httpx.Response:
        return await self._request(
            "PUT",
            path,
            owner,
            repo,
            json_data=json_data,
        )

    async def _post(
        self,
        path: str,
        owner: str,
        repo: str,
        json_data: Any = None,
    ) -> httpx.Response:
        return await self._request(
            "POST",
            path,
            owner,
            repo,
            json_data=json_data,
        )

    async def _list_paginated(
        self,
        path: str,
        owner: str,
        repo: str,
        params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        page = 1
        req_params = dict(params or {})
        req_params["per_page"] = 100
        while True:
            req_params["page"] = page
            response = await self._get(path, owner, repo, params=req_params)
            data = response.json()
            if not data:
                break
            for item in data:
                yield item
            page += 1

    async def get_pr_author(self, owner: str, repo: str, pr_number: int) -> str:
        """Get the GitHub username/login of a pull request author."""
        data = await self.get_pr_details(owner, repo, pr_number)
        return PullRequest.model_validate(data).author

    async def get_pr_body(self, owner: str, repo: str, pr_number: int) -> str:
        """Get the description/changelog body of a pull request."""
        data = await self.get_pr_details(owner, repo, pr_number)
        return PullRequestBody.model_validate(data).body

    async def merge_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Merge a pull request using the squash-and-merge method."""
        response = await self._put(
            f"pulls/{pr_number}/merge",
            owner,
            repo,
            json_data={"merge_method": "squash"},
        )
        return cast("dict[str, Any]", response.json())

    async def comment_on_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a comment on the specified pull request/issue."""
        response = await self._post(
            f"issues/{pr_number}/comments",
            owner,
            repo,
            json_data={"body": body},
        )
        return cast("dict[str, Any]", response.json())

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
    ) -> dict[str, Any]:
        """Open a pull request from ``head`` into ``base``."""
        response = await self._post(
            "pulls",
            owner,
            repo,
            json_data={"title": title, "head": head, "base": base, "body": body},
        )
        return cast("dict[str, Any]", response.json())

    async def find_open_pr_for_head(
        self,
        owner: str,
        repo: str,
        head_branch: str,
    ) -> PullRequest | None:
        """Return the open pull request whose head is ``head_branch``, if any.

        GitHub permits only one open pull request per head branch, so this
        answers the question a duplicate 'create' would otherwise answer with
        an unexplained 422. A branch name may arrive already qualified as
        'owner:branch'; a git ref cannot contain a colon, so the last segment
        is the branch either way.
        """
        branch = head_branch.split(":")[-1]
        response = await self._get(
            "pulls",
            owner,
            repo,
            params={"head": f"{owner}:{branch}", "state": "open"},
        )
        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        return PullRequest.model_validate(data[0])

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Fetch the repository's default branch, falling back to ``main``."""
        response = await self._get("", owner, repo)
        return RepositoryPayload.model_validate(response.json()).default_branch or "main"

    async def list_pr_commits(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Fetch the commits on a pull request's head branch."""
        return [c async for c in self._list_paginated(f"pulls/{pr_number}/commits", owner, repo)]

    async def get_pr_reviews(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Fetch all reviews and comments on a pull request."""
        response = await self._get(f"pulls/{pr_number}/reviews", owner, repo)
        return cast("list[dict[str, Any]]", response.json())

    async def get_pr_details(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> dict[str, Any]:
        """Fetch general details of a specific pull request."""
        response = await self._get(f"pulls/{pr_number}", owner, repo)
        return cast("dict[str, Any]", response.json())

    async def get_commit_check_runs(
        self,
        owner: str,
        repo: str,
        ref: str,
    ) -> dict[str, Any]:
        """Fetch check runs for a commit reference."""
        response = await self._get(f"commits/{ref}/check-runs", owner, repo)
        return cast("dict[str, Any]", response.json())

    async def get_commit_status(
        self,
        owner: str,
        repo: str,
        ref: str,
    ) -> dict[str, Any]:
        """Fetch combined legacy commit status for a commit reference."""
        response = await self._get(f"commits/{ref}/status", owner, repo)
        return cast("dict[str, Any]", response.json())

    async def get_workflow_runs_for_commit(
        self,
        owner: str,
        repo: str,
        ref: str,
    ) -> dict[str, Any]:
        """Fetch workflow runs associated with a commit reference."""
        response = await self._get(
            "actions/runs",
            owner,
            repo,
            params={"head_sha": ref},
        )
        return cast("dict[str, Any]", response.json())

    async def get_workflow_run_jobs(
        self,
        owner: str,
        repo: str,
        run_id: int,
    ) -> dict[str, Any]:
        """Fetch jobs for a workflow run."""
        response = await self._get(f"actions/runs/{run_id}/jobs", owner, repo)
        return cast("dict[str, Any]", response.json())

    async def get_job_logs(self, owner: str, repo: str, job_id: int) -> str:
        """Fetch raw log text for a workflow run job."""
        response = await self._get(
            f"actions/jobs/{job_id}/logs",
            owner,
            repo,
            follow_redirects=True,
        )
        return response.text

    async def list_open_prs(
        self,
        owner: str,
        repo: str,
    ) -> list[PullRequest]:
        """Fetch all open pull requests, parsed into the slice we read.

        Validating here rather than passing raw dicts on keeps the fields this
        project depends on — the base ref above all — declared in one place
        instead of re-derived at every call site. The huge body/changelog
        payloads dependency bots produce are dropped by the model.
        """
        params = {
            "state": "open",
            "sort": "created",
            "direction": "asc",
        }
        return [
            PullRequest.model_validate(pr) async for pr in self._list_paginated("pulls", owner, repo, params=params)
        ]

    async def get_pr_diff(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> str:
        """Fetch the diff of a pull request as plain text."""
        response = await self._get(
            f"pulls/{pr_number}",
            owner,
            repo,
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        return response.text

    async def get_pr_files(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[ChangedFile]:
        """Fetch the list of files changed in a pull request."""
        return [
            ChangedFile(
                filename=f["filename"],
                status=f.get("status", ""),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
            )
            async for f in self._list_paginated(f"pulls/{pr_number}/files", owner, repo)
        ]

    async def get_file_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """Fetch file contents from a repository.

        Returns a dict with 'name', 'path', 'size', 'content' (base64), and 'encoding'.
        """
        params = {"ref": ref} if ref else None
        response = await self._get(f"contents/{path}", owner, repo, params=params)
        return cast("dict[str, Any]", response.json())

    async def list_commits(
        self,
        owner: str,
        repo: str,
        sha: str | None = None,
        per_page: int = 30,
    ) -> list[CommitSummary]:
        """Fetch recent commits for a repository or branch."""
        params: dict[str, Any] = {"per_page": per_page}
        if sha:
            params["sha"] = sha
        response = await self._get("commits", owner, repo, params=params)
        parsed = [CommitResponse.model_validate(item) for item in response.json()]
        return [
            CommitSummary(
                sha=c.sha[:7],
                message=c.commit.message.split("\n")[0],
                author=c.commit.author_name,
                date=c.commit.author_date,
            )
            for c in parsed
        ]

    async def get_commit(
        self,
        owner: str,
        repo: str,
        sha: str,
    ) -> CommitDetails:
        """Fetch details of a specific commit."""
        response = await self._get(f"commits/{sha}", owner, repo)
        parsed = CommitResponse.model_validate(response.json())
        return CommitDetails(
            sha=parsed.sha,
            message=parsed.commit.message,
            author=parsed.commit.author_name,
            date=parsed.commit.author_date,
            files=[
                PatchedFile(
                    filename=f.filename,
                    status=f.status,
                    patch=f.patch,
                )
                for f in parsed.files
            ],
        )

    async def list_branches(
        self,
        owner: str,
        repo: str,
    ) -> list[str]:
        """Fetch branch names for a repository."""
        return [b["name"] async for b in self._list_paginated("branches", owner, repo)]

    async def _check_is_authenticated_user(self, owner: str) -> bool:
        if "Authorization" not in self.headers:
            return False

        token = self.headers["Authorization"]
        if token in self._user_cache:
            cached_login = self._user_cache[token]
            return bool(cached_login and cached_login.lower() == owner.lower())

        try:
            response = await self._send_with_backoff(
                functools.partial(
                    self.client.get,
                    "https://api.github.com/user",
                    headers=self.headers,
                ),
            )
            response.raise_for_status()
            user_data = response.json()
            if isinstance(user_data, dict):
                user = GitHubUser.model_validate(user_data)
                login = user.login or ""
                self._user_cache[token] = login
                return login.lower() == owner.lower()
        except (GitHubAuthenticationError, GitHubRateLimitError):
            # A rate limit says nothing about who the token belongs to.
            # Swallowing it here would silently fall back to the public
            # endpoint and hide half the owner's repositories.
            raise
        except (httpx.HTTPError, GitHubClientError, ValidationError):
            pass
        return False

    async def _fetch_repos_page(
        self,
        owner: str,
        endpoint_prefix: str,
        page: int,
        *,
        is_authenticated_user: bool,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        if is_authenticated_user:
            url = f"{endpoint_prefix}?affiliation=owner&per_page=100&page={page}"
        else:
            url = f"{endpoint_prefix}?type=owner&per_page=100&page={page}"

        try:
            response = await self._send_with_backoff(functools.partial(self.client.get, url, headers=self.headers))
            response.raise_for_status()
            return is_authenticated_user, endpoint_prefix, response.json()
        except (httpx.HTTPStatusError, GitHubNotFoundError) as e:
            is_404 = False
            if isinstance(e, GitHubNotFoundError) or (
                isinstance(e, httpx.HTTPStatusError) and e.response.status_code == HTTPStatus.NOT_FOUND
            ):
                is_404 = True

            # If /user/repos returns 404, fall back to public /users/{owner}/repos
            if is_authenticated_user and page == 1 and is_404:
                endpoint_prefix = f"https://api.github.com/users/{owner}/repos"
                return await self._fetch_repos_page(owner, endpoint_prefix, page, is_authenticated_user=False)

            status_code = getattr(e, "status_code", None)
            if status_code is None and isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            if (
                not is_authenticated_user
                and status_code == HTTPStatus.NOT_FOUND
                and page == 1
                and "users" in endpoint_prefix
            ):
                endpoint_prefix = f"https://api.github.com/orgs/{owner}/repos"
                url = f"{endpoint_prefix}?type=sources&per_page=100&page={page}"
                response = await self._send_with_backoff(functools.partial(self.client.get, url, headers=self.headers))
                response.raise_for_status()
                return is_authenticated_user, endpoint_prefix, response.json()
            raise

    async def get_repositories(self, owner: str) -> list[str]:
        """Fetch active, non-forked repositories from GitHub API.

        Forks are skipped because their dependency PRs belong upstream.
        Archived and disabled repositories are skipped because they reject
        pushes and merges, so scanning them can only waste a run.

        Uses pagination to get all repositories for the given owner.
        """
        validate_repo_params(owner, "placeholder-repo")

        is_authenticated_user = await self._check_is_authenticated_user(owner)

        if is_authenticated_user:
            endpoint_prefix = "https://api.github.com/user/repos"
        else:
            endpoint_prefix = f"https://api.github.com/users/{owner}/repos"

        repos: list[str] = []
        page = 1
        while True:
            is_authenticated_user, endpoint_prefix, repos_data = await self._fetch_repos_page(
                owner,
                endpoint_prefix,
                page,
                is_authenticated_user=is_authenticated_user,
            )
            if not repos_data:
                break

            parsed_repos = [RepositoryPayload.model_validate(raw) for raw in repos_data]
            repos.extend(
                f"{owner}/{repo.name}"
                for repo in parsed_repos
                if repo.name and not (repo.fork or repo.archived or repo.disabled)
            )
            page += 1

        return repos

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()
