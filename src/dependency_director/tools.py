"""Tools and GitHub API client helper functions for dependency bot PR management."""

import asyncio
import contextlib
import copy
import functools
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any, NamedTuple, cast

import httpx

from dependency_director.argv import (
    resolve_exe,
    split_compound_argv,
)
from dependency_director.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_LOG_LIMITS,
    DEFAULT_OUTPUT_LIMITS,
    DEFAULT_SRT_SETTINGS_PATH,
    OUTPUT_HEAD_FRACTION,
    PASSING_RUN_CONCLUSIONS,
    RATE_LIMIT_BASE_DELAY_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    RATE_LIMIT_MAX_DELAY_SECONDS,
    SAFE_ENV_ALLOWLIST,
    BotConfig,
    LogLimits,
    OutputLimits,
)
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
    WorkflowJob,
    WorkflowJobsResponse,
    WorkflowRun,
    WorkflowRunsResponse,
    json_payload,
    summarize_checks,
)


class CommandResult(NamedTuple):
    """Result from executing a single sandboxed command.

    Attributes:
        output: The formatted command output string.
        returncode: The process exit code (-1 for errors before execution).

    """

    output: str
    returncode: int


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


def _validate_repo_params(owner: str, repo: str) -> None:
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
                backoff = RATE_LIMIT_BASE_DELAY_SECONDS * 2 ** (attempt - 1)
                wait = exc.retry_after if exc.retry_after is not None else backoff
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
        _validate_repo_params(owner, repo)
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
        response.raise_for_status()
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

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Fetch the repository's default branch, falling back to ``main``."""
        response = await self._get("", owner, repo)
        data = cast("dict[str, Any]", response.json())
        branch = data.get("default_branch")
        return branch if isinstance(branch, str) and branch else "main"

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
        return str(response.text)

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
        return str(response.text)

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
        data = response.json()
        return [
            CommitSummary(
                sha=c["sha"][:7],
                message=c.get("commit", {}).get("message", "").split("\n")[0],
                author=c.get("commit", {}).get("author", {}).get("name", ""),
                date=c.get("commit", {}).get("author", {}).get("date", ""),
            )
            for c in data
        ]

    async def get_commit(
        self,
        owner: str,
        repo: str,
        sha: str,
    ) -> CommitDetails:
        """Fetch details of a specific commit."""
        response = await self._get(f"commits/{sha}", owner, repo)
        data = response.json()
        return CommitDetails(
            sha=data["sha"],
            message=data.get("commit", {}).get("message", ""),
            author=data.get("commit", {}).get("author", {}).get("name", ""),
            date=data.get("commit", {}).get("author", {}).get("date", ""),
            files=[
                PatchedFile(
                    filename=f["filename"],
                    status=f.get("status", ""),
                    patch=f.get("patch", ""),
                )
                for f in data.get("files", [])
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
                login = str(user_data.get("login") or "")
                self._user_cache[token] = login
                return login.lower() == owner.lower()
        except (GitHubAuthenticationError, GitHubRateLimitError):
            # A rate limit says nothing about who the token belongs to.
            # Swallowing it here would silently fall back to the public
            # endpoint and hide half the owner's repositories.
            raise
        except (httpx.HTTPError, GitHubClientError):
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
        _validate_repo_params(owner, "placeholder-repo")

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

            repos.extend(
                f"{owner}/{repo['name']}"
                for repo in repos_data
                if not any(repo.get(flag, False) for flag in ("fork", "archived", "disabled"))
            )
            page += 1

        return repos

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()


def _check_bot_author(author: str, bots: list[BotConfig]) -> BotConfig:
    """Validate that the PR author is an allowed bot.

    Returns the matching BotConfig if allowed, else raises PermissionError.
    """
    allowed = {b.author for b in bots}
    if author not in allowed:
        allowed_str = ", ".join(f"'{a}'" for a in sorted(allowed))
        msg = (
            f"Security Block: Only pull requests authored by {allowed_str} can "
            f"be processed. This PR was authored by '{author}'."
        )
        raise PermissionError(
            msg,
        )
    return next(b for b in bots if b.author == author)


ToolFn = Callable[..., Awaitable[str]]


class LegacyTools(NamedTuple):
    """Container for legacy GitHub API tool functions."""

    get_pr_diff: ToolFn
    get_pr_files: ToolFn
    get_file_contents: ToolFn
    list_commits: ToolFn
    get_commit_details: ToolFn
    list_branches: ToolFn


class WriteTools(NamedTuple):
    """Container for the tool functions that change state on GitHub."""

    merge_bot_pr: ToolFn
    rebase_bot_pr: ToolFn
    wait_for_reviews: ToolFn


class AgentTools(NamedTuple):
    """Every tool function handed to the agent, named rather than positional.

    Alphabetical, so the insertion point for a new tool is obvious. Iterating
    the tuple yields them in that order, which is what the agent config wants.
    """

    create_pr: ToolFn
    get_branch_ci_status: ToolFn
    get_commit_details: ToolFn
    get_file_contents: ToolFn
    get_pr_diff: ToolFn
    get_pr_files: ToolFn
    get_pr_status: ToolFn
    get_pr_workflow_run_logs: ToolFn
    list_bot_prs: ToolFn
    list_branches: ToolFn
    list_commits: ToolFn
    merge_bot_pr: ToolFn
    rebase_bot_pr: ToolFn
    wait_for_ci: ToolFn
    wait_for_reviews: ToolFn


def _make_merge_bot_pr(client: GitHubClient, bots: list[BotConfig], *, dry_run: bool) -> ToolFn:
    async def merge_bot_pr(owner: str, repo: str, pr_number: int) -> str:
        """Merge a pull request authored by a configured bot.

        Uses squash-and-merge.
        """
        author = await client.get_pr_author(owner, repo, pr_number)
        _check_bot_author(author, bots)

        if dry_run:
            return f"[DRY-RUN] Would have merged PR #{pr_number} in {owner}/{repo}."

        try:
            res = await client.merge_pr(owner, repo, pr_number)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == HTTPStatus.METHOD_NOT_ALLOWED:
                return (
                    f"PR #{pr_number} cannot be merged right now (GitHub 405 — not mergeable). "
                    "A prior merge likely introduced a conflict. "
                    "Call get_pr_status to confirm, then rebase_bot_pr if CONFLICT."
                )
            raise
        message = res.get("message", "PR merged successfully.")
        return f"Successfully merged PR #{pr_number} in {owner}/{repo}: {message}"

    return merge_bot_pr


async def _find_foreign_commit_author(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    bots: list[BotConfig],
) -> str | None:
    """Return a description of the first commit on the PR the bot did not author.

    Returns None when every commit is attributable to a configured bot.
    """
    bot_authors = {b.author for b in bots}
    for commit in await client.list_pr_commits(owner, repo, pr_number):
        gh_author = commit.get("author")
        login = gh_author.get("login") if isinstance(gh_author, dict) else None
        if not login:
            return "an unattributed commit (no GitHub account on record)"
        if login not in bot_authors:
            return f"a commit by '{login}'"
    return None


def _make_rebase_bot_pr(client: GitHubClient, bots: list[BotConfig], *, dry_run: bool) -> ToolFn:
    async def rebase_bot_pr(owner: str, repo: str, pr_number: int) -> str:
        """Post a rebase comment on a pull request.

        Author must be a configured bot. Refuses when the branch carries
        commits the bot did not author, because a bot rebase force-pushes the
        branch from scratch and would discard them.
        """
        author = await client.get_pr_author(owner, repo, pr_number)
        bot = _check_bot_author(author, bots)

        foreign = await _find_foreign_commit_author(client, owner, repo, pr_number, bots)
        if foreign:
            return (
                f"Refused to rebase PR #{pr_number} in {owner}/{repo}: its branch contains "
                f"{foreign}. Commenting '{bot.rebase_command}' force-pushes the branch from "
                "scratch and would discard that work. Resolve the conflict locally and push "
                "to the branch instead, or close this PR and open a standalone one."
            )

        if dry_run:
            return f"[DRY-RUN] Would have commented '{bot.rebase_command}' on PR #{pr_number} in {owner}/{repo}."

        await client.comment_on_pr(owner, repo, pr_number, bot.rebase_command)
        return (
            f"Successfully requested rebase for PR #{pr_number} in {owner}/{repo} by commenting '{bot.rebase_command}'."
        )

    return rebase_bot_pr


def _make_create_pr(client: GitHubClient, *, dry_run: bool) -> ToolFn:
    async def create_pr(
        owner: str,
        repo: str,
        title: str,
        head_branch: str,
        body: str,
        base_branch: str = "",
    ) -> str:
        """Open a pull request from an already-pushed branch.

        Args:
            owner: The owner of the repository.
            repo: The repository name.
            title: The pull request title.
            head_branch: The branch holding the fix, already pushed to origin.
            body: The pull request description. Reference the original PR here.
            base_branch: Target branch. When fixing a branch on behalf of a bot
                PR, pass that PR's 'base_ref' — the fix must land on the branch
                the PR actually targets. Only defaults to the repository's
                default branch when omitted.

        """
        _validate_repo_params(owner, repo)
        base = base_branch or await client.get_default_branch(owner, repo)

        if dry_run:
            return f"[DRY-RUN] Would have opened a PR '{title}' from {head_branch} into {base} in {owner}/{repo}."

        result = await client.create_pull_request(
            owner,
            repo,
            title=title,
            head=head_branch,
            base=base,
            body=body,
        )
        return f"Opened PR #{result.get('number')} in {owner}/{repo}: {result.get('html_url', '')}"

    return create_pr


def _make_wait_for_reviews(client: GitHubClient, review_wait: int) -> ToolFn:
    async def wait_for_reviews(owner: str, repo: str, pr_number: int) -> str:
        """Wait for review comments on a pull request, polling every 30 seconds.

        Args:
            owner: The owner of the repository.
            repo: The repository name.
            pr_number: The pull request number.

        """
        if review_wait <= 0:
            return "Review wait is disabled. Skipping."

        interval = 30
        elapsed = 0
        timeout_seconds = review_wait * 60

        baseline_reviews = await client.get_pr_reviews(owner, repo, pr_number)
        baseline_keys = {
            r.get("id")
            if r.get("id") is not None
            else (r.get("user", {}).get("login"), r.get("state"), r.get("body"), idx)
            for idx, r in enumerate(baseline_reviews)
        }

        while elapsed < timeout_seconds:
            reviews = await client.get_pr_reviews(owner, repo, pr_number)
            new_reviews = []
            for idx, r in enumerate(reviews):
                key = (
                    r.get("id")
                    if r.get("id") is not None
                    else (
                        r.get("user", {}).get("login"),
                        r.get("state"),
                        r.get("body"),
                        idx,
                    )
                )
                if key not in baseline_keys:
                    new_reviews.append(r)

            if new_reviews:
                lines = []
                for r in new_reviews:
                    user = r.get("user", {}).get("login", "unknown")
                    state = r.get("state", "")
                    body = r.get("body", "").strip()
                    if body:
                        lines.append(f"- **{user}** ({state}): {body}")
                    else:
                        lines.append(f"- **{user}** ({state})")
                if lines:
                    return f"Review comments found on PR #{pr_number}:\n" + "\n".join(
                        lines,
                    )
            await asyncio.sleep(interval)
            elapsed += interval

        return f"No review comments received on PR #{pr_number} within {review_wait} minute(s)."

    return wait_for_reviews


def _make_write_tools(
    client: GitHubClient,
    bots: list[BotConfig],
    *,
    dry_run: bool,
    review_wait: int,
) -> WriteTools:
    """Create agent tool functions with config bound via closure."""
    return WriteTools(
        merge_bot_pr=_make_merge_bot_pr(client, bots, dry_run=dry_run),
        rebase_bot_pr=_make_rebase_bot_pr(client, bots, dry_run=dry_run),
        wait_for_reviews=_make_wait_for_reviews(client, review_wait),
    )


def _make_get_pr_status(client: GitHubClient) -> ToolFn:
    async def get_pr_status(owner: str, repo: str, pr_number: int) -> str:
        """Get the mergeability and CI checks status of a pull request.

        Returns a JSON string detailing PR status, mergeable state, and CI results.
        """
        _ci_status, _merge_status, result_json = await _check_ci(client, owner, repo, pr_number)
        return result_json

    return get_pr_status


async def _checks_for_ref(client: GitHubClient, owner: str, repo: str, ref: str) -> tuple[str, list[CheckSummary]]:
    """Fetch and reduce both check APIs for one commit ref."""
    check_runs_data, commit_status_data = await asyncio.gather(
        client.get_commit_check_runs(owner, repo, ref),
        client.get_commit_status(owner, repo, ref),
    )
    return summarize_checks(
        CheckRunsResponse.model_validate(check_runs_data),
        CommitStatusResponse.model_validate(commit_status_data),
    )


def _make_get_branch_ci_status(client: GitHubClient) -> ToolFn:
    # A base branch is checked once and the verdict reused for every red PR
    # sharing it. The cache lives in this closure, which is built once per
    # repository run, so its lifetime is exactly one run and nothing has to
    # invalidate it. The lock keeps two PRs asking about the same base at the
    # same moment from both paying for the round trip.
    cache: dict[tuple[str, str, str], str] = {}
    lock = asyncio.Lock()

    async def get_branch_ci_status(owner: str, repo: str, branch: str = "") -> str:
        """Report CI health for a branch, without cloning it.

        Use this to tell whether a PR's CI failure is the PR's fault or was
        already broken on the branch it targets.

        Args:
            owner: The owner of the repository.
            repo: The repository name.
            branch: Branch to check. Pass the PR's 'base_ref' — the branch it
                actually targets, reported by 'list_bot_prs' and
                'get_pr_status'. Only defaults to the repository's default
                branch when omitted, which is the wrong branch for any PR
                aimed elsewhere.

        """
        target = branch or await client.get_default_branch(owner, repo)
        key = (owner, repo, target)
        async with lock:
            if key not in cache:
                # GitHub resolves a branch name as a commit ref, so this needs no SHA lookup.
                # An exception leaves the cache untouched: a transient API failure
                # must not become this run's permanent answer for the branch.
                ci_status, checks = await _checks_for_ref(client, owner, repo, target)
                cache[key] = json_payload(BranchCiStatus(branch=target, ci_status=ci_status, checks=checks))
            return cache[key]

    return get_branch_ci_status


async def _check_ci(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
) -> tuple[str, str, str]:
    """Single CI check — returns (ci_status, merge_status, json_summary).

    ci_status reflects CI pass/fail: GREEN, RED, PENDING, or NONE.
    merge_status reflects mergeability: CLEAN, CONFLICT, or UNKNOWN.
    These are independent — a PR can be GREEN + CONFLICT (CI passes but
    has merge conflicts from a prior merge).
    """
    pr = PullRequest.model_validate(await client.get_pr_details(owner, repo, pr_number))

    checks_summary: list[CheckSummary] = []
    ci_status = "NONE"
    if pr.head.sha:
        ci_status, checks_summary = await _checks_for_ref(client, owner, repo, pr.head.sha)

    # Merge status is independent of CI status
    if pr.mergeable is False or pr.mergeable_state in ("dirty", "conflict"):
        merge_status = "CONFLICT"
    elif pr.mergeable is True and pr.mergeable_state in ("clean", "has_hooks", "unstable"):
        merge_status = "CLEAN"
    else:
        merge_status = "UNKNOWN"

    summary = PrStatus(
        pr_number=pr_number,
        title=pr.title,
        head_sha=pr.head.sha,
        base_ref=pr.base.ref,
        mergeable=pr.mergeable,
        mergeable_state=pr.mergeable_state,
        ci_status=ci_status,
        merge_status=merge_status,
        checks=checks_summary,
    )
    return ci_status, merge_status, json_payload(summary)


def _make_wait_for_ci(client: GitHubClient) -> ToolFn:
    """Create a wait_for_ci tool that polls CI status with backoff."""

    async def wait_for_ci(owner: str, repo: str, pr_number: int) -> str:
        """Poll CI status until it settles (GREEN/RED/CONFLICT) or times out.

        Uses exponential backoff: 10s, 20s, 30s, 30s, … (max 10 retries).
        Returns the final status JSON when CI resolves.
        """
        _validate_repo_params(owner, repo)
        max_retries = 10
        delays = [min(10 * (i + 1), 30) for i in range(max_retries)]

        ci_status, merge_status, result_json = await _check_ci(client, owner, repo, pr_number)
        if ci_status not in ("NONE", "PENDING") or merge_status == "CONFLICT":
            return result_json

        for delay in delays:
            await asyncio.sleep(delay)
            ci_status, merge_status, result_json = await _check_ci(client, owner, repo, pr_number)
            if ci_status not in ("NONE", "PENDING") or merge_status == "CONFLICT":
                return result_json

        return f"CI still pending for PR #{pr_number} after polling {max_retries} times. " + result_json

    return wait_for_ci


def _select_candidate_runs(runs: list[WorkflowRun]) -> list[WorkflowRun]:
    """Pick the newest non-passing run for each distinct workflow.

    A commit usually triggers several workflows, and re-running one adds
    another run with the same name on the same SHA. Keep one run per workflow
    name so every failing workflow is represented exactly once, by its most
    recent attempt. Timestamps are aware UTC by the time they arrive here, so
    a re-run recorded with a numeric offset still orders against a 'Z' one.
    """
    ordered = sorted(runs, key=lambda r: r.created_at, reverse=True)

    selected: list[WorkflowRun] = []
    seen_workflows: set[str] = set()
    for run in ordered:
        # Fall back to the run id so unnamed runs are never merged together.
        workflow = run.name or str(run.id)
        if workflow in seen_workflows:
            continue
        seen_workflows.add(workflow)
        if (run.conclusion or "").lower() in PASSING_RUN_CONCLUSIONS:
            continue
        selected.append(run)
    return selected


def _make_get_pr_workflow_run_logs(client: GitHubClient, log_limits: LogLimits) -> ToolFn:
    async def get_pr_workflow_run_logs(owner: str, repo: str, pr_number: int) -> str:
        """Fetch raw logs for failed GitHub Actions workflow runs on a pull request.

        Covers every failing workflow on the PR's head commit, not just one.
        Returns the tail of each failed job's log, capped at a few jobs; the
        cap is stated in the output when it is reached.
        """
        pr = PullRequest.model_validate(await client.get_pr_details(owner, repo, pr_number))
        head_sha = pr.head.sha
        if not head_sha:
            return f"Error: No head SHA found for PR #{pr_number}."

        runs_data = WorkflowRunsResponse.model_validate(
            await client.get_workflow_runs_for_commit(owner, repo, head_sha),
        )
        runs = runs_data.workflow_runs
        if not runs:
            return f"No workflow runs found for commit {head_sha}."

        failed_jobs_logs: list[str] = []
        total_failed = 0

        for run in _select_candidate_runs(runs):
            workflow = run.name or f"run {run.id}"
            jobs_data = WorkflowJobsResponse.model_validate(
                await client.get_workflow_run_jobs(owner, repo, run.id),
            )
            for job in jobs_data.jobs:
                if job.conclusion != "failure":
                    continue
                total_failed += 1
                if len(failed_jobs_logs) >= log_limits.max_failed_jobs:
                    continue
                failed_jobs_logs.append(
                    await _format_failed_job(client, owner, repo, workflow, job, log_limits.tail_lines),
                )

        if not failed_jobs_logs:
            return f"No failed jobs found across {len(runs)} workflow run(s) for commit {head_sha}."

        output = "\n".join(failed_jobs_logs)
        omitted = total_failed - len(failed_jobs_logs)
        if omitted:
            output += f"\n--- {omitted} more failed job(s) omitted (cap: {log_limits.max_failed_jobs}) ---\n"
        return output

    return get_pr_workflow_run_logs


async def _format_failed_job(
    client: GitHubClient,
    owner: str,
    repo: str,
    workflow: str,
    job: WorkflowJob,
    tail_lines: int,
) -> str:
    """Render one failed job as a labelled tail of its log.

    The workflow name is part of the label because job names collide across
    workflows ("build" in CI and in Release are different failures).
    """
    header = f"--- FAILED JOB: {workflow} / {job.name} (ID: {job.id}) ---"
    try:
        log_text = await client.get_job_logs(owner, repo, job.id)
    except (httpx.HTTPError, GitHubClientError) as e:
        return f"{header}\nFailed to retrieve log: {e}\n"
    tail = "\n".join(log_text.splitlines()[-tail_lines:])
    return f"{header}\n{tail}\n"


def _make_list_bot_prs(client: GitHubClient, bots: list[BotConfig]) -> ToolFn:
    async def list_bot_prs(owner: str, repo: str) -> str:
        """List open dependency-bot pull requests for a repository.

        Returns a compact JSON list with only the essential fields: number,
        title, author, created_at, and base_ref — the branch the PR targets,
        which is what 'get_branch_ci_status' should be asked about. Sorted
        oldest first. Only PRs authored by configured bots are included.
        """
        allowed_authors = {b.author for b in bots}
        all_prs = await client.list_open_prs(owner, repo)
        bot_prs = [
            BotPrSummary(
                number=pr.number,
                title=pr.title,
                author=pr.author,
                created_at=pr.created_at,
                base_ref=pr.base.ref,
            )
            for pr in all_prs
            if pr.author in allowed_authors
        ]
        return json_payload(BotPrList(bot_prs=bot_prs, count=len(bot_prs)))

    return list_bot_prs


def _make_legacy_tools(client: GitHubClient) -> LegacyTools:
    async def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
        """Get the diff of a pull request as plain text.

        Returns the unified diff showing all changes in the PR.
        """
        return await client.get_pr_diff(owner, repo, pr_number)

    async def get_pr_files(owner: str, repo: str, pr_number: int) -> str:
        """Get the list of files changed in a pull request.

        Returns a JSON list with filename, status (added/modified/removed),
        additions count, and deletions count for each file.
        """
        files = await client.get_pr_files(owner, repo, pr_number)
        return json.dumps([f.model_dump() for f in files], indent=2)

    async def get_file_contents(
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Get the contents of a file from a GitHub repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to the file within the repository.
            ref: Optional git ref (branch, tag, or SHA) to read from.

        Returns the file content as a JSON object with name, path, size,
        content (base64-encoded), and encoding fields.

        """
        data = await client.get_file_contents(owner, repo, path, ref)
        return json_payload(FileContents.model_validate(data))

    async def list_commits(
        owner: str,
        repo: str,
        sha: str | None = None,
        per_page: int = 30,
    ) -> str:
        """List recent commits for a repository or branch.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: Optional branch name or SHA to list commits from.
            per_page: Number of commits to return (default 30).

        Returns a JSON list of commits with sha, message, author, and date.

        """
        commits = await client.list_commits(owner, repo, sha, per_page)
        return json.dumps([c.model_dump() for c in commits], indent=2)

    async def get_commit_details(owner: str, repo: str, sha: str) -> str:
        """Get details of a specific commit including changed files and patches.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: The commit SHA to look up.

        Returns a JSON object with sha, message, author, date, and files
        (each with filename, status, and patch).

        """
        return json_payload(await client.get_commit(owner, repo, sha))

    async def list_branches(owner: str, repo: str) -> str:
        """List branch names for a repository.

        Returns a JSON list of branch name strings.
        """
        branches = await client.list_branches(owner, repo)
        return json.dumps(branches)

    return LegacyTools(
        get_pr_diff=get_pr_diff,
        get_pr_files=get_pr_files,
        get_file_contents=get_file_contents,
        list_commits=list_commits,
        get_commit_details=get_commit_details,
        list_branches=list_branches,
    )


def create_agent_tools(
    client: GitHubClient,
    bots: list[BotConfig],
    *,
    dry_run: bool,
    review_wait: int,
    log_limits: LogLimits = DEFAULT_LOG_LIMITS,
) -> AgentTools:
    """Create all agent tool functions, including legacy tools, status tools, and workflow log tools."""
    write = _make_write_tools(
        client=client,
        bots=bots,
        dry_run=dry_run,
        review_wait=review_wait,
    )
    legacy = _make_legacy_tools(client)

    return AgentTools(
        create_pr=_make_create_pr(client, dry_run=dry_run),
        get_branch_ci_status=_make_get_branch_ci_status(client),
        get_commit_details=legacy.get_commit_details,
        get_file_contents=legacy.get_file_contents,
        get_pr_diff=legacy.get_pr_diff,
        get_pr_files=legacy.get_pr_files,
        get_pr_status=_make_get_pr_status(client),
        get_pr_workflow_run_logs=_make_get_pr_workflow_run_logs(client, log_limits),
        list_bot_prs=_make_list_bot_prs(client, bots),
        list_branches=legacy.list_branches,
        list_commits=legacy.list_commits,
        merge_bot_pr=write.merge_bot_pr,
        rebase_bot_pr=write.rebase_bot_pr,
        wait_for_ci=_make_wait_for_ci(client),
        wait_for_reviews=write.wait_for_reviews,
    )


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if child is equal to or contained within parent."""
    try:
        child.relative_to(parent)
    except ValueError:
        return child == parent
    else:
        return True


def _validate_target_path(
    path_str: str,
    workspace_dir: str,
    context_msg: str,
) -> str | None:
    """Validate a path string against directory traversal and disallowed locations."""
    if path_str == "/":
        return f"Security Error: {context_msg} targeting root directory is denied."
    if ".." in Path(path_str).parts:
        return f"Security Error: {context_msg} with directory traversal is denied."

    try:
        path = Path(path_str).expanduser()
        resolved = (Path(workspace_dir) / path).resolve() if not path.is_absolute() else path.resolve()
    except (ValueError, RuntimeError, OSError) as e:
        return f"Security Error: {context_msg} has invalid path: {e}"

    workspace_path = Path(workspace_dir).resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()
    fallback_temps = [Path(os.sep + p).resolve() for p in ("tmp", f"private{os.sep}tmp")]

    in_workspace = _is_under(resolved, workspace_path)
    in_tmp = _is_under(resolved, system_temp) or any(_is_under(resolved, fp) for fp in fallback_temps)

    if not (in_workspace or in_tmp):
        return (
            f"Security Error: {context_msg} targeting path outside workspace and temp directory is denied: {path_str}"
        )
    return None


def _check_env_var_token(token: str) -> str | None:
    if "=" in token and not token.startswith("-"):
        parts = token.split("=", 1)
        env_key = parts[0].strip().lower()
        env_value = parts[1].strip().lower() if len(parts) > 1 else ""

        blocked_env_vars = {
            "ld_preload",
            "ld_library_path",
            "dyld_insert_libraries",
            "dyld_library_path",
            "git_ssh_command",
            "git_ssh",
        }
        if env_key in blocked_env_vars:
            return f"Security Error: Environment variable '{parts[0]}' is blocked."

        if "git_config" in env_key or env_key == "git_config_parameters":
            blocked_config_prefixes = (
                "credential.helper",
                "core.hookspath",
                "core.sshcommand",
                "url.",
                "http.proxy",
                "https.proxy",
            )
            for prefix in blocked_config_prefixes:
                if prefix in env_value:
                    return (
                        f"Security Error: Environment variable '{parts[0]}' "
                        f"contains blocked git configuration key '{prefix}'."
                    )
    return None


def _check_blocked_executables(exe_name: str) -> str | None:
    blocked_commands = {
        "sudo",
        "su",
        "nc",
        "netcat",
        "curl",
        "wget",
        "systemctl",
        "service",
        "init",
        "reboot",
        "shutdown",
        "halt",
        "poweroff",
    }
    if exe_name in blocked_commands:
        return f"Security Error: Command '{exe_name}' is blocked."
    return None


def _check_rm_command(tokens: list[str], idx: int, workspace_dir: str, operators: set[str]) -> str | None:
    for j in range(idx + 1, len(tokens)):
        arg = tokens[j]
        if arg in operators:
            break
        err = _validate_target_path(arg, workspace_dir, "Command 'rm'")
        if err:
            return err
    return None


def _check_git_config_subcommand(
    tokens: list[str],
    idx: int,
    operators: set[str],
    blocked_config_prefixes: tuple[str, ...],
) -> str | None:
    skip_next = False
    for k in range(idx + 1, len(tokens)):
        config_token = tokens[k]
        if config_token in operators:
            break
        if skip_next:
            skip_next = False
            continue
        if config_token in (
            "-f",
            "--file",
            "--blob",
            "--type",
            "--default",
        ):
            skip_next = True
            continue
        if config_token.startswith("-"):
            continue
        key = config_token.split("=", 1)[0].strip().lower()
        for prefix in blocked_config_prefixes:
            if key.startswith(prefix) or key == prefix:
                return f"Security Error: Git configuration key '{prefix}' is blocked."
        break
    return None


def _check_git_command(tokens: list[str], idx: int, operators: set[str]) -> str | None:
    blocked_git_options = ("--config-env", "--git-dir")
    for tok in tokens[idx + 1 :]:
        if tok.split("=", 1)[0].lower() in blocked_git_options:
            return f"Security Error: Git option '{tok.split('=', 1)[0]}' is blocked."

    blocked_config_prefixes = (
        "credential.helper",
        "core.hookspath",
        "core.sshcommand",
        "url.",
        "http.proxy",
        "https.proxy",
    )
    for j, tok in enumerate(tokens[idx + 1 :], start=idx + 1):
        if tok in ("-c", "--config") and j + 1 < len(tokens):
            config_expr = tokens[j + 1]
            key = config_expr.split("=", 1)[0].strip().lower()
            for prefix in blocked_config_prefixes:
                if key.startswith(prefix) or key == prefix:
                    return f"Security Error: Git configuration key '{prefix}' is blocked."
        elif tok == "config":
            err = _check_git_config_subcommand(tokens, j, operators, blocked_config_prefixes)
            if err:
                return err
    return None


def _check_shell_operators(argv: list[str]) -> str | None:
    """Return an error if dangerous shell operators are present as tokens."""
    dangerous_operators = {";", "|", "&"}
    if dangerous_operators & set(argv):
        return (
            "Security Error: Shell operators (;, |, &) are not "
            "allowed. Use separate run_command_sandboxed calls instead."
        )
    return None


def _resolve_exe(argv: list[str]) -> tuple[int, str] | str:
    """Return (exe_idx, exe_name) or an error string.

    Identify the executable index and normalized name, then validate every
    ``KEY=val`` assignment attached to it — whether written bare or passed
    through ``env``, since both reach the process the same way.
    """
    resolved = resolve_exe(argv)
    if resolved is None:
        return "Security Error: No executable found in command."

    for token in resolved.env_assignments:
        err = _check_env_var_token(token)
        if err:
            return err

    if Path(argv[resolved.wrapper_idx]).name.lower() == "env":
        for token in argv[resolved.wrapper_idx + 1 :]:
            if token in ("-S", "--split-string"):
                return "Security Error: 'env -S/--split-string' is blocked."

    return resolved.wrapper_idx, resolved.name


def _check_exec_pivots(exe_name: str, argv: list[str], exe_idx: int) -> str | None:
    """Block find -exec/-execdir pivots and xargs."""
    if exe_name == "find":
        for arg in argv[exe_idx + 1 :]:
            if arg in ("-exec", "-execdir", "-ok", "-okdir"):
                return f"Security Error: 'find' with '{arg}' is blocked."
    if exe_name == "xargs":
        return "Security Error: Command 'xargs' is blocked."
    return None


def _check_git_extended(argv: list[str], exe_idx: int) -> str | None:
    """Additional git hardening: block --upload-pack/--receive-pack and extra config prefixes."""
    blocked_git_flags = {"--upload-pack", "--receive-pack"}
    for tok in argv[exe_idx + 1 :]:
        flag = tok.split("=", 1)[0].lower()
        if flag in blocked_git_flags:
            return f"Security Error: Git flag '{flag}' is blocked."

    blocked_config_prefixes_extra = (
        "protocol.ext.allow",
        "remote.",
    )
    for j, tok in enumerate(argv[exe_idx + 1 :], start=exe_idx + 1):
        if tok in ("-c", "--config") and j + 1 < len(argv):
            config_key = argv[j + 1].split("=", 1)[0].strip().lower()
            for prefix in blocked_config_prefixes_extra:
                if config_key.startswith(prefix):
                    return f"Security Error: Git config '{prefix}' is blocked."
    return None


def _check_per_exe(exe_name: str, argv: list[str], exe_idx: int, workspace_dir: str) -> str | None:
    """Dispatch per-executable security checks (pivots, denylist, rm, git)."""
    err = _check_exec_pivots(exe_name, argv, exe_idx) or _check_blocked_executables(exe_name)
    if err:
        return err
    if exe_name == "rm":
        return _check_rm_command(argv, exe_idx, workspace_dir, set())
    if exe_name == "git":
        return _check_git_command(argv, exe_idx, set()) or _check_git_extended(argv, exe_idx)
    return None


def _validate_single_argv(argv: list[str], workspace_dir: str) -> str | None:
    """Validate a single (non-compound) argv array for sandboxed execution."""
    if not argv:
        return "Security Error: Empty command."

    # Reject dangerous shell operators (;, |, &) — these are never safe.
    # Note: && and || are handled at the compound level, not here.
    err = _check_shell_operators(argv)
    if err:
        return err

    # Find the executable (skip leading FOO=bar env assignments)
    result = _resolve_exe(argv)
    if isinstance(result, str):
        return result
    exe_idx, exe_name = result

    # Block SHELL interpreters only (no legitimate use after Phase 2).
    # Language runtimes (python3, node, ruby, etc.) are allowed —
    # agent needs them for test execution; srt OS sandbox is the boundary.
    blocked_shells = {"bash", "sh", "dash", "zsh", "ksh"}
    if exe_name in blocked_shells:
        return f"Security Error: Shell interpreter '{exe_name}' is blocked."

    return _check_per_exe(exe_name, argv, exe_idx, workspace_dir)


def validate_argv(argv: list[str], workspace_dir: str) -> str | None:
    """Validate an argv array for sandboxed execution (defense-in-depth).

    Supports compound commands joined by ``&&`` and ``||``.  Each
    sub-command is validated independently.  Dangerous operators
    (``;``, ``|``, ``&``) remain blocked.
    """
    if not argv:
        return "Security Error: Empty command."

    parts = split_compound_argv(argv)
    for part in parts:
        err = _validate_single_argv(part.argv, workspace_dir)
        if err:
            return err
    return None


def _splice_middle[T: (str, list[str])](units: T, cap: int, marker_for: Callable[[int], T]) -> T:
    """Keep the head and tail of ``units``, naming what was dropped between.

    ``units`` is measured in whatever the cap counts: a string for a character
    budget, a list of lines for a line budget. ``marker_for`` returns the
    replacement for the middle in the same currency, so its own size is
    charged against the cap rather than added on top of it — otherwise every
    truncated stream comes back over the limit the caller configured.
    """
    if cap <= 0 or len(units) <= cap:
        return units

    # Reserve against the longest the marker can get. Its length only grows
    # with the omitted count, so sizing on the whole input can never underrun.
    budget = max(cap - len(marker_for(len(units))), 0)
    head_n = int(budget * OUTPUT_HEAD_FRACTION)
    tail_n = budget - head_n
    head = units[:head_n]
    tail = units[len(units) - tail_n :] if tail_n else units[:0]
    return (head + marker_for(len(units) - budget) + tail)[:cap]


def _truncate_middle(text: str, limits: OutputLimits) -> str:
    """Trim the middle out of ``text`` to fit within ``limits``.

    Keeps a head and a tail because each answers a different question: the
    head shows what ran, the tail shows how it ended. The line cap runs first
    so ordinary logs are trimmed on line boundaries; the character cap then
    catches output that is short on lines but enormous on one of them.
    """
    if limits.max_lines > 0:
        text = "\n".join(
            _splice_middle(text.splitlines(), limits.max_lines, lambda n: [f"... [{n} lines omitted] ..."]),
        )

    return _splice_middle(text, limits.max_chars, lambda n: f"\n... [{n} characters omitted] ...\n")


def _format_command_result(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
    limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
) -> str:
    stdout_str = stdout.decode(errors="replace")
    stderr_str = stderr.decode(errors="replace")

    diagnostics = []
    # Read diagnostics off the full text: a blocked request lands wherever it
    # happened, which for a long install is nowhere near either end.
    combined = stdout_str + stderr_str
    if returncode != 0:
        if "Connection blocked by network allowlist" in combined or "X-Proxy-Error: blocked-by-allowlist" in combined:
            diagnostics.append(
                "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy. "
                "Allowed domains are configured in srt-settings.json.",
            )
        if "Operation not permitted" in stderr_str or "Permission denied" in stderr_str:
            diagnostics.append(
                "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied' / 'Operation not permitted'. "
                "If this path is outside the allowed directories, it was blocked by the "
                "sandbox-runtime policy configured in srt-settings.json.",
            )

    parts = [
        "--- STDOUT ---",
        _truncate_middle(stdout_str, limits),
        "--- STDERR ---",
        _truncate_middle(stderr_str, limits),
        f"--- EXIT CODE: {returncode} ---",
    ]
    if diagnostics:
        parts.append("\n" + "\n".join(diagnostics))
    return "\n".join(parts)


def _setup_cache_env(cache_dir: str, env: dict[str, str]) -> None:
    cache_base = Path(cache_dir)
    env["BUN_INSTALL"] = str(cache_base / "bun")
    env["CARGO_HOME"] = str(cache_base / "cargo")
    env["COMPOSER_CACHE_DIR"] = str(cache_base / "composer")
    env["DENO_DIR"] = str(cache_base / "deno")
    env["DOTNET_CLI_HOME"] = str(cache_base / "dotnet")
    env["GEM_HOME"] = str(cache_base / "gems")
    env["GEM_PATH"] = str(cache_base / "gems")
    env["GOMODCACHE"] = str(cache_base / "go" / "pkg" / "mod")
    env["GRADLE_USER_HOME"] = str(cache_base / "gradle")
    env["MIX_HOME"] = str(cache_base / "mix")
    env["NPM_CONFIG_CACHE"] = str(cache_base / "npm")
    env["NUGET_PACKAGES"] = str(cache_base / "nuget")
    env["PIP_CACHE_DIR"] = str(cache_base / "pip")
    env["PIPENV_CACHE_DIR"] = str(cache_base / "pipenv")
    env["PNPM_HOME"] = str(cache_base / "pnpm")
    env["POETRY_CACHE_DIR"] = str(cache_base / "poetry")
    env["PUB_CACHE"] = str(cache_base / "pub-cache")
    env["UV_CACHE_DIR"] = str(cache_base / "uv")
    env["UV_CONFIG_FILE"] = "/dev/null"
    env["UV_PYTHON_INSTALL_DIR"] = str(cache_base / "uv" / "python")
    env["UV_PYTHON_PREFERENCE"] = "only-system"
    env["UV_TOOL_DIR"] = str(cache_base / "uv" / "tools")
    env["XDG_CACHE_HOME"] = str(cache_base)
    env["YARN_CACHE_FOLDER"] = str(cache_base / "yarn")


def _is_git_command_argv(argv: list[str]) -> bool:
    """Check if the first executable in an argv list is git."""
    for token in argv:
        if "=" in token and not token.startswith("-"):
            continue  # skip env var assignments
        return Path(token).name.lower() == "git"
    return False


def _is_git_command(command_line: str) -> bool:
    """Check if a command string starts with git."""
    try:
        argv = shlex.split(command_line)
    except ValueError:
        return False
    return _is_git_command_argv(argv)


class SandboxConfig(NamedTuple):
    """Immutable configuration for sandboxed command execution.

    Attributes:
        workspace_dir: Root directory for sandboxed file operations.
        srt_config: Parsed srt settings dict (filesystem, network rules).
        github_token: Token for authenticated git operations.
        command_timeout: Max seconds per sandboxed command.
        output_limits: Caps on the output returned to the model.
        cache_dir: Shared package cache, outside the workspace so it persists.

    """

    workspace_dir: str
    srt_config: dict[str, Any]
    github_token: str | None = None
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS
    cache_dir: str = DEFAULT_CACHE_DIR


class SandboxedCommandRunner:
    """Callable tool runner that executes command line strings inside srt sandbox."""

    __name__ = "run_command_sandboxed"

    def __init__(
        self,
        config: SandboxConfig,
        config_path: str | None,
        init_error: str | None,
    ) -> None:
        """Initialize the sandboxed command runner."""
        self.cfg = config
        self.config_path = config_path
        self.init_error = init_error

    def cleanup(self) -> None:
        """Remove any temporary sandbox configurations."""
        if self.config_path:
            path = self.config_path
            self.config_path = None
            with contextlib.suppress(Exception):
                Path(path).unlink()

    def __del__(self) -> None:
        """Cleanup on object destruction."""
        self.cleanup()

    async def __call__(
        self,
        command_line: str,
        working_dir: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a shell command. Sandboxed on macOS by default.

        Supports compound commands joined by ``&&`` and ``||``.
        Each sub-command is validated and executed independently through srt.

        Args:
            command_line: The exact command line string to run.
            working_dir: The directory to run the command in.
            cwd: The directory to run the command in.
            Cwd: Legacy fallback parameter for working directory, supported for backward
                compatibility with certain models/SDK configurations.
            **kwargs: Additional keyword arguments for compatibility.

        """
        if self.init_error:
            return self.init_error

        legacy_cwd = kwargs.get("Cwd")
        target_cwd = working_dir or cwd or legacy_cwd or self.cfg.workspace_dir

        try:
            argv = shlex.split(command_line)
        except ValueError:
            return "Security Error: Invalid shell command quoting."

        validation_err = validate_argv(argv, self.cfg.workspace_dir)
        if validation_err:
            return validation_err

        parts = split_compound_argv(argv)

        # Simple case: no compound operators
        if len(parts) == 1:
            return (await self._run_single_argv(parts[0].argv, target_cwd)).output

        # Compound: run each sub-command with && / || semantics
        all_outputs: list[str] = []
        last_returncode = 0
        for part in parts:
            if part.operator == "&&" and last_returncode != 0:
                break  # short-circuit: previous failed
            if part.operator == "||" and last_returncode == 0:
                continue  # short-circuit: previous succeeded

            result = await self._run_single_argv(part.argv, target_cwd)
            all_outputs.append(result.output)
            last_returncode = result.returncode

        return "\n".join(all_outputs)

    async def _run_single_argv(
        self,
        argv: list[str],
        target_cwd: str,
    ) -> CommandResult:
        """Execute a single (non-compound) argv array through srt."""
        env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_ALLOWLIST}
        _setup_cache_env(self.cfg.cache_dir, env)

        is_git_command = _is_git_command_argv(argv)

        if is_git_command:
            env["GIT_CONFIG_GLOBAL"] = "/dev/null"
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_TEMPLATE_DIR"] = "/dev/null"
            token = self.cfg.github_token or os.environ.get("GITHUB_TOKEN")
            if token:
                env["GIT_CONFIG_COUNT"] = "2"
                env["GIT_CONFIG_KEY_0"] = f"url.https://x-access-token:{token}@github.com/.insteadOf"
                env["GIT_CONFIG_VALUE_0"] = "https://github.com/"
                env["GIT_CONFIG_KEY_1"] = f"url.https://x-access-token:{token}@github.com/.insteadOf"
                env["GIT_CONFIG_VALUE_1"] = "git@github.com:"
                env["GIT_TERMINAL_PROMPT"] = "0"

        active_config_path = self.config_path
        git_config_temp_path = None

        if is_git_command:
            git_config = copy.deepcopy(self.cfg.srt_config)
            git_config["filesystem"]["allowGitConfig"] = True

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".json",
                    mode="w",
                    delete=False,
                ) as temp_f:
                    json.dump(git_config, temp_f)
                    git_config_temp_path = temp_f.name
                    active_config_path = git_config_temp_path
            except (OSError, TypeError, ValueError) as e:
                return CommandResult(f"Error: Failed to create temporary config for Git: {e}", -1)

        try:
            if not active_config_path:
                return CommandResult("Error: Sandbox configuration was not initialized.", -1)

            process = await asyncio.create_subprocess_exec(
                "srt",
                "--settings",
                active_config_path,
                "--",
                *argv,
                cwd=target_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            comm_coro = process.communicate()
            try:
                stdout, stderr = await asyncio.wait_for(comm_coro, timeout=self.cfg.command_timeout)
            except TimeoutError:
                comm_coro.close()
                with contextlib.suppress(Exception):
                    process.kill()
                await process.wait()
                return CommandResult(f"Error: Command timed out after {self.cfg.command_timeout} seconds.", -1)
            rc = process.returncode if process.returncode is not None else -1
            return CommandResult(_format_command_result(stdout, stderr, rc, self.cfg.output_limits), rc)
        finally:
            if git_config_temp_path:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(Path(git_config_temp_path).unlink, missing_ok=True)


def create_run_command_tool(
    workspace_dir: str,
    srt_settings_path: str | Path = "",
    github_token: str | None = None,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Callable[..., Awaitable[str]]:
    """Create a sandboxed run_command tool bound to a workspace."""
    config_path = None
    init_error = None
    srt_config = {}

    # The cache outlives the workspace, so it may already exist from an earlier repo.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    base_path = srt_settings_path or DEFAULT_SRT_SETTINGS_PATH
    try:
        with Path(base_path).open() as f:
            srt_config = json.load(f)

        fs_config = srt_config.setdefault("filesystem", {})
        fs_config["allowGitConfig"] = False

        allow_write = fs_config.setdefault("allowWrite", [])
        allow_read = fs_config.setdefault("allowRead", [])
        for path in (workspace_dir, cache_dir):
            if path not in allow_write:
                allow_write.append(path)
            if path not in allow_read:
                allow_read.append(path)

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            mode="w",
            delete=False,
        ) as temp_f:
            json.dump(srt_config, temp_f)
            config_path = temp_f.name

    except FileNotFoundError:
        init_error = f"Error: Sandbox settings file not found at {base_path}."
    except json.JSONDecodeError:
        init_error = f"Error: Sandbox settings file at {base_path} is not valid JSON."

    runner = SandboxedCommandRunner(
        config=SandboxConfig(
            workspace_dir=workspace_dir,
            srt_config=srt_config,
            github_token=github_token,
            command_timeout=command_timeout,
            output_limits=output_limits,
            cache_dir=cache_dir,
        ),
        config_path=config_path,
        init_error=init_error,
    )

    async def run_command_sandboxed(
        command_line: str,
        working_dir: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a shell command. Sandboxed on macOS by default.

        Args:
            command_line: The exact command line string to run.
            working_dir: The directory to run the command in.
            cwd: The directory to run the command in.
            **kwargs: Additional keyword arguments for compatibility.

        """
        return await runner(command_line, working_dir=working_dir, cwd=cwd, **kwargs)

    setattr(run_command_sandboxed, "cleanup", runner.cleanup)  # noqa: B010

    return run_command_sandboxed


def is_srt_available() -> bool:
    """Check if sandbox-runtime (srt) is available and functional."""
    try:
        res = subprocess.run(
            ["srt", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return False
    else:
        return res.returncode == 0


def is_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is installed and available in PATH."""
    try:
        res = subprocess.run(
            ["rg", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    else:
        return res.returncode == 0
