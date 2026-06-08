"""Tools and GitHub API client helper functions for dependency bot PR management."""

import asyncio
import contextlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import httpx

from dependency_director.config import SAFE_ENV_ALLOWLIST, BotConfig

OWNER_RE = re.compile(r"^[a-zA-Z0-9-]{1,39}$")
REPO_RE = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")
_ENV_FLAGS_WITH_ARG = {"-u", "--unset", "-S", "--split-string"}


class GitHubClientError(Exception):
    """Base exception for GitHubClient errors."""


class GitHubAuthenticationError(GitHubClientError):
    """Exception raised for 401/403 errors."""


class GitHubNotFoundError(GitHubClientError):
    """Exception raised for 404 errors."""


def _validate_repo_params(owner: str, repo: str) -> None:
    if not OWNER_RE.match(owner):
        msg = f"Invalid owner name: {owner!r}"
        raise ValueError(msg)
    if not REPO_RE.match(repo):
        msg = f"Invalid repository name: {repo!r}"
        raise ValueError(msg)


class GitHubClient:
    """Shared HTTP client for GitHub API calls. Created once per agent run."""

    def __init__(self, token: str) -> None:
        """Initialize GitHub API client with optional auth token."""
        self.headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "dependency-director-agent",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        async def check_api_errors(response: httpx.Response) -> None:
            if response.status_code in (401, 403, 404):
                # Don't fail-fast on the organization-fallback check in get_repositories
                if (
                    response.status_code == 404
                    and "/users/" in str(response.url)
                    and "/repos" in str(response.url)
                ):
                    return

                msg = f"GitHub API error {response.status_code} on {response.url}"
                if response.status_code == 401:
                    msg += (
                        " - Unauthorized. Please verify your DEPDIRECTOR_GITHUB_TOKEN."
                    )
                    raise GitHubAuthenticationError(msg)
                if response.status_code == 403:
                    msg += " - Forbidden/Rate Limited. Please verify your token scopes and rate limits."
                    raise GitHubAuthenticationError(msg)
                if response.status_code == 404:
                    msg += " - Not Found. Please verify the owner/repository names exist and your token has access."
                    raise GitHubNotFoundError(msg)

        self.client = httpx.AsyncClient(event_hooks={"response": [check_api_errors]})

    async def _request(
        self,
        method: str,
        path: str,
        owner: str,
        repo: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: Any = None,
    ) -> httpx.Response:
        _validate_repo_params(owner, repo)
        url = f"https://api.github.com/repos/{owner}/{repo}/{path.lstrip('/')}"
        req_headers = {**self.headers, **(headers or {})}

        method_upper = method.upper()
        kwargs: dict[str, Any] = {"headers": req_headers}
        if method_upper == "GET":
            if params is not None:
                kwargs["params"] = params
            response = await self.client.get(url, **kwargs)
        elif method_upper == "PUT":
            if json_data is not None:
                kwargs["json"] = json_data
            response = await self.client.put(url, **kwargs)
        elif method_upper == "POST":
            if json_data is not None:
                kwargs["json"] = json_data
            response = await self.client.post(url, **kwargs)
        else:
            if params is not None:
                kwargs["params"] = params
            if json_data is not None:
                kwargs["json"] = json_data
            response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    async def _get(
        self,
        path: str,
        owner: str,
        repo: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await self._request(
            "GET",
            path,
            owner,
            repo,
            params=params,
            headers=headers,
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
        user = data.get("user")
        if isinstance(user, dict):
            login = user.get("login")
            if isinstance(login, str):
                return login
        return ""

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
        response = await self._get(f"actions/jobs/{job_id}/logs", owner, repo)
        return response.text

    async def list_open_prs(
        self,
        owner: str,
        repo: str,
    ) -> list[dict[str, Any]]:
        """Fetch all open pull requests with minimal fields.

        Returns a compact list of dicts with only number, title, author, and
        created_at. This avoids pulling in the huge body/changelog payloads
        that dependency bots typically produce.
        """
        prs: list[dict[str, Any]] = []
        params = {
            "state": "open",
            "sort": "created",
            "direction": "asc",
        }
        async for pr in self._list_paginated("pulls", owner, repo, params=params):
            prs.append(
                {
                    "number": pr["number"],
                    "title": pr.get("title", ""),
                    "author": pr.get("user", {}).get("login", ""),
                    "created_at": pr.get("created_at", ""),
                },
            )
        return prs

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
    ) -> list[dict[str, Any]]:
        """Fetch the list of files changed in a pull request."""
        files: list[dict[str, Any]] = []
        async for f in self._list_paginated(f"pulls/{pr_number}/files", owner, repo):
            files.append(
                {
                    "filename": f["filename"],
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                },
            )
        return files

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
    ) -> list[dict[str, Any]]:
        """Fetch recent commits for a repository or branch."""
        params: dict[str, Any] = {"per_page": per_page}
        if sha:
            params["sha"] = sha
        response = await self._get("commits", owner, repo, params=params)
        data = response.json()
        return [
            {
                "sha": c["sha"][:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name", ""),
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
            }
            for c in data
        ]

    async def get_commit(
        self,
        owner: str,
        repo: str,
        sha: str,
    ) -> dict[str, Any]:
        """Fetch details of a specific commit."""
        response = await self._get(f"commits/{sha}", owner, repo)
        data = response.json()
        return {
            "sha": data["sha"],
            "message": data.get("commit", {}).get("message", ""),
            "author": data.get("commit", {}).get("author", {}).get("name", ""),
            "date": data.get("commit", {}).get("author", {}).get("date", ""),
            "files": [
                {
                    "filename": f["filename"],
                    "status": f.get("status", ""),
                    "patch": f.get("patch", ""),
                }
                for f in data.get("files", [])
            ],
        }

    async def list_branches(
        self,
        owner: str,
        repo: str,
    ) -> list[str]:
        """Fetch branch names for a repository."""
        return [b["name"] async for b in self._list_paginated("branches", owner, repo)]

    async def get_repositories(self, owner: str) -> list[str]:
        """Fetch non-forked repositories from GitHub API.

        Uses pagination to get all repositories for the given owner.
        """
        _validate_repo_params(owner, "dummy-repo")
        endpoint_prefix = f"https://api.github.com/users/{owner}/repos"
        repos: list[str] = []
        page = 1
        while True:
            url = f"{endpoint_prefix}?type=owner&per_page=100&page={page}"
            response = await self.client.get(url, headers=self.headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if (
                    e.response.status_code == 404
                    and page == 1
                    and "users" in endpoint_prefix
                ):
                    endpoint_prefix = f"https://api.github.com/orgs/{owner}/repos"
                    url = f"{endpoint_prefix}?type=sources&per_page=100&page={page}"
                    response = await self.client.get(url, headers=self.headers)
                    response.raise_for_status()
                else:
                    raise
            repos_data = response.json()
            if not repos_data:
                break

            for repo in repos_data:
                if not repo.get("fork", False):
                    repos.append(f"{owner}/{repo['name']}")
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


def _create_write_tools(
    client: GitHubClient,
    bots: list[BotConfig],
    *,
    dry_run: bool,
    review_wait: int,
) -> tuple[ToolFn, ToolFn, ToolFn]:
    """Create agent tool functions with config bound via closure."""

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
            if exc.response.status_code == 405:
                return (
                    f"PR #{pr_number} cannot be merged right now (GitHub 405 — not mergeable). "
                    "A prior merge likely introduced a conflict. "
                    "Call get_pr_status to confirm, then rebase_bot_pr if CONFLICT."
                )
            raise
        message = res.get("message", "PR merged successfully.")
        return f"Successfully merged PR #{pr_number} in {owner}/{repo}: {message}"

    async def rebase_bot_pr(owner: str, repo: str, pr_number: int) -> str:
        """Post a rebase comment on a pull request.

        Author must be a configured bot.
        """
        author = await client.get_pr_author(owner, repo, pr_number)
        bot = _check_bot_author(author, bots)

        if dry_run:
            return (
                f"[DRY-RUN] Would have commented '{bot.rebase_command}' on "
                f"PR #{pr_number} in {owner}/{repo}."
            )

        await client.comment_on_pr(owner, repo, pr_number, bot.rebase_command)
        return (
            f"Successfully requested rebase for PR #{pr_number} in "
            f"{owner}/{repo} by commenting '{bot.rebase_command}'."
        )

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

        return (
            f"No review comments received on PR #{pr_number} "
            f"within {review_wait} minute(s)."
        )

    return merge_bot_pr, rebase_bot_pr, wait_for_reviews


def create_agent_tools(
    client: GitHubClient,
    bots: list[BotConfig],
    *,
    dry_run: bool,
    review_wait: int,
) -> tuple[ToolFn, ...]:
    """Create all agent tool functions, including legacy tools, status tools, and workflow log tools."""
    merge_bot_pr, rebase_bot_pr, wait_for_reviews = _create_write_tools(
        client=client,
        bots=bots,
        dry_run=dry_run,
        review_wait=review_wait,
    )

    async def get_pr_status(owner: str, repo: str, pr_number: int) -> str:
        """Get the mergeability and CI checks status of a pull request.

        Returns a JSON string detailing PR status, mergeable state, and CI results.
        """
        pr_details = await client.get_pr_details(owner, repo, pr_number)
        head_sha = pr_details.get("head", {}).get("sha", "")
        mergeable = pr_details.get("mergeable")
        mergeable_state = pr_details.get("mergeable_state")

        checks_summary = []
        ci_status = "NONE"

        if head_sha:
            check_runs_data, commit_status_data = await asyncio.gather(
                client.get_commit_check_runs(owner, repo, head_sha),
                client.get_commit_status(owner, repo, head_sha),
            )
            check_runs = check_runs_data.get("check_runs", [])
            for run in check_runs:
                checks_summary.append(
                    {
                        "name": run.get("name"),
                        "status": run.get("status"),
                        "conclusion": run.get("conclusion"),
                    },
                )

            legacy_statuses = commit_status_data.get("statuses", [])
            legacy_state = commit_status_data.get("state") if legacy_statuses else None
            for status in legacy_statuses:
                checks_summary.append(
                    {
                        "name": status.get("context"),
                        "status": "completed",
                        "conclusion": "success"
                        if status.get("state") == "success"
                        else ("failure" if status.get("state") == "failure" else None),
                    },
                )

            # Determine CI outcome
            has_failures = any(
                r.get("conclusion")
                in ("failure", "action_required", "cancelled", "timed_out")
                for r in checks_summary
            )
            has_pending = any(
                r.get("status") not in ("completed", "success")
                or r.get("conclusion") is None
                for r in checks_summary
            )

            if has_failures or legacy_state == "failure":
                ci_status = "RED"
            elif has_pending or legacy_state == "pending":
                ci_status = "PENDING"
            elif checks_summary or legacy_state == "success":
                ci_status = "GREEN"

        if mergeable is False or mergeable_state in ("dirty", "conflict"):
            ci_status = "CONFLICT"

        summary = {
            "pr_number": pr_number,
            "title": pr_details.get("title"),
            "mergeable": mergeable,
            "mergeable_state": mergeable_state,
            "ci_status": ci_status,
            "checks": checks_summary,
        }
        return json.dumps(summary, indent=2)

    async def get_pr_workflow_run_logs(owner: str, repo: str, pr_number: int) -> str:
        """Fetch raw logs for failed GitHub Actions workflow runs on a pull request.

        Returns log text for failed jobs, keeping only the last 50 lines of logs per job.
        """
        pr_details = await client.get_pr_details(owner, repo, pr_number)
        head_sha = pr_details.get("head", {}).get("sha", "")
        if not head_sha:
            return f"Error: No head SHA found for PR #{pr_number}."

        runs_data = await client.get_workflow_runs_for_commit(owner, repo, head_sha)
        runs = runs_data.get("workflow_runs", [])
        if not runs:
            return f"No workflow runs found for commit {head_sha}."

        latest_run = runs[0]
        run_id = latest_run["id"]

        jobs_data = await client.get_workflow_run_jobs(owner, repo, run_id)
        jobs = jobs_data.get("jobs", [])

        failed_jobs_logs = []
        for job in jobs:
            if job.get("conclusion") == "failure":
                job_id = job["id"]
                job_name = job["name"]
                try:
                    log_text = await client.get_job_logs(owner, repo, job_id)
                    lines = log_text.splitlines()
                    truncated_log = "\n".join(lines[-50:])
                    failed_jobs_logs.append(
                        f"--- FAILED JOB: {job_name} (ID: {job_id}) ---\n{truncated_log}\n",
                    )
                except Exception as e:
                    failed_jobs_logs.append(
                        f"--- FAILED JOB: {job_name} (ID: {job_id}) ---\nFailed to retrieve log: {e}\n",
                    )

        if not failed_jobs_logs:
            return "No failed jobs found for the latest workflow run."

        return "\n".join(failed_jobs_logs)

    async def list_bot_prs(owner: str, repo: str) -> str:
        """List open dependency-bot pull requests for a repository.

        Returns a compact JSON list with only the essential fields:
        number, title, author, and created_at. Sorted oldest first.
        Only PRs authored by configured bots are included.
        """
        allowed_authors = {b.author for b in bots}
        all_prs = await client.list_open_prs(owner, repo)
        bot_prs = [pr for pr in all_prs if pr["author"] in allowed_authors]
        if not bot_prs:
            return json.dumps({"bot_prs": [], "count": 0})
        return json.dumps({"bot_prs": bot_prs, "count": len(bot_prs)}, indent=2)

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
        return json.dumps(files, indent=2)

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
        return json.dumps(
            {
                "name": data.get("name"),
                "path": data.get("path"),
                "size": data.get("size"),
                "content": data.get("content"),
                "encoding": data.get("encoding"),
            },
            indent=2,
        )

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
        return json.dumps(commits, indent=2)

    async def get_commit_details(owner: str, repo: str, sha: str) -> str:
        """Get details of a specific commit including changed files and patches.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: The commit SHA to look up.

        Returns a JSON object with sha, message, author, date, and files
        (each with filename, status, and patch).

        """
        data = await client.get_commit(owner, repo, sha)
        return json.dumps(data, indent=2)

    async def list_branches(owner: str, repo: str) -> str:
        """List branch names for a repository.

        Returns a JSON list of branch name strings.
        """
        branches = await client.list_branches(owner, repo)
        return json.dumps(branches)

    return (
        merge_bot_pr,
        rebase_bot_pr,
        wait_for_reviews,
        get_pr_status,
        get_pr_workflow_run_logs,
        list_bot_prs,
        get_pr_diff,
        get_pr_files,
        get_file_contents,
        list_commits,
        get_commit_details,
        list_branches,
    )


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if child is equal to or contained within parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return child == parent


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
        if not path.is_absolute():
            resolved = (Path(workspace_dir) / path).resolve()
        else:
            resolved = path.resolve()
    except Exception as e:
        return f"Security Error: {context_msg} has invalid path: {e}"

    workspace_path = Path(workspace_dir).resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()
    fallback_temps = [Path(p).resolve() for p in ("/tmp", "/private/tmp")]

    in_workspace = _is_under(resolved, workspace_path)
    in_tmp = _is_under(resolved, system_temp) or any(
        _is_under(resolved, fp) for fp in fallback_temps
    )

    if not (in_workspace or in_tmp):
        return f"Security Error: {context_msg} targeting path outside workspace and temp directory is denied: {path_str}"
    return None


def tokenize_command(command_line: str) -> list[str]:
    """Tokenize a shell command line string, splitting operators safely outside quotes."""
    cmd_stripped = command_line.strip()
    in_single = False
    in_double = False
    escaped = False
    spaced_cmd = []
    i = 0
    n = len(cmd_stripped)
    while i < n:
        char = cmd_stripped[i]
        if escaped:
            spaced_cmd.append(char)
            escaped = False
            i += 1
            continue
        if char == "\\":
            spaced_cmd.append(char)
            escaped = True
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            spaced_cmd.append(char)
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            spaced_cmd.append(char)
            i += 1
            continue
        if not in_single and not in_double:
            if i + 1 < n and cmd_stripped[i : i + 2] in ("&&", "||"):
                spaced_cmd.append(f" {cmd_stripped[i : i + 2]} ")
                i += 2
                continue
            if char in (";", "|", "&", "\n"):
                spaced_cmd.append(f" {char} ")
                i += 1
                continue
        spaced_cmd.append(char)
        i += 1

    return shlex.split("".join(spaced_cmd))


def validate_sandboxed_command(command_line: str, workspace_dir: str) -> str | None:
    """Validate command_line even when sandboxed, for defense-in-depth."""
    try:
        tokens = tokenize_command(command_line)
    except ValueError:
        return "Security Error: Invalid shell command quoting."

    if not tokens:
        return "Security Error: Empty command."

    # Identify executables: first token, and any token after a shell operator
    operators = {";", "&&", "||", "|", "&"}

    is_next_executable = True
    for i, token in enumerate(tokens):
        if token in operators:
            is_next_executable = True
            continue

        if is_next_executable:
            is_next_executable = False
            exe_name = Path(token).name.lower()

            if exe_name == "env" and len(tokens) > i + 1:
                # Flags that consume the next token as their argument
                j = i + 1
                while j < len(tokens):
                    t = tokens[j]
                    if t == "--":
                        # -- terminates flags; next token is the command
                        j += 1
                        if j < len(tokens):
                            exe_name = Path(tokens[j]).name.lower()
                        break
                    if t in _ENV_FLAGS_WITH_ARG:
                        j += 2  # skip the flag and its argument
                    elif t.startswith("-") or "=" in t:
                        j += 1
                    else:
                        exe_name = Path(t).name.lower()
                        break

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

            if exe_name == "rm":
                # Look ahead to find arguments of rm until next operator
                for j in range(i + 1, len(tokens)):
                    arg = tokens[j]
                    if arg in operators:
                        break
                    err = _validate_target_path(arg, workspace_dir, "Command 'rm'")
                    if err:
                        return err
    return None


def _format_command_result(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
) -> str:
    stdout_str = stdout.decode(errors="replace")
    stderr_str = stderr.decode(errors="replace")

    diagnostics = []
    combined = stdout_str + stderr_str
    if (
        "Connection blocked by network allowlist" in combined
        or "X-Proxy-Error: blocked-by-allowlist" in combined
    ):
        diagnostics.append(
            "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy. "
            "Allowed domains are configured in srt-settings.json.",
        )
    if "Operation not permitted" in combined or "Permission denied" in combined:
        diagnostics.append(
            "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied' / 'Operation not permitted'. "
            "If this path is outside the allowed directories, it was blocked by the sandbox-runtime policy configured in srt-settings.json.",
        )

    parts = [
        "--- STDOUT ---",
        stdout_str,
        "--- STDERR ---",
        stderr_str,
        f"--- EXIT CODE: {returncode} ---",
    ]
    if diagnostics:
        parts.append("\n" + "\n".join(diagnostics))
    return "\n".join(parts)


def create_run_command_tool(
    workspace_dir: str,
    srt_settings_path: str | Path = "",
) -> Callable[..., Awaitable[str]]:
    """Create a sandboxed run_command tool bound to a workspace."""
    from dependency_director.config import DEFAULT_SRT_SETTINGS_PATH

    config_path = None
    init_error = None
    cleanup = None

    base_path = srt_settings_path or DEFAULT_SRT_SETTINGS_PATH
    try:
        with Path(base_path).open() as f:
            srt_config = json.load(f)

        # Patch workspace into allowWrite and allowRead
        fs_config = srt_config.setdefault("filesystem", {})
        allow_write = fs_config.setdefault("allowWrite", [])
        if workspace_dir not in allow_write:
            allow_write.append(workspace_dir)

        allow_read = fs_config.setdefault("allowRead", [])
        if workspace_dir not in allow_read:
            allow_read.append(workspace_dir)

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            mode="w",
            delete=False,
        ) as temp_f:
            json.dump(srt_config, temp_f)
            config_path = temp_f.name

        def cleanup() -> None:
            if config_path:
                with contextlib.suppress(Exception):
                    Path(config_path).unlink()

    except FileNotFoundError:
        init_error = f"Error: Sandbox settings file not found at {base_path}."
    except json.JSONDecodeError:
        init_error = f"Error: Sandbox settings file at {base_path} is not valid JSON."

    async def run_command_sandboxed(
        command_line: str,
        working_dir: str | None = None,
        Cwd: str | None = None,
    ) -> str:
        """Execute a shell command. Sandboxed on macOS by default.

        Args:
            command_line: The exact command line string to run.
            working_dir: The directory to run the command in.
            Cwd: Legacy fallback parameter for working directory, supported for backward compatibility with certain models/SDK configurations.

        """
        if init_error:
            return init_error

        # Support both 'working_dir' and the legacy/alternative 'Cwd' parameter name
        target_cwd = working_dir or Cwd or workspace_dir

        env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_ALLOWLIST}
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_NOSYSTEM"] = "1"

        if not config_path:
            return "Error: Sandbox configuration was not initialized."
        validation_err = validate_sandboxed_command(command_line, workspace_dir)
        if validation_err:
            return validation_err
        process = await asyncio.create_subprocess_exec(
            "srt",
            "--settings",
            config_path,
            "-c",
            command_line,
            cwd=target_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        comm_coro = process.communicate()
        try:
            stdout, stderr = await asyncio.wait_for(comm_coro, timeout=300)
        except TimeoutError:
            comm_coro.close()
            with contextlib.suppress(Exception):
                process.kill()
            await process.wait()
            return "Error: Command timed out after 300 seconds."
        return _format_command_result(
            stdout,
            stderr,
            process.returncode if process.returncode is not None else -1,
        )

    if init_error is None and cleanup is not None:
        setattr(run_command_sandboxed, "cleanup", cleanup)  # noqa: B010

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
        )
        return res.returncode == 0
    except Exception:
        return False


def is_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is installed and available in PATH."""
    try:
        res = subprocess.run(
            ["rg", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res.returncode == 0
    except Exception:
        return False
