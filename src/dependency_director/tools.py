"""Agent tool definitions, factories, and CI/workflow helpers for dependency-director."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from http import HTTPStatus
from typing import NamedTuple

import httpx

from dependency_director.config import (
    DEFAULT_LOG_LIMITS,
    DEFAULT_OUTPUT_LIMITS,
    PASSING_RUN_CONCLUSIONS,
    BotConfig,
    LogLimits,
    OutputLimits,
)
from dependency_director.github_client import (
    GitHubClient,
    GitHubClientError,
    validate_repo_params,
)
from dependency_director.sandbox import truncate_middle
from dependency_director.schemas import (
    BotPrList,
    BotPrSummary,
    BranchCiStatus,
    CheckRunsResponse,
    CheckSummary,
    CommitResponse,
    CommitStatusResponse,
    FileContents,
    OpenPrSummary,
    PrStatus,
    PullRequest,
    PullRequestReview,
    WorkflowJob,
    WorkflowJobsResponse,
    WorkflowRun,
    WorkflowRunsResponse,
    json_payload,
    json_payload_list,
    summarize_checks,
)


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
    find_open_pr_for_branch: ToolFn
    get_branch_ci_status: ToolFn
    get_commit_details: ToolFn
    get_file_contents: ToolFn
    get_pr_body: ToolFn
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
    for raw_commit in await client.list_pr_commits(owner, repo, pr_number):
        login = CommitResponse.model_validate(raw_commit).github_login
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
        validate_repo_params(owner, repo)

        # A branch that already carries a PR is the normal state of a second
        # run over a fix that has not merged yet. Reporting that PR is the
        # correct outcome; asking GitHub to open a duplicate only earns a 422.
        existing = await client.find_open_pr_for_head(owner, repo, head_branch)
        if existing is not None:
            return _existing_pr_message(owner, repo, head_branch, existing)

        base = base_branch or await client.get_default_branch(owner, repo)

        if dry_run:
            return f"[DRY-RUN] Would have opened a PR '{title}' from {head_branch} into {base} in {owner}/{repo}."

        try:
            result = await client.create_pull_request(
                owner,
                repo,
                title=title,
                head=head_branch,
                base=base,
                body=body,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != HTTPStatus.UNPROCESSABLE_ENTITY:
                raise
            # The pre-check can lose a race, and 422 covers several unrelated
            # causes, so re-ask before blaming anything on a duplicate.
            raced = await client.find_open_pr_for_head(owner, repo, head_branch)
            if raced is not None:
                return _existing_pr_message(owner, repo, head_branch, raced)
            return f"GitHub refused (422) to open a PR from {head_branch} into {base} in {owner}/{repo}: {exc}"
        return f"Opened PR #{result.get('number')} in {owner}/{repo}: {result.get('html_url', '')}"

    return create_pr


def _existing_pr_message(owner: str, repo: str, head_branch: str, pr: PullRequest) -> str:
    """Describe the pull request a branch already carries, and what not to do to it."""
    return (
        f"PR #{pr.number} is already open for {head_branch} in {owner}/{repo}: {pr.html_url}. "
        "No new PR was opened. That branch already carries the fix and is waiting on "
        "review — do NOT rebuild, re-push, or force-push it."
    )


def _make_find_open_pr_for_branch(client: GitHubClient) -> ToolFn:
    async def find_open_pr_for_branch(owner: str, repo: str, branch: str) -> str:
        """Report whether a branch already has an open pull request.

        Call this before rebuilding any branch whose name this agent derives
        rather than reads — a base fix branch above all. An open PR from an
        earlier run means the work is done and awaiting review, so redoing it
        wastes a clone and overwrites the branch under review.

        Args:
            owner: The owner of the repository.
            repo: The repository name.
            branch: The head branch to look up, without an 'owner:' prefix.

        """
        found = await client.find_open_pr_for_head(owner, repo, branch)
        if found is None:
            return f"No open pull request has '{branch}' as its head branch in {owner}/{repo}."
        return json_payload(
            OpenPrSummary(
                number=found.number,
                title=found.title,
                html_url=found.html_url,
                base_ref=found.base.ref,
            ),
        )

    return find_open_pr_for_branch


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

        baseline_reviews = [
            PullRequestReview.model_validate(r) for r in await client.get_pr_reviews(owner, repo, pr_number)
        ]
        baseline_keys = {r.dedup_key(idx) for idx, r in enumerate(baseline_reviews)}

        while elapsed < timeout_seconds:
            reviews = [PullRequestReview.model_validate(r) for r in await client.get_pr_reviews(owner, repo, pr_number)]
            new_reviews = [r for idx, r in enumerate(reviews) if r.dedup_key(idx) not in baseline_keys]

            if new_reviews:
                lines = []
                for r in new_reviews:
                    body = r.body.strip()
                    if body:
                        lines.append(f"- **{r.author}** ({r.state}): {body}")
                    else:
                        lines.append(f"- **{r.author}** ({r.state})")
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


DEFAULT_BRANCH_CI_DELAYS: tuple[float, ...] = tuple(float(min(10 * (i + 1), 30)) for i in range(10))


def _make_get_branch_ci_status(
    client: GitHubClient,
    delays: Sequence[float] | None = None,
) -> ToolFn:
    """Create a get_branch_ci_status tool that polls pending checks and caches settled verdicts."""
    # A base branch is checked once and the verdict reused for every red PR
    # sharing it. The cache lives in this closure, which is built once per
    # repository run, so its lifetime is exactly one run and nothing has to
    # invalidate it. In-flight task tracking coalesces concurrent checks for the
    # same branch without blocking checks for other branches or repositories.
    cache: dict[tuple[str, str, str], str] = {}
    in_flight: dict[tuple[str, str, str], asyncio.Task[str]] = {}
    retry_delays = DEFAULT_BRANCH_CI_DELAYS if delays is None else delays

    async def _fetch_and_poll(owner: str, repo: str, target: str, key: tuple[str, str, str]) -> str:
        try:
            # GitHub resolves a branch name as a commit ref, so this needs no SHA lookup.
            # An exception leaves the cache untouched: a transient API failure
            # must not become this run's permanent answer for the branch.
            ci_status, checks = await _checks_for_ref(client, owner, repo, target)
            if ci_status == "PENDING":
                for delay in retry_delays:
                    await asyncio.sleep(delay)
                    ci_status, checks = await _checks_for_ref(client, owner, repo, target)
                    if ci_status != "PENDING":
                        break

            payload = json_payload(BranchCiStatus(branch=target, ci_status=ci_status, checks=checks))
            # Cache settled verdicts only. An in-progress (PENDING) verdict must
            # not become this run's permanent answer if polling timed out.
            if ci_status != "PENDING":
                cache[key] = payload
            return payload
        finally:
            in_flight.pop(key, None)

    async def get_branch_ci_status(owner: str, repo: str, branch: str = "") -> str:
        """Report CI health for a branch, without cloning it.

        Poll automatically with backoff if CI is still running.

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
        if key in cache:
            return cache[key]
        if key in in_flight:
            return await in_flight[key]

        task = asyncio.create_task(_fetch_and_poll(owner, repo, target, key))
        in_flight[key] = task
        return await task

    return get_branch_ci_status


def _classify_pr_status(pr: PullRequest) -> str:
    """Classify pull request mergeability into CLEAN, CONFLICT, or UNKNOWN."""
    if pr.mergeable is False or pr.mergeable_state in ("dirty", "conflict"):
        return "CONFLICT"
    if pr.mergeable is True and pr.mergeable_state in ("clean", "has_hooks", "unstable"):
        return "CLEAN"
    return "UNKNOWN"


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
    merge_status = _classify_pr_status(pr)

    summary = PrStatus(
        pr_number=pr_number,
        title=pr.title,
        head_sha=pr.head.sha,
        head_ref=pr.head.ref,
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
        validate_repo_params(owner, repo)
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


def _tail_lines(text: str, n: int) -> str:
    """Return the last ``n`` lines of ``text``."""
    return "\n".join(text.splitlines()[-n:])


async def _fetch_failed_logs(
    client: GitHubClient,
    owner: str,
    repo: str,
    runs: list[WorkflowRun],
    log_limits: LogLimits,
) -> tuple[list[str], int]:
    """Fetch formatted log tails for failed jobs across candidate workflow runs."""
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

    return failed_jobs_logs, total_failed


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

        failed_jobs_logs, total_failed = await _fetch_failed_logs(client, owner, repo, runs, log_limits)

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
    tail = _tail_lines(log_text, tail_lines)
    return f"{header}\n{tail}\n"


def _make_list_bot_prs(client: GitHubClient, bots: list[BotConfig]) -> ToolFn:
    async def list_bot_prs(owner: str, repo: str) -> str:
        """List open dependency-bot pull requests for a repository.

        Returns a compact JSON list with only the essential fields: number,
        title, author, created_at, base_ref (the branch the PR targets), and
        head_ref (the remote branch name for checkout/push). Sorted oldest
        first. Only PRs authored by configured bots are included.
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
                head_ref=pr.head.ref,
            )
            for pr in all_prs
            if pr.author in allowed_authors
        ]
        return json_payload(BotPrList(bot_prs=bot_prs, count=len(bot_prs)))

    return list_bot_prs


def _make_get_pr_body(
    client: GitHubClient,
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
) -> ToolFn:
    async def get_pr_body(owner: str, repo: str, pr_number: int) -> str:
        """Fetch the body/description (changelog and release notes) of a pull request.

        Use this on demand when diagnosing a RED dependency PR to inspect
        breaking changes, migration notes, or changelog entries without
        paying the token cost on every open PR.
        """
        body = await client.get_pr_body(owner, repo, pr_number)
        return truncate_middle(body, output_limits)

    return get_pr_body


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
        return json_payload_list(files)

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
        return json_payload_list(commits)

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
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
    branch_ci_delays: Sequence[float] | None = None,
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
        find_open_pr_for_branch=_make_find_open_pr_for_branch(client),
        get_branch_ci_status=_make_get_branch_ci_status(client, delays=branch_ci_delays),
        get_commit_details=legacy.get_commit_details,
        get_file_contents=legacy.get_file_contents,
        get_pr_body=_make_get_pr_body(client, output_limits),
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
