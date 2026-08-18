"""A stand-in for GitHubClient that serves scripted CI state.

The live harness needs GitHub to report a specific arrangement of red and
green checks. Only the service is faked: the tools, the instructions, the
model, the sandbox, and the git repository underneath are all real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dependency_director.schemas import PullRequest


def check_runs(*names_and_conclusions: tuple[str, str]) -> dict[str, Any]:
    """Build a check-runs payload in the shape the GitHub API returns."""
    return {
        "check_runs": [
            {"name": name, "status": "completed", "conclusion": conclusion}
            for name, conclusion in names_and_conclusions
        ],
    }


class FakeGitHub:
    """Serve scripted pull requests and per-ref check results.

    Checks are keyed by git ref because that is how the real API is addressed:
    a PR's checks are fetched by head SHA and a branch's by branch name, which
    is exactly the distinction the base-attribution rule turns on.
    """

    def __init__(
        self,
        *,
        prs: list[dict[str, Any]],
        checks_by_ref: dict[str, dict[str, Any]],
        branches: list[str],
        default_branch: str = "main",
    ) -> None:
        """Store the scripted repository state and start a fresh call log."""
        self._prs = prs
        self._checks_by_ref = checks_by_ref
        self._branches = branches
        self._default_branch = default_branch
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.created_prs: list[dict[str, Any]] = []
        self.merged_prs: list[int] = []
        self.comments: list[tuple[int, str]] = []
        # Filled in by the harness once the run has built its workspace, so
        # tests can inspect what the agent left on disk.
        self.workspace: Path | None = None
        self.checkout: Path | None = None
        self.bare: Path | None = None

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def call_count(self, name: str) -> int:
        """Count how many times a client method was called."""
        return sum(1 for called, _ in self.calls if called == name)

    def _pr(self, pr_number: int) -> dict[str, Any]:
        for pr in self._prs:
            if pr["number"] == pr_number:
                return pr
        msg = f"No scripted PR #{pr_number}"
        raise KeyError(msg)

    # --- Reads the agent's decision depends on ---

    def _base_ref(self, pr: dict[str, Any]) -> str:
        """Return the branch a scripted PR targets, defaulting to the repo default."""
        return str(pr.get("base_ref") or self._default_branch)

    async def list_open_prs(self, owner: str, repo: str) -> list[PullRequest]:
        """Return the scripted open pull requests."""
        self._record("list_open_prs", owner, repo)
        return [
            PullRequest.model_validate(
                {
                    "number": pr["number"],
                    "title": pr["title"],
                    "user": {"login": pr["author"]},
                    "created_at": "2026-08-01T00:00:00Z",
                    "head": {"sha": pr["head_sha"], "ref": pr["head_ref"]},
                    "base": {"ref": self._base_ref(pr)},
                },
            )
            for pr in self._prs
        ]

    async def get_pr_details(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Return PR metadata, including the head SHA its checks are keyed by."""
        self._record("get_pr_details", owner, repo, pr_number)
        pr = self._pr(pr_number)
        return {
            "number": pr_number,
            "title": pr["title"],
            "head": {"sha": pr["head_sha"], "ref": pr["head_ref"]},
            "base": {"ref": self._base_ref(pr)},
            "mergeable": True,
            "mergeable_state": "unstable",
        }

    async def get_commit_check_runs(self, owner: str, repo: str, ref: str) -> dict[str, Any]:
        """Return the scripted check runs for a ref, or none if unscripted."""
        self._record("get_commit_check_runs", owner, repo, ref)
        return self._checks_by_ref.get(ref, {"check_runs": []})

    async def get_commit_status(self, owner: str, repo: str, ref: str) -> dict[str, Any]:
        """Report no legacy commit statuses; this repository uses check runs."""
        self._record("get_commit_status", owner, repo, ref)
        return {"statuses": []}

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Return the scripted default branch."""
        self._record("get_default_branch", owner, repo)
        return self._default_branch

    async def list_branches(self, owner: str, repo: str) -> list[str]:
        """Return the scripted branch names."""
        self._record("list_branches", owner, repo)
        return list(self._branches)

    async def find_open_pr_for_head(self, owner: str, repo: str, head_branch: str) -> PullRequest | None:
        """Return the open PR on a head branch, counting any opened this run.

        Including ``created_prs`` keeps the fake honest about GitHub's one-open-
        PR-per-head rule, so a run that opens a PR and then asks for the same
        head again gets the answer the real API would give.
        """
        self._record("find_open_pr_for_head", owner, repo, head_branch)
        branch = head_branch.split(":")[-1]

        for pr in self._prs:
            if pr["head_ref"] == branch:
                return PullRequest.model_validate(
                    {
                        "number": pr["number"],
                        "title": pr["title"],
                        "html_url": f"https://github.com/{owner}/{repo}/pull/{pr['number']}",
                        "head": {"ref": branch},
                        "base": {"ref": self._base_ref(pr)},
                    },
                )

        for i, created in enumerate(self.created_prs, start=1):
            if str(created["head"]).split(":")[-1] == branch:
                number = 900 + i
                return PullRequest.model_validate(
                    {
                        "number": number,
                        "title": created["title"],
                        "html_url": f"https://github.com/{owner}/{repo}/pull/{number}",
                        "head": {"ref": branch},
                        "base": {"ref": created["base"]},
                    },
                )
        return None

    async def get_pr_author(self, owner: str, repo: str, pr_number: int) -> str:
        """Return the scripted PR author."""
        self._record("get_pr_author", owner, repo, pr_number)
        return str(self._pr(pr_number)["author"])

    # --- Reads that must not crash, but carry no signal for these tests ---

    async def get_workflow_runs_for_commit(self, owner: str, repo: str, ref: str) -> dict[str, Any]:
        """Report no workflow runs, pushing the agent to reproduce locally."""
        self._record("get_workflow_runs_for_commit", owner, repo, ref)
        return {"workflow_runs": []}

    async def get_workflow_run_jobs(self, owner: str, repo: str, run_id: int) -> dict[str, Any]:
        """Report no jobs; no run is ever scripted."""
        self._record("get_workflow_run_jobs", owner, repo, run_id)
        return {"jobs": []}

    async def get_job_logs(self, owner: str, repo: str, job_id: int) -> str:
        """Report no logs; no job is ever scripted."""
        self._record("get_job_logs", owner, repo, job_id)
        return ""

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Return the scripted diff, if the scenario supplied one."""
        self._record("get_pr_diff", owner, repo, pr_number)
        return str(self._pr(pr_number).get("diff", ""))

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Return the scripted changed-file list."""
        self._record("get_pr_files", owner, repo, pr_number)
        return list(self._pr(pr_number).get("files", []))

    async def get_pr_reviews(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Report no reviews."""
        self._record("get_pr_reviews", owner, repo, pr_number)
        return []

    async def list_pr_commits(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Report no commits."""
        self._record("list_pr_commits", owner, repo, pr_number)
        return []

    async def list_commits(self, owner: str, repo: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Report no commits."""
        self._record("list_commits", owner, repo)
        return []

    async def get_commit(self, owner: str, repo: str, sha: str) -> dict[str, Any]:
        """Report an empty commit record."""
        self._record("get_commit", owner, repo, sha)
        return {}

    async def get_file_contents(self, owner: str, repo: str, path: str, **kwargs: Any) -> str:
        """Report an empty file; the agent has the real clone on disk."""
        self._record("get_file_contents", owner, repo, path)
        return ""

    # --- Writes, recorded so tests can assert on them ---

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
    ) -> dict[str, Any]:
        """Record a PR creation and return a plausible response."""
        self._record("create_pull_request", owner, repo, title, head, base)
        self.created_prs.append({"title": title, "head": head, "base": base, "body": body})
        number = 900 + len(self.created_prs)
        return {"number": number, "html_url": f"https://github.com/{owner}/{repo}/pull/{number}"}

    async def merge_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Record a merge and report success."""
        self._record("merge_pr", owner, repo, pr_number)
        self.merged_prs.append(pr_number)
        return {"merged": True, "message": "Pull Request successfully merged"}

    async def comment_on_pr(self, owner: str, repo: str, pr_number: int, body: str) -> dict[str, Any]:
        """Record a comment."""
        self._record("comment_on_pr", owner, repo, pr_number, body)
        self.comments.append((pr_number, body))
        return {}

    async def close(self) -> None:
        """Match GitHubClient's shutdown interface."""
        self._record("close")
