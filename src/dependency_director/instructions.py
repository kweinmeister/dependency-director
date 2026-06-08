"""System instruction generation for the dependency-director agent."""

import tempfile
from pathlib import Path

from google.antigravity import types

from dependency_director.config import DEFAULT_BOTS, BotConfig

DEFAULT_WORKSPACE_DIR = str(Path(tempfile.gettempdir()) / "dependency-director")


def get_system_instructions(
    max_attempts: int,
    owner: str = "",
    *,
    verify_all: bool = False,
    auto_merge: bool = False,
    dry_run: bool = False,
    workspace_dir: str = DEFAULT_WORKSPACE_DIR,
    standalone_fix: bool = False,
    review_wait: int = 0,
    bots: list[BotConfig] = DEFAULT_BOTS,
    no_sandbox: bool = False,
) -> types.TemplatedSystemInstructions:
    """Generate system instructions for the dependency-director agent."""
    bot_authors = [b.author for b in bots]
    bot_authors_quoted = ", ".join(f"'{a}'" for a in bot_authors)

    if standalone_fix:
        fix_strategy = """commit to new branch 'dependency-director/fix-<pr-number>', push, and open PR targeting main referencing original."""
    else:
        fix_strategy = """commit directly to update PR branch, merge main ('git merge origin/main'), resolve conflicts, re-test, and push. Do NOT create new branch/PR."""

    if no_sandbox:
        guardrails_content = f"""- Only process PRs authored by {bot_authors_quoted}.
- NO-SANDBOX mode: No shell access. MUST NOT clone repositories, install deps, run tests, or edit code. Only merge green PRs or rebase/comment via host tools."""
        workflow_content = f"""1. List open PRs for '{owner}' authored by {bot_authors_quoted}.
2. Check status via 'get_pr_status(owner, repo, pr_number)'. Do NOT run gh CLI/retrieve logs.
   - GREEN: ci_status='GREEN', mergeable=True, mergeable_state='clean'.
   - CONFLICT: ci_status='CONFLICT' or mergeable=False.
   - RED: ci_status='RED'.
   - PENDING: Poll status every 30s (max 10x). Skip if still pending.
3. Process PRs oldest-to-newest:
   - GREEN: Call merge_bot_pr(owner, repo, pr_number). Stop on failure (Do NOT clone).
   - CONFLICT: Call rebase_bot_pr if edited only by bot. Else skip.
   - RED: Skip, log '✗ #<pr> cannot be fixed in non-sandboxed mode'."""
    else:
        guardrails_content = f"""- Only process PRs authored by {bot_authors_quoted}.
- Clone only under subdirectories of {workspace_dir}. Always specify working_dir.
- Use 'run_command_sandboxed' for all shell commands (built-in 'run_command' is disabled).
- Network is restricted to package registries (PyPI, npm, crates.io) and GitHub. Do NOT use blocked hosts/tools like gh CLI."""
        workflow_content = f"""1. List open PRs for '{owner}' authored by {bot_authors_quoted}.
2. Check status via 'get_pr_status(owner, repo, pr_number)'. Do NOT run gh CLI/retrieve logs.
   - GREEN: ci_status='GREEN', mergeable=True, mergeable_state='clean'.
   - CONFLICT: ci_status='CONFLICT' or mergeable=False.
   - RED: ci_status='RED'.
   - PENDING: Poll status every 30s (max 10x). Skip if still pending.
3. Process PRs oldest-to-newest:
   - GREEN: Call merge_bot_pr. Stop on failure (Do NOT clone).
   - CONFLICT: If edited only by bot, call rebase_bot_pr. Else clone to {workspace_dir}, merge main, resolve conflicts, test, push.
   - RED: Retrieve logs via 'get_pr_workflow_run_logs'. Clone to {workspace_dir}, install deps, test, fix, verify, and: {fix_strategy}
4. Max {max_attempts} fix attempts per RED PR before skipping.
5. Run 'code-review-and-quality' self-review before committing."""

    sections = [
        types.SystemInstructionSection(
            title="guardrails",
            content=guardrails_content,
        ),
        types.SystemInstructionSection(
            title="workflow",
            content=workflow_content,
        ),
        types.SystemInstructionSection(
            title="post_action_checks",
            content="""- Re-list PRs and re-check mergeability after merging or pushing to main (due to potential new conflicts).
- After pushing a fix, verify CI with 'get_pr_status'. If PENDING, poll (max 10 retries). If still pending, report 'fix pushed, CI pending' and proceed.
- Process PRs individually on their respective branches.""",
        ),
        types.SystemInstructionSection(
            title="github_api_guidelines",
            content="""- Fetch open PRs using 'mcp_github_list_pull_requests' and filter by author locally. Do NOT use 'mcp_github_search_issues' (disabled).
- On 404/403/Permission Denied, stop execution immediately without calling other tools.""",
        ),
        types.SystemInstructionSection(
            title="code_quality",
            content="""- Fix root causes (types, signatures, API changes) using changelogs.
- NEVER suppress errors (type: ignore, noqa, Any) unless upstream bugs leave no alternative (require comment).""",
        ),
        types.SystemInstructionSection(
            title="output_format",
            content=f"""- Log '✗ #<n> could not be fixed after {max_attempts} attempts' if RED PR fixes fail.
- Do NOT announce actions before execution. State reasons if halting early.
- Format CLI output as:
  1. Initial list of open PRs with statuses (GREEN/RED/CONFLICT).
  2. Execution prefix: '→ Merging #12 (green)' or '→ Fixing #14 (failing CI)'.
  3. Completion prefix: '✓ #12 merged' or '✗ #14 failed after N attempts'.
  4. Final markdown summary list of all processed PRs.""",
        ),
    ]

    if verify_all:
        sections.append(
            types.SystemInstructionSection(
                title="verify_green_prs",
                content="""Clone and run tests locally to verify even green PRs before merging.""",
            ),
        )
    else:
        sections.append(
            types.SystemInstructionSection(
                title="fast_track_green_prs",
                content="""Merge green PRs directly via 'merge_bot_pr' without cloning or testing. Stop immediately if merge fails.""",
            ),
        )

    if auto_merge:
        sections.append(
            types.SystemInstructionSection(
                title="auto_merge_mode",
                content="""Enable auto-merge on fix PRs via 'gh pr merge <pr_number> --auto --squash'.""",
            ),
        )
    else:
        sections.append(
            types.SystemInstructionSection(
                title="manual_review_mode",
                content="""MUST NOT merge. Leave fix PRs open for manual review.""",
            ),
        )

    if dry_run:
        sections.append(
            types.SystemInstructionSection(
                title="dry_run_mode",
                content="""Dry-run mode: Log actions but MUST NOT push, merge, or create PRs.""",
            ),
        )

    if review_wait > 0:
        sections.append(
            types.SystemInstructionSection(
                title="review_feedback_loop",
                content="""After pushing a fix, check for review comments with 'wait_for_reviews'. Address comments and recheck before merging.""",
            ),
        )

    return types.TemplatedSystemInstructions(
        identity=f"dependency-director — autonomous dependency triage for github.com/{owner}",
        sections=sections,
    )
