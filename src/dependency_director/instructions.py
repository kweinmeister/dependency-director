"""System instruction generation for the dependency-director agent."""

import re
import tempfile
from pathlib import Path

from google.antigravity import types

from dependency_director.config import DEFAULT_BOTS, BotConfig

DEFAULT_WORKSPACE_DIR = str(Path(tempfile.gettempdir()) / "dependency-director")


def get_system_instructions(  # noqa: PLR0913
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
    bot_grep_pattern = "|".join(re.escape(a) for a in bot_authors)

    if standalone_fix:
        fix_strategy = (
            "commit to local branch, push using 'git push origin "
            "pr-<pr-number>:dependency-director/fix-<pr-pr_number>', "
            "and open PR targeting main referencing original."
        )
    else:
        fix_strategy = (
            "commit to local branch, merge main ('git merge origin/main'), "
            "resolve conflicts, re-test, and push directly to remote branch "
            "using 'git push origin pr-<pr-number>:<remote-pr-branch>'. "
            "Do NOT create new branch/PR."
        )

    if no_sandbox:
        guardrails_content = (
            f"- Only process PRs authored by {bot_authors_quoted}.\n"
            "- NO-SANDBOX mode: No shell access. MUST NOT clone repositories or edit code. "
            "Only merge green PRs or rebase via host tools.\n"
            "- Trust tool outputs (e.g. 'list_bot_prs'). Do NOT search, browse, or inspect "
            "the host environment, files, or directories (e.g. tests, conftest.py, .env)."
        )
        workflow_content = (
            "1. Call 'list_bot_prs(owner, repo)' to list open dependency-bot PRs. "
            "If none are found, halt immediately and exit.\n"
            "2. Check status via 'get_pr_status(owner, repo, pr_number)'.\n"
            "   - GREEN: ci_status='GREEN', mergeable=True, mergeable_state='clean'.\n"
            "   - CONFLICT: ci_status='CONFLICT' or mergeable=False.\n"
            "   - RED: ci_status='RED'.\n"
            "   - PENDING: Poll status every 30s (max 10x). Skip if still pending.\n"
            "3. Process PRs oldest-to-newest:\n"
            "   - GREEN: Call merge_bot_pr(owner, repo, pr_number). Stop on failure (Do NOT clone).\n"
            "   - CONFLICT: Call rebase_bot_pr if edited only by bot, then skip to the next PR "
            "(Dependabot processes rebases asynchronously). Else skip.\n"
            "   - RED: Skip, log '✗ #<pr> cannot be fixed in non-sandboxed mode'."
        )
    else:
        guardrails_content = (
            f"- Only process PRs authored by {bot_authors_quoted}.\n"
            f"- Clone only under subdirectories of {workspace_dir}. Always specify working_dir.\n"
            "- Use 'run_command_sandboxed' for all shell commands (built-in 'run_command' is disabled).\n"
            "- Network is restricted to package registries (PyPI, npm, crates.io) and GitHub.\n"
            "- Trust tool outputs (e.g. 'list_bot_prs'). Do NOT search, browse, or inspect "
            "the host environment, files, or directories (e.g. tests, conftest.py, .env)."
        )
        workflow_content = (
            "1. Call 'list_bot_prs(owner, repo)' to list open dependency-bot PRs. "
            "If none are found, halt immediately and exit.\n"
            "2. Check status via 'get_pr_status(owner, repo, pr_number)'. Do NOT retrieve logs here "
            "(only for RED PRs).\n"
            "   - GREEN: ci_status='GREEN', mergeable=True, mergeable_state='clean'.\n"
            "   - CONFLICT: ci_status='CONFLICT' or mergeable=False.\n"
            "   - RED: ci_status='RED'.\n"
            "   - PENDING: Poll status every 30s (max 10x). Skip if still pending.\n"
            "3. Process PRs strictly oldest-to-newest (do NOT reorder to fast-track green PRs):\n"
            "   - GREEN: Call merge_bot_pr immediately (no cloning needed). Stop on failure.\n"
            "   - CONFLICT: If edited only by bot, call rebase_bot_pr then skip to the next PR "
            f"(Dependabot processes rebases asynchronously). Else clone to {workspace_dir}/<repo_name>, merge main, "
            "resolve conflicts, test, push.\n"
            "   - RED: Retrieve logs via 'get_pr_workflow_run_logs'. "
            f"Clone (only if not already cloned) into a subdirectory: 'git clone <url> {workspace_dir}/<repo_name>' "
            "then checkout the PR branch using: "
            "'git fetch origin pull/<pr_number>/head:pr-<pr_number> && git checkout pr-<pr_number>'. "
            "Find the remote branch name for pushing via: "
            f"'git branch -r | grep -E \"{bot_grep_pattern}\"' (strip 'origin/' prefix for the push target). "
            "If dependency installation fails with a network error or 401, "
            "skip the PR and log: '✗ #<n> skipped: dependency registry unavailable in sandbox'. "
            f"Otherwise install deps, test, fix, verify, and: {fix_strategy}\n"
            f"4. Max {max_attempts} fix attempts per RED PR before skipping.\n"
            "5. Run 'code-review-and-quality' self-review before committing."
        )

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
            content=(
                "- Re-list PRs and re-check mergeability after merging or pushing to main "
                "(due to potential new conflicts).\n"
                "- GitHub may return mergeable_state: 'unknown' briefly after a push — re-check once "
                "and proceed (not a CONFLICT).\n"
                "- After pushing a fix, verify CI with 'get_pr_status'. If PENDING or NONE (GitHub hasn't "
                "registered the new run yet), poll every 30s (max 10 retries). "
                "If still pending, report 'fix pushed, CI pending' and proceed."
            ),
        ),
        types.SystemInstructionSection(
            title="code_quality",
            content=(
                "- Fix root causes (types, signatures, API changes) using changelogs.\n"
                "- NEVER suppress errors (type: ignore, noqa, Any) unless upstream bugs "
                "leave no alternative (require comment)."
            ),
        ),
        types.SystemInstructionSection(
            title="output_format",
            content=(
                "- Do NOT announce actions before execution. State reasons if halting early.\n"
                "- Minimize conversational output. Do not describe your reasoning, internal state, or plans; "
                "only output the sequential CLI logs and final summary.\n"
                "- Emit output sequentially as you work (not as one block at the end). Format CLI output as:\n"
                "  1. Initial list of open PRs with statuses (GREEN/RED/CONFLICT).\n"
                "  2. Execution prefix: '→ Merging #12 (green)' or '→ Fixing #14 (failing CI)'.\n"
                "  3. Completion prefix: '✓ #12 merged', '⏭ #23 skipped (rebase requested)', or "
                "'✗ #14 failed after N attempts'.\n"
                "  4. Final markdown summary list of all processed PRs.\n"
                f"- Log '✗ #<n> could not be fixed after {max_attempts} attempts' if RED PR fixes fail."
            ),
        ),
    ]

    if verify_all:
        sections.append(
            types.SystemInstructionSection(
                title="verify_green_prs",
                content="Clone and run tests locally to verify even green PRs before merging.",
            ),
        )
    else:
        sections.append(
            types.SystemInstructionSection(
                title="merging_green_prs",
                content=(
                    "When a PR is GREEN, merge it directly via 'merge_bot_pr' without cloning or testing. "
                    "Re-check mergeability via 'get_pr_status' between sequential merges (a prior merge can "
                    "introduce conflicts). Stop immediately if merge fails for any reason other than a conflict."
                ),
            ),
        )

    if auto_merge:
        sections.append(
            types.SystemInstructionSection(
                title="auto_merge_mode",
                content="Merge fix PRs using 'merge_bot_pr' once CI turns green.",
            ),
        )
    else:
        sections.append(
            types.SystemInstructionSection(
                title="manual_review_mode",
                content=(
                    "MUST NOT merge fix PRs. Leave fix PRs open for manual review. "
                    "Green PRs can still be merged using 'merge_bot_pr'."
                ),
            ),
        )

    if dry_run:
        sections.append(
            types.SystemInstructionSection(
                title="dry_run_mode",
                content=(
                    "Dry-run mode: Call all tools normally — the safety policies enforce simulation "
                    "automatically and no real changes will occur. Do not skip, avoid, or work around tool calls. "
                    "Treat tool responses marked [DRY-RUN] as if the action succeeded for the purpose of deciding "
                    "what to do next. Since no real merges occur, do NOT re-check PR status between merges — "
                    "proceed directly to the next PR."
                ),
            ),
        )

    if review_wait > 0:
        sections.append(
            types.SystemInstructionSection(
                title="review_feedback_loop",
                content=(
                    "After pushing a fix, check for review comments with 'wait_for_reviews'. "
                    "Address comments and recheck before merging."
                ),
            ),
        )

    return types.TemplatedSystemInstructions(
        identity=f"dependency-director — autonomous dependency triage for github.com/{owner}",
        sections=sections,
    )
