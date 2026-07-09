"""System instruction generation for the dependency-director agent."""

from google.antigravity import types

from dependency_director.config import DEFAULT_BOTS, BotConfig


def get_system_instructions(  # noqa: PLR0913
    max_attempts: int,
    *,
    verify_all: bool = False,
    auto_merge: bool = False,
    dry_run: bool = False,
    standalone_fix: bool = False,
    review_wait: int = 0,
    bots: list[BotConfig] = DEFAULT_BOTS,
    no_sandbox: bool = False,
) -> types.TemplatedSystemInstructions:
    """Generate system instructions for the dependency-director agent.

    System instructions are structured for optimal implicit caching:
    stable sections appear first, flag-dependent sections next, and
    mode-specific sections (guardrails/workflow) last. Repo-specific
    details (workspace_dir, owner) are passed via the user prompt
    rather than system instructions so the instruction prefix stays
    identical across repos in multi-repo runs.
    """
    bot_authors = [b.author for b in bots]
    bot_authors_quoted = ", ".join(f"'{a}'" for a in bot_authors)

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
            "- Clone only under subdirectories of the workspace directory (provided in the prompt). "
            "Always specify working_dir.\n"
            "- Use 'run_command_sandboxed' for all shell commands (built-in 'run_command' is disabled).\n"
            "- To set environment variables for a command, use 'env KEY=val cmd args...', "
            "not 'KEY=val cmd args...' (the latter is shell syntax that srt's argv mode does not support).\n"
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
            "(Dependabot processes rebases asynchronously). Else clone to <workspace_dir>/<repo_name>, merge main, "
            "resolve conflicts, test, push.\n"
            "   - RED: Retrieve logs via 'get_pr_workflow_run_logs'. "
            "Then call 'get_pr_diff(owner, repo, pr_number)' and 'get_pr_files(owner, repo, pr_number)' "
            "to identify changed files and understand the scope of the update before cloning. "
            "Clone (only if not already cloned) into a subdirectory: 'git clone <url> <workspace_dir>/<repo_name>' "
            "then checkout the PR branch using two separate run_command_sandboxed calls: "
            "first 'git fetch origin pull/<pr_number>/head:pr-<pr_number>', "
            "then 'git checkout pr-<pr_number>'. "
            "Find the remote branch name for pushing by calling 'list_branches(owner, repo)' "
            f"and matching against the configured bot authors ({bot_authors_quoted}), "
            "then strip 'origin/' prefix for the push target. "
            "If dependency installation fails with a network error or 401, "
            "skip the PR and log: '✗ #<n> skipped: dependency registry unavailable in sandbox'. "
            f"Otherwise install deps, test, fix, verify, and: {fix_strategy}\n"
            f"4. Max {max_attempts} fix attempts per RED PR before skipping.\n"
            "5. Before committing, briefly self-review: no suppressed errors (type: ignore, noqa), "
            "no debug artifacts, no unrelated changes, commit message is concise and descriptive."
        )

    # Sections are ordered for optimal implicit caching: stable content first,
    # then flag-dependent sections, then parameter/mode-specific sections last.
    # This maximizes the common prefix across different repo runs.
    sections = [
        # --- Stable sections (identical across all runs) ---
        types.SystemInstructionSection(
            title="code_quality",
            content=(
                "- Fix root causes (types, signatures, API changes) using changelogs.\n"
                "- NEVER suppress errors (type: ignore, noqa, Any) unless upstream bugs "
                "leave no alternative (require comment)."
            ),
        ),
        types.SystemInstructionSection(
            title="post_action_checks",
            content=(
                "- After merging a PR or pushing a fix, use 'wait_for_ci(owner, repo, pr_number)' "
                "on the next PR to verify its CI status before acting on it. "
                "This tool polls automatically with backoff — do NOT use 'sleep', "
                "do NOT call 'get_pr_status' in a loop, and do NOT poll manually.\n"
                "- If wait_for_ci reports CONFLICT after a merge, rebase or fix as appropriate.\n"
                "- If still pending after timeout, report 'CI pending' and proceed to the next PR."
            ),
        ),
    ]

    # --- Flag-dependent sections (same across repos within a single run) ---
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
                    "Between sequential merges, call 'wait_for_ci(owner, repo, next_pr_number)' on the next PR "
                    "to re-check its status (a prior merge can introduce conflicts or reset CI). "
                    "Stop immediately if merge fails for any reason other than a conflict."
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

    # --- Parameter-dependent sections (output_format embeds max_attempts) ---
    sections.append(
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
    )

    # --- Mode-specific sections (guardrails and workflow go last) ---
    sections.append(
        types.SystemInstructionSection(
            title="guardrails",
            content=guardrails_content,
        ),
    )
    sections.append(
        types.SystemInstructionSection(
            title="workflow",
            content=workflow_content,
        ),
    )

    return types.TemplatedSystemInstructions(
        identity="dependency-director — autonomous dependency triage agent",
        sections=sections,
    )
