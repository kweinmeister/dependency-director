"""Tests for system instructions generation in dependency-director."""

from typing import Any

import pytest
from google.antigravity import types

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.instructions import (
    get_system_instructions,
)


def _get_instructions(
    max_attempts: int = 3,
    bots: list[BotConfig] = DEFAULT_BOTS,
    **kwargs: Any,
) -> types.TemplatedSystemInstructions:
    return get_system_instructions(
        max_attempts=max_attempts,
        bots=bots,
        **kwargs,
    )


@pytest.fixture
def instructions() -> types.TemplatedSystemInstructions:
    """Fixture to return a default templated system instructions object for testing."""
    return _get_instructions()


@pytest.fixture
def no_sandbox_instructions() -> types.TemplatedSystemInstructions:
    """Fixture to return system instructions configured for no-sandbox mode."""
    return _get_instructions(no_sandbox=True)


@pytest.fixture
def fix_base_instructions() -> types.TemplatedSystemInstructions:
    """Fixture to return system instructions configured with base branch fixing enabled."""
    return _get_instructions(fix_base=True)


@pytest.fixture
def standalone_fix_instructions() -> types.TemplatedSystemInstructions:
    """Fixture to return system instructions configured with standalone fix PR strategy."""
    return _get_instructions(standalone_fix=True)


def _section_content(inst: types.TemplatedSystemInstructions, title: str) -> str:
    return next((s.content for s in inst.sections if s.title == title), "")


def _section_titles(inst: types.TemplatedSystemInstructions) -> set[str]:
    return {s.title for s in inst.sections}


# --- Structure ---


def test_returns_templated(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify that get_system_instructions returns a TemplatedSystemInstructions object."""
    assert isinstance(instructions, types.TemplatedSystemInstructions)
    assert instructions.identity is not None
    assert "dependency-director" in instructions.identity


REQUIRED_SECTIONS = [
    "guardrails",
    "workflow",
    "post_action_checks",
    "code_quality",
    "output_format",
]


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_has_required_section(
    instructions: types.TemplatedSystemInstructions,
    section: str,
) -> None:
    """Verify that the system instructions contain all mandatory section headers."""
    assert section in _section_titles(instructions)


# --- Workflow content ---


@pytest.mark.parametrize(
    "expected",
    [
        "get_pr_status",
        "get_pr_workflow_run_logs",
        "merge_bot_pr",
        "rebase_bot_pr",
        "list_bot_prs",
        "self-review",
    ],
)
def test_workflow_contains(
    instructions: types.TemplatedSystemInstructions,
    expected: str,
) -> None:
    """Verify that the primary agent workflow instructions are generated correctly."""
    content = _section_content(instructions, "workflow")
    assert expected.lower() in content.lower()


def test_no_gh_cli_diagnostic_commands_in_workflow(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify that potentially dangerous gh-cli diagnostic commands are excluded from instructions."""
    content = _section_content(instructions, "workflow")
    assert "gh pr checks" not in content
    assert "gh api" not in content


def test_clone_only_for_red(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify repository cloning instructions are only generated when PR is failing."""
    content = _section_content(instructions, "workflow")
    assert any(phrase in content.lower() for phrase in ("do not clone", "no cloning needed"))


def test_conflict_rebase_skips_to_next_pr(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify that instructions suggest skipping to the next PR if rebase conflict occurs."""
    content = _section_content(instructions, "workflow")
    assert "rebase_bot_pr" in content
    assert "asynchronously" in content


# --- Code quality ---


@pytest.mark.parametrize(
    "expected",
    [
        "root cause",
        "NEVER suppress",
    ],
)
def test_code_quality_contains(
    instructions: types.TemplatedSystemInstructions,
    expected: str,
) -> None:
    """Verify that code quality and review rules are included in the generated instructions."""
    content = _section_content(instructions, "code_quality")
    assert expected in content


# --- Post-action checks ---


def test_post_action_checks(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify that post-action checking steps are explicitly detailed in instructions."""
    content = _section_content(instructions, "post_action_checks")
    assert "wait_for_ci" in content
    assert "do NOT" in content
    assert "poll" in content.lower()
    assert "CONFLICT" in content


def test_post_action_owner_not_leaked_as_template_literal(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify that repository owner variables are safely formatted in the instructions."""
    content = _section_content(instructions, "post_action_checks")
    assert "{owner}" not in content


# --- Output format ---


def test_output_format(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify that rules regarding response and output format are included in the instructions."""
    content = _section_content(instructions, "output_format")
    assert "GREEN" in content
    assert "RED" in content
    assert str(3) in content


def test_output_format_has_a_state_for_no_fix_attempted(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify there is a terminal state for a RED PR nobody tried to fix.

    With only 'merged', 'skipped' and 'failed after N attempts' available, a
    PR whose failure never reproduced gets reported as N exhausted attempts —
    a claim about work that did not happen.
    """
    content = _section_content(instructions, "output_format")
    assert "⚠" in content
    assert "not fixed" in content.lower()


def test_output_format_reports_attempts_actually_made(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify the failure line reports real attempt counts rather than always the maximum."""
    content = _section_content(instructions, "output_format")
    assert "actually made" in content.lower()


def test_workflow_handles_a_failure_that_does_not_reproduce(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify the agent is told what to do when local tests pass on a RED PR.

    Without a branch for this case the agent invents one, which is how a PR
    that passed locally on the first try was reported as three failed fixes.
    """
    content = _section_content(instructions, "workflow")
    assert "reproduce" in content.lower()


# --- max_attempts ---


@pytest.mark.parametrize("value", [1, 3, 7, 10])
def test_max_attempts_appears_in_workflow(value: int) -> None:
    """Verify that the max fix attempts config value is correctly formatted into the workflow instructions."""
    inst = get_system_instructions(max_attempts=value)
    assert str(value) in _section_content(inst, "workflow")


# --- identity ---


def test_identity_is_generic(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify that the agent identity is generic and does not contain repo-specific info."""
    assert instructions.identity is not None
    assert "dependency-director" in instructions.identity
    assert "autonomous" in instructions.identity


# --- workspace_dir in prompt (not system instructions) ---


def test_workspace_dir_uses_placeholder(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify that sandbox workflow references <workspace_dir> placeholder instead of a specific path."""
    workflow = _section_content(instructions, "workflow")
    guardrails = _section_content(instructions, "guardrails")
    assert "<workspace_dir>" in workflow
    assert "workspace directory" in guardrails


def test_workspace_dir_not_hardcoded(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify no hardcoded temp paths appear in system instructions."""
    all_content = " ".join(_section_content(instructions, s.title) for s in instructions.sections)
    assert "/tmp/" not in all_content
    assert "dependency-director-" not in all_content


# --- Conditional sections (flag toggles) ---


@pytest.mark.parametrize(
    ("flags", "present", "absent"),
    [
        ({"auto_merge": True}, "auto_merge_mode", "manual_review_mode"),
        ({"auto_merge": False}, "manual_review_mode", "auto_merge_mode"),
        ({"verify_all": True}, "verify_green_prs", "merging_green_prs"),
        ({"verify_all": False}, "merging_green_prs", "verify_green_prs"),
        ({"dry_run": True}, "dry_run_mode", None),
        ({"review_wait": 5}, "review_feedback_loop", None),
    ],
)
def test_conditional_section_present(
    flags: dict[str, Any],
    present: str,
    absent: str | None,
) -> None:
    """Verify conditional sections are present in instructions when their options are active."""
    inst = _get_instructions(**flags)
    titles = _section_titles(inst)
    assert present in titles
    if absent:
        assert absent not in titles


@pytest.mark.parametrize(
    ("flags", "absent"),
    [
        ({"dry_run": False}, "dry_run_mode"),
        ({"review_wait": 0}, "review_feedback_loop"),
        ({"review_wait": -1}, "review_feedback_loop"),
    ],
)
def test_conditional_section_absent_when_disabled(
    flags: dict[str, Any],
    absent: str,
) -> None:
    """Verify conditional sections are omitted when their corresponding flags are disabled."""
    inst = _get_instructions(**flags)
    assert absent not in _section_titles(inst)


# --- Conditional section content ---


def test_auto_merge_content() -> None:
    """Verify system instructions contain auto-merge logic when enabled."""
    inst = _get_instructions(auto_merge=True)
    content = _section_content(inst, "auto_merge_mode")
    assert "merge_bot_pr" in content
    assert "green" in content


def test_manual_review_content() -> None:
    """Verify system instructions contain manual review workflows when enabled."""
    inst = _get_instructions(auto_merge=False)
    content = _section_content(inst, "manual_review_mode")
    assert "MUST NOT merge fix PRs" in content


def test_dry_run_content() -> None:
    """Verify system instructions include dry-run directives when enabled."""
    inst = _get_instructions(dry_run=True)
    content = _section_content(inst, "dry_run_mode")
    assert "safety policies enforce simulation" in content
    assert "Do not skip" in content
    assert "[DRY-RUN]" in content
    assert "do NOT re-check PR status between merges" in content


def test_verify_all_content() -> None:
    """Verify system instructions include full verification workflows when enabled."""
    inst = _get_instructions(verify_all=True)
    content = _section_content(inst, "verify_green_prs")
    assert "clone" in content.lower()


def test_fast_track_content() -> None:
    """Verify system instructions include fast-track workflows when active."""
    inst = _get_instructions(verify_all=False)
    content = _section_content(inst, "merging_green_prs")
    assert "merge_bot_pr" in content


def test_review_wait_content() -> None:
    """Verify system instructions detail wait behavior for reviews when review_wait is non-zero."""
    inst = _get_instructions(review_wait=5)
    content = _section_content(inst, "review_feedback_loop")
    assert "wait_for_reviews" in content


# --- Fix strategy ---


def test_default_pushes_to_original_branch(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify instructions command pushing fixes back to the original branch by default."""
    content = _section_content(instructions, "workflow")
    assert "directly" in content.lower()
    assert "dependency-director/fix-" not in content


def test_default_includes_merge_from_main(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify instructions command merging main branch before attempting fixes by default."""
    content = _section_content(instructions, "workflow")
    assert "git merge origin/main" in content


def test_standalone_fix_creates_new_branch(
    standalone_fix_instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify instructions specify creating a new branch when standalone-fix is active."""
    content = _section_content(standalone_fix_instructions, "workflow")
    assert "dependency-director/fix-" in content


def test_standalone_fix_branch_placeholder_is_substitutable(
    standalone_fix_instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify the fix branch uses the same placeholder as the source branch.

    'fix-<pr-pr_number>' is not a placeholder the agent can fill in, so it
    ends up in the branch name verbatim.
    """
    content = _section_content(standalone_fix_instructions, "workflow")
    assert "dependency-director/fix-<pr-number>" in content
    assert "pr-pr_number" not in content


def test_standalone_fix_names_the_pr_creation_tool(
    standalone_fix_instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify the standalone strategy points at the tool that opens the PR.

    Pushing a branch nobody opens a PR for leaves the fix invisible.
    """
    content = _section_content(standalone_fix_instructions, "workflow")
    assert "create_pr" in content


# --- Flag interactions ---


def test_all_flags_on() -> None:
    """Verify instructions are generated correctly with all feature flags enabled."""
    inst = get_system_instructions(
        max_attempts=5,
        verify_all=True,
        auto_merge=True,
        dry_run=True,
        standalone_fix=True,
        review_wait=10,
    )
    titles = _section_titles(inst)
    assert "verify_green_prs" in titles
    assert "auto_merge_mode" in titles
    assert "dry_run_mode" in titles
    assert "review_feedback_loop" in titles
    assert inst.identity is not None
    assert "dependency-director" in inst.identity
    assert "workspace directory" in _section_content(inst, "guardrails")
    assert "dependency-director/fix-" in _section_content(inst, "workflow")
    assert "5" in _section_content(inst, "workflow")


def test_all_flags_off() -> None:
    """Verify instructions are generated correctly with all feature flags disabled."""
    inst = get_system_instructions(
        max_attempts=3,
        verify_all=False,
        auto_merge=False,
        dry_run=False,
        standalone_fix=False,
        review_wait=0,
    )
    titles = _section_titles(inst)
    assert "merging_green_prs" in titles
    assert "manual_review_mode" in titles
    assert "dry_run_mode" not in titles
    assert "review_feedback_loop" not in titles


# --- Multi-bot support ---


def test_bot_authors_in_guardrails(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify the allowed list of bot authors is defined in instruction guardrails."""
    content = _section_content(instructions, "guardrails")
    assert "dependabot[bot]" in content
    assert "renovate[bot]" in content


def test_bot_prs_tool_in_workflow(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify that instructions list the bot PRs tool in the allowed tools section."""
    content = _section_content(instructions, "workflow")
    assert "list_bot_prs" in content


def test_custom_bots_in_instructions() -> None:
    """Verify custom bot configurations are correctly added to the system instructions."""
    custom = [BotConfig(author="my-bot[bot]", rebase_command="@my-bot rebase")]
    inst = get_system_instructions(max_attempts=3, bots=custom)
    content = _section_content(inst, "guardrails")
    assert "my-bot[bot]" in content
    assert "dependabot[bot]" not in content


def test_no_sandbox_instructions(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """Verify instructions explicitly flag that sandboxing is disabled when configured."""
    guardrails = _section_content(no_sandbox_instructions, "guardrails")
    assert "NO-SANDBOX mode" in guardrails
    assert "MUST NOT clone repositories" in guardrails

    workflow = _section_content(no_sandbox_instructions, "workflow")
    assert "Do NOT clone" in workflow
    assert "cannot be fixed in non-sandboxed mode" in workflow


def test_no_sandbox_instructions_no_shell_access(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """No-sandbox guardrails must state that no shell access is available."""
    guardrails = _section_content(no_sandbox_instructions, "guardrails")
    assert "No shell access" in guardrails


def test_no_sandbox_instructions_no_run_command_reference(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """No-sandbox instructions must NOT reference run_command_sandboxed since the tool is not registered."""
    all_content = " ".join(_section_content(no_sandbox_instructions, s.title) for s in no_sandbox_instructions.sections)
    assert "run_command_sandboxed" not in all_content


def test_sandbox_instructions_reference_run_command(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Sandbox mode instructions MUST reference run_command_sandboxed since the tool IS registered."""
    guardrails = _section_content(instructions, "guardrails")
    assert "run_command_sandboxed" in guardrails


def test_no_sandbox_workflow_skips_red_prs(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """No-sandbox workflow must skip RED PRs (no local fix attempts)."""
    workflow = _section_content(no_sandbox_instructions, "workflow")
    # Should not reference cloning or installing deps for RED PRs
    assert "clone" not in workflow.lower() or "Do NOT clone" in workflow
    assert "install deps" not in workflow.lower()


def test_no_sandbox_workflow_still_has_merge_and_rebase(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """No-sandbox workflow must still reference merge and rebase tools."""
    workflow = _section_content(no_sandbox_instructions, "workflow")
    assert "merge_bot_pr" in workflow
    assert "rebase_bot_pr" in workflow
    assert "get_pr_status" in workflow


def test_halt_on_no_prs(instructions: types.TemplatedSystemInstructions) -> None:
    """System instructions must direct the agent to halt if no PRs are found."""
    workflow = _section_content(instructions, "workflow")
    assert "halt" in workflow
    assert "exit" in workflow


def test_trust_tool_outputs_no_host_inspection(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """System instructions must direct the agent to trust tool outputs and not inspect host environment/files."""
    guardrails = _section_content(instructions, "guardrails")
    assert "Trust tool outputs" in guardrails
    assert "Do NOT search, browse, or inspect the host environment" in guardrails


def test_guardrails_require_absolute_paths_for_file_tools(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Guardrails must require absolute paths under workspace_dir for file tools."""
    guardrails = _section_content(instructions, "guardrails")
    assert "view_file" in guardrails
    assert "absolute paths" in guardrails


def test_minimize_conversational_output(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """System instructions must direct the agent to minimize conversational output."""
    output_format = _section_content(instructions, "output_format")
    assert "minimize" in output_format.lower()
    assert "output" in output_format.lower()


def test_red_pr_branch_lookup_uses_list_branches() -> None:
    """Branch lookup uses list_branches host tool, not shell pipe patterns."""
    custom_bots = [
        BotConfig(author="custom-bot[bot]", rebase_command="@custom-bot rebase"),
        BotConfig(author="another-bot[bot]", rebase_command="@another-bot rebase"),
    ]
    inst = _get_instructions(bots=custom_bots)
    workflow = _section_content(inst, "workflow")
    # Must reference list_branches host tool
    assert "list_branches" in workflow
    # Bot branch prefixes must appear (for matching directive)
    assert "custom-bot/" in workflow
    assert "another-bot/" in workflow
    # Default bot prefixes must NOT appear when overridden
    assert "dependabot/" not in workflow
    assert "renovate/" not in workflow
    # Must NOT use grep pipe pattern
    assert "grep -E" not in workflow
    assert "git branch -r" not in workflow


def test_red_pr_branch_lookup_uses_branch_prefixes(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Branch-lookup instruction must use branch-name prefixes, not [bot] author strings.

    Branch names look like 'dependabot/pip/urllib3-2.0', never 'dependabot[bot]/...'.
    The instruction must tell the agent to match on the prefix (e.g. 'dependabot/')
    derived from the bot author, not the full author string with [bot] suffix.
    """
    workflow = _section_content(instructions, "workflow")
    # Must reference branch prefixes like 'dependabot/' and 'renovate/'
    assert "dependabot/" in workflow
    assert "renovate/" in workflow
    # Must NOT tell agent to match branches against '[bot]' strings
    assert "[bot]" not in workflow.split("list_branches")[1].split(".")[0]


def test_sandbox_workflow_no_shell_operators(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Sandbox-mode workflow must not contain pipe, &&, or || operators."""
    workflow = _section_content(instructions, "workflow")
    # Check for standalone shell operators (not inside quoted strings)
    assert " | " not in workflow, "Workflow contains pipe operator"
    assert " && " not in workflow, "Workflow contains && operator"
    assert " || " not in workflow, "Workflow contains || operator"


@pytest.mark.parametrize("tool_name", ["get_pr_diff", "get_pr_files"])
def test_workflow_inspects_pr_before_clone(
    instructions: types.TemplatedSystemInstructions,
    tool_name: str,
) -> None:
    """RED PR workflow should reference inspection tools before cloning."""
    workflow = _section_content(instructions, "workflow")
    assert tool_name in workflow, f"{tool_name} not found in workflow"
    assert "before cloning" in workflow, "'before cloning' context not found"
    tool_pos = workflow.find(tool_name)
    before_clone_pos = workflow.find("before cloning")
    assert tool_pos < before_clone_pos, f"{tool_name} must appear before 'before cloning'"


def test_post_action_checks_reference_wait_for_ci(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """post_action_checks should reference wait_for_ci tool instead of manual polling."""
    content = _section_content(instructions, "post_action_checks")
    assert "wait_for_ci" in content


def test_guardrails_env_syntax_guidance(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Guardrails should instruct agent to use 'env KEY=val cmd' not 'KEY=val cmd'."""
    guardrails = _section_content(instructions, "guardrails")
    assert "env KEY=val" in guardrails
    assert "srt" in guardrails.lower() or "argv" in guardrails.lower()


def test_merging_green_prs_uses_wait_for_ci() -> None:
    """merging_green_prs should reference wait_for_ci, not get_pr_status for re-checks."""
    inst = _get_instructions(verify_all=False)
    content = _section_content(inst, "merging_green_prs")
    assert "wait_for_ci" in content
    # Should NOT reference manual get_pr_status polling
    assert "get_pr_status" not in content


def test_workflow_inlines_self_review(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Workflow self-review should be inlined, not reference external skill files."""
    workflow = _section_content(instructions, "workflow")
    assert "self-review" in workflow.lower()
    assert "code-review-and-quality" not in workflow
    assert "SKILL.md" not in workflow


# --- Issue #4: Rebase re-check after processing ---


@pytest.mark.parametrize("no_sandbox", [False, True], ids=["sandbox", "no_sandbox"])
def test_workflow_rebase_recheck_after_processing(no_sandbox: bool) -> None:
    """Workflow must include a final pass to re-check rebased PRs.

    After all PRs are processed, the agent should call get_pr_status
    once on each rebased PR and merge if GREEN.
    """
    inst = _get_instructions(no_sandbox=no_sandbox)
    workflow = _section_content(inst, "workflow")
    assert "rebased" in workflow.lower()
    assert "get_pr_status" in workflow
    if not no_sandbox:
        main_processing = workflow.find("Max")
        recheck = workflow.lower().find("rebased")
        assert recheck > main_processing, "Rebase re-check must appear after main processing steps"


# --- Skill reading guidance ---


def test_guardrails_defer_skill_reading(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Guardrails must tell the agent to NOT read skills unless fixing RED PRs.

    The code-review-and-quality skill is 15KB. Reading it on every repo
    wastes tokens when all PRs are GREEN or CONFLICT.
    """
    guardrails = _section_content(instructions, "guardrails")
    assert "skill" in guardrails.lower()
    assert "RED" in guardrails


def test_no_sandbox_guardrails_no_skill_reading(
    no_sandbox_instructions: types.TemplatedSystemInstructions,
) -> None:
    """No-sandbox guardrails must tell agent to never read skills (no code fixing)."""
    guardrails = _section_content(no_sandbox_instructions, "guardrails")
    assert "skill" in guardrails.lower()


# --- wait_for_ci only after merge/push ---


def test_post_action_checks_not_first_pr(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """post_action_checks must clarify wait_for_ci is only for PRs AFTER a merge/push.

    The first PR's status is already known from get_pr_status — calling
    wait_for_ci on it wastes ~10K tokens per repo.
    """
    section = _section_content(instructions, "post_action_checks")
    # Must explicitly say "not" or "do not" in context of first PR
    assert "first" in section.lower()


# --- No artifact creation ---


def test_output_format_no_artifacts(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """output_format must tell the agent not to create artifact files.

    The agent was writing processing_summary.md files to the brain directory,
    wasting tokens on file creation when the summary is already in stdout.
    """
    section = _section_content(instructions, "output_format")
    assert "artifact" in section.lower() or "file" in section.lower()


# --- Summary shape ---


def test_output_format_summary_is_a_bullet_list_not_a_table(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The final summary must be a bullet list, and tables ruled out explicitly.

    'Summary list' alone left the model free to render a markdown table. On
    kweinmeister/youtube-dashboard all five rows carried the identical reason
    string, which tripped the looping-content detector: the turn died mid-table
    and the SDK re-emitted the whole run.
    """
    content = _section_content(instructions, "output_format").lower()
    assert "bullet list" in content
    assert "never a table" in content


# --- Rebase remaining RED after fix merge (auto-merge only) ---


def test_auto_merge_rebase_remaining_red() -> None:
    """In auto-merge mode, after fixing+merging a RED PR, rebase remaining RED PRs.

    This propagates the fix via main so shared root causes (e.g. ruff on main)
    turn remaining RED PRs green without individual clone+fix cycles.
    """
    inst = _get_instructions(auto_merge=True)
    section = _section_content(inst, "auto_merge_mode")
    assert "rebase" in section.lower()
    assert "remaining" in section.lower() or "other" in section.lower()


def test_manual_review_no_rebase_remaining() -> None:
    """In manual review mode, the fix isn't merged, so rebase-remaining doesn't apply."""
    inst = _get_instructions(auto_merge=False)
    section = _section_content(inst, "manual_review_mode")
    assert "rebase" not in section.lower()


# --- uv sync strategy for sandbox ---


def test_sandbox_workflow_uv_sync_first(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Sandbox workflow must tell agent to run 'uv sync' as a separate step.

    `uv run <tool>` silently tries to sync all deps first. For heavy projects
    (e.g. 500MB spaCy models), this exceeds the sandbox timeout with no
    visible error. Running `uv sync` explicitly first makes failures visible.
    After sync, subsequent `uv run` calls detect the venv is current instantly.
    """
    workflow = _section_content(instructions, "workflow")
    assert "uv sync" in workflow


# --- Base branch health ---


def test_workflow_checks_base_health_before_cloning(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The base check must be instructed, and must come before the clone.

    On fast-diff-mcp the model happened to run 'git checkout main && ruff check'
    on its own, which is luck rather than behaviour. Nothing in the workflow
    asked for it, so on another repo it may clone and grind through every PR
    before noticing the base was already red.
    """
    content = _section_content(instructions, "workflow")
    assert "get_branch_ci_status" in content
    assert content.index("get_branch_ci_status") < content.index("Clone (if not already)")


def test_workflow_reports_a_broken_base_once_not_per_pr(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Five PRs blocked on one cause should be diagnosed once, then skipped."""
    content = _section_content(instructions, "workflow")
    assert "ONCE per distinct base_ref" in content
    assert "do NOT re-check the same base" in content
    assert "every remaining RED PR" in content


def test_base_check_uses_the_prs_own_base_ref(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The branch checked must be the one the PR targets, not the repo default.

    'get_branch_ci_status' falls back to the default branch when no branch is
    passed, so an instruction that omits the argument silently diagnoses the
    wrong branch for any PR aimed at 'develop' or a release line.
    """
    content = _section_content(instructions, "workflow")
    assert "base_ref" in content
    assert "list_bot_prs" in content
    assert content.index("get_branch_ci_status(owner, repo, branch)") < content.index("Clone (if not already)")


def test_base_check_is_reused_per_base_not_across_unrelated_bases(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Two PRs on different bases need two checks; two on one base need one."""
    content = _section_content(instructions, "workflow")
    assert "distinct base_ref" in content
    assert "sharing that base" in content


def test_base_fix_pr_targets_the_base_that_was_diagnosed(
    fix_base_instructions: types.TemplatedSystemInstructions,
) -> None:
    """The fix must land on the branch that is actually broken.

    Opening the fix against the repository default when the diagnosed base was
    'develop' repairs a branch nobody reported as failing and leaves the
    dependency PRs just as blocked.
    """
    content = _section_content(fix_base_instructions, "fix_base_branch")
    assert "base_ref" in content


def test_base_health_is_checked_before_logs_are_pulled(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The cheap branch-level check has to precede the expensive per-PR one.

    A base failing the same checks excuses every PR on it, so logs pulled
    first are logs discarded. Reading them for five PRs is what turned one
    repository into 2.6M input tokens.
    """
    content = _section_content(instructions, "workflow")
    assert content.index("get_branch_ci_status") < content.index("get_pr_workflow_run_logs")


def test_base_health_check_handles_pending_and_none(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Base check instructs agent to continue when base is GREEN, PENDING, or NONE."""
    content = _section_content(instructions, "workflow")
    assert "If the base is GREEN, PENDING, or NONE, the failure belongs to the PR: continue." in content


def test_logs_are_read_once_per_distinct_failure(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """PRs failing the identical check must not each cost a full log read.

    kweinmeister/fast-diff-mcp ran five dependency PRs that all failed the same
    'Build and Test' job for the same reason, and the run pulled logs five
    times — the single largest line item in a 4.5M-token run.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "already read logs for" in lowered
    assert "reproduce" in lowered


def test_log_reuse_falls_back_when_the_failure_differs(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Two PRs can fail one check for unrelated reasons.

    Reusing a diagnosis across them is only safe because the local run is the
    real source of truth; without a stated fallback the agent would carry the
    first PR's conclusion onto a failure it does not explain.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "does not reproduce" in lowered
    assert "different error" in lowered


def test_log_reuse_is_keyed_on_the_check_that_failed(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The reuse rule has to be scoped to matching checks, not to 'a second RED PR'."""
    content = _section_content(instructions, "workflow")
    reuse_at = content.lower().index("already read logs for")
    window = content[max(0, reuse_at - 300) : reuse_at + 300].lower()
    assert "check" in window


def test_a_check_that_failed_from_a_runner_error_is_not_cloned(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """A check the runner itself killed says nothing about the dependency update.

    kweinmeister/youtube-dashboard had five PRs whose only failing check was a
    Gemini review action that died on 'FatalTurnLimitedError'. Nothing in the
    repository can reproduce that, so the clone-sync-test cycle the agent ran
    to prove it could only ever end where it started.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "runner error" in lowered
    error_at = lowered.index("runner error")
    window = lowered[error_at : error_at + 400]
    assert "do not clone" in window


@pytest.mark.parametrize(
    "condition",
    ["turn", "rate", "memory", "timed out", "authenticate"],
)
def test_runner_error_triage_names_the_conditions_it_covers(
    instructions: types.TemplatedSystemInstructions,
    condition: str,
) -> None:
    """'Runner error' is only actionable if the agent knows what counts as one."""
    content = _section_content(instructions, "workflow").lower()
    assert condition in content, f"runner-error triage does not mention {condition}"


def test_a_diagnosis_outside_the_dependency_update_is_not_re_cloned(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Reusing a log read is not enough when the clone is the expensive part.

    The existing rule stops the second PR pulling logs but still sends it
    through clone, 'uv sync' and a full test run to re-confirm a verdict the
    agent already holds. For a cause no code change of its own would fix,
    that whole cycle is dead weight.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "skip the clone" in lowered
    assert "outside the dependency update" in lowered


def test_clone_reuse_still_reproduces_a_real_code_failure(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Skipping the clone is only safe for causes outside the PR's own code.

    Two PRs can fail the same check for genuinely different code reasons, so
    the shortcut must not swallow a failure a fix would actually address.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "clone and reproduce" in lowered
    assert "real code failure" in lowered


def test_non_reproduction_is_only_claimed_after_a_local_run(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """'Did not reproduce locally' is a claim about a test run that happened.

    In the youtube-dashboard run the agent logged it for three PRs it never
    cloned, reporting a local result it had not observed.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    claim_at = lowered.index("did not reproduce locally")
    window = lowered[claim_at : claim_at + 400]
    assert "actually ran" in window


def test_dry_run_treats_a_blocked_push_as_expected() -> None:
    """A push denial is the policy working, so it must not be reported as a failure.

    Every RED PR in a dry run ends on one, and the agent narrated each into the
    transcript as though something had gone wrong.
    """
    content = _section_content(_get_instructions(dry_run=True), "dry_run_mode")
    lowered = content.lower()
    assert "denied by policy" in lowered
    assert "expected" in lowered


def test_base_blame_requires_the_same_checks_to_be_failing(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """A red base does not excuse every red PR.

    kweinmeister/voting-agent has a failing 'Deploy Frontend' job on main while
    lint and tests pass. A PR that breaks lint there is the PR's own fault, so
    blaming any RED PR on a merely-RED base would silently stop fixing them.
    """
    content = _section_content(instructions, "workflow")
    lowered = content.lower()
    assert "check names" in lowered
    assert "also failing on the base" in lowered


def test_pr_only_failures_are_still_fixed_when_the_base_is_red(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """The escape hatch: a check the PR fails but the base passes is the PR's."""
    content = _section_content(instructions, "workflow")
    assert "passing on the base" in content
    assert "the PR introduced that failure" in content


def test_base_fix_section_absent_by_default(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """Opening unrelated PRs is a surprise, so it must be opt-in."""
    assert "fix_base_branch" not in _section_titles(instructions)


def test_base_fix_section_opens_a_separate_pr_against_the_base(
    fix_base_instructions: types.TemplatedSystemInstructions,
) -> None:
    """A base fix must never land on a dependency branch.

    Pushing repo-wide changes to 'dependabot/...' is bad review hygiene and
    makes Dependabot stop managing the PR.
    """
    content = _section_content(fix_base_instructions, "fix_base_branch")
    assert "create_pr" in content
    assert "MUST NOT" in content
    assert "dependency" in content.lower()


def test_base_fix_scope_is_bounded_to_what_ci_fails_on(
    fix_base_instructions: types.TemplatedSystemInstructions,
) -> None:
    """'Fix the base' must not become 'clean up the whole repo'."""
    content = _section_content(fix_base_instructions, "fix_base_branch")
    lowered = content.lower()
    assert "only" in lowered
    assert "unrelated" in lowered


def test_base_fix_checks_for_a_pr_it_already_opened(
    fix_base_instructions: types.TemplatedSystemInstructions,
) -> None:
    """A base fix awaiting review must stop the next run before it clones.

    The fix branch name is derived from the base ref, so every run over an
    unmerged base fix rebuilds the same branch, force-pushes over the PR a
    human is reviewing, and then dead-ends when GitHub refuses the duplicate.
    """
    content = _section_content(fix_base_instructions, "fix_base_branch")
    assert "find_open_pr_for_branch" in content
    assert "before" in content.lower()


def test_base_fix_leaves_generated_files_alone(
    fix_base_instructions: types.TemplatedSystemInstructions,
) -> None:
    """'uv sync' rewrites the lockfile, and that is not part of a lint fix."""
    content = _section_content(fix_base_instructions, "fix_base_branch")
    assert "uv.lock" in content


def test_base_fix_is_excluded_from_auto_merge() -> None:
    """Merging the agent's own change to main is a bigger deal than a dep bump."""
    content = _section_content(_get_instructions(fix_base=True, auto_merge=True), "fix_base_branch")
    assert "MUST NOT merge" in content
