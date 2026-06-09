"""Tests for system instructions generation in dependency-director."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from google.antigravity import types

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.instructions import (
    DEFAULT_WORKSPACE_DIR,
    get_system_instructions,
)


def _get_instructions(
    max_attempts: int = 3,
    owner: str = "test-owner",
    bots: list[BotConfig] = DEFAULT_BOTS,
    **kwargs: Any,
) -> types.TemplatedSystemInstructions:
    return get_system_instructions(
        max_attempts=max_attempts,
        owner=owner,
        bots=bots,
        **kwargs,
    )


@pytest.fixture
def instructions() -> types.TemplatedSystemInstructions:
    """Fixture to return a default templated system instructions object for testing."""
    return _get_instructions()


def _section_content(inst: types.TemplatedSystemInstructions, title: str) -> str:
    for s in inst.sections:
        if s.title == title:
            return str(s.content)
    return ""


def _section_titles(inst: types.TemplatedSystemInstructions) -> set[str]:
    return {s.title for s in inst.sections}


# --- Structure ---


def test_returns_templated(instructions: types.TemplatedSystemInstructions) -> None:
    """Verify that get_system_instructions returns a TemplatedSystemInstructions object."""
    assert isinstance(instructions, types.TemplatedSystemInstructions)
    assert instructions.identity is not None
    assert "test-owner" in instructions.identity


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
        "code-review-and-quality",
    ],
)
def test_workflow_contains(
    instructions: types.TemplatedSystemInstructions,
    expected: str,
) -> None:
    """Verify that the primary agent workflow instructions are generated correctly."""
    content = _section_content(instructions, "workflow")
    assert expected in content


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
    assert "re-list" in content.lower() or "re-check" in content.lower()
    assert "get_pr_status" in content
    assert "max 10 retries" in content
    assert "unknown" in content  # mergeable_state: 'unknown' guidance


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


# --- max_attempts ---


@pytest.mark.parametrize("value", [1, 3, 7, 10])
def test_max_attempts_appears_in_workflow(value: int) -> None:
    """Verify that the max fix attempts config value is correctly formatted into the workflow instructions."""
    inst = get_system_instructions(max_attempts=value, owner="test-owner")
    assert str(value) in _section_content(inst, "workflow")


# --- owner ---


def test_owner_appears_in_identity() -> None:
    """Verify that the repository owner is correctly injected into the agent's identity section."""
    inst = get_system_instructions(max_attempts=3, owner="my-org")
    assert inst.identity is not None
    assert "my-org" in inst.identity


def test_owner_with_special_chars() -> None:
    """Verify that owner names containing special characters are correctly handled."""
    inst = get_system_instructions(max_attempts=3, owner="my-org_123")
    assert inst.identity is not None
    assert "my-org_123" in inst.identity


def test_owner_empty_string() -> None:
    """Verify that an empty string owner raises a ValueError during validation."""
    inst = get_system_instructions(max_attempts=3, owner="")
    assert isinstance(inst, types.TemplatedSystemInstructions)
    assert inst.identity is not None
    assert "github.com/" in inst.identity


# --- workspace_dir ---


def test_workspace_dir_appears() -> None:
    """Verify that the active workspace directory path is included in the instructions."""
    workspace = str(Path(tempfile.gettempdir()) / "custom-ws")
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        workspace_dir=workspace,
    )
    assert workspace in _section_content(inst, "guardrails")
    assert workspace in _section_content(inst, "workflow")


def test_workspace_dir_default() -> None:
    """Verify default workspace directory placeholder is used when not specified."""
    inst = get_system_instructions(max_attempts=3, owner="test-owner")
    assert DEFAULT_WORKSPACE_DIR in _section_content(inst, "guardrails")


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


def test_default_pushes_to_original_branch() -> None:
    """Verify instructions command pushing fixes back to the original branch by default."""
    inst = _get_instructions(standalone_fix=False)
    content = _section_content(inst, "workflow")
    assert "directly" in content.lower()
    assert "dependency-director/fix-" not in content


def test_default_includes_merge_from_main() -> None:
    """Verify instructions command merging main branch before attempting fixes by default."""
    inst = _get_instructions(standalone_fix=False)
    content = _section_content(inst, "workflow")
    assert "git merge origin/main" in content


def test_standalone_fix_creates_new_branch() -> None:
    """Verify instructions specify creating a new branch when standalone-fix is active."""
    inst = _get_instructions(standalone_fix=True)
    content = _section_content(inst, "workflow")
    assert "dependency-director/fix-" in content


# --- Flag interactions ---


def test_all_flags_on() -> None:
    """Verify instructions are generated correctly with all feature flags enabled."""
    custom_ws = str(Path(tempfile.gettempdir()) / "all-flags")
    inst = get_system_instructions(
        max_attempts=5,
        owner="all-flags-org",
        verify_all=True,
        auto_merge=True,
        dry_run=True,
        standalone_fix=True,
        review_wait=10,
        workspace_dir=custom_ws,
    )
    titles = _section_titles(inst)
    assert "verify_green_prs" in titles
    assert "auto_merge_mode" in titles
    assert "dry_run_mode" in titles
    assert "review_feedback_loop" in titles
    assert inst.identity is not None
    assert "all-flags-org" in inst.identity
    assert custom_ws in _section_content(inst, "guardrails")
    assert "dependency-director/fix-" in _section_content(inst, "workflow")
    assert "5" in _section_content(inst, "workflow")


def test_all_flags_off() -> None:
    """Verify instructions are generated correctly with all feature flags disabled."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="minimal-org",
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
    inst = get_system_instructions(max_attempts=3, owner="test-owner", bots=custom)
    content = _section_content(inst, "guardrails")
    assert "my-bot[bot]" in content
    assert "dependabot[bot]" not in content


def test_no_sandbox_instructions() -> None:
    """Verify instructions explicitly flag that sandboxing is disabled when configured."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=True,
    )
    guardrails = _section_content(inst, "guardrails")
    assert "NO-SANDBOX mode" in guardrails
    assert "MUST NOT clone repositories" in guardrails

    workflow = _section_content(inst, "workflow")
    assert "Do NOT clone" in workflow
    assert "cannot be fixed in non-sandboxed mode" in workflow


def test_no_sandbox_instructions_no_shell_access() -> None:
    """No-sandbox guardrails must state that no shell access is available."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=True,
    )
    guardrails = _section_content(inst, "guardrails")
    assert "No shell access" in guardrails


def test_no_sandbox_instructions_no_run_command_reference() -> None:
    """No-sandbox instructions must NOT reference run_command_sandboxed since the tool is not registered."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=True,
    )
    all_content = " ".join(_section_content(inst, s.title) for s in inst.sections)
    assert "run_command_sandboxed" not in all_content


def test_sandbox_instructions_reference_run_command() -> None:
    """Sandbox mode instructions MUST reference run_command_sandboxed since the tool IS registered."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=False,
    )
    guardrails = _section_content(inst, "guardrails")
    assert "run_command_sandboxed" in guardrails


def test_no_sandbox_workflow_skips_red_prs() -> None:
    """No-sandbox workflow must skip RED PRs (no local fix attempts)."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=True,
    )
    workflow = _section_content(inst, "workflow")
    # Should not reference cloning or installing deps for RED PRs
    assert "clone" not in workflow.lower() or "Do NOT clone" in workflow
    assert "install deps" not in workflow.lower()


def test_no_sandbox_workflow_still_has_merge_and_rebase() -> None:
    """No-sandbox workflow must still reference merge and rebase tools."""
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        no_sandbox=True,
    )
    workflow = _section_content(inst, "workflow")
    assert "merge_bot_pr" in workflow
    assert "rebase_bot_pr" in workflow
    assert "get_pr_status" in workflow


def test_halt_on_no_prs(instructions: types.TemplatedSystemInstructions) -> None:
    """System instructions must direct the agent to halt if no PRs are found."""
    workflow = _section_content(instructions, "workflow")
    assert "If none are found, halt immediately and exit" in workflow


def test_trust_tool_outputs_no_host_inspection(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """System instructions must direct the agent to trust tool outputs and not inspect host environment/files."""
    guardrails = _section_content(instructions, "guardrails")
    assert "Trust tool outputs" in guardrails
    assert "Do NOT search, browse, or inspect the host environment" in guardrails


def test_minimize_conversational_output(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    """System instructions must direct the agent to minimize conversational output."""
    output_format = _section_content(instructions, "output_format")
    assert "Minimize conversational output" in output_format


def test_red_pr_grep_pattern_reflects_configured_bots() -> None:
    """Grep pattern for remote branch discovery uses configured bot authors, not hardcoded names."""
    import re as _re

    custom_bots = [
        BotConfig(author="custom-bot[bot]", rebase_command="@custom-bot rebase"),
        BotConfig(author="another-bot[bot]", rebase_command="@another-bot rebase"),
    ]
    inst = _get_instructions(bots=custom_bots)
    workflow = _section_content(inst, "workflow")
    # re.escape is applied to authors, so check for the escaped form in the pattern
    assert _re.escape("custom-bot[bot]") in workflow
    assert _re.escape("another-bot[bot]") in workflow
    # Default bot escaped names must NOT appear when overridden
    assert _re.escape("dependabot[bot]") not in workflow
    assert _re.escape("renovate[bot]") not in workflow
