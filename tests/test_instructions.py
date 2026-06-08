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
    content = _section_content(instructions, "workflow")
    assert expected in content


def test_no_gh_cli_diagnostic_commands_in_workflow(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    content = _section_content(instructions, "workflow")
    assert "gh pr checks" not in content
    assert "gh api" not in content


def test_clone_only_for_red(instructions: types.TemplatedSystemInstructions) -> None:
    content = _section_content(instructions, "workflow")
    assert "do not clone" in content.lower()


def test_conflict_rebase_skips_to_next_pr(
    instructions: types.TemplatedSystemInstructions,
) -> None:
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
    content = _section_content(instructions, "code_quality")
    assert expected in content


# --- Post-action checks ---


def test_post_action_checks(instructions: types.TemplatedSystemInstructions) -> None:
    content = _section_content(instructions, "post_action_checks")
    assert "re-list" in content.lower() or "re-check" in content.lower()
    assert "get_pr_status" in content
    assert "max 10 retries" in content
    assert "unknown" in content  # mergeable_state: 'unknown' guidance


def test_post_action_owner_not_leaked_as_template_literal(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    content = _section_content(instructions, "post_action_checks")
    assert "{owner}" not in content


# --- Output format ---


def test_output_format(instructions: types.TemplatedSystemInstructions) -> None:
    content = _section_content(instructions, "output_format")
    assert "GREEN" in content
    assert "RED" in content
    assert str(3) in content


# --- max_attempts ---


@pytest.mark.parametrize("value", [1, 3, 7, 10])
def test_max_attempts_appears_in_workflow(value: int) -> None:
    inst = get_system_instructions(max_attempts=value, owner="test-owner")
    assert str(value) in _section_content(inst, "workflow")


# --- owner ---


def test_owner_appears_in_identity() -> None:
    inst = get_system_instructions(max_attempts=3, owner="my-org")
    assert inst.identity is not None
    assert "my-org" in inst.identity


def test_owner_with_special_chars() -> None:
    inst = get_system_instructions(max_attempts=3, owner="my-org_123")
    assert inst.identity is not None
    assert "my-org_123" in inst.identity


def test_owner_empty_string() -> None:
    inst = get_system_instructions(max_attempts=3, owner="")
    assert isinstance(inst, types.TemplatedSystemInstructions)
    assert inst.identity is not None
    assert "github.com/" in inst.identity


# --- workspace_dir ---


def test_workspace_dir_appears() -> None:
    inst = get_system_instructions(
        max_attempts=3,
        owner="test-owner",
        workspace_dir="/tmp/custom-ws",
    )
    assert "/tmp/custom-ws" in _section_content(inst, "guardrails")
    assert "/tmp/custom-ws" in _section_content(inst, "workflow")


def test_workspace_dir_default() -> None:
    inst = get_system_instructions(max_attempts=3, owner="test-owner")
    assert DEFAULT_WORKSPACE_DIR in _section_content(inst, "guardrails")


# --- Conditional sections (flag toggles) ---


@pytest.mark.parametrize(
    ("flags", "present", "absent"),
    [
        ({"auto_merge": True}, "auto_merge_mode", "manual_review_mode"),
        ({"auto_merge": False}, "manual_review_mode", "auto_merge_mode"),
        ({"verify_all": True}, "verify_green_prs", "fast_track_green_prs"),
        ({"verify_all": False}, "fast_track_green_prs", "verify_green_prs"),
        ({"dry_run": True}, "dry_run_mode", None),
        ({"review_wait": 5}, "review_feedback_loop", None),
    ],
)
def test_conditional_section_present(
    flags: dict[str, Any],
    present: str,
    absent: str | None,
) -> None:
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
    inst = _get_instructions(**flags)
    assert absent not in _section_titles(inst)


# --- Conditional section content ---


def test_auto_merge_content() -> None:
    inst = _get_instructions(auto_merge=True)
    content = _section_content(inst, "auto_merge_mode")
    assert "merge_bot_pr" in content
    assert "green" in content


def test_manual_review_content() -> None:
    inst = _get_instructions(auto_merge=False)
    content = _section_content(inst, "manual_review_mode")
    assert "MUST NOT merge fix PRs" in content


def test_dry_run_content() -> None:
    inst = _get_instructions(dry_run=True)
    content = _section_content(inst, "dry_run_mode")
    assert "safety policies enforce simulation" in content
    assert "Do not skip" in content
    assert "[DRY-RUN]" in content
    assert "do NOT re-check PR status between merges" in content


def test_verify_all_content() -> None:
    inst = _get_instructions(verify_all=True)
    content = _section_content(inst, "verify_green_prs")
    assert "clone" in content.lower()


def test_fast_track_content() -> None:
    inst = _get_instructions(verify_all=False)
    content = _section_content(inst, "fast_track_green_prs")
    assert "merge_bot_pr" in content


def test_review_wait_content() -> None:
    inst = _get_instructions(review_wait=5)
    content = _section_content(inst, "review_feedback_loop")
    assert "wait_for_reviews" in content


# --- Fix strategy ---


def test_default_pushes_to_original_branch() -> None:
    inst = _get_instructions(standalone_fix=False)
    content = _section_content(inst, "workflow")
    assert "directly" in content.lower()
    assert "dependency-director/fix-" not in content


def test_default_includes_merge_from_main() -> None:
    inst = _get_instructions(standalone_fix=False)
    content = _section_content(inst, "workflow")
    assert "git merge origin/main" in content


def test_standalone_fix_creates_new_branch() -> None:
    inst = _get_instructions(standalone_fix=True)
    content = _section_content(inst, "workflow")
    assert "dependency-director/fix-" in content


# --- Flag interactions ---


def test_all_flags_on() -> None:
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
    assert "fast_track_green_prs" in titles
    assert "manual_review_mode" in titles
    assert "dry_run_mode" not in titles
    assert "review_feedback_loop" not in titles


# --- Multi-bot support ---


def test_bot_authors_in_guardrails(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    content = _section_content(instructions, "guardrails")
    assert "dependabot[bot]" in content
    assert "renovate[bot]" in content


def test_bot_prs_tool_in_workflow(
    instructions: types.TemplatedSystemInstructions,
) -> None:
    content = _section_content(instructions, "workflow")
    assert "list_bot_prs" in content


def test_custom_bots_in_instructions() -> None:
    custom = [BotConfig(author="my-bot[bot]", rebase_command="@my-bot rebase")]
    inst = get_system_instructions(max_attempts=3, owner="test-owner", bots=custom)
    content = _section_content(inst, "guardrails")
    assert "my-bot[bot]" in content
    assert "dependabot[bot]" not in content


def test_no_sandbox_instructions() -> None:
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
