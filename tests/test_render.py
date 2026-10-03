"""Tests for how an agent turn's text reaches the terminal.

The agent reports its per-PR outcome as a run of status lines ('⚠ #51 not
fixed: ...'). Markdown treats a plain newline as a soft break, so rendering
those lines verbatim folds five separate outcomes into one run-on paragraph
and the reader has to parse the glyphs apart by eye.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.antigravity import types
from rich.console import Console

from dependency_director.main import (
    _format_tool_args,
    _preserve_status_line_breaks,
    _render_agent_response,
)

STATUS_LINES = (
    "⚠ #51 not fixed: blocked on base fix PR #58\n"
    "⚠ #52 not fixed: blocked on base fix PR #58\n"
    "⚠ #54 not fixed: blocked on base fix PR #58"
)


def _response(*chunks: Any) -> MagicMock:
    """Build a stand-in chat response that streams ``chunks`` as model output."""

    async def stream() -> AsyncGenerator[Any]:
        for index, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                yield types.Text(text=chunk, step_index=index)
            else:
                yield chunk

    response = MagicMock()
    response.chunks = stream()
    return response


async def _render(*chunks: Any, console_width: int = 80) -> str:
    """Render ``chunks`` the way a real turn would, and return what the terminal got."""
    console = Console(width=console_width, no_color=True, highlight=False)
    with patch("dependency_director.main.console", console), console.capture() as capture:
        await _render_agent_response(_response(*chunks))
    return capture.get()


@pytest.mark.asyncio
async def test_status_lines_each_get_their_own_line() -> None:
    """Three outcomes must read as three lines, not one paragraph."""
    output = await _render(STATUS_LINES)
    assert sum("not fixed" in line for line in output.splitlines()) == 3


@pytest.mark.asyncio
async def test_status_lines_survive_arriving_in_separate_chunks() -> None:
    """The model streams text in arbitrary pieces; the split must not decide layout."""
    output = await _render(*(f"{part}\n" for part in STATUS_LINES.split("\n")))
    assert sum("not fixed" in line for line in output.splitlines()) == 3


@pytest.mark.asyncio
async def test_ordinary_prose_is_still_reflowed() -> None:
    """The fix must target status lines, not disable markdown wrapping wholesale."""
    output = await _render("The base branch is red.\nEvery PR on it inherits the failure.")
    assert "red. Every PR" in output


@pytest.mark.asyncio
async def test_tables_still_render_as_tables() -> None:
    """The summary the agent ends on is a markdown table and must stay one."""
    output = await _render("| PR | Result |\n| --- | --- |\n| #51 | ⚠ blocked |\n")
    assert "─" in output
    assert "| PR | Result |" not in output


POLICY_DENIAL = (
    "Denied by policy 'dry_run_block_push_sandboxed'. "
    "(\"denied by pre-tool hook: Denied by policy 'dry_run_block_push_sandboxed'.\")"
)

WORKSPACE_DENIAL = (
    'Access to path "/Users/kweinmeister/Projects/dependency-director" is denied. '
    "It is outside the allowed workspace directories: [/tmp/ws1 /tmp/ws2] "
    '("denied by pre-tool hook: Access to path '
    '"/Users/kweinmeister/Projects/dependency-director" is denied. '
    'It is outside the allowed workspace directories: [/tmp/ws1 /tmp/ws2]")'
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("denial_text", "expected_policy"),
    [
        (POLICY_DENIAL, "dry_run_block_push_sandboxed"),
        (WORKSPACE_DENIAL, "workspace_only"),
    ],
    ids=["policy_rule", "workspace_confinement"],
)
async def test_denial_renders_as_one_labelled_line(denial_text: str, expected_policy: str) -> None:
    """The SDK emits the deny reason as prose; it belongs with the tool lines.

    A dry run or workspace containment blocks an action, which emits pre-tool hook
    prose that would look like a crash if rendered verbatim.
    """
    output = (await _render(denial_text)).strip()
    assert output.count("\n") == 0, f"denial spilled across lines: {output!r}"
    assert expected_policy in output
    assert "denied by pre-tool hook" not in output


@pytest.mark.asyncio
async def test_prose_about_a_denial_is_left_as_prose() -> None:
    """The model discusses the denial in its own words; only the SDK's line is ours."""
    said = "I saw Denied by policy 'x' and treated it as the expected simulated push."
    output = await _render(said)
    assert "treated it as the expected simulated push" in output


@pytest.mark.asyncio
async def test_a_denial_does_not_swallow_text_around_it() -> None:
    """Collapsing a buffer that holds more than the denial would lose output."""
    output = await _render(f"{POLICY_DENIAL}\n\nThe fix is verified and ready.")
    assert "The fix is verified and ready." in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("denial_text", "expected_policy", "trailing"),
    [
        (POLICY_DENIAL, "dry_run_block_push_sandboxed", "✓ #57 fix pushed"),
        (WORKSPACE_DENIAL, "workspace_only", "✓ #43 conflict resolved"),
    ],
    ids=["policy_rule", "workspace_confinement"],
)
async def test_denial_preserves_trailing_text(denial_text: str, expected_policy: str, trailing: str) -> None:
    """Nothing separates the SDK's denial from the model's next sentence.

    Both land in one buffer whenever no tool call intervenes.
    """
    output = await _render(f"{denial_text}{trailing}")
    assert "denied by pre-tool hook" not in output
    assert f"blocked by policy '{expected_policy}'" in output
    assert trailing in output


def test_code_fences_are_left_alone() -> None:
    """Trailing whitespace is content inside a fence, so it must not be added there."""
    fenced = "```\n✓ example output\n```\n"
    assert _preserve_status_line_breaks(fenced) == fenced


def test_a_status_line_that_already_hard_breaks_is_untouched() -> None:
    """Appending to a line that already ends in a break would just add noise."""
    already = "✓ #51 merged  \n✓ #52 merged"
    assert _preserve_status_line_breaks(already) == "✓ #51 merged  \n✓ #52 merged"


@pytest.mark.parametrize(
    ("raw_args", "expected_str"),
    [
        (
            {"pr_number": 154.0, "owner": "kweinmeister", "ratio": 1.5, "flag": True},
            "pr_number=154, owner='kweinmeister', ratio=1.5, flag=True",
        ),
        (
            {"ids": [154.0, 156.0], "nested": {"target": 42.0, "ratio": 2.5}},
            "ids=[154, 156], nested={'target': 42, 'ratio': 2.5}",
        ),
    ],
    ids=["flat_primitives", "nested_containers"],
)
def test_format_tool_args_normalizes_floats(
    raw_args: dict[str, Any],
    expected_str: str,
) -> None:
    """Whole-number floats from protobuf structs must display as ints."""
    assert _format_tool_args(raw_args) == expected_str


def test_format_tool_args_truncates_long_strings() -> None:
    """Long argument strings must truncate with an ellipsis."""
    args = {"data": "x" * 200}
    formatted = _format_tool_args(args, max_len=30)
    assert len(formatted) == 30
    assert formatted.endswith("...")


@pytest.mark.asyncio
async def test_render_tool_call_formats_pr_number_as_int() -> None:
    """Tool calls with whole-number float arguments must render without decimal points."""
    call = types.ToolCall(
        name="get_pr_status",
        args={"owner": "kweinmeister", "pr_number": 154.0, "repo": "agent-design-patterns"},
    )
    output = await _render(call, console_width=120)
    assert "pr_number=154" in output
    assert "154.0" not in output
