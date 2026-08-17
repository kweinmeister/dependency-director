"""Tests for settings and repository configurations in dependency-director."""

import hashlib
import json
import logging
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.antigravity import types
from google.antigravity.hooks import hooks, policy
from google.antigravity.types import BuiltinTools
from pydantic import ValidationError

from dependency_director.config import (
    DEFAULT_BOTS,
    DEFAULT_CACHE_DIR,
    BotConfig,
    OutputLimits,
    Settings,
    get_dry_run_policies,
    get_safety_policies,
)
from dependency_director.main import _check_open_bot_prs, run_agent_for_repo
from dependency_director.tools import GitHubClient

from .conftest import AsyncFSHelper


@pytest.mark.parametrize(
    (
        "env_vars",
        "expected_concurrency",
        "expected_max_attempts",
        "expected_gemini_key",
        "expected_github_token",
        "expected_srt_settings",
        "expected_model",
    ),
    [
        (
            {
                "DEPDIRECTOR_CONCURRENCY": "4",
                "DEPDIRECTOR_MAX_FIX_ATTEMPTS": "5",
                "GEMINI_API_KEY": "test-gemini-key",
                "GITHUB_TOKEN": "test-github-key",
                "DEPDIRECTOR_SRT_SETTINGS": "/path/to/custom.json",
                "DEPDIRECTOR_MODEL": "gemini-3.6-pro",
            },
            4,
            5,
            "test-gemini-key",
            "test-github-key",
            "/path/to/custom.json",
            "gemini-3.6-pro",
        ),
        ({}, 1, 3, "", "", "", "gemini-3.7-flash"),
    ],
)
def test_settings_loading(
    monkeypatch: pytest.MonkeyPatch,
    env_vars: dict[str, str],
    expected_concurrency: int,
    expected_max_attempts: int,
    expected_gemini_key: str,
    expected_github_token: str,
    expected_srt_settings: str,
    expected_model: str,
) -> None:
    """Verify configuration settings load correctly from default values and file."""
    monkeypatch.delenv("DEPDIRECTOR_CONCURRENCY", raising=False)
    monkeypatch.delenv("DEPDIRECTOR_MAX_FIX_ATTEMPTS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DEPDIRECTOR_SRT_SETTINGS", raising=False)
    monkeypatch.delenv("DEPDIRECTOR_MODEL", raising=False)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    settings = Settings()
    assert settings.concurrency == expected_concurrency
    assert settings.max_fix_attempts == expected_max_attempts
    assert settings.gemini_api_key == expected_gemini_key
    assert settings.github_token == expected_github_token
    assert settings.srt_settings == expected_srt_settings
    assert settings.model == expected_model


def test_settings_invalid_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validation errors are raised for incorrect configuration value types."""
    monkeypatch.setenv("DEPDIRECTOR_CONCURRENCY", "not-an-int")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
def test_settings_concurrency_rejects_non_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Concurrency=0 would deadlock asyncio.Semaphore; negative values are also invalid."""
    monkeypatch.setenv("DEPDIRECTOR_CONCURRENCY", value)
    with pytest.raises(ValidationError):
        Settings()


def test_settings_concurrency_accepts_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrency=1 (the minimum) must be accepted."""
    monkeypatch.setenv("DEPDIRECTOR_CONCURRENCY", "1")
    assert Settings().concurrency == 1


def test_settings_empty_bots_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty bots list silently disables all PR processing; reject it early."""
    monkeypatch.setenv("DEPDIRECTOR_BOTS", "[]")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_output_limits_default_to_the_documented_values() -> None:
    """Verify the output caps default to what the README and .env.template state."""
    assert Settings().output_limits == OutputLimits(max_lines=200, max_chars=24000)


def test_settings_output_limits_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify both caps can be overridden from the environment, including to 0."""
    monkeypatch.setenv("DEPDIRECTOR_MAX_OUTPUT_LINES", "0")
    monkeypatch.setenv("DEPDIRECTOR_MAX_OUTPUT_CHARS", "5000")
    assert Settings().output_limits == OutputLimits(max_lines=0, max_chars=5000)


@pytest.mark.parametrize("var", ["DEPDIRECTOR_MAX_OUTPUT_LINES", "DEPDIRECTOR_MAX_OUTPUT_CHARS"])
def test_settings_output_limits_reject_negative(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    """Verify a negative cap is rejected rather than silently slicing backwards."""
    monkeypatch.setenv(var, "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_cache_dir_defaults_outside_any_workspace() -> None:
    """Verify the package cache defaults to a shared path the workspace cleanup cannot reach."""
    cache_dir = Settings().cache_dir
    assert cache_dir == DEFAULT_CACHE_DIR
    assert not Path(cache_dir).name.startswith("dependency-director-workspace")
    assert Path(cache_dir).parent == Path(tempfile.gettempdir())


def test_settings_cache_dir_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the shared package cache location can be overridden from the environment."""
    monkeypatch.setenv("DEPDIRECTOR_CACHE_DIR", "/tmp/custom-depdirector-cache")
    assert Settings().cache_dir == "/tmp/custom-depdirector-cache"


@pytest.mark.asyncio
async def test_repo_without_bot_prs_skips_workspace_and_sandbox_setup(
    mock_agent_class: MagicMock,
    mock_list_open_prs: MagicMock,
    github_token: str,
) -> None:
    """Verify a repo with no bot PRs costs no workspace, no sandbox, and no agent spawn."""
    mock_list_open_prs.side_effect = None
    mock_list_open_prs.return_value = []
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    with (
        patch("dependency_director.main.create_run_command_tool") as mock_run_command_tool,
        patch("dependency_director.main._prepare_workspace") as mock_prepare_workspace,
    ):
        await run_agent_for_repo(
            repo="test-owner/test-repo",
            settings=settings,
            max_attempts=3,
        )
    mock_prepare_workspace.assert_not_called()
    mock_run_command_tool.assert_not_called()
    mock_agent_class.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_for_repo_passes_cache_dir_to_sandbox(
    mock_agent_class: MagicMock,
    github_token: str,
) -> None:
    """Verify the configured shared cache directory reaches the sandboxed command tool."""
    _ = mock_agent_class
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    settings.cache_dir = "/tmp/custom-depdirector-cache"
    with patch("dependency_director.main.create_run_command_tool") as mock_run_command_tool:
        await run_agent_for_repo(
            repo="test-owner/test-repo",
            settings=settings,
            max_attempts=3,
            dry_run=True,
        )
    mock_run_command_tool.assert_called_once()
    assert mock_run_command_tool.call_args.kwargs["cache_dir"] == "/tmp/custom-depdirector-cache"


def test_safety_policy_structure() -> None:
    """Verify that the safety policy configuration conforms to the required schema."""
    policies = get_safety_policies()
    assert len(policies) == 1
    assert policies[0].tool == "*"


@pytest.mark.asyncio
async def test_safety_policies_enforcement() -> None:
    """Integration test: compiled policies correctly evaluate ToolCall objects."""
    policies = get_safety_policies()
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    safe = types.ToolCall(name="run_command", args={"command_line": "uv run pytest"})
    assert (await hook.run(ctx, safe)).allow is True
    injected = types.ToolCall(name="run_command", args={"command_line": "npm test; rm -rf /"})
    assert (await hook.run(ctx, injected)).allow is True
    allowed = types.ToolCall(name="run_command", args={"command_line": "echo hello"})
    assert (await hook.run(ctx, allowed)).allow is True


def test_dry_run_policies_enforcement() -> None:
    """Verify that dry-run configuration policies are correctly enforced."""
    policies = get_dry_run_policies()
    names = [p.name for p in policies]
    assert "dry_run_block_push" in names
    block_push = next(p for p in policies if p.name == "dry_run_block_push")
    assert block_push.when({"CommandLine": "git push origin main"}) is True
    assert block_push.when({"command_line": "git push origin main"}) is True
    assert block_push.when({"CommandLine": "git status"}) is False
    assert block_push.when({"command_line": "git status"}) is False


@pytest.mark.asyncio
async def test_agent_config_includes_skills_path(
    mock_agent_class: MagicMock,
    async_fs: type[AsyncFSHelper],
    github_token: str,
) -> None:
    """Verify that the generated agent configuration contains the correct skills path."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    mock_agent_instance = mock_agent_class.return_value
    mock_response = AsyncMock()
    mock_response.text.return_value = "Triage completed successfully."
    mock_agent_instance.chat = AsyncMock(return_value=mock_response)
    mock_usage = mock_agent_instance.conversation.total_usage
    mock_usage.prompt_token_count = 100
    mock_usage.cached_content_token_count = 0
    mock_usage.candidates_token_count = 50
    mock_usage.thoughts_token_count = 10
    mock_usage.total_token_count = 160
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    mock_agent_class.assert_called_once()
    config_passed = mock_agent_class.call_args[1]["config"]
    assert hasattr(config_passed, "skills_paths")
    assert len(config_passed.skills_paths) == 1
    skills_dir = config_passed.skills_paths[0]
    assert ".agents/skills" in skills_dir
    assert await async_fs.exists(skills_dir), f"Skills directory {skills_dir} does not exist"
    skill_folder = Path(skills_dir) / "code-review-and-quality"
    assert skill_folder.exists(), f"Skill folder {skill_folder} does not exist"
    skill_file = skill_folder / "SKILL.md"
    assert skill_file.exists(), f"Skill definition file {skill_file} does not exist"


@pytest.mark.asyncio
async def test_agent_spawn_status_output(
    mock_agent_class: MagicMock,
    github_token: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that agent spawning status output includes model and mode info."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    mock_agent_instance = mock_agent_class.return_value
    mock_response = AsyncMock()
    mock_response.text.return_value = "Done."
    mock_agent_instance.chat = AsyncMock(return_value=mock_response)

    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        model="gemini-3.7-flash",
    )
    captured = capsys.readouterr()
    assert "Spawning Agent for test-owner/test-repo [model: gemini-3.7-flash | mode: Developer API]" in captured.out


@pytest.mark.parametrize(
    ("env_var", "value", "expected_attr", "expected_val"),
    [
        ("DEPDIRECTOR_OWNER", "my-org", "owner", "my-org"),
        ("DEPDIRECTOR_REVIEW_WAIT", "10", "review_wait", 10),
        ("DEPDIRECTOR_MODEL", "gemini-3.6-pro", "model", "gemini-3.6-pro"),
    ],
)
def test_settings_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    value: str,
    expected_attr: str,
    expected_val: Any,
) -> None:
    """Verify configuration settings can be overridden by environment variables."""
    monkeypatch.setenv(env_var, value)
    settings = Settings()
    assert getattr(settings, expected_attr) == expected_val


def test_dry_run_git_push_substring() -> None:
    """Verify dry-run policy allows git push dry-run sub-commands."""
    policies = get_dry_run_policies()
    block_push = next(p for p in policies if p.name == "dry_run_block_push")
    assert block_push.when({"command_line": "git push origin main"}) is True
    assert block_push.when({"command_line": "git pushup"}) is False
    assert block_push.when({"command_line": "git pull"}) is False
    assert block_push.when({"command_line": "git -C dir push"}) is True


@pytest.mark.parametrize(
    ("command_line", "blocked"),
    [
        # --- Plain invocations ---
        ("git push origin main", True),
        ("git push origin pr-90:dependabot/pip/ty-0.0.64", True),
        ("/usr/bin/git push origin main", True),
        ("git -C dir push", True),
        ("git -c user.name=x push origin main", True),
        # --- env(1) wrappers: the form the system instructions tell the agent to use ---
        ("env GIT_TERMINAL_PROMPT=0 git push origin main", True),
        ("env -u SSH_AUTH_SOCK git push origin main", True),
        ("env -- git push origin main", True),
        ("/usr/bin/env GIT_TERMINAL_PROMPT=0 git push origin main", True),
        # --- Bare KEY=val prefixes ---
        ("GIT_AUTHOR_NAME=x git push origin main", True),
        ("A=1 B=2 git push origin main", True),
        # --- Compound commands where git is not the leading token ---
        ("ls && git push origin main", True),
        ("git status && git push origin main", True),
        ("uv run pytest || git push origin main", True),
        ("git add -A && env FOO=1 git push origin main", True),
        # --- Must stay allowed ---
        ("git status", False),
        ("git pull", False),
        ("git pushup", False),
        ("git fetch origin pull/90/head:pr-90", False),
        ("git merge origin/main", False),
        ("echo push", False),
        ("env FOO=bar uv sync", False),
        ("uv run pytest && ruff check .", False),
        ("git commit -m push", False),
        ("env FOO=push git status", False),
        ("", False),
    ],
)
def test_dry_run_push_guard_resists_wrappers(command_line: str, *, blocked: bool) -> None:
    """Verify the dry-run push guard cannot be evaded by env or compound wrappers.

    The agent is instructed to prefix commands with ``env KEY=val`` (see
    ``get_system_instructions``), so a guard that only inspects ``argv[0]``
    would let real pushes through during a dry run.
    """
    block_push = next(p for p in get_dry_run_policies() if p.name == "dry_run_block_push")
    assert block_push.when({"command_line": command_line}) is blocked


def test_dry_run_push_guard_covers_both_arg_spellings() -> None:
    """Verify the guard reads both 'command_line' and legacy 'CommandLine' args."""
    block_push = next(p for p in get_dry_run_policies() if p.name == "dry_run_block_push")
    assert block_push.when({"CommandLine": "env FOO=1 git push origin main"}) is True
    assert block_push.when({}) is False


def test_dry_run_push_guard_applies_to_sandboxed_tool() -> None:
    """Verify the sandboxed runner is guarded identically to the builtin runner."""
    policies = get_dry_run_policies()
    sandboxed = next(p for p in policies if p.name == "dry_run_block_push_sandboxed")
    assert sandboxed.when({"command_line": "env FOO=1 git push origin main"}) is True
    assert sandboxed.when({"command_line": "git status"}) is False


def test_dry_run_push_guard_handles_unparseable_quoting() -> None:
    """Verify an unbalanced quote fails closed when it mentions a push."""
    block_push = next(p for p in get_dry_run_policies() if p.name == "dry_run_block_push")
    assert block_push.when({"command_line": 'git push origin "unclosed'}) is True


@pytest.mark.asyncio
async def test_workspace_cleanup_on_start(
    mock_agent_class: MagicMock,
    async_fs: type[AsyncFSHelper],
    github_token: str,
) -> None:
    """Verify that run_agent_for_repo cleans up stale workspace directories."""
    _ = mock_agent_class
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    repo = "test-owner/test-repo"
    repo_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
    workspace_dir = str(Path(tempfile.gettempdir()) / f"dependency-director-{repo_hash}")
    await async_fs.mkdir(Path(workspace_dir) / "stale-clone")
    stale_file = str(Path(workspace_dir) / "stale-clone" / "file.txt")
    await async_fs.write_text(stale_file, "leftover")
    assert await async_fs.exists(stale_file)
    await run_agent_for_repo(
        repo=repo,
        settings=settings,
        max_attempts=3,
        dry_run=False,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    assert not await async_fs.exists(stale_file)
    assert not await async_fs.exists(workspace_dir)


def test_bot_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default configuration values for automated bot setups."""
    monkeypatch.delenv("DEPDIRECTOR_BOTS", raising=False)
    settings = Settings()
    assert len(settings.bots) == 2
    authors = [b.author for b in settings.bots]
    assert "dependabot[bot]" in authors
    assert "renovate[bot]" in authors
    rebase_cmds = {b.author: b.rebase_command for b in settings.bots}
    assert rebase_cmds["dependabot[bot]"] == "@dependabot rebase"
    assert rebase_cmds["renovate[bot]"] == "@renovatebot rebase"


def test_bot_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify bot settings can be overridden using environment variables."""
    custom = [{"author": "custom[bot]", "rebase_command": "@custom rebase"}]
    monkeypatch.setenv("DEPDIRECTOR_BOTS", json.dumps(custom))
    settings = Settings()
    assert len(settings.bots) == 1
    assert settings.bots[0].author == "custom[bot]"
    assert settings.bots[0].rebase_command == "@custom rebase"


def test_default_bots_constant() -> None:
    """Verify the predefined list of default bot configurations is correct."""
    assert len(DEFAULT_BOTS) == 2
    assert DEFAULT_BOTS[0].author == "dependabot[bot]"
    assert DEFAULT_BOTS[1].author == "renovate[bot]"


@pytest.mark.asyncio
async def test_agent_config_vertex_fields(mock_agent_class: MagicMock, github_token: str) -> None:
    """Verify that Vertex AI settings are passed through to LocalAgentConfig."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = ""
    settings.vertex = True
    settings.google_cloud_project = "my-project"
    settings.google_cloud_location = "us-central1"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]
    assert config_passed.vertex is True
    assert config_passed.project == "my-project"
    assert config_passed.location == "us-central1"


@pytest.mark.asyncio
async def test_agent_config_no_vertex_excludes_project(mock_agent_class: MagicMock, github_token: str) -> None:
    """When vertex=False, project/location must NOT be passed even if env vars are set."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    settings.vertex = False
    settings.google_cloud_project = "stale-project-from-env"
    settings.google_cloud_location = "us-central1"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]
    assert config_passed.vertex is not True
    assert config_passed.project is None
    assert config_passed.location is None


@pytest.mark.parametrize(
    ("env_vars", "expected_vertex", "expected_project", "expected_location"),
    [
        (
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
                "GOOGLE_CLOUD_PROJECT": "my-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            },
            True,
            "my-project",
            "us-central1",
        ),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "true"}, True, "", ""),
        ({}, False, "", ""),
    ],
)
def test_settings_vertex_loading(
    monkeypatch: pytest.MonkeyPatch,
    env_vars: dict[str, str],
    *,
    expected_vertex: bool,
    expected_project: str,
    expected_location: str,
) -> None:
    """Verify configuration settings for Vertex AI are loaded correctly."""
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    settings = Settings()
    assert settings.vertex is expected_vertex
    assert settings.google_cloud_project == expected_project
    assert settings.google_cloud_location == expected_location


@pytest.mark.asyncio
async def test_agent_config_disables_builtin_run_command(mock_agent_class: MagicMock, github_token: str) -> None:
    """Verify that the agent config disables the built-in run_command tool and registers the custom one."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    settings.vertex = False
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]

    assert BuiltinTools.RUN_COMMAND in config_passed.capabilities.disabled_tools
    registered_tool_names = [getattr(t, "__name__", "") for t in config_passed.tools]
    assert "run_command_sandboxed" in registered_tool_names


@pytest.mark.asyncio
async def test_run_agent_for_repo_with_hint(mock_agent_class: MagicMock, github_token: str) -> None:
    """Verify the agent runs with the provided custom user hint."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    mock_agent_instance = mock_agent_class.return_value
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
        hint="Some test hint",
    )
    mock_agent_instance.chat.assert_called_once()
    prompt_arg = mock_agent_instance.chat.call_args[0][0]
    assert "Additional context: Some test hint" in prompt_arg


@pytest.mark.asyncio
async def test_run_agent_for_repo_processes_chunks(mock_agent_class: MagicMock, github_token: str) -> None:
    """Verify the agent correctly processes repository work chunks in order."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"

    async def _mock_chunks() -> AsyncGenerator[types.Text | types.Thought | types.ToolCall | types.ToolResult]:
        yield types.Text(text="Starting triage...", step_index=0)
        yield types.Thought(text="I need to check the PR status.", step_index=0)
        yield types.ToolCall(name="get_pull_request", args={"pr_number": 42})
        yield types.ToolResult(name="get_pull_request", error="404 Not Found")
        yield types.ToolCall(
            name="rebase_bot_pr",
            args={
                "pr_number": 42,
                "extremely_long_arg_to_test_truncation_logic_here_so_it_exceeds_eighty_characters": True,
            },
        )
        yield types.ToolResult(name="rebase_bot_pr", error=None)
        yield types.Text(text="Done!", step_index=1)

    mock_agent_instance = mock_agent_class.return_value
    mock_response = MagicMock()
    mock_response.chunks = _mock_chunks()
    mock_agent_instance.chat = AsyncMock(return_value=mock_response)
    mock_usage = mock_agent_instance.conversation.total_usage
    mock_usage.prompt_token_count = 10
    mock_usage.cached_content_token_count = 0
    mock_usage.candidates_token_count = 5
    mock_usage.thoughts_token_count = 2
    mock_usage.total_token_count = 17
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )


async def _empty_chunks() -> AsyncGenerator[types.Text]:
    """Yield no chunks, as a turn that produced nothing renderable does."""
    return
    yield types.Text(text="", step_index=0)


@pytest.mark.asyncio
async def test_run_agent_for_repo_reports_a_clean_turn_as_completed(
    mock_agent_class: MagicMock,
    github_token: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify a turn that logs nothing is still reported as a clean completion."""
    _ = mock_agent_class
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
    )
    assert "Agent execution completed for test-owner/test-repo" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_agent_for_repo_surfaces_errors_the_sdk_only_logged(
    mock_agent_class: MagicMock,
    github_token: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The SDK reports some terminal failures by logging, not raising or yielding a chunk.

    A model-output loop is the case seen in practice: the turn ends early with
    truncated output while the run still claims to have completed.
    """
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"

    async def _chat_that_logs_a_system_error(_prompt: str) -> MagicMock:
        logging.getLogger().warning(
            "System step error (HTTP %s): %s",
            0,
            "Detected a loop in the model's output.",
        )
        response = MagicMock()
        response.chunks = _empty_chunks()
        return response

    mock_agent_class.return_value.chat = AsyncMock(side_effect=_chat_that_logs_a_system_error)
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
    )
    output = capsys.readouterr().out
    assert "Detected a loop in the model's output." in output
    assert "Agent execution completed" not in output


@pytest.mark.asyncio
async def test_run_agent_for_repo_ignores_its_own_log_records(
    mock_agent_class: MagicMock,
    github_token: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Our own warnings are already on screen; re-reporting them would double up."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"

    async def _chat_that_triggers_our_own_warning(_prompt: str) -> MagicMock:
        logging.getLogger("dependency_director.main").warning("A warning we emitted ourselves")
        response = MagicMock()
        response.chunks = _empty_chunks()
        return response

    mock_agent_class.return_value.chat = AsyncMock(side_effect=_chat_that_triggers_our_own_warning)
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
    )
    assert "Agent execution completed for test-owner/test-repo" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_agent_for_repo_tool_error_hook(mock_agent_class: MagicMock, github_token: str) -> None:
    """Verify the agent triggers the error hook callback when a tool fails."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]
    assert len(config_passed.hooks) == 1
    hook_fn = config_passed.hooks[0]
    await hook_fn(Exception("Test tool failure exception"))


@pytest.mark.asyncio
async def test_agent_config_no_mcp_servers(mock_agent_class: MagicMock, github_token: str) -> None:
    """Agent should not use any MCP servers — all GitHub API access is via host tools."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]
    assert not config_passed.mcp_servers


@pytest.mark.asyncio
async def test_agent_config_registers_all_host_tools(mock_agent_class: MagicMock, github_token: str) -> None:
    """All 13 host tools (+ optional run_command) should be registered."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
    config_passed = mock_agent_class.call_args[1]["config"]
    tool_names = {t.__name__ for t in config_passed.tools}
    expected = {
        "list_bot_prs",
        "merge_bot_pr",
        "rebase_bot_pr",
        "wait_for_reviews",
        "get_pr_status",
        "wait_for_ci",
        "get_pr_workflow_run_logs",
        "get_pr_diff",
        "get_pr_files",
        "get_file_contents",
        "list_commits",
        "get_commit_details",
        "list_branches",
        "create_pr",
        "run_command_sandboxed",
    }
    assert tool_names == expected


@pytest.mark.asyncio
async def test_run_agent_for_repo_early_halt(
    mock_agent_class: MagicMock,
    capsys: pytest.CaptureFixture[str],
    github_token: str,
) -> None:
    """Verify run_agent_for_repo exits early when no matching PRs are open."""
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"
    with patch("dependency_director.tools.GitHubClient.list_open_prs", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = []
        await run_agent_for_repo(repo="test-owner/test-repo", settings=settings, max_attempts=3)
        mock_agent_class.assert_not_called()
        captured = capsys.readouterr()
        assert "Open Pull Requests (Initial List)" in captured.out
        assert "No open dependency update PRs were found for test-owner/test-repo" in captured.out
        assert "Final Summary" not in captured.out
        assert "No dependency update PRs were processed." not in captured.out


# --- Issue #7: _check_open_bot_prs uses provided client ---


@pytest.mark.asyncio
async def test_check_open_bot_prs_uses_provided_client() -> None:
    """_check_open_bot_prs must use the provided GitHubClient, not create its own.

    This eliminates the redundant client creation that happened on every repo.
    """
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.list_open_prs = AsyncMock(
        return_value=[
            {"number": 1, "title": "bump foo", "author": "dependabot[bot]", "created_at": "2026-01-01"},
            {"number": 2, "title": "bump bar", "author": "human-user", "created_at": "2026-01-02"},
            {"number": 3, "title": "bump baz", "author": "renovate[bot]", "created_at": "2026-01-03"},
        ],
    )

    result = await _check_open_bot_prs("owner", "repo", mock_client, DEFAULT_BOTS)

    mock_client.list_open_prs.assert_called_once_with("owner", "repo")
    assert len(result) == 2
    assert result[0]["number"] == 1
    assert result[1]["number"] == 3


@pytest.mark.asyncio
async def test_check_open_bot_prs_filters_by_custom_bots() -> None:
    """_check_open_bot_prs must respect the provided bots list for filtering."""
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.list_open_prs = AsyncMock(
        return_value=[
            {"number": 1, "title": "bump foo", "author": "dependabot[bot]", "created_at": "2026-01-01"},
            {"number": 2, "title": "bump bar", "author": "custom[bot]", "created_at": "2026-01-02"},
        ],
    )
    custom_bots = [BotConfig(author="custom[bot]", rebase_command="@custom rebase")]

    result = await _check_open_bot_prs("owner", "repo", mock_client, custom_bots)

    assert len(result) == 1
    assert result[0]["number"] == 2


# --- Issue #3: agent.chat error does not crash multi-repo sweep ---


@pytest.mark.asyncio
async def test_run_agent_for_repo_chat_error_does_not_crash(
    mock_agent_class: MagicMock,
    github_token: str,
) -> None:
    """A transient error from agent.chat() must not crash the entire run.

    It should log the error and complete gracefully so multi-repo sweeps
    continue processing the remaining repositories.
    """
    settings = Settings()
    settings.github_token = github_token
    settings.gemini_api_key = "placeholder-key"

    mock_agent_instance = mock_agent_class.return_value
    mock_agent_instance.chat = AsyncMock(side_effect=RuntimeError("Model produced invalid output"))

    # Must not raise — the error should be caught and logged
    await run_agent_for_repo(
        repo="test-owner/test-repo",
        settings=settings,
        max_attempts=3,
        dry_run=True,
        auto_merge=False,
        verify_all=False,
        standalone_fix=False,
        review_wait=0,
    )
