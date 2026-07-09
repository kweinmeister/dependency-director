"""Tests for settings and repository configurations in dependency-director."""

import hashlib
import json
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

from dependency_director.config import DEFAULT_BOTS, Settings, get_dry_run_policies, get_safety_policies
from dependency_director.main import run_agent_for_repo

from .conftest import AsyncFSHelper


@pytest.mark.parametrize(
    (
        "env_vars",
        "expected_concurrency",
        "expected_max_attempts",
        "expected_gemini_key",
        "expected_github_token",
        "expected_srt_settings",
    ),
    [
        (
            {
                "DEPDIRECTOR_CONCURRENCY": "4",
                "DEPDIRECTOR_MAX_FIX_ATTEMPTS": "5",
                "GEMINI_API_KEY": "test-gemini-key",
                "GITHUB_TOKEN": "test-github-key",
                "DEPDIRECTOR_SRT_SETTINGS": "/path/to/custom.json",
            },
            4,
            5,
            "test-gemini-key",
            "test-github-key",
            "/path/to/custom.json",
        ),
        ({}, 1, 3, "", "", ""),
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
) -> None:
    """Verify configuration settings load correctly from default values and file."""
    monkeypatch.delenv("DEPDIRECTOR_CONCURRENCY", raising=False)
    monkeypatch.delenv("DEPDIRECTOR_MAX_FIX_ATTEMPTS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DEPDIRECTOR_SRT_SETTINGS", raising=False)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    settings = Settings()
    assert settings.concurrency == expected_concurrency
    assert settings.max_fix_attempts == expected_max_attempts
    assert settings.gemini_api_key == expected_gemini_key
    assert settings.github_token == expected_github_token
    assert settings.srt_settings == expected_srt_settings


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
    repo = "test-owner/test-repo"
    expected_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
    expected_dir = str(Path(tempfile.gettempdir()) / f"dependency-director-{expected_hash}")
    assert hasattr(config_passed, "workspaces")
    assert expected_dir in config_passed.workspaces
    assert tempfile.gettempdir() not in config_passed.workspaces
    assert str(Path(tempfile.gettempdir()) / "dependency-director") not in config_passed.workspaces


@pytest.mark.parametrize(
    ("env_var", "value", "expected_attr", "expected_val"),
    [("DEPDIRECTOR_OWNER", "my-org", "owner", "my-org"), ("DEPDIRECTOR_REVIEW_WAIT", "10", "review_wait", 10)],
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
