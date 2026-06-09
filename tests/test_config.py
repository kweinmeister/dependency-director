import hashlib
import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.antigravity import types
from google.antigravity.hooks import hooks, policy

from dependency_director.config import (
    DEFAULT_BOTS,
    Settings,
    get_dry_run_policies,
    get_safety_policies,
)
from dependency_director.main import run_agent_for_repo


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
    monkeypatch.setenv("DEPDIRECTOR_CONCURRENCY", "not-an-int")
    # Pydantic Settings should raise ValidationError on type mismatch
    with pytest.raises(Exception):
        Settings()


# ============================================================
# Policy integration — verifies wiring, not logic
# ============================================================


def test_safety_policy_structure() -> None:
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

    injected = types.ToolCall(
        name="run_command",
        args={"command_line": "npm test; rm -rf /"},
    )
    assert (await hook.run(ctx, injected)).allow is True

    allowed = types.ToolCall(name="run_command", args={"command_line": "echo hello"})
    assert (await hook.run(ctx, allowed)).allow is True


def test_dry_run_policies_enforcement() -> None:
    policies = get_dry_run_policies()
    names = [p.name for p in policies]
    assert "dry_run_block_push" in names

    # Find the block push policy and test it
    block_push = next(p for p in policies if p.name == "dry_run_block_push")
    assert block_push.when({"CommandLine": "git push origin main"}) is True
    assert block_push.when({"command_line": "git push origin main"}) is True
    assert block_push.when({"CommandLine": "git status"}) is False
    assert block_push.when({"command_line": "git status"}) is False


@pytest.mark.asyncio
async def test_agent_config_includes_skills_path(mock_agent_class: MagicMock) -> None:
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

    mock_agent_instance = mock_agent_class.return_value
    mock_response = AsyncMock()
    mock_response.text.return_value = "Triage completed successfully."
    mock_agent_instance.chat = AsyncMock(return_value=mock_response)

    mock_usage = mock_agent_instance.conversation.total_usage
    mock_usage.prompt_token_count = 100
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

    # Verify the directory exists and contains the expected skill
    assert Path(skills_dir).exists(), f"Skills directory {skills_dir} does not exist"
    skill_folder = Path(skills_dir) / "code-review-and-quality"
    assert skill_folder.exists(), f"Skill folder {skill_folder} does not exist"
    skill_file = skill_folder / "SKILL.md"
    assert skill_file.exists(), f"Skill definition file {skill_file} does not exist"

    # Verify workspaces configuration

    repo = "test-owner/test-repo"
    expected_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
    expected_dir = str(
        Path(tempfile.gettempdir()) / f"dependency-director-{expected_hash}",
    )
    assert hasattr(config_passed, "workspaces")
    assert expected_dir in config_passed.workspaces
    assert tempfile.gettempdir() not in config_passed.workspaces
    assert (
        str(Path(tempfile.gettempdir()) / "dependency-director")
        not in config_passed.workspaces
    )


# ============================================================
# Settings edge cases
# ============================================================


@pytest.mark.parametrize(
    ("env_var", "value", "expected_attr", "expected_val"),
    [
        ("DEPDIRECTOR_OWNER", "my-org", "owner", "my-org"),
        ("DEPDIRECTOR_REVIEW_WAIT", "10", "review_wait", 10),
    ],
)
def test_settings_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    value: str,
    expected_attr: str,
    expected_val: Any,
) -> None:
    monkeypatch.setenv(env_var, value)
    settings = Settings()
    assert getattr(settings, expected_attr) == expected_val


# ============================================================
# Policy edge cases
# ============================================================


def test_dry_run_git_push_substring() -> None:
    policies = get_dry_run_policies()
    block_push = next(p for p in policies if p.name == "dry_run_block_push")
    assert block_push.when({"command_line": "git push origin main"}) is True
    assert block_push.when({"command_line": "git pushup"}) is False
    assert block_push.when({"command_line": "git pull"}) is False
    assert block_push.when({"command_line": "git -C dir push"}) is True


# ============================================================
# Workspace tests
# ============================================================


@pytest.mark.asyncio
async def test_workspace_cleanup_on_start(mock_agent_class: MagicMock) -> None:
    """Verify that run_agent_for_repo cleans up stale workspace directories."""
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

    repo = "test-owner/test-repo"
    repo_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
    workspace_dir = str(
        Path(tempfile.gettempdir()) / f"dependency-director-{repo_hash}",
    )

    # Create a stale file in the workspace to simulate leftover from a prior run
    Path(workspace_dir, "stale-clone").mkdir(parents=True, exist_ok=True)
    stale_file = str(Path(workspace_dir) / "stale-clone" / "file.txt")
    with Path(stale_file).open("w") as f:
        f.write("leftover")

    assert Path(stale_file).exists()

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

    # Stale content should be gone, workspace cleaned up after run
    assert not Path(stale_file).exists()
    assert not Path(workspace_dir).exists()


# ============================================================
# BotConfig defaults and env var override
# ============================================================


def test_bot_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
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
    custom = [{"author": "custom[bot]", "rebase_command": "@custom rebase"}]
    monkeypatch.setenv("DEPDIRECTOR_BOTS", json.dumps(custom))
    settings = Settings()
    assert len(settings.bots) == 1
    assert settings.bots[0].author == "custom[bot]"
    assert settings.bots[0].rebase_command == "@custom rebase"


def test_default_bots_constant() -> None:
    assert len(DEFAULT_BOTS) == 2
    assert DEFAULT_BOTS[0].author == "dependabot[bot]"
    assert DEFAULT_BOTS[1].author == "renovate[bot]"


# ============================================================
# Vertex AI settings
# ============================================================


@pytest.mark.asyncio
async def test_agent_config_vertex_fields(mock_agent_class: MagicMock) -> None:
    """Verify that Vertex AI settings are passed through to LocalAgentConfig."""
    settings = Settings()
    settings.github_token = "dummy-token"
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
async def test_agent_config_no_vertex_excludes_project(
    mock_agent_class: MagicMock,
) -> None:
    """When vertex=False, project/location must NOT be passed even if env vars are set."""
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"
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
async def test_agent_config_disables_builtin_run_command(
    mock_agent_class: MagicMock,
) -> None:
    """Verify that the agent config disables the built-in run_command tool and registers the custom one."""
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"
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
    from google.antigravity.types import BuiltinTools

    assert BuiltinTools.RUN_COMMAND in config_passed.capabilities.disabled_tools
    registered_tool_names = [getattr(t, "__name__", "") for t in config_passed.tools]
    assert "run_command_sandboxed" in registered_tool_names


@pytest.mark.asyncio
async def test_run_agent_for_repo_with_hint(mock_agent_class: MagicMock) -> None:
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

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
async def test_run_agent_for_repo_processes_chunks(mock_agent_class: MagicMock) -> None:
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

    async def _mock_chunks() -> AsyncGenerator[
        types.Text | types.Thought | types.ToolCall | types.ToolResult
    ]:
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
async def test_run_agent_for_repo_tool_error_hook(mock_agent_class: MagicMock) -> None:
    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

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


# --- agent tool registration ---


@pytest.mark.asyncio
async def test_agent_config_no_mcp_servers(mock_agent_class: MagicMock) -> None:
    """Agent should not use any MCP servers — all GitHub API access is via host tools."""
    settings = Settings()
    settings.github_token = "test-pat-token"
    settings.gemini_api_key = "dummy-key"

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
async def test_agent_config_registers_all_host_tools(
    mock_agent_class: MagicMock,
) -> None:
    """All 12 host tools (+ optional run_command) should be registered."""
    settings = Settings()
    settings.github_token = "test-pat-token"
    settings.gemini_api_key = "dummy-key"

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
) -> None:
    """If list_open_prs returns no PRs matching bots, the agent should not be spawned and run_agent_for_repo should return early."""
    from unittest.mock import patch

    from dependency_director.config import Settings
    from dependency_director.main import run_agent_for_repo

    settings = Settings()
    settings.github_token = "dummy-token"
    settings.gemini_api_key = "dummy-key"

    with patch(
        "dependency_director.tools.GitHubClient.list_open_prs",
        new_callable=AsyncMock,
    ) as mock_list:
        mock_list.return_value = []

        await run_agent_for_repo(
            repo="test-owner/test-repo",
            settings=settings,
            max_attempts=3,
        )

        mock_agent_class.assert_not_called()

        captured = capsys.readouterr()
        assert "Open Pull Requests (Initial List)" in captured.out
        assert (
            "No open dependency update PRs were found for test-owner/test-repo"
            in captured.out
        )
        assert "Final Summary" not in captured.out
        assert "No dependency update PRs were processed." not in captured.out
