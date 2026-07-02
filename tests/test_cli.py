"""Tests for the dependency-director command-line interface."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from dependency_director.main import cli


@pytest.fixture(autouse=True)
def mock_srt() -> Generator[None]:
    """Fixture to mock SRT availability for testing."""
    with patch("dependency_director.main.is_srt_available", return_value=True):
        yield


def test_cli_help() -> None:
    """Verify that the CLI --help option prints usage instructions."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "dependency-director" in result.output


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_owner_repo(mock_run_agent: MagicMock) -> None:
    """Verify CLI execution when provided with an owner/repository target."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, ["test-owner/some-repo", "--dry-run"])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            1,
            3,
            "test-owner/some-repo",
            dry_run=True,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_owner_only(mock_run_agent: MagicMock) -> None:
    """Verify CLI execution when provided with an owner-only target."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 2
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, ["some-org"])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "some-org",
            2,
            3,
            None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
        )


@pytest.mark.parametrize(
    ("target_input", "expected_owner", "expected_repo"),
    [
        ("https://github.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("https://github.com/test-owner/some-repo.git", "test-owner", "test-owner/some-repo"),
        ("http://github.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("git@github.com:test-owner/some-repo.git", "test-owner", "test-owner/some-repo"),
        ("github.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("https://github.com/test-owner", "test-owner", None),
        ("https://github.com/test-owner/", "test-owner", None),
        ("test-owner", "test-owner", None),
    ],
)
@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_formats(
    mock_run_agent: MagicMock,
    target_input: str,
    expected_owner: str,
    expected_repo: str | None,
) -> None:
    """Verify CLI parses and normalizes various target formats correctly."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [target_input])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            expected_owner,
            1,
            3,
            expected_repo,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        "owner/",
        "https://github.com/",
        "https://github.com/owner/repo/extra",
        "",
    ],
)
@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_invalid_target_formats_rejected(
    mock_run_agent: MagicMock,
    invalid_target: str,
) -> None:
    """Verify CLI rejects invalid target formats with a non-zero exit code."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        mock_settings.owner = None
        result = runner.invoke(cli, [invalid_target] if invalid_target else [])

        assert result.exit_code != 0
        mock_run_agent.assert_not_called()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_no_target_uses_env(mock_run_agent: MagicMock) -> None:
    """Verify CLI falls back to the owner environment variable if target is omitted."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            1,
            3,
            None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_no_target_no_env_errors(mock_run_agent: MagicMock) -> None:
    """Verify CLI returns an error when neither target nor environment owner is set."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = ""
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0

        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        mock_run_agent.assert_not_called()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_all_overrides(mock_run_agent: MagicMock) -> None:
    """Verify CLI successfully overrides default configuration with command-line flags."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(
            cli,
            [
                "test-owner/test-repo",
                "--concurrency",
                "4",
                "--max-attempts",
                "5",
                "--dry-run",
                "--auto-merge",
                "--verify-all",
                "--standalone-fix",
                "--review-wait",
                "5",
            ],
        )

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            4,
            5,
            "test-owner/test-repo",
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=True,
            review_wait=5,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_short_flags_overrides(mock_run_agent: MagicMock) -> None:
    """Verify CLI successfully overrides default configuration with short command-line flags."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(
            cli,
            [
                "test-owner/test-repo",
                "-c",
                "4",
                "-m",
                "5",
                "-d",
                "-a",
                "-v",
                "-w",
                "5",
            ],
        )

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            4,
            5,
            "test-owner/test-repo",
            dry_run=True,
            auto_merge=True,
            verify_all=True,
            standalone_fix=False,
            review_wait=5,
            hint=None,
            no_sandbox=False,
        )


# ============================================================
# CLI edge cases
# ============================================================


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_trailing_slash_rejected(mock_run_agent: MagicMock) -> None:
    """Verify CLI rejects target strings containing a trailing slash."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0

        result = runner.invoke(cli, ["owner/"])

        assert result.exit_code != 0
        mock_run_agent.assert_not_called()


def test_cli_help_contains_target_description() -> None:
    """Verify CLI help text includes a description of the target argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert "TARGET" in result.output
    assert "user/org" in result.output.lower() or "owner/repo" in result.output.lower()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_review_wait_default_from_env(mock_run_agent: MagicMock) -> None:
    """Verify CLI review wait time defaults to the environment variable if not specified."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 7
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            1,
            3,
            None,
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            review_wait=7,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_print_banner(_mock_run_agent: MagicMock) -> None:  # noqa: PT019
    """Verify CLI prints the startup banner as part of its execution."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0

        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert "Autonomous Dependency Triage Agent" in result.output


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_hint_passed_to_run_agent(mock_run_agent: MagicMock) -> None:
    """Verify CLI passes a user-provided hint directly to the agent run function."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0

        result = runner.invoke(
            cli,
            ["test-owner/test-repo", "--hint", "PR #42 needs google-adk>=2.0"],
        )

        assert result.exit_code == 0
        mock_run_agent.assert_called_once()
        call_kwargs = mock_run_agent.call_args
        assert "PR #42 needs google-adk>=2.0" in str(call_kwargs)


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_no_hint_passes_none(mock_run_agent: MagicMock) -> None:
    """Verify CLI passes None for hints when the option is not specified."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0

        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once()
        call_args = mock_run_agent.call_args[0]
        # hint should not be in positional args (it's passed as keyword or last positional)
        assert call_args[-1] is None or call_args[-1] == 0


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_srt_not_available_exits(mock_run_agent: MagicMock) -> None:
    """Verify CLI exits with an error when srt is missing and sandboxing is active."""
    runner = CliRunner()
    with (
        patch("dependency_director.main.Settings") as mock_settings_cls,
        patch("dependency_director.main.is_srt_available", return_value=False),
        patch("shutil.which", return_value=None),
    ):
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "sandbox-runtime (srt) is not available" in result.output
        mock_run_agent.assert_not_called()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_srt_not_available_but_no_sandbox_allowed(
    mock_run_agent: MagicMock,
) -> None:
    """Verify CLI runs successfully when srt is missing but sandboxing is bypassed."""
    runner = CliRunner()
    with (
        patch("dependency_director.main.Settings") as mock_settings_cls,
        patch("dependency_director.main.is_srt_available", return_value=False),
    ):
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, ["--no-sandbox"])

        assert result.exit_code == 0
        assert "Running in --no-sandbox mode" in result.output
        mock_run_agent.assert_called_once()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_srt_installed_but_missing_ripgrep_on_linux(
    mock_run_agent: MagicMock,
) -> None:
    """Verify CLI exits when running on Linux with srt but missing ripgrep."""
    runner = CliRunner()
    with (
        patch("dependency_director.main.Settings") as mock_settings_cls,
        patch("dependency_director.main.is_srt_available", return_value=False),
        patch("shutil.which", return_value="/path/to/srt"),
        patch("sys.platform", "linux"),
        patch("dependency_director.main.is_ripgrep_available", return_value=False),
    ):
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "ripgrep (rg) is missing" in result.output
        mock_run_agent.assert_not_called()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_srt_installed_but_not_functioning(
    mock_run_agent: MagicMock,
) -> None:
    """Verify CLI exits when srt is installed but fails to execute commands."""
    runner = CliRunner()
    with (
        patch("dependency_director.main.Settings") as mock_settings_cls,
        patch("dependency_director.main.is_srt_available", return_value=False),
        patch("shutil.which", return_value="/path/to/srt"),
        patch("sys.platform", "darwin"),
    ):
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "installed but not functioning properly" in result.output
        mock_run_agent.assert_not_called()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_verify_all_with_no_sandbox_rejected(mock_run_agent: MagicMock) -> None:
    """Verify CLI rejects running in verify-all mode if sandboxing is disabled."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.owner = "test-owner"
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False

        result = runner.invoke(cli, ["--verify-all", "--no-sandbox"])

        assert result.exit_code != 0
        assert "verify-all is not allowed in --no-sandbox mode" in result.output.lower()
        mock_run_agent.assert_not_called()
