"""Tests for the dependency-director command-line interface."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from dependency_director.main import GITHUB_HOSTS, cli


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


def test_cli_max_attempts_help_describes_the_unit_it_counts() -> None:
    """--max-attempts bounds fix attempts per failing PR; 'repository chunk' is not a thing."""
    result = CliRunner().invoke(cli, ["--help"])
    assert "repository chunk" not in result.output
    assert "per failing PR" in result.output


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
            fix_base=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_model_option(mock_run_agent: MagicMock) -> None:
    """Verify CLI execution with explicit --model flag."""
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False
        mock_settings.model = "gemini-3.7-flash"

        result = runner.invoke(cli, ["test-owner/some-repo", "--model", "gemini-3.6-pro"])

        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with(
            "test-owner",
            1,
            3,
            "test-owner/some-repo",
            dry_run=False,
            auto_merge=False,
            verify_all=False,
            standalone_fix=False,
            fix_base=False,
            review_wait=0,
            hint=None,
            no_sandbox=False,
            model="gemini-3.6-pro",
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
            fix_base=False,
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
        ("https://www.github.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        # Hostnames are case-insensitive, in every form we accept one.
        ("https://GitHub.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("git@GitHub.com:test-owner/some-repo", "test-owner", "test-owner/some-repo"),
        ("GitHub.com/test-owner/some-repo", "test-owner", "test-owner/some-repo"),
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
            fix_base=False,
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
        # A bare host names no owner; it used to be read as an org called
        # "github.com" and scanned.
        "github.com",
        "github.com/",
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


@pytest.mark.parametrize(
    "foreign_target",
    [
        "https://evil.com/test-owner/some-repo",
        "http://github.com.evil.com/test-owner/some-repo",
        # github.com is the userinfo here; the host is evil.com.
        "https://github.com@evil.com/test-owner/some-repo",
        "git@gitlab.com:test-owner/some-repo",
    ],
)
@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_rejects_targets_hosted_elsewhere(
    mock_run_agent: MagicMock,
    foreign_target: str,
) -> None:
    """A URL naming another host must be refused, not resolved against GitHub anyway.

    Only the path was ever read out of these, so a GitLab clone URL or a
    lookalike host silently operated on whatever github.com/<same-path>
    happened to be — a different repo than the one that was asked for.
    """
    runner = CliRunner()
    with patch("dependency_director.main.Settings") as mock_settings_cls:
        mock_settings = mock_settings_cls.return_value
        mock_settings.concurrency = 1
        mock_settings.max_fix_attempts = 3
        mock_settings.review_wait = 0
        mock_settings.no_sandbox = False
        mock_settings.owner = None

        result = runner.invoke(cli, [foreign_target])

        assert result.exit_code != 0
        # Say which hosts would have worked, rather than just "invalid".
        assert all(host in result.output for host in GITHUB_HOSTS)
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
            fix_base=False,
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
            fix_base=False,
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
            fix_base=False,
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
            fix_base=False,
            review_wait=7,
            hint=None,
            no_sandbox=False,
        )


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_print_banner(_mock_run_agent: MagicMock) -> None:
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


def test_cli_exposes_fix_base_flag() -> None:
    """--fix-base must be discoverable and describe its blast radius."""
    result = CliRunner().invoke(cli, ["--help"])
    assert "--fix-base" in result.output
    assert "separate PR" in result.output


def test_cli_fix_base_defaults_to_off() -> None:
    """An agent opening PRs unrelated to any dependency update must be opted into."""
    with patch("dependency_director.main.run_agent", new_callable=AsyncMock) as mock_run:
        CliRunner().invoke(cli, ["owner/repo", "--dry-run"])
    assert mock_run.call_args.kwargs["fix_base"] is False


def test_cli_fix_base_reaches_run_agent() -> None:
    """The flag must actually thread through, not just exist in --help."""
    with patch("dependency_director.main.run_agent", new_callable=AsyncMock) as mock_run:
        CliRunner().invoke(cli, ["owner/repo", "--dry-run", "--fix-base"])
    assert mock_run.call_args.kwargs["fix_base"] is True
