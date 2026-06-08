from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from dependency_director.main import cli


@pytest.fixture(autouse=True)
def mock_srt() -> Generator[None]:
    with patch("dependency_director.main.is_srt_available", return_value=True):
        yield


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "dependency-director" in result.output


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_owner_repo(mock_run_agent: MagicMock) -> None:
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


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_no_target_uses_env(mock_run_agent: MagicMock) -> None:
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


# ============================================================
# CLI edge cases
# ============================================================


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_target_trailing_slash_rejected(mock_run_agent: MagicMock) -> None:
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
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert "TARGET" in result.output
    assert "user/org" in result.output.lower() or "owner/repo" in result.output.lower()


@patch("dependency_director.main.run_agent", new_callable=AsyncMock)
def test_cli_review_wait_default_from_env(mock_run_agent: MagicMock) -> None:
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
def test_cli_print_banner(_mock_run_agent: MagicMock) -> None:
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
