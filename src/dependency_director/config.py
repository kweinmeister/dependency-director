"""Configuration settings and command filtering logic for dependency-director."""

import importlib.resources
import shlex
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from google.antigravity.hooks import policy
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dependency_director.argv import is_git_push_argv


class BotConfig(BaseModel):
    """Configuration for an allowed dependency bot.

    Defines its author name and rebase command.
    """

    author: str
    rebase_command: str


DEFAULT_BOTS: list[BotConfig] = [
    BotConfig(author="dependabot[bot]", rebase_command="@dependabot rebase"),
    BotConfig(author="renovate[bot]", rebase_command="@renovatebot rebase"),
]
DEFAULT_COMMAND_TIMEOUT = 300

# Caps on the command output handed back to the model. A dependency install
# can emit tens of thousands of lines whose middle says nothing; the head names
# what ran and the tail carries the failure. Either cap may be set to 0 to
# disable it.
DEFAULT_MAX_OUTPUT_LINES = 200
DEFAULT_MAX_OUTPUT_CHARS = 24000

# Share of each cap spent on the head. The tail gets the rest, because that is
# where errors land.
OUTPUT_HEAD_FRACTION = 0.2


class OutputLimits(NamedTuple):
    """Caps applied to a single command's stdout and stderr.

    Attributes:
        max_lines: Maximum lines kept per stream; 0 disables the line cap.
        max_chars: Maximum characters kept per stream; 0 disables the char cap.

    """

    max_lines: int = DEFAULT_MAX_OUTPUT_LINES
    max_chars: int = DEFAULT_MAX_OUTPUT_CHARS


DEFAULT_OUTPUT_LIMITS = OutputLimits()

# Failed CI jobs whose logs are returned for a single PR. Several workflows can
# fail on one commit, and they usually fail for the same underlying reason, so
# a handful of tails is enough to diagnose without flooding the context window.
DEFAULT_MAX_FAILED_JOBS = 3

# Lines kept from the end of each failed job's log. Failures report at the
# bottom; the setup output above it is noise.
DEFAULT_WORKFLOW_LOG_TAIL_LINES = 50


class LogLimits(NamedTuple):
    """Caps applied to the CI logs fetched for one pull request.

    Attributes:
        max_failed_jobs: Failed jobs whose logs are returned; the rest are
            counted and reported, never dropped silently.
        tail_lines: Lines kept from the end of each job's log.

    Neither may be 0: a run that fetches logs and then returns none of them is
    a wasted round trip, and 'no failures found' would be a lie.
    """

    max_failed_jobs: int = DEFAULT_MAX_FAILED_JOBS
    tail_lines: int = DEFAULT_WORKFLOW_LOG_TAIL_LINES


DEFAULT_LOG_LIMITS = LogLimits()

# Workflow run conclusions that cannot contain a failed job, so the run's jobs
# are never fetched. Everything else — including an unset conclusion on a run
# still in progress — is examined, since the per-job conclusion is the real
# filter and an extra API call is cheaper than a missed failure.
PASSING_RUN_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})

# Shared package cache for every sandboxed command. It deliberately sits outside
# the per-repo workspace, which is deleted before and after each repository —
# a cache inside it would make every repo re-download every dependency.
DEFAULT_CACHE_DIR: str = str(Path(tempfile.gettempdir()) / "dependency-director-cache")

DEFAULT_SRT_SETTINGS_PATH: str = str(
    importlib.resources.files("dependency_director") / "srt-settings.json",
)

SAFE_ENV_ALLOWLIST: set[str] = {
    "BUNDLE_PATH",
    "CC",
    "CFLAGS",
    "CI",
    "CONDA_PREFIX",
    "CXX",
    "CXXFLAGS",
    "DOTNET_ROOT",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_NAME",
    "GIT_COMMITTER_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_TERMINAL_PROMPT",
    "GITHUB_ACTIONS",
    "GITHUB_REF",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ID",
    "GITHUB_SHA",
    "GOBIN",
    "GOPATH",
    "GOROOT",
    "HOME",
    "JAVA_HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LDFLAGS",
    "LOGNAME",
    "M2_HOME",
    "MAKEFLAGS",
    "MAVEN_HOME",
    "NODE_PATH",
    "NVM_DIR",
    "OLDPWD",
    "PATH",
    "PIP_EXTRA_INDEX_URL",
    "PIP_INDEX_URL",
    "PKG_CONFIG_PATH",
    "PWD",
    "PYENV_ROOT",
    "RBENV_ROOT",
    "RUNNER_ARCH",
    "RUNNER_OS",
    "RUSTUP_HOME",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX_URL",
    "XDG_RUNTIME_DIR",
}


class Settings(BaseSettings):
    """Global configuration settings for dependency-director.

    Loaded from environment variables.
    """

    gemini_api_key: str = Field(default="", validation_alias="gemini_api_key")
    github_token: str = Field(default="", validation_alias="github_token")
    max_fix_attempts: int = Field(
        default=3,
        validation_alias="depdirector_max_fix_attempts",
    )
    owner: str = Field(default="", validation_alias="depdirector_owner")
    concurrency: int = Field(default=1, ge=1, validation_alias="depdirector_concurrency")
    review_wait: int = Field(default=0, validation_alias="depdirector_review_wait")
    bots: list[BotConfig] = Field(
        default=DEFAULT_BOTS,
        min_length=1,
        validation_alias="depdirector_bots",
    )
    vertex: bool = Field(default=False, validation_alias="google_genai_use_vertexai")
    google_cloud_project: str = Field(
        default="",
        validation_alias="google_cloud_project",
    )
    google_cloud_location: str = Field(
        default="",
        validation_alias="google_cloud_location",
    )
    no_sandbox: bool = Field(
        default=False,
        validation_alias="depdirector_no_sandbox",
    )
    srt_settings: str = Field(
        default="",
        validation_alias="depdirector_srt_settings",
    )
    command_timeout: int = Field(
        default=DEFAULT_COMMAND_TIMEOUT,
        ge=10,
        validation_alias="depdirector_command_timeout",
    )
    model: str = Field(
        default="gemini-3.7-flash",
        validation_alias="depdirector_model",
    )
    max_output_lines: int = Field(
        default=DEFAULT_MAX_OUTPUT_LINES,
        ge=0,
        validation_alias="depdirector_max_output_lines",
    )
    max_output_chars: int = Field(
        default=DEFAULT_MAX_OUTPUT_CHARS,
        ge=0,
        validation_alias="depdirector_max_output_chars",
    )
    max_failed_jobs: int = Field(
        default=DEFAULT_MAX_FAILED_JOBS,
        ge=1,
        validation_alias="depdirector_max_failed_jobs",
    )
    workflow_log_tail_lines: int = Field(
        default=DEFAULT_WORKFLOW_LOG_TAIL_LINES,
        ge=1,
        validation_alias="depdirector_workflow_log_tail_lines",
    )
    cache_dir: str = Field(
        default=DEFAULT_CACHE_DIR,
        validation_alias="depdirector_cache_dir",
    )

    @property
    def output_limits(self) -> OutputLimits:
        """Return the configured command output caps."""
        return OutputLimits(max_lines=self.max_output_lines, max_chars=self.max_output_chars)

    @property
    def log_limits(self) -> LogLimits:
        """Return the configured CI-log caps."""
        return LogLimits(max_failed_jobs=self.max_failed_jobs, tail_lines=self.workflow_log_tail_lines)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_safety_policies() -> list[Any]:
    """Define the SDK safety policies for run_command and file edits."""
    return [
        policy.allow("*"),
    ]


def get_dry_run_policies() -> list[Any]:
    """Define the extra safety policies inserted when running in dry-run mode."""

    def is_git_push(args: dict[str, Any]) -> bool:
        cmd = args.get("command_line") or args.get("CommandLine") or ""
        cmd_stripped = cmd.strip()
        if not cmd_stripped:
            return False
        try:
            tokens = shlex.split(cmd_stripped)
        except ValueError:
            # Unparseable quoting: fail closed rather than guess.
            return "git" in cmd_stripped and "push" in cmd_stripped
        return is_git_push_argv(tokens)

    return [
        policy.deny("run_command", when=is_git_push, name="dry_run_block_push"),
        policy.deny(
            "run_command_sandboxed",
            when=is_git_push,
            name="dry_run_block_push_sandboxed",
        ),
    ]
