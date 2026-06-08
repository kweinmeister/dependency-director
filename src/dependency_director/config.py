"""Configuration settings and command filtering logic for dependency-director."""

import importlib.resources
from typing import Any

from google.antigravity.hooks import policy
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
DEFAULT_SRT_SETTINGS_PATH: str = str(
    importlib.resources.files("dependency_director") / "srt-settings.json",
)

SAFE_ENV_ALLOWLIST: set[str] = {
    "BUNDLE_PATH",
    "CARGO_HOME",
    "CC",
    "CFLAGS",
    "CI",
    "CONDA_PREFIX",
    "CXX",
    "CXXFLAGS",
    "DOTNET_CLI_HOME",
    "DOTNET_ROOT",
    "GEM_HOME",
    "GEM_PATH",
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
    "GRADLE_USER_HOME",
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
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
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
    concurrency: int = Field(default=1, validation_alias="depdirector_concurrency")
    review_wait: int = Field(default=0, validation_alias="depdirector_review_wait")
    bots: list[BotConfig] = Field(
        default=DEFAULT_BOTS,
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
        import shlex
        from pathlib import Path

        try:
            tokens = shlex.split(cmd_stripped)
        except ValueError:
            return "git" in cmd_stripped and "push" in cmd_stripped
        if not tokens:
            return False
        exe_name = Path(tokens[0]).name.lower()
        if exe_name == "git":
            return "push" in [t.lower() for t in tokens[1:]]
        return False

    return [
        policy.deny("merge_pull_request", name="dry_run_block_merge"),
        policy.deny("mcp_github_merge_pull_request", name="dry_run_block_mcp_merge"),
        policy.deny("github_merge_pull_request", name="dry_run_block_github_merge"),
        policy.deny("run_command", when=is_git_push, name="dry_run_block_push"),
        policy.deny(
            "run_command_sandboxed",
            when=is_git_push,
            name="dry_run_block_push_sandboxed",
        ),
    ]
