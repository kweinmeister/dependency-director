"""Sandboxed command execution and argv validation for dependency-director."""

import asyncio
import contextlib
import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from dependency_director.argv import (
    extract_git_config_arg,
    is_forced_git_push_argv,
    resolve_exe,
    split_compound_argv,
)
from dependency_director.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_OUTPUT_LIMITS,
    DEFAULT_SRT_SETTINGS_PATH,
    OUTPUT_HEAD_FRACTION,
    SAFE_ENV_ALLOWLIST,
    OutputLimits,
)
from dependency_director.schemas import SrtSettings


class CommandResult(NamedTuple):
    """Result from executing a single sandboxed command.

    Attributes:
        output: The formatted command output string.
        returncode: The process exit code (-1 for errors before execution).

    """

    output: str
    returncode: int


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if child is equal to or contained within parent."""
    try:
        child.relative_to(parent)
    except ValueError:
        return child == parent
    else:
        return True


def _validate_target_path(
    path_str: str,
    workspace_dir: str,
    context_msg: str,
) -> str | None:
    """Validate a path string against directory traversal and disallowed locations."""
    if path_str == "/":
        return f"Security Error: {context_msg} targeting root directory is denied."
    if ".." in Path(path_str).parts:
        return f"Security Error: {context_msg} with directory traversal is denied."

    try:
        path = Path(path_str).expanduser()
        resolved = (Path(workspace_dir) / path).resolve() if not path.is_absolute() else path.resolve()
    except (ValueError, RuntimeError, OSError) as e:
        return f"Security Error: {context_msg} has invalid path: {e}"

    workspace_path = Path(workspace_dir).resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()
    fallback_temps = [Path(os.sep + p).resolve() for p in ("tmp", f"private{os.sep}tmp")]

    in_workspace = _is_under(resolved, workspace_path)
    in_tmp = _is_under(resolved, system_temp) or any(_is_under(resolved, fp) for fp in fallback_temps)

    if not (in_workspace or in_tmp):
        return (
            f"Security Error: {context_msg} targeting path outside workspace and temp directory is denied: {path_str}"
        )
    return None


# git options that hand control of the transport or the repository location
# to the caller. Matched against the token's pre-'=' half, so both the
# '--opt=value' and '--opt value' spellings are covered.
BLOCKED_GIT_OPTIONS = (
    "--config-env",
    "--git-dir",
    "--receive-pack",
    "--upload-pack",
)

# Configuration keys that turn a git invocation into arbitrary execution, a
# credential leak, or a request to somewhere other than the intended remote.
# Matched as prefixes, so 'url.' covers every 'url.<base>.insteadOf'.
BLOCKED_GIT_CONFIG_PREFIXES = (
    "core.fsmonitor",
    "core.hookspath",
    "core.sshcommand",
    "credential.helper",
    "http.proxy",
    "https.proxy",
    "protocol.ext.allow",
    "remote.",
    "url.",
)

# 'git config' flags that take a separate value, which must not be mistaken
# for the key being set.
_GIT_CONFIG_FLAGS_WITH_ARG = ("-f", "--file", "--blob", "-t", "--type", "--default", "--comment")
_GIT_CONFIG_SUBCOMMANDS = frozenset({"set", "get", "unset"})


def _check_env_var_token(token: str) -> str | None:
    if "=" in token and not token.startswith("-"):
        parts = token.split("=", 1)
        env_key = parts[0].strip().lower()
        env_value = parts[1].strip().lower() if len(parts) > 1 else ""

        blocked_env_vars = {
            "ld_preload",
            "ld_library_path",
            "dyld_insert_libraries",
            "dyld_library_path",
            "git_ssh_command",
            "git_ssh",
        }
        if env_key in blocked_env_vars:
            return f"Security Error: Environment variable '{parts[0]}' is blocked."

        if "git_config" in env_key or env_key == "git_config_parameters":
            for prefix in BLOCKED_GIT_CONFIG_PREFIXES:
                if prefix in env_value:
                    return (
                        f"Security Error: Environment variable '{parts[0]}' "
                        f"contains blocked git configuration key '{prefix}'."
                    )
    return None


def _check_blocked_executables(exe_name: str) -> str | None:
    blocked_commands = {
        "sudo",
        "su",
        "nc",
        "netcat",
        "curl",
        "wget",
        "systemctl",
        "service",
        "init",
        "reboot",
        "shutdown",
        "halt",
        "poweroff",
    }
    if exe_name in blocked_commands:
        return f"Security Error: Command '{exe_name}' is blocked."
    return None


def _check_rm_command(tokens: list[str], idx: int, workspace_dir: str) -> str | None:
    """Return an error if 'rm' names a path outside the workspace."""
    for j in range(idx + 1, len(tokens)):
        err = _validate_target_path(tokens[j], workspace_dir, "Command 'rm'")
        if err:
            return err
    return None


def _blocked_config_key(key: str) -> str | None:
    """Return the blocked prefix ``key`` falls under, or None if it is allowed."""
    normalized = key.split("=", 1)[0].strip().lower()
    return next(
        (prefix for prefix in BLOCKED_GIT_CONFIG_PREFIXES if normalized.startswith(prefix)),
        None,
    )


def _check_git_config_subcommand(tokens: list[str], idx: int) -> str | None:
    """Return an error if 'git config' sets one of the blocked keys."""
    skip_next = False
    for config_token in tokens[idx + 1 :]:
        if skip_next:
            skip_next = False
            continue
        if config_token in _GIT_CONFIG_FLAGS_WITH_ARG:
            skip_next = True
            continue
        if config_token.startswith("-"):
            continue
        if config_token.lower() in _GIT_CONFIG_SUBCOMMANDS:
            continue
        prefix = _blocked_config_key(config_token)
        if prefix:
            return f"Security Error: Git configuration key '{prefix}' is blocked."
        break
    return None


def _check_git_command(tokens: list[str], idx: int) -> str | None:
    """Return an error if a git invocation sets a blocked option or config key.

    Covers both routes to the same capability: an option on the git command
    itself, and a configuration key set either with '-c' or by the 'config'
    subcommand. They are checked together because a key blocked on one route
    and allowed on the other is not blocked at all.
    """
    for tok in tokens[idx + 1 :]:
        option = tok.split("=", 1)[0].lower()
        if option in BLOCKED_GIT_OPTIONS:
            return f"Security Error: Git option '{option}' is blocked."

    for j, tok in enumerate(tokens[idx + 1 :], start=idx + 1):
        config_expr = extract_git_config_arg(tokens, j)
        if config_expr is not None:
            prefix = _blocked_config_key(config_expr)
            if prefix:
                return f"Security Error: Git configuration key '{prefix}' is blocked."
        elif tok == "config":
            err = _check_git_config_subcommand(tokens, j)
            if err:
                return err
    return None


def _check_shell_operators(argv: list[str]) -> str | None:
    """Return an error if dangerous shell operators are present as tokens."""
    dangerous_operators = {";", "|", "&"}
    if dangerous_operators & set(argv):
        return (
            "Security Error: Shell operators (;, |, &) are not "
            "allowed. Use separate run_command_sandboxed calls instead."
        )
    return None


class CheckedExe(NamedTuple):
    """The executable a command runs, once its env assignments have passed.

    Attributes:
        idx: Index of the executable token in the argv list.
        name: Basename of that executable, lowercased.

    """

    idx: int
    name: str


def _resolve_exe(argv: list[str]) -> CheckedExe | str:
    """Return the executable a command runs, or an error string.

    Identify the executable index and normalized name, then validate every
    ``KEY=val`` assignment attached to it — whether written bare or passed
    through ``env``, since both reach the process the same way.
    """
    resolved = resolve_exe(argv)
    if resolved is None:
        return "Security Error: No executable found in command."

    for token in resolved.env_assignments:
        err = _check_env_var_token(token)
        if err:
            return err

    if Path(argv[resolved.wrapper_idx]).name.lower() == "env":
        for token in argv[resolved.wrapper_idx + 1 :]:
            if token in ("-S", "--split-string"):
                return "Security Error: 'env -S/--split-string' is blocked."

    return CheckedExe(idx=resolved.exe_idx, name=resolved.name)


def _check_exec_pivots(exe_name: str, argv: list[str], exe_idx: int) -> str | None:
    """Block find -exec/-execdir pivots and xargs."""
    if exe_name == "find":
        for arg in argv[exe_idx + 1 :]:
            if arg in ("-exec", "-execdir", "-ok", "-okdir"):
                return f"Security Error: 'find' with '{arg}' is blocked."
    if exe_name == "xargs":
        return "Security Error: Command 'xargs' is blocked."
    return None


def _check_git_force_push(argv: list[str]) -> str | None:
    """Refuse a forced push: it discards commits nobody agreed to lose."""
    if not is_forced_git_push_argv(argv):
        return None
    return (
        "Security Error: Forced push is blocked. It overwrites whatever the remote "
        "branch already carried, including commits a reviewer or an earlier run "
        "pushed there. Push without forcing; if the remote rejects that as "
        "non-fast-forward, the branch already holds work you must not discard — "
        "call 'find_open_pr_for_branch' and reconcile instead."
    )


def _check_per_exe(exe: CheckedExe, argv: list[str], workspace_dir: str) -> str | None:
    """Dispatch per-executable security checks (pivots, denylist, rm, git)."""
    err = _check_exec_pivots(exe.name, argv, exe.idx) or _check_blocked_executables(exe.name)
    if err:
        return err
    if exe.name == "rm":
        return _check_rm_command(argv, exe.idx, workspace_dir)
    if exe.name == "git":
        return _check_git_command(argv, exe.idx) or _check_git_force_push(argv)
    return None


def _validate_single_argv(argv: list[str], workspace_dir: str) -> str | None:
    """Validate a single (non-compound) argv array for sandboxed execution."""
    if not argv:
        return "Security Error: Empty command."

    # Reject dangerous shell operators (;, |, &) — these are never safe.
    # Note: && and || are handled at the compound level, not here.
    err = _check_shell_operators(argv)
    if err:
        return err

    # Find the executable (skip leading FOO=bar env assignments)
    exe = _resolve_exe(argv)
    if isinstance(exe, str):
        return exe

    # Block SHELL interpreters only (no legitimate use after Phase 2).
    # Language runtimes (python3, node, ruby, etc.) are allowed —
    # agent needs them for test execution; srt OS sandbox is the boundary.
    blocked_shells = {"bash", "sh", "dash", "zsh", "ksh"}
    if exe.name in blocked_shells:
        return f"Security Error: Shell interpreter '{exe.name}' is blocked."

    return _check_per_exe(exe, argv, workspace_dir)


def validate_argv(argv: list[str], workspace_dir: str) -> str | None:
    """Validate an argv array for sandboxed execution (defense-in-depth).

    Supports compound commands joined by ``&&`` and ``||``.  Each
    sub-command is validated independently.  Dangerous operators
    (``;``, ``|``, ``&``) remain blocked.
    """
    if not argv:
        return "Security Error: Empty command."

    parts = split_compound_argv(argv)
    for part in parts:
        err = _validate_single_argv(part.argv, workspace_dir)
        if err:
            return err
    return None


def _splice_middle[T: (str, list[str])](units: T, cap: int, marker_for: Callable[[int], T]) -> T:
    """Keep the head and tail of ``units``, naming what was dropped between.

    ``units`` is measured in whatever the cap counts: a string for a character
    budget, a list of lines for a line budget. ``marker_for`` returns the
    replacement for the middle in the same currency, so its own size is
    charged against the cap rather than added on top of it — otherwise every
    truncated stream comes back over the limit the caller configured.
    """
    if cap <= 0 or len(units) <= cap:
        return units

    # Reserve against the longest the marker can get. Its length only grows
    # with the omitted count, so sizing on the whole input can never underrun.
    budget = max(cap - len(marker_for(len(units))), 0)
    head_n = int(budget * OUTPUT_HEAD_FRACTION)
    tail_n = budget - head_n
    head = units[:head_n]
    tail = units[len(units) - tail_n :] if tail_n else units[:0]
    return (head + marker_for(len(units) - budget) + tail)[:cap]


def truncate_middle(text: str, limits: OutputLimits) -> str:
    """Trim the middle out of ``text`` to fit within ``limits``.

    Keeps a head and a tail because each answers a different question: the
    head shows what ran, the tail shows how it ended. The line cap runs first
    so ordinary logs are trimmed on line boundaries; the character cap then
    catches output that is short on lines but enormous on one of them.
    """
    if limits.max_lines > 0:
        text = "\n".join(
            _splice_middle(text.splitlines(), limits.max_lines, lambda n: [f"... [{n} lines omitted] ..."]),
        )

    return _splice_middle(text, limits.max_chars, lambda n: f"\n... [{n} characters omitted] ...\n")


def _format_command_result(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
    limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
) -> str:
    stdout_str = stdout.decode(errors="replace")
    stderr_str = stderr.decode(errors="replace")

    diagnostics = []
    # Read diagnostics off the full text: a blocked request lands wherever it
    # happened, which for a long install is nowhere near either end.
    combined = stdout_str + stderr_str
    if returncode != 0:
        if "Connection blocked by network allowlist" in combined or "X-Proxy-Error: blocked-by-allowlist" in combined:
            diagnostics.append(
                "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy. "
                "Allowed domains are configured in srt-settings.json.",
            )
        if "Operation not permitted" in stderr_str or "Permission denied" in stderr_str:
            diagnostics.append(
                "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied' / 'Operation not permitted'. "
                "If this path is outside the allowed directories, it was blocked by the "
                "sandbox-runtime policy configured in srt-settings.json.",
            )

    parts = [
        "--- STDOUT ---",
        truncate_middle(stdout_str, limits),
        "--- STDERR ---",
        truncate_middle(stderr_str, limits),
        f"--- EXIT CODE: {returncode} ---",
    ]
    if diagnostics:
        parts.append("\n" + "\n".join(diagnostics))
    return "\n".join(parts)


def _setup_cache_env(cache_dir: str, env: dict[str, str]) -> None:
    cache_base = Path(cache_dir)
    env["BUN_INSTALL"] = str(cache_base / "bun")
    env["CARGO_HOME"] = str(cache_base / "cargo")
    env["COMPOSER_CACHE_DIR"] = str(cache_base / "composer")
    env["DENO_DIR"] = str(cache_base / "deno")
    env["DOTNET_CLI_HOME"] = str(cache_base / "dotnet")
    env["GEM_HOME"] = str(cache_base / "gems")
    env["GEM_PATH"] = str(cache_base / "gems")
    env["GOMODCACHE"] = str(cache_base / "go" / "pkg" / "mod")
    env["GRADLE_USER_HOME"] = str(cache_base / "gradle")
    env["MIX_HOME"] = str(cache_base / "mix")
    env["NPM_CONFIG_CACHE"] = str(cache_base / "npm")
    env["NUGET_PACKAGES"] = str(cache_base / "nuget")
    env["PIP_CACHE_DIR"] = str(cache_base / "pip")
    env["PIPENV_CACHE_DIR"] = str(cache_base / "pipenv")
    env["PNPM_HOME"] = str(cache_base / "pnpm")
    env["POETRY_CACHE_DIR"] = str(cache_base / "poetry")
    env["PUB_CACHE"] = str(cache_base / "pub-cache")
    env["UV_CACHE_DIR"] = str(cache_base / "uv")
    env["UV_CONFIG_FILE"] = "/dev/null"
    env["UV_PYTHON_INSTALL_DIR"] = str(cache_base / "uv" / "python")
    env["UV_PYTHON_PREFERENCE"] = "only-system"
    env["UV_TOOL_DIR"] = str(cache_base / "uv" / "tools")
    env["XDG_CACHE_HOME"] = str(cache_base)
    env["YARN_CACHE_FOLDER"] = str(cache_base / "yarn")


def _ensure_workspace_tmp(workspace_dir: str) -> Path:
    """Create and return the per-workspace `.tmp` directory."""
    sandbox_tmp = Path(workspace_dir) / ".tmp"
    sandbox_tmp.mkdir(parents=True, exist_ok=True)
    return sandbox_tmp


def _setup_tmp_env(workspace_dir: str, env: dict[str, str]) -> Path:
    """Configure temporary directory environment variables inside the workspace.

    Sets CLAUDE_CODE_TMPDIR (and CLAUDE_TMPDIR), as well as standard POSIX and
    Windows environment variables (TEMP, TMP), to a dedicated .tmp directory
    inside the workspace. srt automatically injects CLAUDE_CODE_TMPDIR as the
    sandboxed child's TMPDIR, ensuring test runners (like pytest's pytest-of-<user>)
    and compilers write strictly within the sandbox's allowed boundaries.

    Keeps the parent srt process's TMPDIR bounded to a short standard system
    temp directory (e.g. tempfile.gettempdir() or /tmp) to prevent srt's internal
    Unix domain socket (srt-mux-*.sock) from exceeding POSIX sockaddr_un sun_path
    limits (104 bytes on macOS/BSD, 108 bytes on Linux) when running inside
    deeply nested workspace paths.
    """
    sandbox_tmp = _ensure_workspace_tmp(workspace_dir)
    tmp_path_str = str(sandbox_tmp)
    env["CLAUDE_CODE_TMPDIR"] = tmp_path_str
    env["CLAUDE_TMPDIR"] = tmp_path_str
    env["TEMP"] = tmp_path_str
    env["TMP"] = tmp_path_str

    # Ensure the parent srt process uses a short standard system temp directory
    # to avoid POSIX socket length limits (e.g. sockaddr_un sun_path 104-108 bytes)
    # in deeply nested workspaces, while sandboxed children receive CLAUDE_CODE_TMPDIR.
    system_tmp = tempfile.gettempdir()
    env["TMPDIR"] = system_tmp if len(system_tmp) < 50 else "/tmp"
    return sandbox_tmp


def _is_git_command_argv(argv: list[str]) -> bool:
    """Check if the executable resolved from an argv list is git."""
    resolved = resolve_exe(argv)
    return resolved is not None and resolved.name == "git"


class SandboxConfig(NamedTuple):
    """Immutable configuration for sandboxed command execution.

    Attributes:
        workspace_dir: Root directory for sandboxed file operations.
        srt_config: Parsed srt settings model (filesystem, network rules).
        github_token: Token for authenticated git operations.
        command_timeout: Max seconds per sandboxed command.
        output_limits: Caps on the output returned to the model.
        cache_dir: Shared package cache, outside the workspace so it persists.

    """

    workspace_dir: str
    srt_config: SrtSettings
    github_token: str | None = None
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS
    cache_dir: str = DEFAULT_CACHE_DIR


class SandboxedCommandRunner:
    """Callable tool runner that executes command line strings inside srt sandbox."""

    __name__ = "run_command_sandboxed"

    def __init__(
        self,
        config: SandboxConfig,
        config_path: str | None,
        init_error: str | None,
    ) -> None:
        """Initialize the sandboxed command runner."""
        self.cfg = config
        self.config_path = config_path
        self.init_error = init_error

    def cleanup(self) -> None:
        """Remove any temporary sandbox configurations."""
        if self.config_path:
            path = self.config_path
            self.config_path = None
            with contextlib.suppress(Exception):
                Path(path).unlink()

    def __del__(self) -> None:
        """Cleanup on object destruction."""
        self.cleanup()

    async def __call__(
        self,
        command_line: str,
        working_dir: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a shell command. Sandboxed on macOS by default.

        Supports compound commands joined by ``&&`` and ``||``.
        Each sub-command is validated and executed independently through srt.

        Args:
            command_line: The exact command line string to run.
            working_dir: The directory to run the command in.
            cwd: The directory to run the command in.
            Cwd: Legacy fallback parameter for working directory, supported for backward
                compatibility with certain models/SDK configurations.
            **kwargs: Additional keyword arguments for compatibility.

        """
        if self.init_error:
            return self.init_error

        legacy_cwd = kwargs.get("Cwd")
        target_cwd = working_dir or cwd or legacy_cwd or self.cfg.workspace_dir

        try:
            argv = shlex.split(command_line)
        except ValueError:
            return "Security Error: Invalid shell command quoting."

        validation_err = validate_argv(argv, self.cfg.workspace_dir)
        if validation_err:
            return validation_err

        parts = split_compound_argv(argv)

        # Simple case: no compound operators
        if len(parts) == 1:
            return (await self._run_single_argv(parts[0].argv, target_cwd)).output

        # Compound: run each sub-command with && / || semantics
        all_outputs: list[str] = []
        last_returncode = 0
        for part in parts:
            if part.operator == "&&" and last_returncode != 0:
                break  # short-circuit: previous failed
            if part.operator == "||" and last_returncode == 0:
                continue  # short-circuit: previous succeeded

            result = await self._run_single_argv(part.argv, target_cwd)
            all_outputs.append(result.output)
            last_returncode = result.returncode

        return "\n".join(all_outputs)

    async def _run_single_argv(
        self,
        argv: list[str],
        target_cwd: str,
    ) -> CommandResult:
        """Execute a single (non-compound) argv array through srt."""
        env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_ALLOWLIST}
        _setup_cache_env(self.cfg.cache_dir, env)
        _setup_tmp_env(self.cfg.workspace_dir, env)

        is_git_command = _is_git_command_argv(argv)

        if is_git_command:
            env["GIT_CONFIG_GLOBAL"] = "/dev/null"
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_TEMPLATE_DIR"] = "/dev/null"
            token = self.cfg.github_token or os.environ.get("GITHUB_TOKEN")
            if token:
                env["GIT_CONFIG_COUNT"] = "2"
                env["GIT_CONFIG_KEY_0"] = f"url.https://x-access-token:{token}@github.com/.insteadOf"
                env["GIT_CONFIG_VALUE_0"] = "https://github.com/"
                env["GIT_CONFIG_KEY_1"] = f"url.https://x-access-token:{token}@github.com/.insteadOf"
                env["GIT_CONFIG_VALUE_1"] = "git@github.com:"
                env["GIT_TERMINAL_PROMPT"] = "0"

        active_config_path = self.config_path
        git_config_temp_path = None

        if is_git_command:
            git_config = self.cfg.srt_config.model_copy(deep=True)
            git_config.filesystem.allowGitConfig = True

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".json",
                    mode="w",
                    delete=False,
                ) as temp_f:
                    temp_f.write(git_config.model_dump_json())
                    git_config_temp_path = temp_f.name
                    active_config_path = git_config_temp_path
            except (OSError, TypeError, ValueError) as e:
                return CommandResult(f"Error: Failed to create temporary config for Git: {e}", -1)

        try:
            if not active_config_path:
                return CommandResult("Error: Sandbox configuration was not initialized.", -1)

            process = await asyncio.create_subprocess_exec(
                "srt",
                "--settings",
                active_config_path,
                "--",
                *argv,
                cwd=target_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            comm_coro = process.communicate()
            try:
                stdout, stderr = await asyncio.wait_for(comm_coro, timeout=self.cfg.command_timeout)
            except TimeoutError:
                comm_coro.close()
                with contextlib.suppress(Exception):
                    process.kill()
                await process.wait()
                return CommandResult(f"Error: Command timed out after {self.cfg.command_timeout} seconds.", -1)
            rc = process.returncode if process.returncode is not None else -1
            return CommandResult(_format_command_result(stdout, stderr, rc, self.cfg.output_limits), rc)
        finally:
            if git_config_temp_path:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(Path(git_config_temp_path).unlink, missing_ok=True)


def create_run_command_tool(
    workspace_dir: str,
    srt_settings_path: str | Path = "",
    github_token: str | None = None,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    output_limits: OutputLimits = DEFAULT_OUTPUT_LIMITS,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Callable[..., Awaitable[str]]:
    """Create a sandboxed run_command tool bound to a workspace."""
    config_path = None
    init_error = None
    srt_config = SrtSettings()

    # The cache outlives the workspace, so it may already exist from an earlier repo.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _ensure_workspace_tmp(workspace_dir)

    base_path = srt_settings_path or DEFAULT_SRT_SETTINGS_PATH
    try:
        with Path(base_path).open() as f:
            raw_srt_config = json.load(f)

        srt_config = SrtSettings.model_validate(raw_srt_config)
        srt_config.filesystem.allowGitConfig = False

        for path in (workspace_dir, cache_dir):
            if path not in srt_config.filesystem.allowWrite:
                srt_config.filesystem.allowWrite.append(path)
            if path not in srt_config.filesystem.allowRead:
                srt_config.filesystem.allowRead.append(path)

        if workspace_dir not in srt_config.git.safeDirectories:
            srt_config.git.safeDirectories.append(workspace_dir)

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            mode="w",
            delete=False,
        ) as temp_f:
            temp_f.write(srt_config.model_dump_json())
            config_path = temp_f.name

    except FileNotFoundError:
        init_error = f"Error: Sandbox settings file not found at {base_path}."
    except json.JSONDecodeError:
        init_error = f"Error: Sandbox settings file at {base_path} is not valid JSON."
    except ValidationError as e:
        init_error = f"Error: Sandbox settings file at {base_path} failed schema validation: {e}"

    runner = SandboxedCommandRunner(
        config=SandboxConfig(
            workspace_dir=workspace_dir,
            srt_config=srt_config,
            github_token=github_token,
            command_timeout=command_timeout,
            output_limits=output_limits,
            cache_dir=cache_dir,
        ),
        config_path=config_path,
        init_error=init_error,
    )

    async def run_command_sandboxed(
        command_line: str,
        working_dir: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a shell command. Sandboxed on macOS by default.

        Args:
            command_line: The exact command line string to run.
            working_dir: The directory to run the command in.
            cwd: The directory to run the command in.
            **kwargs: Additional keyword arguments for compatibility.

        """
        return await runner(command_line, working_dir=working_dir, cwd=cwd, **kwargs)

    setattr(run_command_sandboxed, "cleanup", runner.cleanup)  # noqa: B010

    return run_command_sandboxed


def is_srt_available() -> bool:
    """Check if sandbox-runtime (srt) is available and functional."""
    try:
        res = subprocess.run(
            ["srt", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return False
    else:
        return res.returncode == 0


def is_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is installed and available in PATH."""
    try:
        res = subprocess.run(
            ["rg", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    else:
        return res.returncode == 0
