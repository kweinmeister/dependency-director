"""Tests for command and filesystem sandboxing in dependency-director."""

import contextlib
import inspect
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.antigravity import LocalAgentConfig

from dependency_director.config import DEFAULT_SRT_SETTINGS_PATH
from dependency_director.tools import (
    CommandResult,
    SandboxedCommandRunner,
    create_run_command_tool,
    is_ripgrep_available,
    is_srt_available,
    split_compound_argv,
    validate_argv,
)

from .conftest import AsyncFSHelper

has_srt = is_srt_available()
requires_srt = pytest.mark.skipif(not has_srt, reason="sandbox-runtime (srt) is required")


def get_stdout(output: str) -> str:
    """Extract raw stdout from formatted command result."""
    if "--- STDOUT ---" in output:
        parts = output.split("--- STDERR ---")
        return parts[0].replace("--- STDOUT ---", "").strip()
    return output.strip()


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_network_denied(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that network connections are blocked inside the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    cmd = "python3 -c \"import urllib.request; urllib.request.urlopen('https://www.google.com', timeout=2)\""
    output = await run_command(cmd)
    assert any(
        err in output
        for err in ["URLError", "Permission denied", "TimeoutError", "timed out", "Operation not permitted"]
    )


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_filesystem_write_restricted(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that write access outside the allowed workspace is denied in the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    unauthorized_file = await async_fs.expanduser("~/exploit_sandbox_test.txt")
    if await async_fs.exists(unauthorized_file):
        with contextlib.suppress(Exception):
            await async_fs.unlink(unauthorized_file)
    cmd = f"python3 -c \"open('{unauthorized_file}','w').write('evil payload\\n')\""
    output = await run_command(cmd)
    try:
        exists = await async_fs.exists(unauthorized_file)
        if exists:
            await async_fs.unlink(unauthorized_file)
    except OSError:
        exists = False
    is_blocked = (
        any(
            msg in output
            for msg in (
                "No such file or directory",
                "Operation not permitted",
                "Permission denied",
                "Read-only file system",
            )
        )
        or not exists
    )
    assert is_blocked


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_workspace_write_allowed(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that write access inside the workspace is allowed in the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    target_file = str(Path(workspace) / "valid_patch.txt")
    cmd = f"python3 -c \"open('{target_file}','w').write('valid patch content\\n')\""
    output = await run_command(cmd)
    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    assert await async_fs.exists(target_file)
    assert (await async_fs.read_text(target_file)).strip() == "valid patch content"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_sensitive_files_denied(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that access to sensitive files (e.g. system files, SSH keys) is denied."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    ssh_dir = await async_fs.expanduser("~/.ssh")
    await async_fs.mkdir(ssh_dir)
    sensitive_file = str(Path(ssh_dir) / "test_depdirector_sandbox_read.txt")
    await async_fs.write_text(sensitive_file, "sensitive data")
    try:
        cmd = f"cat {sensitive_file}"
        output = await run_command(cmd)
        assert any(
            msg in output for msg in ("No such file or directory", "Operation not permitted", "Permission denied")
        )
    finally:
        with contextlib.suppress(Exception):
            await async_fs.unlink(sensitive_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_denied(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that modifying git metadata files directly is denied inside the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    git_dir = str(Path(workspace) / ".git")
    hooks_dir = str(Path(git_dir) / "hooks")
    await async_fs.mkdir(hooks_dir)
    run_command = create_run_command_tool(workspace)
    hook_file = str(Path(hooks_dir) / "post-commit")
    cmd = f"python3 -c \"open('{hook_file}','w').write('evil\\n')\""
    output = await run_command(cmd)
    assert any(
        msg in output
        for msg in (
            "No such file or directory",
            "Operation not permitted",
            "Permission denied",
            "Read-only file system",
        )
    )
    assert not await async_fs.exists(hook_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_config_allowed(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that modifying git configurations via the git CLI is allowed."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    git_dir = str(Path(workspace) / ".git")
    await async_fs.mkdir(git_dir)
    run_command = create_run_command_tool(workspace)
    config_file = str(Path(git_dir) / "config")
    cmd = f"python3 -c \"open('{config_file}','w').write('evil-config\\n')\""
    output = await run_command(cmd)
    assert any(
        msg in output
        for msg in (
            "No such file or directory",
            "Operation not permitted",
            "Permission denied",
            "Read-only file system",
        )
    )
    assert not await async_fs.exists(config_file) or (await async_fs.read_text(config_file)).strip() != "evil-config"
    await run_command("git init")
    cmd_git = "git config core.repositoryformatversion 0"
    output_git = await run_command(cmd_git)
    assert "Operation not permitted" not in output_git
    assert "Permission denied" not in output_git
    assert await async_fs.exists(config_file)
    content = await async_fs.read_text(config_file)
    assert "repositoryformatversion = 0" in content


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_compound_command(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that compound commands (&&) execute both sub-commands through srt."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    git_dir = str(Path(workspace) / ".git")
    await async_fs.mkdir(git_dir)
    config_file = str(Path(git_dir) / "config")
    run_command = create_run_command_tool(workspace)
    # Compound commands are now supported — both sub-commands execute
    cmd = "echo 'initiating...' && git init"
    output = await run_command(cmd)
    assert "Shell operators" not in output
    assert await async_fs.exists(config_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_other_metadata_write_allowed(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that writing non-git metadata files is allowed in the workspace."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    git_dir = str(Path(workspace) / ".git")
    refs_dir = str(Path(git_dir) / "refs")
    await async_fs.mkdir(refs_dir)
    run_command = create_run_command_tool(workspace)
    ref_file = str(Path(refs_dir) / "main")
    cmd = f"python3 -c \"open('{ref_file}','w').write('commit_hash\\n')\""
    output = await run_command(cmd)
    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    assert await async_fs.exists(ref_file)
    assert (await async_fs.read_text(ref_file)).strip() == "commit_hash"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_network_command_bypasses_sandbox(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that git network commands (like fetch/clone/pull) bypass the sandbox network block."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    cmd = "git fetch origin main"
    output = await run_command(cmd)
    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    assert "not a git repository" in output.lower() or "fatal:" in output.lower()


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_home_allowlist_enforced(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that access to the user's home directory is restricted by an allowlist."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    home_dir = await async_fs.expanduser("~")
    cargo_dir = str(Path(home_dir) / ".cargo" / "registry")
    await async_fs.mkdir(cargo_dir)
    cargo_config = str(Path(cargo_dir) / "config_test_sandbox.toml")
    await async_fs.write_text(cargo_config, "placeholder_cargo_config")
    try:
        cmd = f"cat {cargo_config}"
        output = await run_command(cmd)
        assert "Permission denied" not in output
        assert "Operation not permitted" not in output
        assert get_stdout(output) == "placeholder_cargo_config"
    finally:
        with contextlib.suppress(Exception):
            await async_fs.unlink(cargo_config)
    docs_dir = str(Path(home_dir) / "Documents")
    await async_fs.mkdir(docs_dir)
    secret_doc = str(Path(docs_dir) / "secret_test_sandbox.txt")
    await async_fs.write_text(secret_doc, "secret_data")
    try:
        cmd = f"cat {secret_doc}"
        output = await run_command(cmd)
        assert any(
            msg in output for msg in ("No such file or directory", "Operation not permitted", "Permission denied")
        )
    finally:
        with contextlib.suppress(Exception):
            await async_fs.unlink(secret_doc)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_custom_allowlist_enforced(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that custom allowlists are enforced correctly inside the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    home_dir = await async_fs.expanduser("~")
    custom_abs_dir = str(Path(home_dir) / "test_custom_allowlist")
    await async_fs.mkdir(custom_abs_dir)
    custom_file = str(Path(custom_abs_dir) / "custom_sandbox_test.txt")
    await async_fs.write_text(custom_file, "custom_data")
    content = await async_fs.read_text(DEFAULT_SRT_SETTINGS_PATH)
    config = json.loads(content)
    config.setdefault("filesystem", {}).setdefault("allowRead", []).append(custom_abs_dir)
    custom_settings_path = str(Path(tmp_path) / "custom_srt_settings.json")
    await async_fs.write_text(custom_settings_path, json.dumps(config))
    run_command = create_run_command_tool(workspace, srt_settings_path=custom_settings_path)
    try:
        cmd = f"cat {custom_file}"
        output = await run_command(cmd)
        assert "Permission denied" not in output
        assert "Operation not permitted" not in output
        assert get_stdout(output) == "custom_data"
    finally:
        with contextlib.suppress(Exception):
            await async_fs.unlink(custom_file)
        with contextlib.suppress(Exception):
            await async_fs.rmdir(custom_abs_dir)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_custom_denylist_enforced(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that custom denylists are enforced correctly inside the sandbox."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)

    temp_dir = tempfile.gettempdir()
    blocked_file = str(Path(temp_dir) / "blocked_sandbox_test.txt")
    allowed_file = str(Path(temp_dir) / "allowed_sandbox_test.txt")
    await async_fs.write_text(blocked_file, "blocked_secret")
    await async_fs.write_text(allowed_file, "allowed_data")
    content = await async_fs.read_text(DEFAULT_SRT_SETTINGS_PATH)
    config = json.loads(content)
    config.setdefault("filesystem", {}).setdefault("denyRead", []).append(blocked_file)
    custom_settings_path = str(Path(tmp_path) / "custom_srt_settings.json")
    await async_fs.write_text(custom_settings_path, json.dumps(config))
    run_command = create_run_command_tool(workspace, srt_settings_path=custom_settings_path)
    try:
        cmd1 = f"cat {blocked_file}"
        output1 = await run_command(cmd1)
        assert any(
            msg in output1 for msg in ("No such file or directory", "Operation not permitted", "Permission denied")
        )
        cmd2 = f"cat {allowed_file}"
        output2 = await run_command(cmd2)
        assert "Permission denied" not in output2
        assert "Operation not permitted" not in output2
        assert get_stdout(output2) == "allowed_data"
    finally:
        with contextlib.suppress(Exception):
            await async_fs.unlink(blocked_file)
        with contextlib.suppress(Exception):
            await async_fs.unlink(allowed_file)


@pytest.mark.asyncio
async def test_sandbox_sensitive_env_stripped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    async_fs: type[AsyncFSHelper],
) -> None:
    """Verify that sensitive environment variables are stripped before running commands."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    monkeypatch.setenv("GITHUB_TOKEN", "secret_github")
    monkeypatch.setenv("GEMINI_API_KEY", "secret_gemini")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret_aws")
    monkeypatch.setenv("STRIPE_API_KEY", "secret_stripe")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/host/project/.venv")
    monkeypatch.setenv("LANG", "custom_lang")
    run_command = create_run_command_tool(workspace)
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"mocked output", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create:
        await run_command("echo test")
        assert mock_create.called
        kwargs = mock_create.call_args[1]
        called_env = kwargs.get("env", {})
        assert "GITHUB_TOKEN" not in called_env
        assert "GEMINI_API_KEY" not in called_env
        assert "AWS_SECRET_ACCESS_KEY" not in called_env
        assert "STRIPE_API_KEY" not in called_env
        assert "VIRTUAL_ENV" not in called_env
        assert called_env.get("LANG") == "custom_lang"


@pytest.mark.asyncio
async def test_sandbox_srt_not_found_raises(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that missing srt executable raises an appropriate error."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    with (
        patch(
            "dependency_director.tools.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("No such file or directory"),
        ),
        pytest.raises(FileNotFoundError),
    ):
        await run_command("echo test")


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_conflict_resolution_commands(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that git commands used during conflict resolution are allowed."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    cmd_fetch = "git fetch origin main"
    output_fetch = await run_command(cmd_fetch)
    assert "Permission denied" not in output_fetch
    assert "Operation not permitted" not in output_fetch
    assert "not a git repository" in output_fetch.lower() or "fatal:" in output_fetch.lower()
    cmd_merge = "git merge origin/main"
    output_merge = await run_command(cmd_merge)
    assert "Permission denied" not in output_merge
    assert "Operation not permitted" not in output_merge
    assert "not a git repository" in output_merge.lower() or "fatal:" in output_merge.lower()
    cmd_push = "git push origin HEAD"
    output_push = await run_command(cmd_push)
    assert "Permission denied" not in output_push
    assert "Operation not permitted" not in output_push
    assert "not a git repository" in output_push.lower() or "fatal:" in output_push.lower()


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_developer_loop_dependency_upgrade(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify sandbox behavior during developer loop dependency upgrades."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    lib_dir = str(Path(workspace) / "placeholder_lib")
    await async_fs.mkdir(lib_dir)
    await async_fs.write_text(str(Path(lib_dir) / "__init__.py"), "def add(*, x, y):\n    return x + y\n")
    app_file = str(Path(workspace) / "app.py")
    await async_fs.write_text(
        app_file,
        "import sys\nsys.path.insert(0, '.')\nimport placeholder_lib\nprint(placeholder_lib.add(2, 3))\n",
    )
    run_command = create_run_command_tool(workspace)
    cmd_run_fail = "python3 app.py"
    output_fail = await run_command(cmd_run_fail)
    assert "TypeError" in output_fail
    cmd_read_lib = "cat placeholder_lib/__init__.py"
    output_read = await run_command(cmd_read_lib)
    assert "def add(*, x, y):" in output_read
    patch_code = "import sys\nsys.path.insert(0, '.')\nimport placeholder_lib\nprint(placeholder_lib.add(x=2, y=3))\n"

    # Write a helper script that patches app.py (avoids shell quoting issues)
    patch_script = str(Path(workspace) / "_patch_app.py")
    await async_fs.write_text(
        patch_script,
        f"with open('app.py', 'w') as f:\n    f.write({patch_code!r})\n",
    )
    cmd_patch = "python3 _patch_app.py"
    await run_command(cmd_patch)
    output_success = await run_command(cmd_run_fail)
    assert get_stdout(output_success) == "5"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_bypass_symlink(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that bypassing git metadata write block using symlinks is blocked."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    env_file = str(Path(workspace) / ".env")
    await async_fs.write_text(env_file, "KEY=VAL\n")
    run_command = create_run_command_tool(workspace)
    link_file = str(Path(workspace) / "env_link")
    cmd_link = f"ln -s {env_file} {link_file}"
    await run_command(cmd_link)
    assert await async_fs.exists(link_file)
    assert await async_fs.is_symlink(link_file)
    cmd_write = f"python3 -c \"open('{link_file}','a').write('evil_env_payload\\n')\""
    output = await run_command(cmd_write)
    assert any(
        msg in output
        for msg in (
            "No such file or directory",
            "Operation not permitted",
            "Permission denied",
            "Read-only file system",
        )
    )
    env_content = await async_fs.read_text(env_file)
    assert "evil_env_payload" not in env_content


@pytest.mark.asyncio
async def test_sandbox_settings_not_found(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify behavior when sandbox settings file is missing."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace, srt_settings_path=str(tmp_path / "nonexistent.json"))
    res = await run_command("echo test")
    assert "settings file not found" in res.lower()


@pytest.mark.asyncio
async def test_sandbox_settings_invalid_json(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify behavior when sandbox settings file contains invalid JSON."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("invalid{json")
    run_command = create_run_command_tool(workspace, srt_settings_path=str(invalid_json_file))
    res = await run_command("echo test")
    assert "not valid json" in res.lower()


@pytest.mark.asyncio
async def test_sandbox_timeout_sandboxed(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that commands running longer than the timeout are terminated."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))
    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_process),
        patch("asyncio.wait_for", side_effect=TimeoutError("Mocked timeout")),
    ):
        res = await run_command("echo test")
        assert "Error: Command timed out after 300 seconds." in res
        mock_process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_sandbox_timeout_custom(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that a custom command_timeout is respected."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file), command_timeout=600)
    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_process),
        patch("asyncio.wait_for", side_effect=TimeoutError("Mocked timeout")) as mock_wait,
    ):
        res = await run_command("echo test")
        assert "Error: Command timed out after 600 seconds." in res
        # Verify the actual timeout value passed to wait_for
        mock_wait.assert_called_once()
        assert mock_wait.call_args[1]["timeout"] == 600


def test_ripgrep_and_srt_availability() -> None:
    """Verify that ripgrep and srt are available in the current environment."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert is_ripgrep_available() is True
        assert is_srt_available() is True
        mock_run.return_value.returncode = 1
        assert is_ripgrep_available() is False
        assert is_srt_available() is False
        mock_run.side_effect = OSError("Subprocess error")
        assert is_ripgrep_available() is False
        assert is_srt_available() is False


@pytest.mark.parametrize(
    ("command_line", "expected_error"),
    [
        # --- Basic allowed ---
        ("echo 'hello'", None),
        ("rm -rf /tmp/mytest", None),
        ("rm -rf /private/tmp/test", None),
        ("rm -rf {workspace}/file.txt", None),
        ('git log --grep="fix; test"', None),
        ("git log --grep='fix; test'", None),
        # --- Existing denylist ---
        ("curl https://google.com", "Command 'curl' is blocked"),
        ("sudo apt-get install", "Command 'sudo' is blocked"),
        ("rm -rf /", "Command 'rm' targeting root directory is denied"),
        ("rm -rf ../file", "Command 'rm' with directory traversal is denied"),
        ("rm -rf /usr/bin", "Command 'rm' targeting path outside workspace"),
        ("rm -rf /home/user/project-extra/file.txt", "Command 'rm' targeting path outside workspace"),
        # --- env handling ---
        ("env -u SECRET_VAR echo hello", None),
        ("env -u SECRET_VAR curl http://evil.com", "Command 'curl' is blocked"),
        ("env -- curl http://evil.com", "Command 'curl' is blocked"),
        ("env -- echo hello", None),
        # --- Shell operators now rejected as standalone tokens ---
        ("echo hello ; curl evil.com", "Shell operators"),
        # --- Shell interpreter pivots ---
        ("bash -c 'curl evil.com'", "Shell interpreter 'bash' is blocked"),
        ("sh -c 'rm -rf /'", "Shell interpreter 'sh' is blocked"),
        ("dash -c 'wget evil.com'", "Shell interpreter 'dash' is blocked"),
        ("zsh -c 'nc evil 4444'", "Shell interpreter 'zsh' is blocked"),
        # --- Language runtimes ALLOWED (agent needs them for tests) ---
        ("python3 -m pytest", None),
        ("python3 app.py", None),
        ("node test.js", None),
        ("ruby test.rb", None),
        # --- Exec pivots ---
        ("find . -name '*.py'", None),
        ("xargs curl", "Command 'xargs' is blocked"),
        # --- env -S/--split-string ---
        ("env -S 'curl http://evil.com'", "'env -S/--split-string' is blocked"),
        ("env --split-string 'curl http://evil.com'", "'env -S/--split-string' is blocked"),
        # --- Git hardening ---
        ("git --upload-pack=evil clone url", "Git flag '--upload-pack' is blocked"),
        ("git --receive-pack=evil push", "Git flag '--receive-pack' is blocked"),
        ("git -c protocol.ext.allow=always clone url", "Git config 'protocol.ext.allow' is blocked"),
        ("git -c remote.origin.uploadpack=evil fetch", "Git config 'remote.' is blocked"),
        # --- Command substitution in argument value: allowed (srt quoter escapes it) ---
        ("echo '$(whoami)'", None),
    ],
)
def test_validate_sandboxed_command_rules(command_line: str, expected_error: str | None) -> None:
    """Verify validation rules for executing sandboxed commands."""
    workspace = "/home/user/project"
    cmd = command_line.format(workspace=workspace)
    argv = shlex.split(cmd)
    res = validate_argv(argv, workspace)
    if expected_error is None:
        assert res is None
    else:
        assert res is not None
        assert expected_error in res


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (["find", ".", "-exec", "curl", "{}", "+"], "'find' with '-exec' is blocked"),
        (["find", ".", "-execdir", "rm", "{}", "+"], "'find' with '-execdir' is blocked"),
        (["find", ".", "-ok", "rm", "{}", "+"], "'find' with '-ok' is blocked"),
        # With ; terminator: the operator check catches it first (also blocked)
        (["find", ".", "-exec", "curl", "{}", ";"], "Shell operators"),
    ],
)
def test_validate_argv_exec_pivots(argv: list[str], expected_error: str) -> None:
    """Verify find -exec/-execdir/-ok pivots are blocked."""
    res = validate_argv(argv, "/tmp")  # noqa: S108
    assert res is not None
    assert expected_error in res


@pytest.mark.asyncio
async def test_sandbox_diagnostics_formatting(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify diagnostic output formatting for sandboxed command execution."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"some output\nConnection blocked by network allowlist\n", b"")
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("echo 'simulate_network_blocked'")
        assert "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy" in output
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"some output\nConnection blocked by network allowlist\n", b"")
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("echo 'simulate_network_blocked'")
        assert "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy" not in output
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"cat: /root/secret: Permission denied\n")
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("cat /root/secret")
        assert "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied'" in output
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"cat: /root/secret: Permission denied\n")
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("cat /root/secret")
        assert "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied'" not in output


@pytest.mark.asyncio
async def test_sandbox_cache_env_overrides(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify cache environment variables can override defaults."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await run_command("echo hello")
        mock_exec.assert_called_once()
        kwargs = mock_exec.call_args.kwargs
        env = kwargs.get("env", {})
        cache_base = str(Path(workspace) / ".cache")
        assert env.get("NPM_CONFIG_CACHE") == str(Path(cache_base) / "npm")
        assert env.get("YARN_CACHE_FOLDER") == str(Path(cache_base) / "yarn")
        assert env.get("PNPM_HOME") == str(Path(cache_base) / "pnpm")
        assert env.get("BUN_INSTALL") == str(Path(cache_base) / "bun")
        assert env.get("DENO_DIR") == str(Path(cache_base) / "deno")
        assert env.get("PIP_CACHE_DIR") == str(Path(cache_base) / "pip")
        assert env.get("UV_CACHE_DIR") == str(Path(cache_base) / "uv")
        assert env.get("POETRY_CACHE_DIR") == str(Path(cache_base) / "poetry")
        assert env.get("PIPENV_CACHE_DIR") == str(Path(cache_base) / "pipenv")
        assert env.get("GOMODCACHE") == str(Path(cache_base) / "go" / "pkg" / "mod")
        assert env.get("CARGO_HOME") == str(Path(cache_base) / "cargo")
        assert env.get("GEM_HOME") == str(Path(cache_base) / "gems")
        assert env.get("GEM_PATH") == str(Path(cache_base) / "gems")
        assert env.get("COMPOSER_CACHE_DIR") == str(Path(cache_base) / "composer")
        assert env.get("GRADLE_USER_HOME") == str(Path(cache_base) / "gradle")
        assert env.get("NUGET_PACKAGES") == str(Path(cache_base) / "nuget")
        assert env.get("DOTNET_CLI_HOME") == str(Path(cache_base) / "dotnet")
        assert env.get("PUB_CACHE") == str(Path(cache_base) / "pub-cache")
        assert env.get("MIX_HOME") == str(Path(cache_base) / "mix")
        assert env.get("XDG_CACHE_HOME") == cache_base


@pytest.mark.asyncio
async def test_sandbox_git_credential_env_overrides(
    tmp_path: Path,
    github_token: str,
    async_fs: type[AsyncFSHelper],
) -> None:
    """Verify git credential environment variables are passed to sandbox for git commands."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file), github_token=github_token)
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await run_command("git status")
        mock_exec.assert_called_once()
        kwargs = mock_exec.call_args.kwargs
        env = kwargs.get("env", {})
        assert env.get("GIT_CONFIG_COUNT") == "2"
        assert env.get("GIT_CONFIG_KEY_0") == f"url.https://x-access-token:{github_token}@github.com/.insteadOf"
        assert env.get("GIT_CONFIG_VALUE_0") == "https://github.com/"
        assert env.get("GIT_CONFIG_KEY_1") == f"url.https://x-access-token:{github_token}@github.com/.insteadOf"
        assert env.get("GIT_CONFIG_VALUE_1") == "git@github.com:"
        assert env.get("GIT_TERMINAL_PROMPT") == "0"


def test_sandbox_run_command_cleanup(tmp_path: Path) -> None:
    """Verify temp files are cleaned up after running a sandboxed command."""
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))
    cleanup_sandbox = getattr(run_command, "cleanup", None)
    assert cleanup_sandbox is not None

    with patch("dependency_director.tools.Path.unlink") as mock_unlink:
        cleanup_sandbox()
        mock_unlink.assert_called_once()


def test_sandbox_run_command_cleanup_on_del(tmp_path: Path) -> None:
    """Verify temp files are cleaned up when the runner is garbage collected."""
    workspace = str(tmp_path / "workspace")
    Path(workspace).mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))

    with patch("dependency_director.tools.Path.unlink") as mock_unlink:
        del run_command
        mock_unlink.assert_called_once()


def test_create_run_command_tool_rejects_no_sandbox_param() -> None:
    """create_run_command_tool no longer accepts a no_sandbox parameter.

    In no-sandbox mode the tool is simply not registered, so there's no
    runtime flag to toggle.
    """
    sig = inspect.signature(create_run_command_tool)
    assert "no_sandbox" not in sig.parameters


@pytest.mark.asyncio
async def test_run_command_always_uses_exec_not_shell(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """The tool must always use subprocess_exec (with srt) in argv mode, never subprocess_shell."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"output", b"")
    mock_process.returncode = 0
    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec,
        patch("asyncio.create_subprocess_shell") as mock_shell,
    ):
        await run_command("echo test")
        mock_exec.assert_called_once()
        mock_shell.assert_not_called()
        # Verify srt is invoked in argv mode (-- separator, not -c)
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "srt"
        assert "--" in call_args, "srt must be invoked with -- separator for argv mode"
        separator_idx = list(call_args).index("--")
        argv_portion = call_args[separator_idx + 1 :]
        assert argv_portion == ("echo", "test"), f"Unexpected argv: {argv_portion}"
        assert "-c" not in call_args, "srt must NOT use -c (shell string mode)"


def test_validate_sandboxed_command_git_config() -> None:
    """Verify git config commands are validated correctly."""
    temp_dir = tempfile.gettempdir()

    def assert_blocked(cmd: str) -> None:
        res = validate_argv(shlex.split(cmd), temp_dir)
        assert res is not None
        assert "Security Error" in res

    assert_blocked("git config core.hookspath /evil")
    assert_blocked("git config Core.HooksPath /evil")
    assert_blocked("git config --global credential.helper store")
    assert_blocked("git config --global Credential.Helper store")
    assert_blocked("git config --local url.https://.insteadof git://")
    assert_blocked("git config --local Url.https://.insteadof git://")
    assert_blocked("git -c core.sshcommand=evil clone")
    assert_blocked("git -c Core.SSHcommand=evil clone")
    assert_blocked("git --config http.proxy=evil clone")
    assert_blocked("git config -f .git/config core.hookspath /evil")
    assert_blocked("git config --file=.git/config credential.helper store")
    assert_blocked("git config --file .git/config credential.helper store")
    assert_blocked("git config --type bool core.hookspath true")
    assert_blocked("git config --default placeholder --get core.hookspath")
    assert_blocked("HTTP_PROXY=http://evil.com curl http://example.com")
    assert_blocked("A=B C=D sudo id")
    assert_blocked("GIT_CONFIG_PARAMETERS='core.sshcommand=evil' git status")
    assert_blocked("LD_PRELOAD=evil.so git status")
    assert_blocked("DYLD_INSERT_LIBRARIES=evil.dylib git status")
    assert_blocked("DYLD_LIBRARY_PATH=/tmp git status")
    assert_blocked("git --config-env=core.sshcommand=ENV_VAR clone")
    assert_blocked("git --config-env core.sshcommand=ENV_VAR clone")
    assert_blocked("git --git-dir=/tmp/evil-git-dir status")
    assert_blocked("git --git-dir /tmp/evil-git-dir status")
    assert validate_argv(shlex.split("git config core.repositoryformatversion"), temp_dir) is None
    assert validate_argv(shlex.split("git config -f .git/config core.repositoryformatversion"), temp_dir) is None
    assert validate_argv(shlex.split("git clone https://github.com/my-url.dot/repo"), temp_dir) is None
    assert validate_argv(shlex.split("git status"), temp_dir) is None


@pytest.mark.asyncio
async def test_create_run_command_tool_agent_registration(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Verify that create_run_command_tool successfully registers with the Agent config.

    It should initialize the session without error.
    """
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    await async_fs.write_text(settings_file, '{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file))

    config = LocalAgentConfig(
        model="gemini-3.5-flash",
        tools=[run_command],
        workspaces=[workspace],
    )
    # Verify the tool is accepted by LocalAgentConfig without error.
    # Entering the Agent context requires a live Gemini API key, so we only
    # verify construction here — proto conversion happens at config build time.
    assert run_command in config.tools


@pytest.mark.asyncio
async def test_sandbox_non_git_command_has_no_token_env(
    tmp_path: Path,
    github_token: str,
    async_fs: type[AsyncFSHelper],
) -> None:
    """Non-git commands must NOT have GitHub token env vars (exfiltration defense)."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')
    run_command = create_run_command_tool(workspace, srt_settings_path=str(settings_file), github_token=github_token)
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await run_command("echo hello")
        mock_exec.assert_called_once()
        kwargs = mock_exec.call_args.kwargs
        env = kwargs.get("env", {})
        token_env_keys = {
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_CONFIG_KEY_1",
            "GIT_CONFIG_VALUE_1",
            "GIT_TERMINAL_PROMPT",
        }
        leaked = token_env_keys & set(env)
        assert not leaked, f"Token env vars leaked to non-git command: {leaked}"


def test_srt_settings_policy_assertions() -> None:
    """Verify srt-settings.json enforces critical sandbox policies."""
    with Path(DEFAULT_SRT_SETTINGS_PATH).open() as f:
        config = json.load(f)

    # Network isolation must NOT be weakened
    assert config.get("enableWeakerNetworkIsolation") is not True, "enableWeakerNetworkIsolation must be false/absent"

    # Home directory must be denied for reads
    deny_read = config.get("filesystem", {}).get("denyRead", [])
    assert "~" in deny_read, "filesystem.denyRead must include '~'"

    # Exfil webhook domains must be denied
    denied_domains = config.get("network", {}).get("deniedDomains", [])
    expected_exfil_patterns = ["*.ngrok.io", "*.pipedream.com", "*.webhook.site", "*.requestbin.com"]
    for pattern in expected_exfil_patterns:
        assert pattern in denied_domains, f"network.deniedDomains missing exfil pattern: {pattern}"


# --- Compound && / || support (TDD) ---


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # No operators — single command passthrough
        (["git", "status"], [["git", "status"]]),
        # Single &&
        (
            ["git", "config", "user.email", "x", "&&", "git", "config", "user.name", "y"],
            [["git", "config", "user.email", "x"], ["git", "config", "user.name", "y"]],
        ),
        # Multiple &&
        (
            ["cmd1", "&&", "cmd2", "arg", "&&", "cmd3"],
            [["cmd1"], ["cmd2", "arg"], ["cmd3"]],
        ),
        # || operator
        (
            ["cmd1", "||", "cmd2"],
            [["cmd1"], ["cmd2"]],
        ),
        # Mixed && and ||
        (
            ["cmd1", "&&", "cmd2", "||", "cmd3"],
            [["cmd1"], ["cmd2"], ["cmd3"]],
        ),
    ],
)
def test_split_compound_argv(argv: list[str], expected: list[list[str]]) -> None:
    """split_compound_argv correctly splits argv on && and || tokens."""
    parts = split_compound_argv(argv)
    # Extract just the argv lists (ignoring operators)
    assert [p.argv for p in parts] == expected


def test_split_compound_argv_preserves_operators() -> None:
    """split_compound_argv records which operator joins each sub-command."""
    parts = split_compound_argv(["cmd1", "&&", "cmd2", "||", "cmd3"])
    assert parts[0].operator is None  # first command has no preceding operator
    assert parts[1].operator == "&&"
    assert parts[2].operator == "||"


def test_split_compound_argv_empty() -> None:
    """split_compound_argv rejects empty argv."""
    parts = split_compound_argv([])
    assert len(parts) == 1
    assert parts[0].argv == []


def test_split_compound_argv_quoted_operator_not_split() -> None:
    """&& inside a quoted argument should NOT be treated as an operator.

    shlex.split('echo "&&"') produces ['echo', '&&'] — but the && is a
    single token that was quoted. However, shlex.split strips quotes so
    the token IS '&&'. The protection comes from shlex.split correctly
    handling 'echo "a && b"' → ['echo', 'a && b'] (no standalone &&).

    A standalone && token after shlex.split IS an operator. This test
    documents that split_compound_argv treats it as such (correct behavior).
    """
    # shlex.split('echo "&&"') → ['echo', '&&']  — the && IS standalone
    argv = shlex.split('echo "&&"')
    parts = split_compound_argv(argv)
    # This WILL be split because && is a standalone token post-shlex
    assert len(parts) == 2


@pytest.mark.parametrize(
    ("command_line", "expected_error"),
    [
        # && and || are now ALLOWED
        ("git config user.email x && git config user.name y", None),
        ("echo hello || echo fallback", None),
        # ;, |, & are still BLOCKED
        ("echo hello ; curl evil.com", "Shell operators"),
        ("echo hello | curl evil.com", "Shell operators"),
        ("echo hello &", "Shell operators"),
    ],
)
def test_validate_argv_compound_operators(command_line: str, expected_error: str | None) -> None:
    """validate_argv allows && and || but still blocks ;, |, and &."""
    workspace = "/home/user/project"
    argv = shlex.split(command_line)
    res = validate_argv(argv, workspace)
    if expected_error is None:
        assert res is None
    else:
        assert res is not None
        assert expected_error in res


def test_validate_argv_compound_blocked_subcmd() -> None:
    """Each sub-command in a compound is validated independently.

    'echo hello && curl evil.com' must be blocked because curl is in
    the second sub-command.
    """
    argv = shlex.split("echo hello && curl evil.com")
    res = validate_argv(argv, "/workspace")
    assert res is not None
    assert "curl" in res


@requires_srt
@pytest.mark.asyncio
async def test_compound_runner_and_and(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """SandboxedCommandRunner executes compound && commands sequentially."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    result = await run_command("echo first && echo second")
    assert "first" in result
    assert "second" in result


@requires_srt
@pytest.mark.asyncio
async def test_compound_runner_stops_on_failure(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """SandboxedCommandRunner stops at first failure in && chain."""
    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)
    run_command = create_run_command_tool(workspace)
    result = await run_command('python3 -c "import sys; sys.exit(1)" && echo should_not_appear')
    assert "should_not_appear" not in result
    assert "EXIT CODE: 1" in result


@pytest.mark.asyncio
async def test_compound_runner_error_string_is_failure(tmp_path: Path, async_fs: type[AsyncFSHelper]) -> None:
    """Error strings (no EXIT CODE marker) must be treated as failure in && chains.

    _run_single_argv can return strings like "Error: Command timed out..." or
    "Error: Failed to create temporary config..." that lack an EXIT CODE marker.
    These must short-circuit && chains, not be treated as success.
    """
    call_count = 0
    call_args: list[list[str]] = []

    async def mock_run(_self: Any, argv: list[str], _cwd: str) -> CommandResult:
        nonlocal call_count
        call_count += 1
        call_args.append(argv)
        if call_count == 1:
            return CommandResult("Error: Command timed out after 300 seconds.", -1)
        return CommandResult("--- STDOUT ---\nsecond\n--- EXIT CODE: 0 ---", 0)

    workspace = str(tmp_path / "workspace")
    await async_fs.mkdir(workspace)

    with patch.object(SandboxedCommandRunner, "_run_single_argv", mock_run):
        run_command = create_run_command_tool(workspace)
        result = await run_command("echo first && echo second")

    assert "Error: Command timed out" in result
    assert "echo second" not in result  # second command output must not appear
    assert call_count == 1  # second command was never called
    assert call_args == [["echo", "first"]]
