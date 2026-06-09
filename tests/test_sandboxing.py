import contextlib
import os
from pathlib import Path

import pytest

from dependency_director.tools import create_run_command_tool, is_srt_available

has_srt = is_srt_available()
requires_srt = pytest.mark.skipif(
    not has_srt,
    reason="sandbox-runtime (srt) is required",
)


def get_stdout(output: str) -> str:
    """Helper to extract raw stdout from formatted command result."""
    if "--- STDOUT ---" in output:
        parts = output.split("--- STDERR ---")
        return parts[0].replace("--- STDOUT ---", "").strip()
    return output.strip()


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_network_denied(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Create the run_command tool bound to the workspace, sandboxed
    run_command = create_run_command_tool(workspace)

    # Command to test network access
    cmd = "python3 -c \"import urllib.request; urllib.request.urlopen('https://www.google.com', timeout=2)\""

    output = await run_command(cmd)

    assert any(
        err in output
        for err in [
            "URLError",
            "Permission denied",
            "TimeoutError",
            "timed out",
            "Operation not permitted",
        ]
    )


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_filesystem_write_restricted(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Try writing to a path outside the workspace (like user's home directory)
    unauthorized_file = os.path.expanduser("~/exploit_sandbox_test.txt")
    if os.path.exists(unauthorized_file):
        with contextlib.suppress(Exception):
            os.unlink(unauthorized_file)

    cmd = f"echo 'evil payload' > {unauthorized_file}"
    output = await run_command(cmd)

    try:
        exists = os.path.exists(unauthorized_file)
        if exists:
            os.unlink(unauthorized_file)
    except Exception:
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
async def test_sandbox_workspace_write_allowed(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Write to a path inside the workspace
    target_file = os.path.join(workspace, "valid_patch.txt")
    cmd = f"echo 'valid patch content' > {target_file}"

    output = await run_command(cmd)

    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    assert os.path.exists(target_file)
    with open(target_file) as f:
        assert f.read().strip() == "valid patch content"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_sensitive_files_denied(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Create a dummy sensitive file in ~/.ssh
    ssh_dir = os.path.expanduser("~/.ssh")
    os.makedirs(ssh_dir, exist_ok=True)
    sensitive_file = os.path.join(ssh_dir, "test_depdirector_sandbox_read.txt")

    with open(sensitive_file, "w") as f:
        f.write("sensitive data")

    try:
        cmd = f"cat {sensitive_file}"
        output = await run_command(cmd)
        assert any(
            msg in output
            for msg in (
                "No such file or directory",
                "Operation not permitted",
                "Permission denied",
            )
        )
    finally:
        with contextlib.suppress(Exception):
            os.unlink(sensitive_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_denied(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Create the .git directory structure
    git_dir = os.path.join(workspace, ".git")
    hooks_dir = os.path.join(git_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Try writing to .git/hooks/post-commit
    hook_file = os.path.join(hooks_dir, "post-commit")
    cmd = f"echo 'evil' > {hook_file}"
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
    assert not os.path.exists(hook_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_config_allowed(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Create the .git directory structure
    git_dir = os.path.join(workspace, ".git")
    os.makedirs(git_dir, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # 1. Try writing to .git/config via non-git command (which should be blocked)
    config_file = os.path.join(git_dir, "config")
    cmd = f"echo 'evil-config' > {config_file}"
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
    assert (
        not os.path.exists(config_file)
        or open(config_file).read().strip() != "evil-config"
    )

    # 2. Try writing to .git/config via git command (which should be allowed)
    await run_command("git init")
    cmd_git = "git config core.repositoryformatversion 0"
    output_git = await run_command(cmd_git)
    assert "Operation not permitted" not in output_git
    assert "Permission denied" not in output_git
    assert os.path.exists(config_file)
    with open(config_file) as f:
        content = f.read()
    assert "repositoryformatversion = 0" in content


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_compound_command(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    git_dir = os.path.join(workspace, ".git")
    os.makedirs(git_dir, exist_ok=True)
    config_file = os.path.join(git_dir, "config")

    run_command = create_run_command_tool(workspace)

    # Compound command starting with echo but executing git init afterwards.
    # It should be detected as a git command, enabling allowGitConfig, and succeeding.
    cmd = "echo 'initiating...' && git init"
    output = await run_command(cmd)

    assert "Operation not permitted" not in output
    assert "Permission denied" not in output
    assert os.path.exists(config_file)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_other_metadata_write_allowed(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    git_dir = os.path.join(workspace, ".git")
    refs_dir = os.path.join(git_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Writing to refs/heads/main should be allowed
    ref_file = os.path.join(refs_dir, "main")
    cmd = f"echo 'commit_hash' > {ref_file}"
    output = await run_command(cmd)

    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    assert os.path.exists(ref_file)
    with open(ref_file) as f:
        assert f.read().strip() == "commit_hash"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_network_command_bypasses_sandbox(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Run a git fetch command (should try to access network and fail because it is not a git repo,
    # but it should NOT fail with Seatbelt's "Permission denied" or "Operation not permitted")
    cmd = "git fetch origin main"
    output = await run_command(cmd)

    # It should not say Permission denied for network access
    assert "Permission denied" not in output
    assert "Operation not permitted" not in output
    # Instead it should be a standard git error like "not a git repository"
    assert "not a git repository" in output.lower() or "fatal:" in output.lower()


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_home_allowlist_enforced(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Cargo configuration file (allowlisted under ~/.cargo/registry in srt-settings.json)
    home_dir = os.path.expanduser("~")
    cargo_dir = os.path.join(home_dir, ".cargo", "registry")
    os.makedirs(cargo_dir, exist_ok=True)
    cargo_config = os.path.join(cargo_dir, "config_test_sandbox.toml")
    with open(cargo_config, "w") as f:
        f.write("dummy_cargo_config")

    try:
        cmd = f"cat {cargo_config}"
        output = await run_command(cmd)
        assert "Permission denied" not in output
        assert "Operation not permitted" not in output
        assert get_stdout(output) == "dummy_cargo_config"
    finally:
        with contextlib.suppress(Exception):
            os.unlink(cargo_config)

    # Random document file (blocked by denyRead: ~ in srt-settings.json)
    docs_dir = os.path.join(home_dir, "Documents")
    os.makedirs(docs_dir, exist_ok=True)
    secret_doc = os.path.join(docs_dir, "secret_test_sandbox.txt")
    with open(secret_doc, "w") as f:
        f.write("secret_data")

    try:
        cmd = f"cat {secret_doc}"
        output = await run_command(cmd)
        assert any(
            msg in output
            for msg in (
                "No such file or directory",
                "Operation not permitted",
                "Permission denied",
            )
        )
    finally:
        with contextlib.suppress(Exception):
            os.unlink(secret_doc)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_custom_allowlist_enforced(tmp_path: Path) -> None:
    import json

    from dependency_director.config import DEFAULT_SRT_SETTINGS_PATH

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Custom directory to allowlist (outside workspace, so normally blocked)
    home_dir = os.path.expanduser("~")
    custom_abs_dir = os.path.join(home_dir, "test_custom_allowlist")
    os.makedirs(custom_abs_dir, exist_ok=True)

    # Custom file inside it
    custom_file = os.path.join(custom_abs_dir, "custom_sandbox_test.txt")
    with open(custom_file, "w") as f:
        f.write("custom_data")

    # Load default settings, add custom directory to allowRead, and save
    with open(DEFAULT_SRT_SETTINGS_PATH) as f:
        config = json.load(f)
    config.setdefault("filesystem", {}).setdefault("allowRead", []).append(
        custom_abs_dir,
    )

    custom_settings_path = os.path.join(tmp_path, "custom_srt_settings.json")
    with open(custom_settings_path, "w") as f:
        json.dump(config, f)

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=custom_settings_path,
    )

    try:
        cmd = f"cat {custom_file}"
        output = await run_command(cmd)
        assert "Permission denied" not in output
        assert "Operation not permitted" not in output
        assert get_stdout(output) == "custom_data"
    finally:
        with contextlib.suppress(Exception):
            os.unlink(custom_file)
        with contextlib.suppress(Exception):
            os.rmdir(custom_abs_dir)


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_custom_denylist_enforced(tmp_path: Path) -> None:
    import json

    from dependency_director.config import DEFAULT_SRT_SETTINGS_PATH

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Custom files in /tmp to check denylist enforcement
    blocked_file = "/tmp/blocked_sandbox_test.txt"
    allowed_file = "/tmp/allowed_sandbox_test.txt"

    with open(blocked_file, "w") as f:
        f.write("blocked_secret")
    with open(allowed_file, "w") as f:
        f.write("allowed_data")

    # Load default settings, add custom path to denyRead, and save
    with open(DEFAULT_SRT_SETTINGS_PATH) as f:
        config = json.load(f)
    config.setdefault("filesystem", {}).setdefault("denyRead", []).append(blocked_file)

    custom_settings_path = os.path.join(tmp_path, "custom_srt_settings.json")
    with open(custom_settings_path, "w") as f:
        json.dump(config, f)

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=custom_settings_path,
    )

    try:
        # Check blocked file (should be denied)
        cmd1 = f"cat {blocked_file}"
        output1 = await run_command(cmd1)
        assert any(
            msg in output1
            for msg in (
                "No such file or directory",
                "Operation not permitted",
                "Permission denied",
            )
        )

        # Check allowed file (should be allowed)
        cmd2 = f"cat {allowed_file}"
        output2 = await run_command(cmd2)
        assert "Permission denied" not in output2
        assert "Operation not permitted" not in output2
        assert get_stdout(output2) == "allowed_data"
    finally:
        with contextlib.suppress(Exception):
            os.unlink(blocked_file)
        with contextlib.suppress(Exception):
            os.unlink(allowed_file)


@pytest.mark.asyncio
async def test_sandbox_sensitive_env_stripped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Set sensitive and safe environment variables
    monkeypatch.setenv("GITHUB_TOKEN", "secret_github")
    monkeypatch.setenv("GEMINI_API_KEY", "secret_gemini")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret_aws")
    monkeypatch.setenv("STRIPE_API_KEY", "secret_stripe")
    monkeypatch.setenv("LANG", "custom_lang")

    run_command = create_run_command_tool(workspace)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"mocked output", b"")
    mock_process.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=mock_process,
    ) as mock_create:
        await run_command("echo test")

        assert mock_create.called
        kwargs = mock_create.call_args[1]
        called_env = kwargs.get("env", {})

        # Verify targeted host credentials are stripped
        assert "GITHUB_TOKEN" not in called_env
        assert "GEMINI_API_KEY" not in called_env
        assert "AWS_SECRET_ACCESS_KEY" not in called_env
        assert "STRIPE_API_KEY" not in called_env
        # Verify other safe / project-specific config variables are preserved
        assert called_env.get("LANG") == "custom_lang"


@pytest.mark.asyncio
async def test_sandbox_srt_not_found_raises(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Create the run_command tool bound to the workspace, sandboxed
    run_command = create_run_command_tool(workspace)

    # Executing any command when srt is not found should raise FileNotFoundError
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
async def test_sandbox_git_conflict_resolution_commands(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_command = create_run_command_tool(workspace)

    # Test git fetch command inside the sandbox
    cmd_fetch = "git fetch origin main"
    output_fetch = await run_command(cmd_fetch)
    assert "Permission denied" not in output_fetch
    assert "Operation not permitted" not in output_fetch
    assert (
        "not a git repository" in output_fetch.lower()
        or "fatal:" in output_fetch.lower()
    )

    # Test git merge command inside the sandbox
    cmd_merge = "git merge origin/main"
    output_merge = await run_command(cmd_merge)
    assert "Permission denied" not in output_merge
    assert "Operation not permitted" not in output_merge
    assert (
        "not a git repository" in output_merge.lower()
        or "fatal:" in output_merge.lower()
    )

    # Test git push command inside the sandbox
    cmd_push = "git push origin HEAD"
    output_push = await run_command(cmd_push)
    assert "Permission denied" not in output_push
    assert "Operation not permitted" not in output_push
    assert (
        "not a git repository" in output_push.lower() or "fatal:" in output_push.lower()
    )


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_developer_loop_dependency_upgrade(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Create the dummy library representing the upgraded dependency
    lib_dir = os.path.join(workspace, "dummy_lib")
    os.makedirs(lib_dir, exist_ok=True)
    with open(os.path.join(lib_dir, "__init__.py"), "w") as f:
        f.write("def add(*, x, y):\n    return x + y\n")

    # Create the application code that uses old positional call style
    app_file = os.path.join(workspace, "app.py")
    with open(app_file, "w") as f:
        f.write(
            "import sys\nsys.path.insert(0, '.')\nimport dummy_lib\nprint(dummy_lib.add(2, 3))\n",
        )

    run_command = create_run_command_tool(workspace)

    # 1. Run the app, expecting TypeError due to signature mismatch
    cmd_run_fail = "python3 app.py"
    output_fail = await run_command(cmd_run_fail)
    assert "TypeError" in output_fail

    # 2. Emulate the agent reading the library code to inspect the signature
    cmd_read_lib = "cat dummy_lib/__init__.py"
    output_read = await run_command(cmd_read_lib)
    assert "def add(*, x, y):" in output_read

    # 3. Emulate the agent patching the app.py to match the new signature
    # (Writing the python code to rewrite app.py or using python -c directly)
    patch_code = "import sys\nsys.path.insert(0, '.')\nimport dummy_lib\nprint(dummy_lib.add(x=2, y=3))\n"
    # To write this safely inside the sandbox, we use echo with shlex.quote
    import shlex

    cmd_patch = f"echo {shlex.quote(patch_code)} > app.py"
    await run_command(cmd_patch)

    # 4. Run the app again, it should pass and output 5
    output_success = await run_command(cmd_run_fail)
    assert get_stdout(output_success) == "5"


@requires_srt
@pytest.mark.asyncio
async def test_sandbox_git_metadata_write_bypass_symlink(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    env_file = os.path.join(workspace, ".env")

    # Pre-create .env with some initial content
    with open(env_file, "w") as f:
        f.write("KEY=VAL\n")

    run_command = create_run_command_tool(workspace)

    # 1. Create a symbolic link to the env file outside the .git folder but inside the workspace
    link_file = os.path.join(workspace, "env_link")
    cmd_link = f"ln -s {env_file} {link_file}"
    await run_command(cmd_link)
    assert os.path.exists(link_file)
    assert os.path.islink(link_file)

    # 2. Try writing to the link file inside the sandbox
    cmd_write = f"echo 'evil_env_payload' >> {link_file}"
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

    # If the write fails, the original env file will NOT be modified
    with open(env_file) as f:
        env_content = f.read()

    assert "evil_env_payload" not in env_content


@pytest.mark.asyncio
async def test_sandbox_settings_not_found(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(tmp_path / "nonexistent.json"),
    )
    res = await run_command("echo test")
    assert "settings file not found" in res.lower()


@pytest.mark.asyncio
async def test_sandbox_settings_invalid_json(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("invalid{json")
    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(invalid_json_file),
    )
    res = await run_command("echo test")
    assert "not valid json" in res.lower()


@pytest.mark.asyncio
async def test_sandbox_timeout_sandboxed(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
    )

    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_process),
        patch("asyncio.wait_for", side_effect=TimeoutError("Mocked timeout")),
    ):
        res = await run_command("echo test")
        assert "Error: Command timed out after 300 seconds." in res
        mock_process.kill.assert_called_once()


def test_ripgrep_and_srt_availability() -> None:
    from unittest.mock import patch

    from dependency_director.tools import is_ripgrep_available, is_srt_available

    with patch("subprocess.run") as mock_run:
        # Test success paths
        mock_run.return_value.returncode = 0
        assert is_ripgrep_available() is True
        assert is_srt_available() is True

        # Test failure paths
        mock_run.return_value.returncode = 1
        assert is_ripgrep_available() is False
        assert is_srt_available() is False

        # Test exception paths
        mock_run.side_effect = Exception("Subprocess error")
        assert is_ripgrep_available() is False
        assert is_srt_available() is False


@pytest.mark.parametrize(
    ("command_line", "expected_error"),
    [
        # 1. Allowed commands
        ("echo 'hello'", None),
        ("pytest && python3 app.py", None),
        ("rm -rf /tmp/mytest", None),
        ("rm -rf /private/tmp/test", None),
        ("rm -rf {workspace}/file.txt", None),
        ('git log --grep="fix; test"', None),
        ("git log --grep='fix; test'", None),
        # 2. Blocked executables
        ("curl https://google.com", "Command 'curl' is blocked"),
        ("sudo apt-get install", "Command 'sudo' is blocked"),
        ("pytest && nc localhost 4444", "Command 'nc' is blocked"),
        # 3. rm safety checks
        ("rm -rf /", "Command 'rm' targeting root directory is denied"),
        ("rm -rf ../file", "Command 'rm' with directory traversal is denied"),
        ("rm -rf /usr/bin", "Command 'rm' targeting path outside workspace"),
        (
            "pytest && rm -rf /etc/hosts",
            "Command 'rm' targeting path outside workspace",
        ),
        # 4. Chained commands injection protection (Command Injection)
        ("git checkout branch;curl http://attacker.com", "Command 'curl' is blocked"),
        (
            "git checkout branch;rm -rf /usr/bin",
            "Command 'rm' targeting path outside workspace",
        ),
        # 5. Prefix-containment: /home/user/project-extra must NOT match /home/user/project
        # Use a non-/tmp workspace since _validate_target_path allows anything under /tmp
        (
            "rm -rf /home/user/project-extra/file.txt",
            "Command 'rm' targeting path outside workspace",
        ),
        # 6. env -u unwrap: should correctly skip -u and its argument
        ("env -u SECRET_VAR echo hello", None),
        ("env -u SECRET_VAR curl http://evil.com", "Command 'curl' is blocked"),
        ("env -- curl http://evil.com", "Command 'curl' is blocked"),
        ("env -- echo hello", None),
    ],
)
def test_validate_sandboxed_command_rules(
    command_line: str,
    expected_error: str | None,
) -> None:
    from dependency_director.tools import validate_sandboxed_command

    # Use /home/user/project for prefix-containment tests (not /tmp,
    # since _validate_target_path intentionally allows all of /tmp)
    workspace = "/home/user/project"
    cmd = command_line.format(workspace=workspace)
    res = validate_sandboxed_command(cmd, workspace)
    if expected_error is None:
        assert res is None
    else:
        assert res is not None
        assert expected_error in res


@pytest.mark.asyncio
async def test_sandbox_diagnostics_formatting(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Setup dummy settings file
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
    )

    # 1. Mock process returning network blocked message (with non-zero exit code)
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (
        b"some output\nConnection blocked by network allowlist\n",
        b"",
    )
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("echo 'simulate_network_blocked'")
        assert (
            "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy"
            in output
        )

    # 2. Mock process returning network blocked message (with zero exit code)
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (
        b"some output\nConnection blocked by network allowlist\n",
        b"",
    )
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("echo 'simulate_network_blocked'")
        assert (
            "[Sandbox Violation] Outbound network connection blocked by sandbox-runtime policy"
            not in output
        )

    # 3. Mock process returning filesystem permission denied message (with non-zero exit code)
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (
        b"",
        b"cat: /root/secret: Permission denied\n",
    )
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("cat /root/secret")
        assert (
            "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied'"
            in output
        )

    # 4. Mock process returning filesystem permission denied message (with zero exit code)
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (
        b"",
        b"cat: /root/secret: Permission denied\n",
    )
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        output = await run_command("cat /root/secret")
        assert (
            "[Sandbox Diagnostic] Filesystem access failed with 'Permission denied'"
            not in output
        )


@pytest.mark.asyncio
async def test_sandbox_cache_env_overrides(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
    )

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=mock_process,
    ) as mock_exec:
        await run_command("echo hello")
        mock_exec.assert_called_once()
        kwargs = mock_exec.call_args.kwargs
        env = kwargs.get("env", {})

        cache_base = str(Path(workspace) / ".cache")
        assert env.get("NPM_CONFIG_CACHE") == os.path.join(cache_base, "npm")
        assert env.get("YARN_CACHE_FOLDER") == os.path.join(cache_base, "yarn")
        assert env.get("PNPM_HOME") == os.path.join(cache_base, "pnpm")
        assert env.get("BUN_INSTALL") == os.path.join(cache_base, "bun")
        assert env.get("DENO_DIR") == os.path.join(cache_base, "deno")
        assert env.get("PIP_CACHE_DIR") == os.path.join(cache_base, "pip")
        assert env.get("UV_CACHE_DIR") == os.path.join(cache_base, "uv")
        assert env.get("POETRY_CACHE_DIR") == os.path.join(cache_base, "poetry")
        assert env.get("PIPENV_CACHE_DIR") == os.path.join(cache_base, "pipenv")
        assert env.get("GOMODCACHE") == os.path.join(cache_base, "go", "pkg", "mod")
        assert env.get("CARGO_HOME") == os.path.join(cache_base, "cargo")
        assert env.get("GEM_HOME") == os.path.join(cache_base, "gems")
        assert env.get("GEM_PATH") == os.path.join(cache_base, "gems")
        assert env.get("COMPOSER_CACHE_DIR") == os.path.join(cache_base, "composer")
        assert env.get("GRADLE_USER_HOME") == os.path.join(cache_base, "gradle")
        assert env.get("NUGET_PACKAGES") == os.path.join(cache_base, "nuget")
        assert env.get("DOTNET_CLI_HOME") == os.path.join(cache_base, "dotnet")
        assert env.get("PUB_CACHE") == os.path.join(cache_base, "pub-cache")
        assert env.get("MIX_HOME") == os.path.join(cache_base, "mix")
        assert env.get("XDG_CACHE_HOME") == cache_base


@pytest.mark.asyncio
async def test_sandbox_git_credential_env_overrides(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
        github_token="test-token-123",
    )

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=mock_process,
    ) as mock_exec:
        await run_command("echo hello")
        mock_exec.assert_called_once()
        kwargs = mock_exec.call_args.kwargs
        env = kwargs.get("env", {})

        assert env.get("GIT_CONFIG_COUNT") == "2"
        assert (
            env.get("GIT_CONFIG_KEY_0")
            == "url.https://x-access-token:test-token-123@github.com/.insteadOf"
        )
        assert env.get("GIT_CONFIG_VALUE_0") == "https://github.com/"
        assert (
            env.get("GIT_CONFIG_KEY_1")
            == "url.https://x-access-token:test-token-123@github.com/.insteadOf"
        )
        assert env.get("GIT_CONFIG_VALUE_1") == "git@github.com:"
        assert env.get("GIT_TERMINAL_PROMPT") == "0"


def test_sandbox_run_command_cleanup(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
    )

    cleanup_sandbox = getattr(run_command, "cleanup", None)
    assert cleanup_sandbox is not None

    from unittest.mock import patch

    with patch("dependency_director.tools.Path.unlink") as mock_unlink:
        cleanup_sandbox()
        mock_unlink.assert_called_once()


def test_create_run_command_tool_rejects_no_sandbox_param() -> None:
    """create_run_command_tool no longer accepts a no_sandbox parameter.

    In no-sandbox mode the tool is simply not registered, so there's no
    runtime flag to toggle.
    """
    import inspect

    sig = inspect.signature(create_run_command_tool)
    assert "no_sandbox" not in sig.parameters


@pytest.mark.asyncio
async def test_run_command_always_uses_exec_not_shell(tmp_path: Path) -> None:
    """The tool must always use subprocess_exec (with srt), never subprocess_shell."""
    from unittest.mock import AsyncMock, patch

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"filesystem": {}}')

    run_command = create_run_command_tool(
        workspace,
        srt_settings_path=str(settings_file),
    )

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


def test_validate_sandboxed_command_git_config() -> None:
    from dependency_director.tools import validate_sandboxed_command

    def assert_blocked(cmd: str) -> None:
        res = validate_sandboxed_command(cmd, "/tmp")
        assert res is not None
        assert "Security Error" in res

    # Blocked keys (including mixed case to test case-insensitivity)
    assert_blocked("git config core.hookspath /evil")
    assert_blocked("git config Core.HooksPath /evil")
    assert_blocked("git config --global credential.helper store")
    assert_blocked("git config --global Credential.Helper store")
    assert_blocked("git config --local url.https://.insteadof git://")
    assert_blocked("git config --local Url.https://.insteadof git://")
    assert_blocked("git -c core.sshcommand=evil clone")
    assert_blocked("git -c Core.SSHcommand=evil clone")
    assert_blocked("git --config http.proxy=evil clone")

    # Verify that argument-consuming flags like -f and --file don't bypass checks
    assert_blocked("git config -f .git/config core.hookspath /evil")
    assert_blocked("git config --file=.git/config credential.helper store")
    assert_blocked("git config --file .git/config credential.helper store")
    assert_blocked("git config --type bool core.hookspath true")
    assert_blocked("git config --default dummy --get core.hookspath")

    # Verify environment variable prefix bypass check (env vars shouldn't blind executable validation)
    # Note: These are blocked because 'curl' and 'sudo' are blocked executables, not because of the env variables.
    assert_blocked("HTTP_PROXY=http://evil.com curl http://example.com")
    assert_blocked("A=B C=D sudo id")

    # Verify blocked environment variables
    assert_blocked("GIT_CONFIG_PARAMETERS='core.sshcommand=evil' git status")
    assert_blocked("LD_PRELOAD=evil.so git status")
    assert_blocked("DYLD_INSERT_LIBRARIES=evil.dylib git status")
    assert_blocked("DYLD_LIBRARY_PATH=/tmp git status")

    # Verify blocked git options
    assert_blocked("git --config-env=core.sshcommand=ENV_VAR clone")
    assert_blocked("git --config-env core.sshcommand=ENV_VAR clone")
    assert_blocked("git --git-dir=/tmp/evil-git-dir status")
    assert_blocked("git --git-dir /tmp/evil-git-dir status")

    # Allowed commands
    assert (
        validate_sandboxed_command("git config core.repositoryformatversion", "/tmp")
        is None
    )
    assert (
        validate_sandboxed_command(
            "git config -f .git/config core.repositoryformatversion",
            "/tmp",
        )
        is None
    )
    assert (
        validate_sandboxed_command(
            "git clone https://github.com/my-url.dot/repo",
            "/tmp",
        )
        is None
    )
    assert validate_sandboxed_command("git status", "/tmp") is None
