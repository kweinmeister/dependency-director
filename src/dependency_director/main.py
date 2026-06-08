"""dependency-director: Autonomous dependency triage and patching agent."""

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import anyio.abc
import click
import mcp.client.stdio
from anyio.streams.text import TextReceiveStream
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import hooks
from google.antigravity.mcp.bridge import McpBridge
from mcp.client import stdio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

from dependency_director.config import (
    Settings,
    get_dry_run_policies,
    get_safety_policies,
)
from dependency_director.tools import (
    GitHubClient,
    create_agent_tools,
    create_run_command_tool,
    is_ripgrep_available,
    is_srt_available,
)


# Monkey-patch: the upstream Antigravity SDK's McpStdioServer does not
# support passing custom environment variables to the MCP server subprocess.
# This patch adds `env` support so the GitHub MCP server can receive
# GITHUB_PERSONAL_ACCESS_TOKEN without inheriting the full host environment.
# TODO: Remove once the SDK adds native `env` support to McpStdioServer.
class PatchedMcpStdioServer(types.McpStdioServer):  # type: ignore[misc]
    env: dict[str, str] | None = None


async def patched_connect_stdio(
    self: McpBridge,
    command: str,
    args: Sequence[str],
    server_cfg: types.McpStdioServer | types.McpStreamableHttpServer | None = None,
) -> None:
    env = getattr(server_cfg, "env", None)
    params = stdio.StdioServerParameters(
        command=command,
        args=list(args),
        env=env,
    )
    await self._connect(params, server_cfg)


McpBridge.connect_stdio = cast("Any", patched_connect_stdio)


# Monkey-patch: Wrap the MCP client's stdio subprocess creation to capture
# and prefix the stderr logs (such as the GitHub MCP server startup message)
# with a clean emoji.
async def _stderr_reader(process: anyio.abc.Process, errlog: Any) -> None:
    if process.stderr is None:
        return
    text_stream = TextReceiveStream(process.stderr, encoding="utf-8")
    try:
        async for chunk in text_stream:
            if "GitHub MCP Server running on stdio" in chunk:
                chunk = chunk.replace(
                    "GitHub MCP Server running on stdio",
                    "🔌 GitHub MCP Server running on stdio",
                )
            errlog.write(chunk)
            errlog.flush()
    except Exception:  # noqa: BLE001
        pass


async def patched_create_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    errlog: Any = sys.stderr,
    cwd: Any = None,
) -> anyio.abc.Process:
    process = await anyio.open_process(
        [command, *args],
        env=env,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )
    asyncio.create_task(_stderr_reader(process, errlog))
    return process


mcp.client.stdio._create_platform_compatible_process = patched_create_process  # type: ignore # pytype: disable=invalid-assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_PATH = PROJECT_ROOT / ".agents" / "skills"

console = Console(
    theme=Theme(
        {
            "thinking": "dim",
            "tool.name": "bold yellow",
            "tool.args": "dim",
            "tool.ok": "green",
            "tool.fail": "bold red",
            "banner": "bold cyan",
            "status": "cyan",
        },
    ),
)


class _SuppressMcpProbeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            "Could not fetch prompts" not in record.getMessage()
            and "Could not fetch resources" not in record.getMessage()
        )


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logging.getLogger().addFilter(_SuppressMcpProbeFilter())
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("dependency_director")
logger.setLevel(logging.INFO)


from dependency_director.instructions import get_system_instructions  # noqa: E402


async def get_repositories(owner: str, token: str) -> list[str]:
    """Fetch non-forked repositories from GitHub API."""
    client = GitHubClient(token=token)
    try:
        return await client.get_repositories(owner)
    finally:
        await client.close()


def _prepare_workspace(workspace_dir: str) -> None:
    """Prepare a clean workspace directory synchronously."""
    path = Path(workspace_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cleanup_workspace(workspace_dir: str) -> None:
    """Clean up a workspace directory synchronously."""
    path = Path(workspace_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


async def run_agent_for_repo(
    repo: str,
    settings: Settings,
    max_attempts: int,
    *,
    dry_run: bool = False,
    auto_merge: bool = False,
    verify_all: bool = False,
    standalone_fix: bool = False,
    review_wait: int = 0,
    hint: str | None = None,
) -> None:
    """Run the triage agent for a single GitHub repository."""
    client = GitHubClient(token=settings.github_token)
    (
        merge_bot_pr,
        rebase_bot_pr,
        wait_for_reviews,
        get_pr_status,
        get_pr_workflow_run_logs,
    ) = create_agent_tools(
        client=client,
        bots=settings.bots,
        dry_run=dry_run,
        review_wait=review_wait,
    )

    try:
        click.secho(
            f"🔗 Configuring GitHub MCP server connection for {repo}...",
            fg="blue",
        )
        mcp_servers: list[types.McpServerConfig] = [
            PatchedMcpStdioServer(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token,
                    "PATH": os.environ.get("PATH", ""),
                },
                disabled_tools=[
                    "merge_pull_request",
                    "search_issues",
                    "search_repositories",
                    "search_code",
                    "search_users",
                    "create_repository",
                    "fork_repository",
                    "create_or_update_file",
                    "push_files",
                    "create_branch",
                    "create_issue",
                    "update_issue",
                ],
            ),
        ]

        # Gather safety policies
        policies = get_safety_policies()

        # Add dry-run policy if active to deny any git push / PR merge tool
        # calls programmatically
        if dry_run:
            dry_run_policies = get_dry_run_policies()
            for p in reversed(dry_run_policies):
                policies.insert(0, p)

        repo_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
        workspace_tmp = str(
            Path(tempfile.gettempdir()) / f"dependency-director-{repo_hash}"
        )
        await asyncio.to_thread(_prepare_workspace, workspace_tmp)

        # In no-sandbox mode the agent operates purely through GitHub API
        # host tools (merge_bot_pr, rebase_bot_pr, get_pr_status, etc.).
        # No shell access is provided — this eliminates the entire attack
        # surface of command injection.
        if settings.no_sandbox:
            run_command = None
        else:
            run_command = create_run_command_tool(
                workspace_tmp,
                srt_settings_path=settings.srt_settings,
            )

        owner, repo_name = repo.split("/", 1)

        # Get agent system instructions
        system_instructions = get_system_instructions(
            max_attempts=max_attempts,
            owner=owner,
            verify_all=verify_all,
            auto_merge=auto_merge,
            dry_run=dry_run,
            workspace_dir=workspace_tmp,
            standalone_fix=standalone_fix,
            review_wait=review_wait,
            bots=settings.bots,
            no_sandbox=settings.no_sandbox,
        )

        class RepoToolErrorHook(hooks.OnToolErrorHook):  # type: ignore[misc]
            async def run(self, context: hooks.HookContext | None, data: Any) -> None:
                console.print(f"  [tool.fail]Tool error: {data!r}[/tool.fail]")

            async def __call__(self, error: Exception) -> None:
                await self.run(None, error)

        project_root = str(PROJECT_ROOT)
        skills_path = str(SKILLS_PATH)

        agent_tools: list[Any] = [
            merge_bot_pr,
            rebase_bot_pr,
            wait_for_reviews,
            get_pr_status,
            get_pr_workflow_run_logs,
        ]
        if run_command is not None:
            agent_tools.append(run_command)

        config = LocalAgentConfig(
            model="gemini-3.5-flash",
            vertex=settings.vertex or None,
            project=(settings.google_cloud_project or None)
            if settings.vertex
            else None,
            location=(settings.google_cloud_location or None)
            if settings.vertex
            else None,
            system_instructions=system_instructions,
            mcp_servers=mcp_servers,
            policies=policies,
            hooks=[RepoToolErrorHook()],
            tools=agent_tools,
            skills_paths=[skills_path],
            workspaces=[project_root, workspace_tmp],
            capabilities=types.CapabilitiesConfig(
                enable_subagents=False,
                disabled_tools=[types.BuiltinTools.RUN_COMMAND],
            ),
        )

        click.secho(
            f"🚀 Spawning Antigravity Agent for {repo}...",
            fg="cyan",
            bold=True,
        )
        async with Agent(config=config) as agent:
            bot_names = ", ".join(b.author for b in settings.bots)
            prompt = (
                f"Process all open dependency update PRs "
                f"(authored by {bot_names}) for '{owner}/{repo_name}'."
            )

            if dry_run:
                prompt += (
                    " Perform this run in DRY-RUN mode "
                    "(simulate all merge and push actions)."
                )

            if hint:
                prompt += f" Additional context: {hint}"

            wrapped_prompt = textwrap.fill(
                prompt,
                width=100,
                initial_indent="💬 Agent Prompt: ",
                subsequent_indent="                 ",
            )
            click.secho(wrapped_prompt, fg="blue")

            response = await agent.chat(prompt)

            text_buffer: list[str] = []

            def _flush_text() -> None:
                if text_buffer:
                    console.print(Markdown("".join(text_buffer)))
                    text_buffer.clear()

            async for chunk in response.chunks:
                match chunk:
                    case types.Thought(text=text) if text.strip():
                        _flush_text()
                        console.print(
                            Markdown(text.strip()),
                            style="thinking",
                            width=100,
                        )
                    case types.Text(text=text):
                        text_buffer.append(text)
                    case types.ToolCall(name=name, args=args):
                        _flush_text()
                        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                        if len(args_str) > 80:
                            args_str = args_str[:77] + "..."
                        console.print(
                            f"  🔧 [tool.name]{name}[/tool.name]"
                            f"([tool.args]{args_str}[/tool.args])",
                        )
                    case types.ToolResult(name=name, error=error) if error:
                        _flush_text()
                        console.print(
                            f"  ❌ [tool.fail]{name} failed: {error}[/tool.fail]",
                        )
                    case types.ToolResult(name=name):
                        _flush_text()
                        console.print(f"  ✅ [tool.ok]{name}[/tool.ok]")
            _flush_text()

            console.print(
                f"\n[bold green]✨ Agent execution completed for {repo}.[/bold green]",
            )

            usage = agent.conversation.total_usage
            console.print(
                Panel(
                    f"Prompt: {usage.prompt_token_count:,}  |  "
                    f"Output: {usage.candidates_token_count:,}  |  "
                    f"Thinking: {usage.thoughts_token_count:,}  |  "
                    f"Total: {usage.total_token_count:,}",
                    title=f"Token Usage — {repo}",
                    style="dim cyan",
                ),
            )
    finally:
        await client.close()
        cleanup_sandbox = getattr(run_command, "cleanup", None)
        if cleanup_sandbox:
            cleanup_sandbox()
        await asyncio.to_thread(_cleanup_workspace, workspace_tmp)


async def run_agent(
    owner: str,
    concurrency: int,
    max_attempts: int,
    repo: str | None = None,
    *,
    dry_run: bool = False,
    auto_merge: bool = False,
    verify_all: bool = False,
    standalone_fix: bool = False,
    review_wait: int = 0,
    hint: str | None = None,
    no_sandbox: bool = False,
) -> None:
    """Run dependency-director triage and patch execution across repositories."""
    click.secho("⚙️  Initializing dependency-director configuration...", fg="cyan")
    settings = Settings()
    if no_sandbox:
        settings.no_sandbox = True

    if verify_all and settings.no_sandbox:
        click.secho(
            "❌ Error: --verify-all is not allowed in --no-sandbox mode.\n"
            "   Local test verification requires OS-level sandboxing (sandbox-runtime) "
            "to safely execute untrusted package dependencies.",
            fg="red",
            bold=True,
        )
        sys.exit(1)

    if settings.vertex:
        if not settings.gemini_api_key and not (
            settings.google_cloud_project and settings.google_cloud_location
        ):
            click.secho(
                "❌ Vertex AI mode requires GOOGLE_CLOUD_PROJECT and "
                "GOOGLE_CLOUD_LOCATION (or a GEMINI_API_KEY for Express Mode).",
                fg="red",
                bold=True,
            )
            sys.exit(1)
    elif not settings.gemini_api_key:
        click.secho(
            "❌ GEMINI_API_KEY is required. Get one at "
            "https://aistudio.google.com/app/api-keys\n"
            "   Or set GOOGLE_GENAI_USE_VERTEXAI=TRUE to use Vertex AI "
            "with Application Default Credentials.",
            fg="red",
            bold=True,
        )
        sys.exit(1)

    if not settings.github_token:
        click.secho(
            "⚠️  GITHUB_TOKEN is not set. API calls to private repos will fail.\n"
            "   Get a token: run 'gh auth token' or visit https://github.com/settings/tokens",
            fg="yellow",
        )

    if repo:
        click.secho(
            f"📦 Restricting triage run to repository: {repo}",
            fg="magenta",
            bold=True,
        )
        # Verify repository exists and is accessible before spawning the agent
        # Skip this check when running unit tests to avoid unauthorized network requests
        if "pytest" not in sys.modules:
            owner_name, repo_name = repo.split("/", 1)
            client = GitHubClient(token=settings.github_token)
            try:
                url = f"https://api.github.com/repos/{owner_name}/{repo_name}"
                response = await client.client.get(url, headers=client.headers)
                response.raise_for_status()
            except Exception as e:
                click.secho(
                    f"❌  Failed to access repository '{repo}': {e}",
                    fg="red",
                    bold=True,
                )
                await client.close()
                sys.exit(1)
            finally:
                await client.close()

        await run_agent_for_repo(
            repo,
            settings,
            max_attempts,
            dry_run=dry_run,
            auto_merge=auto_merge,
            verify_all=verify_all,
            standalone_fix=standalone_fix,
            review_wait=review_wait,
            hint=hint,
        )
    else:
        click.secho(f"🔍 Fetching repositories for owner '{owner}'...", fg="cyan")
        try:
            repos = await get_repositories(owner, settings.github_token)
        except Exception as e:  # noqa: BLE001
            click.secho(
                f"❌  Failed to fetch repositories for owner '{owner}': {e}",
                fg="red",
                bold=True,
            )
            sys.exit(1)

        if not repos:
            click.secho(
                f"ℹ️  No active repositories found for owner '{owner}'.",
                fg="yellow",
            )
            return

        click.secho(
            f"📚  Found {len(repos)} repositories to process with "
            f"concurrency={concurrency}.",
            fg="green",
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def worker(r: str) -> None:
            async with semaphore:
                click.secho(f"▶️  Starting processing for repository: {r}", fg="yellow")
                try:
                    await run_agent_for_repo(
                        r,
                        settings,
                        max_attempts,
                        dry_run=dry_run,
                        auto_merge=auto_merge,
                        verify_all=verify_all,
                        standalone_fix=standalone_fix,
                        review_wait=review_wait,
                        hint=hint,
                    )
                    click.secho(
                        f"✅  Finished processing for repository: {r}",
                        fg="green",
                        bold=True,
                    )
                except Exception as e:  # noqa: BLE001
                    click.secho(
                        f"❌  Error processing repository {r}: {e}",
                        fg="red",
                        bold=True,
                    )

        await asyncio.gather(*(worker(r) for r in repos))


def print_banner() -> None:
    """Print the dependency-director ascii banner and initialization messages."""
    banner = r"""
    ____                            __                      ____  _                __
   / __ \___  ____  ___  ____  ____/ /__  ____  _______  __/ __ \(_)_______  _____/ /_____  _____
  / / / / _ \/ __ \/ _ \/ __ \/ __  / _ \/ __ \/ ___/ / / / / / / / ___/ _ \/ ___/ __/ __ \/ ___/
 / /_/ /  __/ /_/ /  __/ / / / /_/ /  __/ / / / /__/ /_/ / /_/ / / /  /  __/ /__/ /_/ /_/ / /
/_____/\___/ .___/\___/_/ /_/\__,_/\___/_/ /_/\___/\__, /_____/_/_/   \___/\___/\__/\____/_/
          /_/                                     /____/
"""
    console.print(banner, style="banner", highlight=False)
    console.print("Autonomous Dependency Triage Agent", style="status")
    console.rule(style="dim")


@click.command()
@click.argument("target", required=False, default=None)
@click.option(
    "--concurrency",
    "-c",
    type=int,
    help="Number of concurrent repository scans (overrides env).",
)
@click.option(
    "--max-attempts",
    "-m",
    type=int,
    help="Max edit/test loops per failure (overrides env).",
)
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Simulate execution without merging or pushing fixes.",
)
@click.option(
    "--auto-merge",
    "-a",
    is_flag=True,
    help="Enable native GitHub auto-merge on any created patch PRs.",
)
@click.option(
    "--verify-all",
    "-v",
    is_flag=True,
    help=(
        "Force local test verification of all PRs "
        "(including green ones) before merging."
    ),
)
@click.option(
    "--standalone-fix",
    is_flag=True,
    help=(
        "Create fixes on a new branch with a separate PR "
        "instead of pushing to the original dependency update branch."
    ),
)
@click.option(
    "--review-wait",
    "-w",
    type=int,
    help=(
        "Minutes to wait for review comments after pushing a fix "
        "(0 = disabled, overrides env)."
    ),
)
@click.option(
    "--hint",
    "-H",
    type=str,
    default=None,
    help=(
        "Extra context appended to the agent prompt "
        "(e.g., 'skip PR #5, known upstream issue')."
    ),
)
@click.option(
    "--no-sandbox",
    is_flag=True,
    help="Disable sandbox-runtime (srt) sandboxing (runs with host privileges).",
)
def cli(
    target: str | None,
    concurrency: int | None,
    max_attempts: int | None,
    *,
    dry_run: bool,
    auto_merge: bool,
    verify_all: bool,
    standalone_fix: bool,
    review_wait: int | None,
    hint: str | None,
    no_sandbox: bool,
) -> None:
    """dependency-director: Autonomous dependency triage and patching agent.

    TARGET is a GitHub user/org (scan all repos) or owner/repo (scan one repo).
    If omitted, uses DEPDIRECTOR_OWNER from env.
    """
    print_banner()
    settings = Settings()

    no_sandbox_val = no_sandbox or settings.no_sandbox

    if verify_all and no_sandbox_val:
        click.secho(
            "❌ Error: --verify-all is not allowed in --no-sandbox mode.\n"
            "   Local test verification requires OS-level sandboxing (sandbox-runtime) "
            "to safely execute untrusted package dependencies.",
            fg="red",
            bold=True,
        )
        sys.exit(1)

    if not no_sandbox_val:
        if not is_srt_available():
            srt_installed = shutil.which("srt") is not None
            if srt_installed:
                if sys.platform == "linux" and not is_ripgrep_available():
                    click.secho(
                        "❌ Error: sandbox-runtime (srt) is installed, but ripgrep (rg) is missing.\n"
                        "   srt requires ripgrep on Linux to enforce write restrictions.\n"
                        "   Please install ripgrep (e.g. sudo apt-get install ripgrep).",
                        fg="red",
                        bold=True,
                    )
                else:
                    click.secho(
                        "❌ Error: sandbox-runtime (srt) is installed but not functioning properly.\n"
                        "   Ensure bubblewrap and socat are installed on Linux.",
                        fg="red",
                        bold=True,
                    )
            else:
                click.secho(
                    "❌ Error: sandbox-runtime (srt) is not available.\n"
                    "   Install it with: npm install -g @anthropic-ai/sandbox-runtime\n"
                    "   Or pass --no-sandbox to run without sandboxing (run cautiously).",
                    fg="red",
                    bold=True,
                )
            sys.exit(1)

    if no_sandbox_val:
        click.secho(
            "⚠️  Warning: Running in --no-sandbox mode. Commands will run with host privileges.",
            fg="yellow",
        )

    if target and "/" in target:
        owner, repo_part = target.split("/", 1)
        if not repo_part:
            msg = (
                f"Invalid target '{target}'. Use 'owner/repo' format "
                "or just 'owner' to scan all repos."
            )
            raise click.UsageError(
                msg,
            )
        repo: str | None = target
    elif target:
        owner = target
        repo = None
    elif settings.owner:
        owner = settings.owner
        repo = None
    else:
        msg = (
            "No target specified. Provide a GitHub user/org or owner/repo "
            "as an argument, or set DEPDIRECTOR_OWNER in your environment."
        )
        raise click.UsageError(
            msg,
        )

    concurrency_val = concurrency if concurrency is not None else settings.concurrency
    max_attempts_val = (
        max_attempts if max_attempts is not None else settings.max_fix_attempts
    )
    review_wait_val = review_wait if review_wait is not None else settings.review_wait

    try:
        asyncio.run(
            run_agent(
                owner,
                concurrency_val,
                max_attempts_val,
                repo,
                dry_run=dry_run,
                auto_merge=auto_merge,
                verify_all=verify_all,
                standalone_fix=standalone_fix,
                review_wait=review_wait_val,
                hint=hint,
                no_sandbox=no_sandbox_val,
            ),
        )
    except* KeyboardInterrupt:
        click.secho(
            "\n🛑 Execution interrupted by user. Shutting down...",
            fg="yellow",
            bold=True,
        )
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    cli()
